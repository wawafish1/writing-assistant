from __future__ import annotations

import argparse
import sys

from app.config import get_settings
from app.db import Database
from app.llm import LlmError
from app.style import analyze_style_with_llm, build_heuristic_profile, generate_article_with_llm
from app.x_api import XApiError, XClient


def main() -> int:
    parser = argparse.ArgumentParser(description="写作助手")
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect_parser = subparsers.add_parser("collect", help="Collect recent posts")
    collect_parser.add_argument("username")
    collect_parser.add_argument("--limit", type=int, default=500)
    collect_parser.add_argument("--include-replies", action="store_true")
    collect_parser.add_argument("--include-retweets", action="store_true")

    analyze_parser = subparsers.add_parser("analyze", help="Analyze writing style")
    analyze_parser.add_argument("username")
    analyze_parser.add_argument("--sample-limit", type=int, default=120)
    analyze_parser.add_argument("--heuristic", action="store_true")
    analyze_parser.add_argument("--model")

    generate_parser = subparsers.add_parser("generate", help="Generate an original article")
    generate_parser.add_argument("username")
    generate_parser.add_argument("brief")
    generate_parser.add_argument("--platform")
    generate_parser.add_argument("--target-length")
    generate_parser.add_argument("--extra-constraints")
    generate_parser.add_argument("--model")

    args = parser.parse_args()
    settings = get_settings()
    database = Database(settings.database_path)

    try:
        if args.command == "collect":
            client = XClient(settings.x_bearer_token or "")
            user = client.get_user(args.username)
            posts = client.fetch_recent_posts(
                user_id=user["id"],
                limit=args.limit,
                exclude_replies=not args.include_replies,
                exclude_retweets=not args.include_retweets,
            )
            username = user["username"].lower()
            database.upsert_author(username, user["id"], user.get("name"))
            changed = database.upsert_posts(username, posts)
            print(f"Fetched {len(posts)} posts for @{username}; rows changed: {changed}.")
            return 0

        if args.command == "analyze":
            posts = database.get_posts(args.username, limit=args.sample_limit)
            if not posts:
                print("No posts found. Run collect first.", file=sys.stderr)
                return 1
            model = args.model or settings.openai_model
            if args.heuristic:
                profile = build_heuristic_profile(args.username, posts)
                profile_model = "heuristic"
            else:
                profile = analyze_style_with_llm(args.username, posts, model=model)
                profile_model = model
            database.save_style_profile(args.username, profile, len(posts), profile_model)
            print(f"Saved style profile for @{args.username.lower()} from {len(posts)} posts.")
            return 0

        if args.command == "generate":
            profile_record = database.get_style_profile(args.username)
            if not profile_record:
                print("No style profile found. Run analyze first.", file=sys.stderr)
                return 1
            model = args.model or settings.openai_model
            draft = generate_article_with_llm(
                username=args.username,
                style_profile=profile_record["profile"],
                brief=args.brief,
                platform=args.platform,
                target_length=args.target_length,
                extra_constraints=args.extra_constraints,
                model=model,
            )
            generation_id = database.save_generation(
                username=args.username,
                brief=args.brief,
                platform=args.platform,
                target_length=args.target_length,
                draft=draft,
                model=model,
            )
            print(f"Generation #{generation_id}\n")
            print(draft)
            return 0
    except (XApiError, LlmError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
