from __future__ import annotations

import argparse
import sys

from app.config import get_settings
from app.db import Database
from app.llm import LlmError
from app.style import analyze_style_with_llm, build_heuristic_profile, generate_article_with_llm


def main() -> int:
    parser = argparse.ArgumentParser(description="写作助手")
    subparsers = parser.add_subparsers(dest="command", required=True)

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
        if args.command == "analyze":
            posts = database.get_posts(args.username, limit=args.sample_limit)
            if not posts:
                print("No posts found. Import samples first.", file=sys.stderr)
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
    except LlmError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
