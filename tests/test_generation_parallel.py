from __future__ import annotations

import threading
import unittest
from unittest.mock import Mock, patch

from app.main import generate_model_drafts


class ParallelGenerationTests(unittest.TestCase):
    def test_models_generate_concurrently_and_save_results(self) -> None:
        jobs = [
            {
                "name": "gpt",
                "provider": "openai",
                "model": "gpt-test",
                "profile_username": "writer__gpt",
            },
            {
                "name": "deepseek",
                "provider": "deepseek",
                "model": "deepseek-test",
                "profile_username": "writer__deepseek",
            },
        ]
        barrier = threading.Barrier(2)

        def generate(**kwargs: object) -> str:
            barrier.wait(timeout=2)
            return f"draft from {kwargs['model']}"

        fake_db = Mock()
        fake_db.get_style_profile.side_effect = lambda username: {
            "profile": {"username": username}
        }
        fake_db.save_generation.side_effect = [101, 102]

        with (
            patch("app.main.model_jobs_for_style", return_value=jobs),
            patch("app.main.generate_article_with_llm", side_effect=generate),
            patch("app.main.db", fake_db),
        ):
            results = generate_model_drafts(
                style_name="writer",
                brief="brief",
                platform="X",
                target_length="100字",
                extra_constraints=None,
            )

        self.assertTrue(results["gpt"]["ok"])
        self.assertTrue(results["deepseek"]["ok"])
        self.assertEqual(fake_db.save_generation.call_count, 2)


if __name__ == "__main__":
    unittest.main()
