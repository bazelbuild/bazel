#!/usr/bin/env python3
# Copyright 2026 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for post_review.py."""

import json
import os
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import post_review


class PostReviewTest(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmp_dir.name)
        self.old_cwd = os.getcwd()
        os.chdir(self.root)
        (self.root / ".gemini").mkdir()
        (self.root / "gemini-artifacts").mkdir()
        (self.root / ".gemini" / "context.json").write_text(
            json.dumps(
                {
                    "repository": "bazelbuild/bazel",
                    "owner": "bazelbuild",
                    "repo": "bazel",
                    "pull_request_number": 42,
                    "head_sha": "d" * 40,
                    "overall_cherry_pick_classification": "CLEAN",
                    "changed_files": [{"path": "src/Foo.java", "status": "modified"}],
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self):
        os.chdir(self.old_cwd)
        self.tmp_dir.cleanup()

    def test_posts_approve_review_for_clean_cherry_pick(self):
        review_obj = {
            "decision": "APPROVE",
            "body": "## 📋 Gemini Cherry-Pick Review Summary\n\n**Verdict:** ✅ Approved",
            "comments": [],
        }
        (self.root / "gemini-artifacts" / "stdout.log").write_text(
            json.dumps({"response": json.dumps(review_obj), "status": "SUCCESS", "duration_seconds": 12.3}),
            encoding="utf-8",
        )
        posted = []

        def fake_post(endpoint, payload, token=None):
            posted.append((endpoint, payload, token))
            res = MagicMock()
            res.returncode = 0
            res.stdout = json.dumps({"html_url": "https://github.com/bazelbuild/bazel/pull/42#pullrequestreview-1"})
            res.stderr = ""
            return res

        summary_file = self.root / "step_summary.md"
        with patch.dict(os.environ, {"GITHUB_STEP_SUMMARY": str(summary_file)}), patch.object(
            post_review, "gh_api_post", side_effect=fake_post
        ):
            self.assertEqual(post_review.main(), 0)

        self.assertEqual(len(posted), 1)
        self.assertEqual(posted[0][1]["event"], "APPROVE")
        self.assertIn("✅ Approved PR", summary_file.read_text(encoding="utf-8"))

    def test_safety_guardrail_downgrades_approve_when_critical_comment_present(self):
        review_obj = {
            "decision": "APPROVE",
            "body": "## 📋 Gemini Cherry-Pick Review Summary",
            "comments": [
                {
                    "path": "src/Foo.java",
                    "line": 2,
                    "side": "RIGHT",
                    "body": "🔴 Unrelated change sneaked in.",
                }
            ],
        }
        (self.root / "gemini-artifacts" / "stdout.log").write_text(
            json.dumps({"response": json.dumps(review_obj)}),
            encoding="utf-8",
        )
        posted = []

        def fake_post(endpoint, payload, token=None):
            posted.append(payload)
            res = MagicMock()
            res.returncode = 0
            res.stdout = json.dumps({"html_url": "https://github.com/bazelbuild/bazel/pull/42#pullrequestreview-2"})
            res.stderr = ""
            return res

        with patch.object(post_review, "gh_api_post", side_effect=fake_post):
            self.assertEqual(post_review.main(), 0)

        self.assertEqual(len(posted), 1)
        self.assertEqual(posted[0]["event"], "REQUEST_CHANGES")

    def test_self_review_http_422_retries_with_fallback_token_then_comment(self):
        review_obj = {
            "decision": "APPROVE",
            "body": "## 📋 Gemini Cherry-Pick Review Summary",
            "comments": [],
        }
        (self.root / "gemini-artifacts" / "stdout.log").write_text(
            json.dumps({"response": json.dumps(review_obj)}),
            encoding="utf-8",
        )
        calls = []

        def fake_post(endpoint, payload, token=None):
            calls.append((payload["event"], token))
            res = MagicMock()
            if payload["event"] == "APPROVE":
                res.returncode = 1
                res.stdout = '{"message": "Unprocessable Entity"}'
                res.stderr = "gh: Can not approve your own pull request (HTTP 422)"
            else:
                res.returncode = 0
                res.stdout = json.dumps({"html_url": "https://github.com/bazelbuild/bazel/pull/42#pullrequestreview-3"})
                res.stderr = ""
            return res

        env = {"GITHUB_TOKEN": "token-primary", "FALLBACK_GITHUB_TOKEN": "token-fallback"}
        with patch.dict(os.environ, env, clear=False), patch.object(post_review, "gh_api_post", side_effect=fake_post):
            self.assertEqual(post_review.main(), 0)

        self.assertEqual(calls, [("APPROVE", None), ("APPROVE", "token-fallback"), ("COMMENT", None)])


if __name__ == "__main__":
    unittest.main()
