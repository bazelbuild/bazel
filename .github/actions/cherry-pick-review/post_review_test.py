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
        self._write_context()

    def _write_context(self, **overrides):
        ctx = {
            "repository": "bazelbuild/bazel",
            "owner": "bazelbuild",
            "repo": "bazel",
            "workspace_root": str(self.root),
            "pull_request_number": 42,
            "head_sha": "d" * 40,
            "trigger_type": "automatic",
            "has_existing_gemini_review": False,
            "overall_cherry_pick_classification": "CLEAN",
            "changed_files": [{"path": "src/Foo.java", "status": "modified"}],
        }
        ctx.update(overrides)
        (self.root / ".gemini" / "context.json").write_text(json.dumps(ctx), encoding="utf-8")

    def tearDown(self):
        os.chdir(self.old_cwd)
        self.tmp_dir.cleanup()

    def test_posts_approve_review_for_clean_cherry_pick(self):
        review_obj = {
            "should_post": True,
            "decision": "APPROVE",
            "body": "## 📋 Gemini Cherry-Pick Review Summary\n\n**Verdict:** ✅ Approved",
            "comments": [],
        }
        (self.root / "gemini-artifacts" / "stdout.log").write_text(
            json.dumps({"response": json.dumps(review_obj), "status": "SUCCESS", "duration_seconds": 12.3}),
            encoding="utf-8",
        )
        posted = []

        def fake_post(endpoint, payload):
            posted.append((endpoint, payload))
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
            "body": "## 📋 Gemini Cherry-Pick Review Summary\n\n**Verdict:** ✅ Approved",
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

        def fake_post(endpoint, payload):
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
        self.assertIn("**Verdict:** ❌ Changes Requested", posted[0]["body"])
        self.assertIn("Automated Guardrail", posted[0]["body"])

    def test_self_review_http_422_falls_back_directly_to_comment(self):
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

        def fake_post(endpoint, payload):
            calls.append(payload["event"])
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

        with patch.object(post_review, "gh_api_post", side_effect=fake_post):
            self.assertEqual(post_review.main(), 0)

        self.assertEqual(calls, ["APPROVE", "COMMENT"])

    def test_should_post_false_skips_when_existing_review_present_but_posts_on_initial_or_manual(self):
        review_obj = {
            "should_post": False,
            "skip_reason": "PR was rebased with no substantive changes.",
            "decision": "APPROVE",
            "body": "## 📋 Gemini Cherry-Pick Review Summary",
            "comments": [],
        }
        (self.root / "gemini-artifacts" / "stdout.log").write_text(
            json.dumps({"response": json.dumps(review_obj)}),
            encoding="utf-8",
        )
        posted = []

        def fake_post(endpoint, payload):
            posted.append(payload)
            res = MagicMock()
            res.returncode = 0
            res.stdout = json.dumps({"html_url": "https://github.com/bazelbuild/bazel/pull/42#pullrequestreview-4"})
            res.stderr = ""
            return res

        # 1) When has_existing_gemini_review is True and trigger_type is automatic -> skips posting
        self._write_context(has_existing_gemini_review=True, trigger_type="automatic")
        summary_file = self.root / "step_summary.md"
        with patch.dict(os.environ, {"GITHUB_STEP_SUMMARY": str(summary_file)}), patch.object(
            post_review, "gh_api_post", side_effect=fake_post
        ):
            self.assertEqual(post_review.main(), 0)
        self.assertEqual(len(posted), 0)
        self.assertIn("Skipped posting duplicate review", summary_file.read_text(encoding="utf-8"))

        # 2) When has_existing_gemini_review is False -> guardrail overrides should_post=False and posts
        self._write_context(has_existing_gemini_review=False, trigger_type="automatic")
        with patch.object(post_review, "gh_api_post", side_effect=fake_post):
            self.assertEqual(post_review.main(), 0)
        self.assertEqual(len(posted), 1)

    def test_strips_runner_file_links_and_rejects_traversal_paths(self):
        review_obj = {
            "decision": "COMMENT",
            "body": "See [Foo.java](file:///home/runner/work/bazel/bazel/gemini-artifacts/pr/head/src/Foo.java#L1-L5).",
            "comments": [
                {
                    "path": "file:///home/runner/work/bazel/bazel/gemini-artifacts/pr/head/src/Foo.java#L2",
                    "line": 2,
                    "side": "RIGHT",
                    "body": "🟡 Check [Bar.java](file:///tmp/Bar.java).",
                },
                {
                    "path": "../../etc/passwd",
                    "line": 1,
                    "side": "RIGHT",
                    "body": "🔴 Traversal attempt",
                },
            ],
        }
        (self.root / "gemini-artifacts" / "stdout.log").write_text(
            json.dumps({"response": json.dumps(review_obj)}),
            encoding="utf-8",
        )
        posted = []

        def fake_post(endpoint, payload):
            posted.append(payload)
            res = MagicMock()
            res.returncode = 0
            res.stdout = json.dumps({"html_url": "https://github.com/bazelbuild/bazel/pull/42#pullrequestreview-5"})
            res.stderr = ""
            return res

        with patch.object(post_review, "gh_api_post", side_effect=fake_post):
            self.assertEqual(post_review.main(), 0)

        self.assertEqual(len(posted), 1)
        self.assertNotIn("file://", posted[0]["body"])
        self.assertIn("`Foo.java`", posted[0]["body"])
        self.assertEqual(len(posted[0]["comments"]), 1)
        self.assertEqual(posted[0]["comments"][0]["path"], "src/Foo.java")
        self.assertNotIn("file://", posted[0]["comments"][0]["body"])
        self.assertIn("`Bar.java`", posted[0]["comments"][0]["body"])


if __name__ == "__main__":
    unittest.main()
