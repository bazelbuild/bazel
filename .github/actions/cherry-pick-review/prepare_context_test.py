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
"""Unit tests for prepare_context.py."""

import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import prepare_context


class PrepareContextTest(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmp_dir.name)
        self.old_cwd = os.getcwd()
        os.chdir(self.root)
        self._init_git_repo()

    def tearDown(self):
        os.chdir(self.old_cwd)
        self.tmp_dir.cleanup()

    def _git(self, *args):
        subprocess.run(["git"] + list(args), cwd=self.root, check=True, capture_output=True, text=True)

    def _git_out(self, *args):
        return subprocess.run(
            ["git"] + list(args), cwd=self.root, check=True, capture_output=True, text=True
        ).stdout.strip()

    def _init_git_repo(self):
        self._git("init", "-b", "master")
        self._git("config", "user.name", "Test User")
        self._git("config", "user.email", "test@example.com")
        (self.root / "src").mkdir()
        (self.root / "src" / "Foo.java").write_text(
            "class Foo {\n  int a() { return 1; }\n  int b() { return 2; }\n}\n",
            encoding="utf-8",
        )
        self._git("add", ".")
        self._git("commit", "-m", "Initial commit")
        # Create release-8.9.0 branch
        self._git("branch", "release-8.9.0")
        self.base_sha = self._git_out("rev-parse", "release-8.9.0")

    def test_skips_non_release_branch_pr(self):
        def fake_gh_api(endpoint, paginate=False, slurp=False):
            return {
                "state": "open",
                "draft": False,
                "base": {"ref": "master", "sha": self.base_sha},
                "head": {"ref": "feat", "sha": "a" * 40},
            }

        out_file = self.root / "output.txt"
        env = {
            "REPOSITORY": "bazelbuild/bazel",
            "EVENT_NAME": "issue_comment",
            "PR_NUMBER": "100",
            "GITHUB_OUTPUT": str(out_file),
        }
        with patch.dict(os.environ, env, clear=False), patch.object(prepare_context, "gh_api", side_effect=fake_gh_api):
            self.assertEqual(prepare_context.main(), 0)
            self.assertIn("should_run=false", out_file.read_text(encoding="utf-8"))

    def test_clean_cherry_pick_classification(self):
        # Commit a fix on master
        (self.root / "src" / "Foo.java").write_text(
            "class Foo {\n  int a() { return 10; }\n  int b() { return 2; }\n}\n",
            encoding="utf-8",
        )
        self._git("add", ".")
        self._git("commit", "-m", "Fix Foo.a return value (#500)\n\nRELNOTES: None\nPiperOrigin-RevId: 123456789")
        mainline_sha = self._git_out("rev-parse", "master")

        # Cherry-pick cleanly onto a PR branch off release-8.9.0
        self._git("checkout", "-b", "cp-clean", "release-8.9.0")
        self._git("cherry-pick", "-x", mainline_sha)
        head_sha = self._git_out("rev-parse", "cp-clean")
        commit_msg = self._git_out("show", "-s", "--format=%B", head_sha)
        patch_text = "@@ -1,4 +1,4 @@\n class Foo {\n-  int a() { return 1; }\n+  int a() { return 10; }\n   int b() { return 2; }\n }"
        self._git("checkout", "master")

        def fake_gh_api(endpoint, paginate=False, slurp=False):
            if endpoint == "repos/bazelbuild/bazel/pulls/101":
                return {
                    "state": "open",
                    "draft": False,
                    "title": "[8.9.0] Fix Foo.a return value",
                    "body": f"Fixes Foo.a\n(cherry picked from commit {mainline_sha})",
                    "changed_files": 1,
                    "user": {"login": "wyv"},
                    "base": {"ref": "release-8.9.0", "sha": self.base_sha},
                    "head": {"ref": "cp-clean", "sha": head_sha},
                }
            if "pulls/101/commits" in endpoint:
                return [[{"sha": head_sha, "commit": {"message": commit_msg}, "author": {"login": "wyv"}}]]
            if "pulls/101/files" in endpoint:
                return [
                    [
                        {
                            "filename": "src/Foo.java",
                            "status": "modified",
                            "additions": 1,
                            "deletions": 1,
                            "patch": patch_text,
                        }
                    ]
                ]
            if "issues/101/comments" in endpoint or "pulls/101/reviews" in endpoint or "pulls/101/comments" in endpoint:
                return [[]]
            raise AssertionError(f"Unexpected endpoint: {endpoint}")

        out_file = self.root / "output.txt"
        env = {
            "REPOSITORY": "bazelbuild/bazel",
            "EVENT_NAME": "pull_request_target",
            "EVENT_ACTION": "opened",
            "PR_NUMBER": "101",
            "GITHUB_OUTPUT": str(out_file),
        }
        with patch.dict(os.environ, env, clear=False), patch.object(prepare_context, "gh_api", side_effect=fake_gh_api):
            self.assertEqual(prepare_context.main(), 0)

        ctx = json.loads((self.root / ".gemini" / "context.json").read_text(encoding="utf-8"))
        self.assertEqual(ctx["overall_cherry_pick_classification"], "CLEAN")
        self.assertEqual(len(ctx["mainline_commits"]), 1)
        self.assertEqual(ctx["mainline_commits"][0]["sha"], mainline_sha)
        self.assertTrue(ctx["mainline_commits"][0]["applied_cleanly_on_base"])
        self.assertEqual(ctx["file_comparisons"][0]["classification"], "identical")
        self.assertIsNone(ctx["vs_mainline_diff_path"])
        self.assertEqual(ctx["breaking_change_signals"], [])
        self.assertFalse(ctx["has_existing_gemini_review"])

    def test_conflict_and_extra_file_and_breaking_signal_detection(self):
        # First make an intermediate change on master that is NOT on release-8.9.0
        (self.root / "src" / "Foo.java").write_text(
            "class Foo {\n  int a(String ctx) { return 1; }\n  int b() { return 2; }\n}\n",
            encoding="utf-8",
        )
        self._git("add", ".")
        self._git("commit", "-m", "Refactor Foo.a signature on master")

        # Now make the target commit on master touching the refactored line with RELNOTES[INC]
        (self.root / "src" / "Foo.java").write_text(
            "class Foo {\n  int a(String ctx) { return 42; }\n  int b() { return 2; }\n}\n",
            encoding="utf-8",
        )
        self._git("add", ".")
        self._git(
            "commit",
            "-m",
            "Change Foo.a behavior\n\nRELNOTES[INC]: Foo.a now returns 42 instead of 1.\nPiperOrigin-RevId: 987654321",
        )
        mainline_sha = self._git_out("rev-parse", "master")

        # Create a PR branch on release-8.9.0 that resolves the conflict in Foo.java AND adds an unrelated file
        self._git("checkout", "-b", "cp-conflict-extra", "release-8.9.0")
        (self.root / "src" / "Foo.java").write_text(
            "class Foo {\n  int a() { return 42; }\n  int b() { return 2; }\n}\n",
            encoding="utf-8",
        )
        (self.root / "src" / "Sneaky.java").write_text("class Sneaky {}\n", encoding="utf-8")
        self._git("add", ".")
        self._git("commit", "-m", f"[8.9.0] Change Foo.a behavior\n\n(cherry picked from commit {mainline_sha})")
        head_sha = self._git_out("rev-parse", "cp-conflict-extra")
        commit_msg = self._git_out("show", "-s", "--format=%B", head_sha)
        self._git("checkout", "master")

        def fake_gh_api(endpoint, paginate=False, slurp=False):
            if endpoint == "repos/bazelbuild/bazel/pulls/102":
                return {
                    "state": "open",
                    "draft": False,
                    "title": "[8.9.0] Change Foo.a behavior",
                    "body": f"(cherry picked from commit {mainline_sha})",
                    "changed_files": 2,
                    "user": {"login": "wyv"},
                    "base": {"ref": "release-8.9.0", "sha": self.base_sha},
                    "head": {"ref": "cp-conflict-extra", "sha": head_sha},
                }
            if "pulls/102/commits" in endpoint:
                return [[{"sha": head_sha, "commit": {"message": commit_msg}, "author": {"login": "wyv"}}]]
            if "pulls/102/files" in endpoint:
                return [
                    [
                        {
                            "filename": "src/Foo.java",
                            "status": "modified",
                            "additions": 1,
                            "deletions": 1,
                            "patch": "@@ -1,4 +1,4 @@\n class Foo {\n-  int a() { return 1; }\n+  int a() { return 42; }\n   int b() { return 2; }\n }",
                        },
                        {
                            "filename": "src/Sneaky.java",
                            "status": "added",
                            "additions": 1,
                            "deletions": 0,
                            "patch": "@@ -0,0 +1 @@\n+class Sneaky {}",
                        },
                    ]
                ]
            if "issues/102/comments" in endpoint or "pulls/102/reviews" in endpoint or "pulls/102/comments" in endpoint:
                return [[]]
            raise AssertionError(f"Unexpected endpoint: {endpoint}")

        out_file = self.root / "output.txt"
        env = {
            "REPOSITORY": "bazelbuild/bazel",
            "EVENT_NAME": "pull_request_target",
            "EVENT_ACTION": "opened",
            "PR_NUMBER": "102",
            "GITHUB_OUTPUT": str(out_file),
        }
        with patch.dict(os.environ, env, clear=False), patch.object(prepare_context, "gh_api", side_effect=fake_gh_api):
            self.assertEqual(prepare_context.main(), 0)

        ctx = json.loads((self.root / ".gemini" / "context.json").read_text(encoding="utf-8"))
        self.assertEqual(ctx["overall_cherry_pick_classification"], "MODIFIED_OR_EXTRA_CHANGES")
        self.assertFalse(ctx["mainline_commits"][0]["applied_cleanly_on_base"])
        self.assertEqual(ctx["mainline_commits"][0]["conflicted_files"], ["src/Foo.java"])
        by_path = {fc["path"]: fc["classification"] for fc in ctx["file_comparisons"]}
        self.assertEqual(by_path["src/Foo.java"], "adapted_with_conflict")
        self.assertEqual(by_path["src/Sneaky.java"], "extra_in_pr")
        self.assertTrue(any("RELNOTES[INC]" in s for s in ctx["breaking_change_signals"]))
        self.assertIsNotNone(ctx["vs_mainline_diff_path"])
        vs_diff = pathlib.Path(ctx["vs_mainline_diff_path"]).read_text(encoding="utf-8")
        self.assertIn("CLASSIFICATION: adapted_with_conflict", vs_diff)
        self.assertIn("CLASSIFICATION: extra_in_pr", vs_diff)
        self.assertIn("<<<<<<<", vs_diff)

    def test_rejects_unsafe_paths_and_symlinks(self):
        for bad in (
            "",
            "/etc/passwd",
            "../secret.txt",
            "src/../../secret.txt",
            "src/./Foo.java",
            ".git/config",
            "src/.git/hooks/pre-commit",
            "-flag-injection",
            ":(top)src/Foo.java",
            "src\\Foo.java",
            "src/Foo\x00.java",
            "src/dir/",
        ):
            self.assertFalse(prepare_context.is_safe_repo_rel_path(bad), f"Should reject {bad!r}")
            self.assertIsNone(prepare_context.resolve_path_within(self.root, bad), f"Should reject {bad!r}")

        self.assertIsNone(prepare_context.is_valid_release_branch("release-1.0..2.0"))
        self.assertIsNone(prepare_context.is_valid_release_branch("release-.hidden"))
        self.assertIsNone(prepare_context.is_valid_release_branch("release-8.9.0.lock"))
        self.assertEqual(prepare_context.is_valid_release_branch("release-8.9.0"), "8.9.0")

        # Symlink inside base_dir must be rejected by resolve_path_within and safe_write_staged_file
        stage_dir = self.root / "stage_test"
        stage_dir.mkdir()
        (stage_dir / "link_dir").symlink_to("/tmp")
        self.assertIsNone(prepare_context.resolve_path_within(stage_dir, "link_dir/escape.txt"))
        self.assertIsNone(prepare_context.safe_write_staged_file(stage_dir, "link_dir/escape.txt", "pwn"))

        # Git symlink (mode 120000) must be rejected by read_git_regular_file_blob
        (self.root / "src" / "evil_link").symlink_to("/etc/hosts")
        self._git("add", "src/evil_link")
        self._git("commit", "-m", "Add symlink")
        sha_with_link = self._git_out("rev-parse", "HEAD")
        self.assertIsNone(prepare_context.read_git_regular_file_blob(sha_with_link, "src/evil_link"))
        self.assertIsNotNone(prepare_context.read_git_regular_file_blob(sha_with_link, "src/Foo.java"))

    def test_prior_review_delta_detects_unchanged_rebase_and_modified_followup(self):
        # Commit 1 on PR branch
        self._git("checkout", "-b", "pr-branch", "release-8.9.0")
        (self.root / "src" / "Foo.java").write_text(
            "class Foo {\n  int a() { return 10; }\n  int b() { return 2; }\n}\n",
            encoding="utf-8",
        )
        self._git("add", ".")
        self._git("commit", "-m", "PR commit 1")
        rev1_sha = self._git_out("rev-parse", "HEAD")
        pr_patches_v1 = {"src/Foo.java": "@@ -1,4 +1,4 @@\n class Foo {\n-  int a() { return 1; }\n+  int a() { return 10; }\n   int b() { return 2; }\n }"}

        reviews = [
            {
                "id": 1,
                "state": "CHANGES_REQUESTED",
                "commit_id": rev1_sha,
                "submitted_at": "2026-11-12T20:00:00Z",
                "body": "## 📋 Gemini Cherry-Pick Review Summary\n\n**Verdict:** ❌ Changes Requested",
            }
        ]
        inline_comments = [
            {
                "id": 10,
                "user": {"login": "bazel-io"},
                "path": "src/Foo.java",
                "line": 2,
                "created_at": "2026-11-12T20:00:00Z",
                "body": "🔴 Some issue",
            }
        ]

        # Case 1: Rebase with identical patch changes
        (self.root / "src" / "Foo.java").write_text(
            "class Foo {\n  int a() { return 10; }\n  int b() { return 2; }\n}\n",
            encoding="utf-8",
        )
        self._git("commit", "--amend", "-m", "PR commit 1 (rebased)")
        rev1_rebased_sha = self._git_out("rev-parse", "HEAD")

        has_existing, last_review, unchanged, since_path, comments_out = prepare_context.analyze_prior_review_delta(
            pr_reviews=reviews,
            pr_inline_comments=inline_comments,
            base_ref="release-8.9.0",
            base_sha=self.base_sha,
            head_sha=rev1_rebased_sha,
            pr_patches_by_file=pr_patches_v1,
            stage_dir=self.root / "gemini-artifacts" / "pr",
        )
        self.assertTrue(has_existing)
        self.assertEqual(last_review["commit_id"], rev1_sha)
        self.assertTrue(unchanged)
        self.assertIsNotNone(since_path)
        self.assertEqual(len(comments_out), 1)

        # Case 2: Follow-up commit modifies the PR patch
        (self.root / "src" / "Foo.java").write_text(
            "class Foo {\n  int a() { return 20; }\n  int b() { return 2; }\n}\n",
            encoding="utf-8",
        )
        self._git("add", ".")
        self._git("commit", "-m", "PR commit 2")
        rev2_sha = self._git_out("rev-parse", "HEAD")
        pr_patches_v2 = {"src/Foo.java": "@@ -1,4 +1,4 @@\n class Foo {\n-  int a() { return 1; }\n+  int a() { return 20; }\n   int b() { return 2; }\n }"}

        _, _, unchanged2, since_path2, _ = prepare_context.analyze_prior_review_delta(
            pr_reviews=reviews,
            pr_inline_comments=inline_comments,
            base_ref="release-8.9.0",
            base_sha=self.base_sha,
            head_sha=rev2_sha,
            pr_patches_by_file=pr_patches_v2,
            stage_dir=self.root / "gemini-artifacts" / "pr",
        )
        self.assertFalse(unchanged2)
        self.assertIn("return 20;", pathlib.Path(since_path2).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
