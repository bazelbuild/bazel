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

    def test_multi_commit_discovery_patch_release_and_flag_addition_vs_flip(self):
        self.assertFalse(prepare_context.is_patch_release_version("8.9.0"))
        self.assertTrue(prepare_context.is_patch_release_version("8.5.1"))
        self.assertTrue(prepare_context.is_patch_release_version("7.7.2"))

        # Create two commits on master
        (self.root / "src" / "Foo.java").write_text(
            "class Foo {\n  int a() { return 11; }\n  int b() { return 2; }\n}\n",
            encoding="utf-8",
        )
        self._git("add", "src/Foo.java")
        self._git("commit", "-m", "First mainline fix for Foo.a (#601)")
        sha1 = self._git_out("rev-parse", "HEAD")

        (self.root / "src" / "Bar.java").write_text("class Bar { int x() { return 7; } }\n", encoding="utf-8")
        self._git("add", "src/Bar.java")
        self._git("commit", "-m", "Second mainline fix for Bar.x (#602)")
        sha2 = self._git_out("rev-parse", "HEAD")

        # Create PR branch with two commits: first has trailer, second omits trailer (subject only)
        pr_commits = [
            {
                "sha": "1111111111111111111111111111111111111111",
                "commit": {
                    "message": f"First mainline fix for Foo.a (#601)\n\n(cherry picked from commit {sha1})"
                },
            },
            {
                "sha": "2222222222222222222222222222222222222222",
                "commit": {
                    "message": "[8.9.0] Second mainline fix for Bar.x"
                },
            },
        ]
        found = prepare_context.find_mainline_commits(
            pr_title="[8.9.0] Two cherry-picks",
            pr_body="",
            pr_commits=pr_commits,
            pr_comments=[],
            pr_author="wyv",
            mainline_ref="master",
            base_sha=self.base_sha,
        )
        self.assertEqual([sha for sha, _ in found], [sha1, sha2])

        # Adding a new default-off incompatible flag should NOT trigger a breaking signal
        signals_add_flag = prepare_context.extract_breaking_change_signals(
            mainline_commits_info=[],
            pr_title="[8.9.0] Add stub incompatible flag",
            pr_body="",
            pr_files=[
                {
                    "filename": "src/Options.java",
                    "patch": (
                        '@@ -10,0 +11,5 @@\n'
                        '+  @Option(\n'
                        '+      name = "incompatible_new_stub",\n'
                        '+      defaultValue = "false",\n'
                        '+      metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE})\n'
                    ),
                }
            ],
        )
        self.assertEqual(signals_add_flag, [])

        # Flipping an existing incompatible flag's defaultValue SHOULD trigger a breaking signal
        signals_flip_flag = prepare_context.extract_breaking_change_signals(
            mainline_commits_info=[],
            pr_title="[8.9.0] Flip incompatible flag",
            pr_body="",
            pr_files=[
                {
                    "filename": "src/Options.java",
                    "patch": (
                        '@@ -10,5 +10,5 @@\n'
                        '   @Option(\n'
                        '       name = "incompatible_existing_flag",\n'
                        '-      defaultValue = "false",\n'
                        '+      defaultValue = "true",\n'
                        '       metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE})\n'
                    ),
                }
            ],
        )
        self.assertEqual(len(signals_flip_flag), 1)

        # normalize_patch_changes preserves context lines so moving a change to another method is detected
        p_method_a = "@@ -1,3 +1,3 @@\n void methodA() {\n-  return 1;\n+  return 2;\n }"
        p_method_b = "@@ -10,3 +10,3 @@\n void methodB() {\n-  return 1;\n+  return 2;\n }"
        self.assertNotEqual(
            prepare_context.normalize_patch_changes(p_method_a),
            prepare_context.normalize_patch_changes(p_method_b),
        )

        # Dismissed prior reviews, spoofed author COMMENT reviews, or quoted replies should not count as active Gemini reviews
        has_existing, last_review, _, _, _ = prepare_context.analyze_prior_review_delta(
            pr_reviews=[
                {
                    "id": 99,
                    "state": "DISMISSED",
                    "commit_id": sha1,
                    "submitted_at": "2026-11-12T20:00:00Z",
                    "body": "## 📋 Gemini Cherry-Pick Review Summary",
                },
                {
                    "id": 100,
                    "state": "COMMENTED",
                    "user": {"login": "untrusted-author"},
                    "author_association": "NONE",
                    "commit_id": sha2,
                    "submitted_at": "2026-11-12T20:01:00Z",
                    "body": "## 📋 Gemini Cherry-Pick Review Summary\n\n**Verdict:** ✅ Approved",
                },
                {
                    "id": 101,
                    "state": "COMMENTED",
                    "user": {"login": "bazel-maintainer"},
                    "author_association": "MEMBER",
                    "commit_id": sha2,
                    "submitted_at": "2026-11-12T20:02:00Z",
                    "body": "Quoting previous review:\n> ## 📋 Gemini Cherry-Pick Review Summary",
                },
            ],
            pr_inline_comments=[],
            base_ref="release-8.9.0",
            base_sha=self.base_sha,
            head_sha=sha2,
            pr_patches_by_file={},
            pr_author="untrusted-author",
            stage_dir=self.root / "gemini-artifacts" / "pr",
        )
        self.assertFalse(has_existing)
        self.assertIsNone(last_review)

        # Revert commits on master should be skipped when matching a forward cherry-pick by subject/PR number
        self._git("revert", "--no-edit", sha2)
        revert_sha = self._git_out("rev-parse", "HEAD")
        self.assertNotEqual(revert_sha, sha2)
        found_no_revert = prepare_context.find_mainline_commits(
            pr_title="[8.9.0] Second mainline fix for Bar.x (#602)",
            pr_body="",
            pr_commits=[
                {
                    "sha": "3333333333333333333333333333333333333333",
                    "commit": {"message": "[8.9.0] Second mainline fix for Bar.x (#602)"},
                }
            ],
            pr_comments=[],
            pr_author="wyv",
            mainline_ref="master",
            base_sha=self.base_sha,
        )
        self.assertEqual([sha for sha, _ in found_no_revert], [sha2])

        # HTML comments in PR body (e.g. from .github/pull_request_template.md) must NOT trigger RELNOTES[INC]
        pr_template_body = (
            "### Release Notes\n"
            "<!--\n"
            "If this is a breaking change, please add 'RELNOTES[INC]: <reason>' here.\n"
            "-->\n"
            "RELNOTES: None\n"
        )
        self.assertEqual(
            prepare_context.extract_breaking_change_signals(
                mainline_commits_info=[],
                pr_title="[8.9.0] Clean fix",
                pr_body=pr_template_body,
                pr_files=[],
            ),
            [],
        )

    def test_excludes_non_master_commits_from_clean_baseline_and_fails_without_master_ref(self):
        # Commit A is on master
        (self.root / "src" / "Foo.java").write_text(
            "class Foo {\n  int a() { return 10; }\n  int b() { return 2; }\n}\n",
            encoding="utf-8",
        )
        self._git("add", "src/Foo.java")
        self._git("commit", "-m", "Master fix Foo.a")
        master_sha = self._git_out("rev-parse", "master")

        # Commit B is on an unmerged side branch (NOT an ancestor of master)
        self._git("checkout", "-b", "unmerged-feature", "release-8.9.0")
        (self.root / "src" / "Unmerged.java").write_text("class Unmerged {}\n", encoding="utf-8")
        self._git("add", "src/Unmerged.java")
        self._git("commit", "-m", "Unmerged side-branch commit")
        unmerged_sha = self._git_out("rev-parse", "HEAD")

        # PR branch combines master_sha + unmerged_sha
        self._git("checkout", "-b", "pr-mixed", "release-8.9.0")
        self._git("cherry-pick", master_sha)
        self._git("cherry-pick", unmerged_sha)
        head_sha = self._git_out("rev-parse", "HEAD")
        self._git("checkout", "master")

        pr_files = [
            {
                "filename": "src/Foo.java",
                "status": "modified",
                "additions": 1,
                "deletions": 1,
                "patch": "@@ -1,4 +1,4 @@\n class Foo {\n-  int a() { return 1; }\n+  int a() { return 10; }\n   int b() { return 2; }\n }",
            },
            {
                "filename": "src/Unmerged.java",
                "status": "added",
                "additions": 1,
                "deletions": 0,
                "patch": "@@ -0,0 +1 @@\n+class Unmerged {}",
            },
        ]
        stage_dir = self.root / "gemini-artifacts" / "pr"
        _, head_root, base_root, _, pr_patches = prepare_context.stage_pr_and_base_files(
            repo="bazelbuild/bazel",
            base_sha=self.base_sha,
            head_sha=head_sha,
            pr_files=pr_files,
            stage_dir=stage_dir,
        )
        mc_info, file_comps, overall, _ = prepare_context.analyze_mainline_and_compare(
            candidates=[(master_sha, "trailer"), (unmerged_sha, "trailer")],
            mainline_ref="master",
            base_sha=self.base_sha,
            pr_files=pr_files,
            pr_patches_by_file=pr_patches,
            head_root_path=head_root,
            base_root_path=base_root,
            stage_dir=stage_dir,
        )
        by_sha = {mc["sha"]: mc["is_on_master"] for mc in mc_info}
        self.assertTrue(by_sha[master_sha])
        self.assertFalse(by_sha[unmerged_sha])
        by_path = {fc["path"]: fc["classification"] for fc in file_comps}
        self.assertEqual(by_path["src/Foo.java"], "identical")
        self.assertEqual(by_path["src/Unmerged.java"], "extra_in_pr")
        self.assertEqual(overall, "MODIFIED_OR_EXTRA_CHANGES")

        # Unsafe PR file paths (e.g. leading `-` or `:`) must NEVER be silently dropped into CLEAN
        pr_files_with_unsafe = [
            pr_files[0],
            {"filename": "-sneaky.sh", "status": "added", "additions": 1, "deletions": 0, "patch": "@@ -0,0 +1 @@\n+echo pwn"},
        ]
        _, file_comps_unsafe, overall_unsafe, vs_path_unsafe = prepare_context.analyze_mainline_and_compare(
            candidates=[(master_sha, "trailer")],
            mainline_ref="master",
            base_sha=self.base_sha,
            pr_files=pr_files_with_unsafe,
            pr_patches_by_file={"src/Foo.java": pr_patches["src/Foo.java"]},
            head_root_path=head_root,
            base_root_path=base_root,
            stage_dir=stage_dir,
        )
        self.assertEqual(overall_unsafe, "MODIFIED_OR_EXTRA_CHANGES")
        self.assertTrue(any(fc.get("unsafe_path") and fc["classification"] == "extra_in_pr" for fc in file_comps_unsafe))
        self.assertIn("-sneaky.sh", pathlib.Path(vs_path_unsafe).read_text(encoding="utf-8"))

        # Lockfile added/modified relative to mainline is classified as lockfile_regenerated (not identical)
        pr_files_with_lock = [
            pr_files[0],
            {"filename": "MODULE.bazel.lock", "status": "modified", "additions": 2, "deletions": 2, "patch": "@@ -1 +1 @@\n-old\n+new"},
        ]
        _, file_comps_lock, overall_lock, vs_path_lock = prepare_context.analyze_mainline_and_compare(
            candidates=[(master_sha, "trailer")],
            mainline_ref="master",
            base_sha=self.base_sha,
            pr_files=pr_files_with_lock,
            pr_patches_by_file={"src/Foo.java": pr_patches["src/Foo.java"], "MODULE.bazel.lock": "@@ -1 +1 @@\n-old\n+new"},
            head_root_path=head_root,
            base_root_path=base_root,
            stage_dir=stage_dir,
        )
        self.assertEqual(overall_lock, "CLEAN")
        by_lock_path = {fc["path"]: fc["classification"] for fc in file_comps_lock}
        self.assertEqual(by_lock_path["MODULE.bazel.lock"], "lockfile_regenerated")
        self.assertIn("MODULE.bazel.lock", pathlib.Path(vs_path_lock).read_text(encoding="utf-8"))

        # Commit message with a cherry-pick trailer must NOT also extract an earlier regression commit URL
        self._git("checkout", "master")
        (self.root / "src" / "Regression.java").write_text("class Regression {}\n", encoding="utf-8")
        self._git("add", "src/Regression.java")
        self._git("commit", "-m", "Earlier regression commit")
        regression_sha = self._git_out("rev-parse", "HEAD")

        found_trailer_only = prepare_context.find_mainline_commits(
            pr_title="[8.9.0] Master fix Foo.a",
            pr_body="",
            pr_commits=[
                {
                    "sha": "4444444444444444444444444444444444444444",
                    "commit": {
                        "message": (
                            f"Master fix Foo.a\n\n"
                            f"Fixes regression from https://github.com/bazelbuild/bazel/commit/{regression_sha}\n\n"
                            f"(cherry picked from commit {master_sha})"
                        )
                    },
                }
            ],
            pr_comments=[],
            pr_author="wyv",
            mainline_ref="master",
            base_sha=self.base_sha,
        )
        self.assertEqual([sha for sha, _ in found_trailer_only], [master_sha])

        # detect_mainline_ref must raise RuntimeError rather than falling back to HEAD (release branch)
        self._git("checkout", "release-8.9.0")
        self._git("branch", "-D", "master")
        with self.assertRaises(RuntimeError):
            prepare_context.detect_mainline_ref()

    def test_renamed_file_deletion_bypass_caught_while_legitimate_rename_stays_clean(self):
        # Case A: Mainline adds src/NewHelper.java; PR adds src/NewHelper.java AND deletes src/Foo.java,
        # which GitHub's /files API collapses into a single status="renamed" entry.
        (self.root / "src" / "NewHelper.java").write_text(
            "class Foo {\n  int a() { return 1; }\n  int b() { return 3; }\n}\n",
            encoding="utf-8",
        )
        self._git("add", "src/NewHelper.java")
        self._git("commit", "-m", "Add NewHelper")
        add_sha = self._git_out("rev-parse", "master")

        self._git("checkout", "-b", "pr-sneaky-rename", "release-8.9.0")
        (self.root / "src" / "Foo.java").unlink()
        (self.root / "src" / "NewHelper.java").write_text(
            "class Foo {\n  int a() { return 1; }\n  int b() { return 3; }\n}\n",
            encoding="utf-8",
        )
        self._git("add", "-A")
        self._git("commit", "-m", "Sneaky delete disguised as rename")
        sneaky_head_sha = self._git_out("rev-parse", "HEAD")
        self._git("checkout", "master")

        pr_files_sneaky = [
            {
                "filename": "src/NewHelper.java",
                "previous_filename": "src/Foo.java",
                "status": "renamed",
                "additions": 1,
                "deletions": 1,
                "patch": "@@ -1,4 +1,4 @@\n class Foo {\n   int a() { return 1; }\n-  int b() { return 2; }\n+  int b() { return 3; }\n }",
            }
        ]
        stage_dir_a = self.root / "stage_rename_a"
        _, head_root_a, base_root_a, _, pr_patches_a = prepare_context.stage_pr_and_base_files(
            repo="bazelbuild/bazel",
            base_sha=self.base_sha,
            head_sha=sneaky_head_sha,
            pr_files=pr_files_sneaky,
            stage_dir=stage_dir_a,
        )
        _, comps_a, overall_a, _ = prepare_context.analyze_mainline_and_compare(
            candidates=[(add_sha, "trailer")],
            mainline_ref="master",
            base_sha=self.base_sha,
            pr_files=pr_files_sneaky,
            pr_patches_by_file=pr_patches_a,
            head_root_path=head_root_a,
            base_root_path=base_root_a,
            head_sha=sneaky_head_sha,
            stage_dir=stage_dir_a,
        )
        self.assertEqual(overall_a, "MODIFIED_OR_EXTRA_CHANGES")
        by_path_a = {fc["path"]: fc["classification"] for fc in comps_a}
        self.assertEqual(by_path_a["src/Foo.java"], "extra_in_pr")
        self.assertEqual(by_path_a["src/NewHelper.java"], "identical")

        # Case B: Mainline legitimately renames src/Foo.java -> src/FooRenamed.java and PR cherry-picks cleanly.
        self._git("mv", "src/Foo.java", "src/FooRenamed.java")
        self._git("commit", "-m", "Rename Foo to FooRenamed")
        rename_sha = self._git_out("rev-parse", "master")

        self._git("checkout", "-b", "pr-legit-rename", "release-8.9.0")
        self._git("cherry-pick", rename_sha)
        legit_head_sha = self._git_out("rev-parse", "HEAD")
        self._git("checkout", "master")

        pr_files_legit = [
            {
                "filename": "src/FooRenamed.java",
                "previous_filename": "src/Foo.java",
                "status": "renamed",
                "additions": 0,
                "deletions": 0,
                "patch": "",
            }
        ]
        stage_dir_b = self.root / "stage_rename_b"
        _, head_root_b, base_root_b, _, pr_patches_b = prepare_context.stage_pr_and_base_files(
            repo="bazelbuild/bazel",
            base_sha=self.base_sha,
            head_sha=legit_head_sha,
            pr_files=pr_files_legit,
            stage_dir=stage_dir_b,
        )
        _, comps_b, overall_b, _ = prepare_context.analyze_mainline_and_compare(
            candidates=[(rename_sha, "trailer")],
            mainline_ref="master",
            base_sha=self.base_sha,
            pr_files=pr_files_legit,
            pr_patches_by_file=pr_patches_b,
            head_root_path=head_root_b,
            base_root_path=base_root_b,
            head_sha=legit_head_sha,
            stage_dir=stage_dir_b,
        )
        self.assertEqual(overall_b, "CLEAN")
        by_path_b = {fc["path"]: fc["classification"] for fc in comps_b}
        self.assertEqual(by_path_b["src/Foo.java"], "identical")
        self.assertEqual(by_path_b["src/FooRenamed.java"], "identical")

    def test_binary_files_and_executable_mode_flip_detected_via_head_tree(self):
        bin_path = self.root / "src" / "tool.bin"
        bin_path.write_bytes(b"\x89PNG\r\n\x1a\n\x00\xff\xfe\xfd")
        sh_path = self.root / "src" / "run.sh"
        sh_path.write_text("#!/bin/sh\necho ok\n", encoding="utf-8")
        sh_path.chmod(0o644)
        self._git("add", "src/tool.bin", "src/run.sh")
        self._git("commit", "-m", "Add binary asset and helper script")
        master_sha = self._git_out("rev-parse", "master")

        # Binary blob must not crash with UnicodeDecodeError and returns None for text staging
        self.assertIsNone(prepare_context.read_git_regular_file_blob(master_sha, "src/tool.bin"))

        # Clean cherry-pick of binary + script is CLEAN via Git tree comparison
        self._git("checkout", "-b", "pr-bin-clean", "release-8.9.0")
        self._git("cherry-pick", master_sha)
        clean_head = self._git_out("rev-parse", "HEAD")

        # Flip executable mode (+x: 100644 -> 100755) without changing text content
        self._git("checkout", "-b", "pr-mode-flip", clean_head)
        sh_path.chmod(0o755)
        self._git("add", "--chmod=+x", "src/run.sh")
        self._git("commit", "-m", "Flip +x on run.sh")
        mode_flip_head = self._git_out("rev-parse", "HEAD")
        self._git("checkout", "master")

        pr_files = [
            {"filename": "src/tool.bin", "status": "added", "additions": 0, "deletions": 0, "patch": ""},
            {
                "filename": "src/run.sh",
                "status": "added",
                "additions": 2,
                "deletions": 0,
                "patch": "@@ -0,0 +1,2 @@\n+#!/bin/sh\n+echo ok",
            },
        ]
        stage_clean = self.root / "stage_bin_clean"
        with patch.object(prepare_context, "gh_api", side_effect=RuntimeError("no network")):
            _, hr_clean, br_clean, _, patches_clean = prepare_context.stage_pr_and_base_files(
                repo="bazelbuild/bazel",
                base_sha=self.base_sha,
                head_sha=clean_head,
                pr_files=pr_files,
                stage_dir=stage_clean,
            )
        _, _, overall_clean, _ = prepare_context.analyze_mainline_and_compare(
            candidates=[(master_sha, "trailer")],
            mainline_ref="master",
            base_sha=self.base_sha,
            pr_files=pr_files,
            pr_patches_by_file=patches_clean,
            head_root_path=hr_clean,
            base_root_path=br_clean,
            head_sha=clean_head,
            stage_dir=stage_clean,
        )
        self.assertEqual(overall_clean, "CLEAN")

        stage_flip = self.root / "stage_mode_flip"
        with patch.object(prepare_context, "gh_api", side_effect=RuntimeError("no network")):
            _, hr_flip, br_flip, _, patches_flip = prepare_context.stage_pr_and_base_files(
                repo="bazelbuild/bazel",
                base_sha=self.base_sha,
                head_sha=mode_flip_head,
                pr_files=pr_files,
                stage_dir=stage_flip,
            )
        _, comps_flip, overall_flip, vs_flip = prepare_context.analyze_mainline_and_compare(
            candidates=[(master_sha, "trailer")],
            mainline_ref="master",
            base_sha=self.base_sha,
            pr_files=pr_files,
            pr_patches_by_file=patches_flip,
            head_root_path=hr_flip,
            base_root_path=br_flip,
            head_sha=mode_flip_head,
            stage_dir=stage_flip,
        )
        self.assertEqual(overall_flip, "MODIFIED_OR_EXTRA_CHANGES")
        by_path_flip = {fc["path"]: fc["classification"] for fc in comps_flip}
        self.assertEqual(by_path_flip["src/tool.bin"], "identical")
        self.assertEqual(by_path_flip["src/run.sh"], "adapted_without_conflict")
        vs_text = pathlib.Path(vs_flip).read_text(encoding="utf-8")
        self.assertIn("mode 100644", vs_text)
        self.assertIn("mode 100755", vs_text)

    def test_merge_commit_on_master_and_anchored_discovery(self):
        # Advance master with an unrelated file first
        (self.root / "src" / "MasterOnly.java").write_text("class MasterOnly {}\n", encoding="utf-8")
        self._git("add", "src/MasterOnly.java")
        self._git("commit", "-m", "Unrelated master commit\n\nCloses #602.")

        # Create a feature branch from release-8.9.0 and merge it (--no-ff) into master
        self._git("checkout", "-b", "feat-branch", "release-8.9.0")
        (self.root / "src" / "MergedFix.java").write_text("class MergedFix {}\n", encoding="utf-8")
        self._git("add", "src/MergedFix.java")
        self._git("commit", "-m", "Merged feature fix")
        self._git("checkout", "master")
        self._git("merge", "--no-ff", "feat-branch", "-m", "Merge pull request #700 from user/feat-branch")
        merge_sha = self._git_out("rev-parse", "master")

        # Also commit a Copybara-style automated rollback so we verify revert filtering
        (self.root / "src" / "RolledBack.java").write_text("class RolledBack {}\n", encoding="utf-8")
        self._git("add", "src/RolledBack.java")
        self._git("commit", "-m", "Add important feature for test")
        orig_feat_sha = self._git_out("rev-parse", "master")
        self._git("rm", "src/RolledBack.java")
        self._git("commit", "-m", "Automated rollback of commit Add important feature for test.")

        # Cherry-pick merge_sha (-m 1) onto release-8.9.0
        self._git("checkout", "-b", "pr-merge-cp", "release-8.9.0")
        self._git("cherry-pick", "-m", "1", merge_sha)
        pr_head = self._git_out("rev-parse", "HEAD")
        self._git("checkout", "master")

        pr_files = [
            {
                "filename": "src/MergedFix.java",
                "status": "added",
                "additions": 1,
                "deletions": 0,
                "patch": "@@ -0,0 +1 @@\n+class MergedFix {}",
            }
        ]
        stage_dir = self.root / "stage_merge"
        _, hr, br, _, patches = prepare_context.stage_pr_and_base_files(
            repo="bazelbuild/bazel",
            base_sha=self.base_sha,
            head_sha=pr_head,
            pr_files=pr_files,
            stage_dir=stage_dir,
        )
        mc_info, comps, overall, _ = prepare_context.analyze_mainline_and_compare(
            candidates=[(merge_sha, "trailer")],
            mainline_ref="master",
            base_sha=self.base_sha,
            pr_files=pr_files,
            pr_patches_by_file=patches,
            head_root_path=hr,
            base_root_path=br,
            head_sha=pr_head,
            stage_dir=stage_dir,
        )
        self.assertEqual(overall, "CLEAN")
        self.assertEqual(mc_info[0]["files_modified"], ["src/MergedFix.java"])
        self.assertTrue(mc_info[0]["applied_cleanly_on_base"])
        self.assertEqual(len(comps), 1)
        self.assertEqual(comps[0]["classification"], "identical")

        # Anchored PR number: (#60) must NOT match Closes #602 on master
        no_prefix_match = prepare_context.find_mainline_commits(
            pr_title="[8.9.0] Unmatched PR (#60)",
            pr_body="",
            pr_commits=[{"sha": "5" * 40, "commit": {"message": "[8.9.0] Unmatched PR (#60)"}}],
            pr_comments=[],
            pr_author="wyv",
            mainline_ref="master",
            base_sha=self.base_sha,
        )
        self.assertEqual(no_prefix_match, [])

        # Automated rollback commit must be skipped when matching forward commit by exact subject
        found_orig = prepare_context.find_mainline_commits(
            pr_title="[8.9.0] Add important feature for test",
            pr_body="",
            pr_commits=[{"sha": "6" * 40, "commit": {"message": "[8.9.0] Add important feature for test"}}],
            pr_comments=[],
            pr_author="wyv",
            mainline_ref="master",
            base_sha=self.base_sha,
        )
        self.assertEqual([sha for sha, _ in found_orig], [orig_feat_sha])

    def test_reverted_mainline_detection_long_and_at_paths_and_review_auth(self):
        long_rel_path = (
            "src/test/java/com/google/devtools/build/lib/analysis/testdata/bzlmod/pkg@1.2.3/"
            + "deep_nested_package_directory/" * 5
            + "VeryLongCherryPickTest.java"
        )
        self.assertGreater(len(long_rel_path), 210)
        full_long_file = self.root / long_rel_path
        full_long_file.parent.mkdir(parents=True, exist_ok=True)
        full_long_file.write_text("class VeryLongCherryPickTest {}\n", encoding="utf-8")
        self._git("add", long_rel_path)
        self._git("commit", "-m", "Add long @-containing test file")
        c1_sha = self._git_out("rev-parse", "master")

        # Roll back c1 on master using Copybara rollback format
        self._git("rm", long_rel_path)
        self._git(
            "commit",
            "-m",
            f"Automated rollback of commit {c1_sha}.\n\n*** Reason for rollback ***\n\nBroke postsubmit",
        )
        r1_sha = self._git_out("rev-parse", "master")
        self.assertEqual(prepare_context.find_revert_on_master(c1_sha, "master"), r1_sha)

        #Cherry-pick c1 onto release-8.9.0 and verify exact >200-char @ path preservation + reverted_by_on_master
        self._git("checkout", "-b", "pr-long-reverted", "release-8.9.0")
        self._git("cherry-pick", c1_sha)
        pr_head = self._git_out("rev-parse", "HEAD")
        self._git("checkout", "master")

        pr_files = [
            {
                "filename": long_rel_path,
                "status": "added",
                "additions": 1,
                "deletions": 0,
                "patch": "@@ -0,0 +1 @@\n+class VeryLongCherryPickTest {}",
            }
        ]
        stage_dir = self.root / "stage_long"
        _, hr, br, changed_info, patches = prepare_context.stage_pr_and_base_files(
            repo="bazelbuild/bazel",
            base_sha=self.base_sha,
            head_sha=pr_head,
            pr_files=pr_files,
            stage_dir=stage_dir,
        )
        self.assertEqual(changed_info[0]["path"], long_rel_path)
        self.assertNotIn("\u200b", changed_info[0]["path"])

        mc_info, comps, overall, vs_path = prepare_context.analyze_mainline_and_compare(
            candidates=[(c1_sha, "trailer")],
            mainline_ref="master",
            base_sha=self.base_sha,
            pr_files=pr_files,
            pr_patches_by_file=patches,
            head_root_path=hr,
            base_root_path=br,
            head_sha=pr_head,
            stage_dir=stage_dir,
        )
        self.assertEqual(overall, "CLEAN")
        self.assertEqual(mc_info[0]["reverted_by_on_master"], r1_sha)
        self.assertEqual(comps[0]["path"], long_rel_path)
        self.assertIn("REVERTED OR ROLLED BACK ON MASTER", pathlib.Path(vs_path).read_text(encoding="utf-8"))

        # Rolling back r1 itself (re-landing c1 via rollback-of-rollback) clears active revert status
        full_long_file.parent.mkdir(parents=True, exist_ok=True)
        full_long_file.write_text("class VeryLongCherryPickTest {}\n", encoding="utf-8")
        self._git("add", long_rel_path)
        self._git(
            "commit",
            "-m",
            (
                f"Automated rollback of commit {r1_sha}.\n\n"
                "*** Original change description ***\n\n"
                f"Automated rollback of commit {c1_sha}."
            ),
        )
        self.assertIsNone(prepare_context.find_revert_on_master(c1_sha, "master"))

        # Non-bot login with empty author_association must be rejected by is_authentic_gemini_review
        self.assertFalse(
            prepare_context.is_authentic_gemini_review(
                {
                    "state": "COMMENTED",
                    "user": {"login": "random-user"},
                    "author_association": "",
                    "body": "## 📋 Gemini Cherry-Pick Review Summary",
                },
                pr_author="other-user",
            )
        )
        self.assertTrue(
            prepare_context.is_authentic_gemini_review(
                {
                    "state": "COMMENTED",
                    "user": {"login": "maintainer-user"},
                    "author_association": "MEMBER",
                    "body": "## 📋 Gemini Cherry-Pick Review Summary",
                },
                pr_author="other-user",
            )
        )


if __name__ == "__main__":
    unittest.main()
