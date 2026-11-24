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
        subprocess.run(["git", *args], cwd=self.root, check=True, capture_output=True, text=True)

    def _git_out(self, *args):
        return subprocess.run(["git", *args], cwd=self.root, check=True, capture_output=True, text=True).stdout.strip()

    def _init_git_repo(self):
        self._git("init", "-b", "master")
        self._git("config", "user.name", "Test User")
        self._git("config", "user.email", "test@example.com")
        self._git("config", "commit.gpgsign", "false")
        (self.root / "src").mkdir()
        (self.root / "src" / "Foo.java").write_text(
            "class Foo {\n  int a() { return 1; }\n  int b() { return 2; }\n}\n",
            encoding="utf-8",
        )
        self._git("add", ".")
        self._git("commit", "-m", "Initial commit")
        self._git("branch", "release-8.9.0")
        self.base_sha = self._git_out("rev-parse", "release-8.9.0")

    def test_skips_non_release_branch_pr(self):
        def fake_gh_api(endpoint, *, paginate=False, optional=False):
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
        (self.root / "src" / "Foo.java").write_text(
            "class Foo {\n  int a() { return 10; }\n  int b() { return 2; }\n}\n",
            encoding="utf-8",
        )
        self._git("add", ".")
        self._git("commit", "-m", "Fix Foo.a return value (#500)\n\nRELNOTES: None\nPiperOrigin-RevId: 123456789")
        mainline_sha = self._git_out("rev-parse", "master")

        self._git("checkout", "-b", "cp-clean", "release-8.9.0")
        self._git("cherry-pick", "-x", mainline_sha)
        head_sha = self._git_out("rev-parse", "cp-clean")
        self._git("checkout", "master")

        def fake_gh_api(endpoint, *, paginate=False, optional=False):
            if endpoint == "repos/bazelbuild/bazel/pulls/101":
                return {
                    "state": "open",
                    "draft": False,
                    "title": "[8.9.0] Fix Foo.a return value",
                    "body": f"Fixes Foo.a\n(cherry picked from commit {mainline_sha})",
                    "user": {"login": "wyv"},
                    "base": {"ref": "release-8.9.0", "sha": self.base_sha},
                    "head": {"ref": "cp-clean", "sha": head_sha},
                }
            if any(p in endpoint for p in ("issues/101/comments", "pulls/101/reviews", "pulls/101/comments")):
                return []
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
        (self.root / "src" / "Foo.java").write_text(
            "class Foo {\n  int a(String ctx) { return 1; }\n  int b() { return 2; }\n}\n",
            encoding="utf-8",
        )
        self._git("add", ".")
        self._git("commit", "-m", "Refactor Foo.a signature on master")

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

        self._git("checkout", "-b", "cp-conflict-extra", "release-8.9.0")
        (self.root / "src" / "Foo.java").write_text(
            "class Foo {\n  int a() { return 42; }\n  int b() { return 2; }\n}\n",
            encoding="utf-8",
        )
        (self.root / "src" / "Sneaky.java").write_text("class Sneaky {}\n", encoding="utf-8")
        self._git("add", ".")
        self._git("commit", "-m", f"[8.9.0] Change Foo.a behavior\n\n(cherry picked from commit {mainline_sha})")
        head_sha = self._git_out("rev-parse", "cp-conflict-extra")
        self._git("checkout", "master")

        def fake_gh_api(endpoint, *, paginate=False, optional=False):
            if endpoint == "repos/bazelbuild/bazel/pulls/102":
                return {
                    "state": "open",
                    "draft": False,
                    "title": "[8.9.0] Change Foo.a behavior",
                    "body": f"(cherry picked from commit {mainline_sha})",
                    "user": {"login": "wyv"},
                    "base": {"ref": "release-8.9.0", "sha": self.base_sha},
                    "head": {"ref": "cp-conflict-extra", "sha": head_sha},
                }
            if any(p in endpoint for p in ("issues/102/comments", "pulls/102/reviews", "pulls/102/comments")):
                return []
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

        self.assertIsNone(prepare_context.is_valid_release_branch("release-1.0..2.0"))
        self.assertIsNone(prepare_context.is_valid_release_branch("release-.hidden"))
        self.assertIsNone(prepare_context.is_valid_release_branch("release-8.9.0.lock"))
        self.assertEqual(prepare_context.is_valid_release_branch("release-8.9.0"), "8.9.0")

        # Pre-existing symlink inside stage_dir must block extraction outside stage_dir via tarfile filter="data"
        (self.root / "link_dir").mkdir()
        (self.root / "link_dir" / "escape.txt").write_text("test\n", encoding="utf-8")
        (self.root / "src" / "evil_link").symlink_to("/etc/hosts")
        self._git("add", "link_dir/escape.txt", "src/evil_link")
        self._git("commit", "-m", "Add file under link_dir and git symlink")
        sha_with_link = self._git_out("rev-parse", "HEAD")

        outside_dir = self.root / "outside_target"
        outside_dir.mkdir()
        stage_dir = self.root / "stage_test"
        stage_dir.mkdir()
        (stage_dir / "link_dir").symlink_to(outside_dir)

        staged = prepare_context.extract_git_files_to_dir(
            sha_with_link, ["link_dir/escape.txt", "src/evil_link", "src/Foo.java"], stage_dir
        )
        self.assertNotIn("link_dir/escape.txt", staged)
        self.assertFalse((outside_dir / "escape.txt").exists())
        self.assertNotIn("src/evil_link", staged)
        self.assertIn("src/Foo.java", staged)

    def test_prior_review_delta_detects_unchanged_rebase_and_modified_followup(self):
        self._git("checkout", "-b", "pr-branch", "release-8.9.0")
        (self.root / "src" / "Foo.java").write_text(
            "class Foo {\n  int a() { return 10; }\n  int b() { return 2; }\n}\n",
            encoding="utf-8",
        )
        self._git("add", ".")
        self._git("commit", "-m", "PR commit 1")
        rev1_sha = self._git_out("rev-parse", "HEAD")

        # Advance release-8.9.0 on an unrelated file and rebase pr-branch onto the new base
        self._git("checkout", "release-8.9.0")
        (self.root / "src" / "ReleaseNote.md").write_text("8.9.0\n", encoding="utf-8")
        self._git("add", "src/ReleaseNote.md")
        self._git("commit", "-m", "Advance release-8.9.0")
        new_base_sha = self._git_out("rev-parse", "release-8.9.0")

        self._git("checkout", "pr-branch")
        self._git("rebase", "release-8.9.0")
        rev1_rebased_sha = self._git_out("rev-parse", "HEAD")

        reviews = [
            {
                "id": 1,
                "user": {"login": "bazel-io"},
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

        has_existing, last_review, unchanged, since_path, comments_out = prepare_context.analyze_prior_review_delta(
            pr_reviews=reviews,
            pr_inline_comments=inline_comments,
            base_ref="release-8.9.0",
            base_sha=new_base_sha,
            head_sha=rev1_rebased_sha,
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

        _, _, unchanged2, since_path2, _ = prepare_context.analyze_prior_review_delta(
            pr_reviews=reviews,
            pr_inline_comments=inline_comments,
            base_ref="release-8.9.0",
            base_sha=new_base_sha,
            head_sha=rev2_sha,
            stage_dir=self.root / "gemini-artifacts" / "pr",
        )
        self.assertFalse(unchanged2)
        self.assertIn("return 20;", pathlib.Path(since_path2).read_text(encoding="utf-8"))

    def test_multi_commit_discovery_patch_release_and_flag_addition_vs_flip(self):
        self.assertFalse(prepare_context.is_patch_release_version("8.9.0"))
        self.assertTrue(prepare_context.is_patch_release_version("8.5.1"))
        self.assertTrue(prepare_context.is_patch_release_version("7.7.2"))

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

        pr_commits = [
            {
                "sha": "1" * 40,
                "commit": {"message": f"First mainline fix for Foo.a (#601)\n\n(cherry picked from commit {sha1})"},
            },
            {
                "sha": "2" * 40,
                "commit": {"message": "[8.9.0] Second mainline fix for Bar.x"},
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

        signals_add_flag = prepare_context.extract_breaking_change_signals(
            mainline_commits_info=[],
            pr_title="[8.9.0] Add stub incompatible flag",
            pr_body="",
            java_patches={
                "src/Options.java": (
                    '@@ -10,0 +11,5 @@\n'
                    '+  @Option(\n'
                    '+      name = "incompatible_new_stub",\n'
                    '+      defaultValue = "false",\n'
                    '+      metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE})\n'
                )
            },
        )
        self.assertEqual(signals_add_flag, [])

        signals_flip_flag = prepare_context.extract_breaking_change_signals(
            mainline_commits_info=[],
            pr_title="[8.9.0] Flip incompatible flag",
            pr_body="",
            java_patches={
                "src/Options.java": (
                    '@@ -10,5 +10,5 @@\n'
                    '   @Option(\n'
                    '       name = "incompatible_existing_flag",\n'
                    '-      defaultValue = "false",\n'
                    '+      defaultValue = "true",\n'
                    '       metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE})\n'
                )
            },
        )
        self.assertEqual(len(signals_flip_flag), 1)

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
            pr_author="untrusted-author",
            stage_dir=self.root / "gemini-artifacts" / "pr",
        )
        self.assertFalse(has_existing)
        self.assertIsNone(last_review)

        self._git("revert", "--no-edit", sha2)
        revert_sha = self._git_out("rev-parse", "HEAD")
        self.assertNotEqual(revert_sha, sha2)
        found_no_revert = prepare_context.find_mainline_commits(
            pr_title="[8.9.0] Second mainline fix for Bar.x (#602)",
            pr_body="",
            pr_commits=[{"sha": "3" * 40, "commit": {"message": "[8.9.0] Second mainline fix for Bar.x (#602)"}}],
            pr_comments=[],
            pr_author="wyv",
            mainline_ref="master",
            base_sha=self.base_sha,
        )
        self.assertEqual([sha for sha, _ in found_no_revert], [sha2])

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
                java_patches={},
            ),
            [],
        )

    def test_excludes_non_master_commits_unsafe_paths_and_lockfile_regeneration(self):
        (self.root / "src" / "Foo.java").write_text(
            "class Foo {\n  int a() { return 10; }\n  int b() { return 2; }\n}\n",
            encoding="utf-8",
        )
        self._git("add", "src/Foo.java")
        self._git("commit", "-m", "Master fix Foo.a")
        master_sha = self._git_out("rev-parse", "master")

        self._git("checkout", "-b", "unmerged-feature", "release-8.9.0")
        (self.root / "src" / "Unmerged.java").write_text("class Unmerged {}\n", encoding="utf-8")
        self._git("add", "src/Unmerged.java")
        self._git("commit", "-m", "Unmerged side-branch commit")
        unmerged_sha = self._git_out("rev-parse", "HEAD")

        # 1) PR branch combines master_sha + unmerged_sha
        self._git("checkout", "-b", "pr-mixed", "release-8.9.0")
        self._git("cherry-pick", master_sha)
        self._git("cherry-pick", unmerged_sha)
        head_sha = self._git_out("rev-parse", "HEAD")

        stage_dir = self.root / "gemini-artifacts" / "pr"
        _, _, base_root, _ = prepare_context.stage_pr_and_base_files(
            base_sha=self.base_sha,
            head_sha=head_sha,
            stage_dir=stage_dir,
        )
        mc_info, file_comps, overall, _, _ = prepare_context.analyze_mainline_and_compare(
            candidates=[(master_sha, "trailer"), (unmerged_sha, "trailer")],
            mainline_ref="master",
            base_sha=self.base_sha,
            head_sha=head_sha,
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

        # 2) Unsafe PR file paths (e.g. leading `-`) committed in Git must NEVER be silently dropped into CLEAN
        self._git("checkout", "-b", "pr-unsafe-path", "release-8.9.0")
        self._git("cherry-pick", master_sha)
        (self.root / "-sneaky.sh").write_text("echo pwn\n", encoding="utf-8")
        self._git("add", "--", "-sneaky.sh")
        self._git("commit", "-m", "Add leading-dash file")
        unsafe_head_sha = self._git_out("rev-parse", "HEAD")

        stage_unsafe = self.root / "stage_unsafe"
        _, _, br_unsafe, changed_unsafe = prepare_context.stage_pr_and_base_files(
            base_sha=self.base_sha,
            head_sha=unsafe_head_sha,
            stage_dir=stage_unsafe,
        )
        self.assertTrue(any(f.get("unsafe_path") and f["path"] == "-sneaky.sh" for f in changed_unsafe))
        _, file_comps_unsafe, overall_unsafe, vs_path_unsafe, _ = prepare_context.analyze_mainline_and_compare(
            candidates=[(master_sha, "trailer")],
            mainline_ref="master",
            base_sha=self.base_sha,
            head_sha=unsafe_head_sha,
            base_root_path=br_unsafe,
            stage_dir=stage_unsafe,
        )
        self.assertEqual(overall_unsafe, "MODIFIED_OR_EXTRA_CHANGES")
        self.assertTrue(any(fc.get("unsafe_path") and fc["classification"] == "extra_in_pr" for fc in file_comps_unsafe))
        self.assertIn("-sneaky.sh", pathlib.Path(vs_path_unsafe).read_text(encoding="utf-8"))

        # 3) Lockfile added/modified relative to mainline is classified as lockfile_regenerated (CLEAN overall)
        self._git("checkout", "-b", "pr-with-lockfile", "release-8.9.0")
        self._git("cherry-pick", master_sha)
        (self.root / "MODULE.bazel.lock").write_text('{"lockFileVersion": 1}\n', encoding="utf-8")
        self._git("add", "MODULE.bazel.lock")
        self._git("commit", "-m", "Update lockfile")
        lock_head_sha = self._git_out("rev-parse", "HEAD")

        stage_lock = self.root / "stage_lock"
        _, _, br_lock, _ = prepare_context.stage_pr_and_base_files(
            base_sha=self.base_sha,
            head_sha=lock_head_sha,
            stage_dir=stage_lock,
        )
        _, file_comps_lock, overall_lock, vs_path_lock, _ = prepare_context.analyze_mainline_and_compare(
            candidates=[(master_sha, "trailer")],
            mainline_ref="master",
            base_sha=self.base_sha,
            head_sha=lock_head_sha,
            base_root_path=br_lock,
            stage_dir=stage_lock,
        )
        self.assertEqual(overall_lock, "CLEAN")
        by_lock_path = {fc["path"]: fc["classification"] for fc in file_comps_lock}
        self.assertEqual(by_lock_path["MODULE.bazel.lock"], "lockfile_regenerated")
        self.assertIn("MODULE.bazel.lock", pathlib.Path(vs_path_lock).read_text(encoding="utf-8"))

        # 4) Commit message with a cherry-pick trailer must NOT also extract an earlier regression commit URL
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
                    "sha": "4" * 40,
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

        # 5) detect_mainline_ref must raise RuntimeError rather than falling back to HEAD (release branch)
        self._git("checkout", "release-8.9.0")
        self._git("branch", "-D", "master")
        with self.assertRaises(RuntimeError):
            prepare_context.detect_mainline_ref()

    def test_renamed_file_deletion_bypass_caught_while_legitimate_rename_stays_clean(self):
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

        stage_dir_a = self.root / "stage_rename_a"
        _, _, base_root_a, _ = prepare_context.stage_pr_and_base_files(
            base_sha=self.base_sha,
            head_sha=sneaky_head_sha,
            stage_dir=stage_dir_a,
        )
        _, comps_a, overall_a, _, _ = prepare_context.analyze_mainline_and_compare(
            candidates=[(add_sha, "trailer")],
            mainline_ref="master",
            base_sha=self.base_sha,
            head_sha=sneaky_head_sha,
            base_root_path=base_root_a,
            stage_dir=stage_dir_a,
        )
        self.assertEqual(overall_a, "MODIFIED_OR_EXTRA_CHANGES")
        by_path_a = {fc["path"]: fc["classification"] for fc in comps_a}
        self.assertEqual(by_path_a["src/Foo.java"], "extra_in_pr")
        self.assertEqual(by_path_a["src/NewHelper.java"], "identical")

        self._git("mv", "src/Foo.java", "src/FooRenamed.java")
        self._git("commit", "-m", "Rename Foo to FooRenamed")
        rename_sha = self._git_out("rev-parse", "master")

        self._git("checkout", "-b", "pr-legit-rename", "release-8.9.0")
        self._git("cherry-pick", rename_sha)
        legit_head_sha = self._git_out("rev-parse", "HEAD")
        self._git("checkout", "master")

        stage_dir_b = self.root / "stage_rename_b"
        _, _, base_root_b, _ = prepare_context.stage_pr_and_base_files(
            base_sha=self.base_sha,
            head_sha=legit_head_sha,
            stage_dir=stage_dir_b,
        )
        _, comps_b, overall_b, _, _ = prepare_context.analyze_mainline_and_compare(
            candidates=[(rename_sha, "trailer")],
            mainline_ref="master",
            base_sha=self.base_sha,
            head_sha=legit_head_sha,
            base_root_path=base_root_b,
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

        self._git("checkout", "-b", "pr-bin-clean", "release-8.9.0")
        self._git("cherry-pick", master_sha)
        clean_head = self._git_out("rev-parse", "HEAD")

        self._git("checkout", "-b", "pr-mode-flip", clean_head)
        sh_path.chmod(0o755)
        self._git("add", "--chmod=+x", "src/run.sh")
        self._git("commit", "-m", "Flip +x on run.sh")
        mode_flip_head = self._git_out("rev-parse", "HEAD")
        self._git("checkout", "master")

        stage_clean = self.root / "stage_bin_clean"
        _, _, br_clean, changed_clean = prepare_context.stage_pr_and_base_files(
            base_sha=self.base_sha,
            head_sha=clean_head,
            stage_dir=stage_clean,
        )
        by_clean = {f["path"]: f for f in changed_clean}
        self.assertIsNone(by_clean["src/tool.bin"]["head_file_path"])
        self.assertIsNotNone(by_clean["src/run.sh"]["head_file_path"])

        _, _, overall_clean, _, _ = prepare_context.analyze_mainline_and_compare(
            candidates=[(master_sha, "trailer")],
            mainline_ref="master",
            base_sha=self.base_sha,
            head_sha=clean_head,
            base_root_path=br_clean,
            stage_dir=stage_clean,
        )
        self.assertEqual(overall_clean, "CLEAN")

        stage_flip = self.root / "stage_mode_flip"
        _, _, br_flip, _ = prepare_context.stage_pr_and_base_files(
            base_sha=self.base_sha,
            head_sha=mode_flip_head,
            stage_dir=stage_flip,
        )
        _, comps_flip, overall_flip, vs_flip, _ = prepare_context.analyze_mainline_and_compare(
            candidates=[(master_sha, "trailer")],
            mainline_ref="master",
            base_sha=self.base_sha,
            head_sha=mode_flip_head,
            base_root_path=br_flip,
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
        (self.root / "src" / "MasterOnly.java").write_text("class MasterOnly {}\n", encoding="utf-8")
        self._git("add", "src/MasterOnly.java")
        self._git("commit", "-m", "Unrelated master commit\n\nCloses #602.")

        self._git("checkout", "-b", "feat-branch", "release-8.9.0")
        (self.root / "src" / "MergedFix.java").write_text("class MergedFix {}\n", encoding="utf-8")
        self._git("add", "src/MergedFix.java")
        self._git("commit", "-m", "Merged feature fix")
        self._git("checkout", "master")
        self._git("merge", "--no-ff", "feat-branch", "-m", "Merge pull request #700 from user/feat-branch")
        merge_sha = self._git_out("rev-parse", "master")

        (self.root / "src" / "RolledBack.java").write_text("class RolledBack {}\n", encoding="utf-8")
        self._git("add", "src/RolledBack.java")
        self._git("commit", "-m", "Add important feature for test")
        orig_feat_sha = self._git_out("rev-parse", "master")
        self._git("rm", "src/RolledBack.java")
        self._git("commit", "-m", "Automated rollback of commit Add important feature for test.")

        self._git("checkout", "-b", "pr-merge-cp", "release-8.9.0")
        self._git("cherry-pick", "-m", "1", merge_sha)
        pr_head = self._git_out("rev-parse", "HEAD")
        self._git("checkout", "master")

        stage_dir = self.root / "stage_merge"
        _, _, br, _ = prepare_context.stage_pr_and_base_files(
            base_sha=self.base_sha,
            head_sha=pr_head,
            stage_dir=stage_dir,
        )
        mc_info, comps, overall, _, _ = prepare_context.analyze_mainline_and_compare(
            candidates=[(merge_sha, "trailer")],
            mainline_ref="master",
            base_sha=self.base_sha,
            head_sha=pr_head,
            base_root_path=br,
            stage_dir=stage_dir,
        )
        self.assertEqual(overall, "CLEAN")
        self.assertEqual(mc_info[0]["files_modified"], ["src/MergedFix.java"])
        self.assertTrue(mc_info[0]["applied_cleanly_on_base"])
        self.assertEqual(len(comps), 1)
        self.assertEqual(comps[0]["classification"], "identical")

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

        self._git("rm", long_rel_path)
        self._git(
            "commit",
            "-m",
            f"Automated rollback of commit {c1_sha}.\n\n*** Reason for rollback ***\n\nBroke postsubmit",
        )
        r1_sha = self._git_out("rev-parse", "master")
        self.assertEqual(prepare_context.find_revert_on_master(c1_sha, "master"), r1_sha)

        self._git("checkout", "-b", "pr-long-reverted", "release-8.9.0")
        self._git("cherry-pick", c1_sha)
        pr_head = self._git_out("rev-parse", "HEAD")
        self._git("checkout", "master")

        stage_dir = self.root / "stage_long"
        _, _, br, changed_info = prepare_context.stage_pr_and_base_files(
            base_sha=self.base_sha,
            head_sha=pr_head,
            stage_dir=stage_dir,
        )
        self.assertEqual(changed_info[0]["path"], long_rel_path)
        self.assertNotIn("\u200b", changed_info[0]["path"])

        mc_info, comps, overall, vs_path, _ = prepare_context.analyze_mainline_and_compare(
            candidates=[(c1_sha, "trailer")],
            mainline_ref="master",
            base_sha=self.base_sha,
            head_sha=pr_head,
            base_root_path=br,
            stage_dir=stage_dir,
        )
        self.assertEqual(overall, "CLEAN")
        self.assertEqual(mc_info[0]["reverted_by_on_master"], r1_sha)
        self.assertEqual(comps[0]["path"], long_rel_path)
        self.assertIn("REVERTED OR ROLLED BACK ON MASTER", pathlib.Path(vs_path).read_text(encoding="utf-8"))

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
        self.assertFalse(
            prepare_context.is_authentic_gemini_review(
                {
                    "state": "COMMENTED",
                    "user": None,
                    "author_association": "NONE",
                    "body": "## 📋 Gemini Cherry-Pick Review Summary",
                },
                pr_author="other-user",
            )
        )

    def test_multi_commit_merge_tree_and_real_bazel_provenance_formats(self):
        (self.root / "src" / "BUILD").write_text("java_library(name = 'base')\n", encoding="utf-8")
        self._git("add", "src/BUILD")
        self._git("commit", "-m", "Add src/BUILD")
        self._git("branch", "-f", "release-8.9.0", "HEAD")
        self.base_sha = self._git_out("rev-parse", "release-8.9.0")

        (self.root / "src" / "BUILD").write_text("java_library(name = 'master_only_refactor')\n", encoding="utf-8")
        self._git("add", "src/BUILD")
        self._git("commit", "-m", "Refactor src/BUILD on master")

        (self.root / "src" / "BUILD").write_text("java_library(name = 'm1_target')\n", encoding="utf-8")
        (self.root / "src" / "Foo.java").write_text(
            "class Foo {\n  int a() { return 100; }\n  int b() { return 2; }\n}\n",
            encoding="utf-8",
        )
        self._git("add", "src/BUILD", "src/Foo.java")
        self._git("commit", "-m", "Primary fix M1 (#801)")
        m1_sha = self._git_out("rev-parse", "HEAD")

        (self.root / "src" / "Foo.java").write_text(
            "class Foo {\n  int a() { return 101; }\n  int b() { return 20; }\n}\n",
            encoding="utf-8",
        )
        self._git("add", "src/Foo.java")
        self._git("commit", "-m", "Follow-up fix M2 (#802)")
        m2_sha = self._git_out("rev-parse", "HEAD")

        (self.root / "CHANGELOG.md").write_text("notes\n", encoding="utf-8")
        self._git("add", "CHANGELOG.md")
        self._git(
            "commit",
            "-m",
            f"Release 11.0.0-pre.1\n\nCherry picks:\n\n   + deadbeef:\n     Automated rollback of commit {m1_sha}.",
        )
        self.assertIsNone(prepare_context.find_revert_on_master(m1_sha, "master"))

        self._git("checkout", "-b", "pr-multi-merge-tree", "release-8.9.0")
        (self.root / "src" / "BUILD").write_text("java_library(name = 'm1_target_890')\n", encoding="utf-8")
        (self.root / "src" / "Foo.java").write_text(
            "class Foo {\n  int a() { return 101; }\n  int b() { return 20; }\n}\n",
            encoding="utf-8",
        )
        self._git("add", "src/BUILD", "src/Foo.java")
        self._git("commit", "-m", "[8.9.0] Multi-commit cherry-pick")
        pr_head = self._git_out("rev-parse", "HEAD")
        self._git("checkout", "master")

        found_bullets = prepare_context.find_mainline_commits(
            pr_title="[8.9.0] Cherry-pick fixes",
            pr_body=f"Cherry-picks:\n- {m1_sha}\n- {m2_sha}\n",
            pr_commits=[{"sha": pr_head, "commit": {"message": "[8.9.0] Cherry-pick fixes"}}],
            pr_comments=[],
            pr_author="wyv",
            mainline_ref="master",
            base_sha=self.base_sha,
        )
        self.assertEqual([s for s, _ in found_bullets], [m1_sha, m2_sha])

        found_backport_prose = prepare_context.find_mainline_commits(
            pr_title="[8.9.0] Follow-up",
            pr_body="Backport of #802 to release-8.9.0.",
            pr_commits=[{"sha": pr_head, "commit": {"message": "[8.9.0] Follow-up"}}],
            pr_comments=[],
            pr_author="wyv",
            mainline_ref="master",
            base_sha=self.base_sha,
        )
        self.assertEqual([s for s, _ in found_backport_prose], [m2_sha])

        stage_dir = self.root / "stage_mt"
        diff_path, _, br, _ = prepare_context.stage_pr_and_base_files(
            base_sha=self.base_sha,
            head_sha=pr_head,
            stage_dir=stage_dir,
        )
        diff_text = pathlib.Path(diff_path).read_text(encoding="utf-8")
        self.assertIn("return 101;", diff_text)
        self.assertIn("m1_target_890", diff_text)

        # Pass candidates in reverse order (m2 before m1) to verify rev-list --topo-order sorting
        _, comps, overall, _, truncated = prepare_context.analyze_mainline_and_compare(
            candidates=[(m2_sha, "bullet"), (m1_sha, "bullet")],
            mainline_ref="master",
            base_sha=self.base_sha,
            head_sha=pr_head,
            base_root_path=br,
            stage_dir=stage_dir,
        )
        self.assertFalse(truncated)
        by_path = {fc["path"]: fc["classification"] for fc in comps}
        self.assertEqual(by_path["src/Foo.java"], "identical")
        self.assertEqual(by_path["src/BUILD"], "adapted_with_conflict")
        self.assertEqual(overall, "CONFLICT_RESOLVED")

        # Prior review delta must detect binary payload swaps between pushes (unchanged=False)
        self._git("checkout", "-b", "pr-bin-delta", "release-8.9.0")
        (self.root / "src" / "asset.bin").write_bytes(b"\x00\x01\x02")
        self._git("add", "src/asset.bin")
        self._git("commit", "-m", "bin v1")
        bin_v1 = self._git_out("rev-parse", "HEAD")
        (self.root / "src" / "asset.bin").write_bytes(b"\x00\x09\x09")
        self._git("add", "src/asset.bin")
        self._git("commit", "-m", "bin v2")
        bin_v2 = self._git_out("rev-parse", "HEAD")
        self._git("checkout", "master")

        _, _, bin_unchanged, _, _ = prepare_context.analyze_prior_review_delta(
            pr_reviews=[
                {
                    "id": 50,
                    "user": {"login": "bazel-io"},
                    "state": "APPROVED",
                    "commit_id": bin_v1,
                    "submitted_at": "2026-11-18T00:00:00Z",
                    "body": "## 📋 Gemini Cherry-Pick Review Summary\n\n**Verdict:** ✅ Approved",
                }
            ],
            pr_inline_comments=[],
            base_ref="release-8.9.0",
            base_sha=self.base_sha,
            head_sha=bin_v2,
            stage_dir=self.root / "stage_bin_delta",
        )
        self.assertFalse(bin_unchanged)


if __name__ == "__main__":
    unittest.main()
