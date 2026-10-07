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

"""Unit tests for relnotes.py."""

import os
import sys
import unittest

# Ensure the scripts/release directory is in sys.path when executed directly.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
  sys.path.insert(0, _SCRIPT_DIR)

import relnotes  # pylint: disable=g-import-not-at-top


class RelnotesTest(unittest.TestCase):

  def test_parse_version_standard(self):
    self.assertEqual(relnotes.parse_version("1.2.3"), (1, 2, 3))
    self.assertEqual(relnotes.parse_version("7.0.0"), (7, 0, 0))
    self.assertEqual(relnotes.parse_version("9.1.2"), (9, 1, 2))

  def test_parse_version_two_digit_major(self):
    self.assertEqual(relnotes.parse_version("10.0.0"), (10, 0, 0))
    self.assertEqual(relnotes.parse_version("10.1.2"), (10, 1, 2))
    self.assertEqual(relnotes.parse_version("11.0.0"), (11, 0, 0))

  def test_parse_version_with_v_prefix(self):
    self.assertEqual(relnotes.parse_version("v1.0.0"), (1, 0, 0))
    self.assertEqual(relnotes.parse_version("v10.0.0"), (10, 0, 0))

  def test_parse_version_prerelease_and_rc(self):
    self.assertEqual(relnotes.parse_version("10.0.0rc1"), (10, 0, 0))
    self.assertEqual(
        relnotes.parse_version("10.0.0-pre.20261001.1"), (10, 0, 0)
    )

  def test_parse_version_invalid(self):
    self.assertEqual(relnotes.parse_version(""), ())
    self.assertEqual(relnotes.parse_version("not-a-version"), ())

  def test_version_sorting_two_digit_major(self):
    tags = ["1.0.0", "10.0.0", "2.0.0", "9.0.0", "2.1.0"]
    tags.sort(key=relnotes.parse_version)
    expected = ["1.0.0", "2.0.0", "2.1.0", "9.0.0", "10.0.0"]
    self.assertEqual(tags, expected)

    current_release = "10.0.0"
    last_release = tags[tags.index(current_release) - 1]
    self.assertEqual(last_release, "9.0.0")

  def test_extract_title(self):
    lines = [
        "[10.0.0] Fix issue with query (#1234)",
        "Details about the fix",
    ]
    self.assertEqual(
        relnotes.extract_title(lines), "Fix issue with query (#1234)"
    )

  def test_extract_relnotes_standard(self):
    commit_lines = [
        "Add feature X (#456)",
        "",
        "RELNOTES: Feature X is now supported.",
        "",
        "PiperOrigin-RevId: 123456789",
    ]
    self.assertEqual(
        relnotes.extract_relnotes(commit_lines),
        "Feature X is now supported. (#456)",
    )

  def test_extract_relnotes_incompatible(self):
    commit_lines = [
        "Change behavior of flag Y (#789)",
        "",
        "RELNOTES[INC]: Flag Y is flipped to true.",
    ]
    self.assertEqual(
        relnotes.extract_relnotes(commit_lines),
        "**[Incompatible]** Flag Y is flipped to true. (#789)",
    )

  def test_extract_relnotes_none(self):
    commit_lines = [
        "Internal refactoring",
        "",
        "RELNOTES: None.",
    ]
    self.assertIsNone(relnotes.extract_relnotes(commit_lines))


if __name__ == "__main__":
  unittest.main()
