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

import io
import os
import sys
import unittest
from unittest import mock

# Ensure the scripts/release directory is in sys.path when executed directly.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
  sys.path.insert(0, _SCRIPT_DIR)

import relnotes  # pylint: disable=g-import-not-at-top
import requests  # pylint: disable=g-import-not-at-top


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

  def test_is_rolling_release(self):
    self.assertTrue(relnotes.is_rolling_release("11.0.0-pre.20261022.3"))
    self.assertTrue(relnotes.is_rolling_release("11.0.0-pre.20261022.3.1"))
    self.assertTrue(relnotes.is_rolling_release("11.0.0-pre.20261022.3rc1"))
    self.assertFalse(relnotes.is_rolling_release("11.0.0"))
    self.assertFalse(relnotes.is_rolling_release("11.0.0rc1"))
    self.assertFalse(relnotes.is_rolling_release("9.3.1"))
    self.assertFalse(relnotes.is_rolling_release(""))

  @mock.patch.object(requests, "get", autospec=True)
  def test_get_label_returns_team_label_on_success(self, mock_get):
    mock_response = mock.Mock()
    mock_response.ok = True
    mock_response.status_code = 200
    mock_response.json.return_value = [
        {"name": "P2"},
        {"name": "team-Rules-Java "},
    ]
    mock_get.return_value = mock_response

    label = relnotes.get_label("12345", auth="fake-token")

    self.assertEqual(label, "team-Rules-Java")
    mock_get.assert_called_once_with(
        "https://api.github.com/repos/bazelbuild/bazel/issues/12345/labels",
        headers={
            "Authorization": "Bearer fake-token",
            "Accept": "application/vnd.github+json",
        },
        timeout=30,
    )

  @mock.patch.object(requests, "get", autospec=True)
  def test_get_label_returns_none_on_http_error_dict(self, mock_get):
    # Regression test: when GitHub returns an HTTP 422/404 JSON error object
    # (dict) instead of a list of label dicts, get_label must return None
    # rather than iterating over string keys and raising AttributeError.
    mock_response = mock.Mock()
    mock_response.ok = False
    mock_response.status_code = 422
    mock_response.json.return_value = {
        "message": "Validation Failed",
        "errors": [{"code": "unprocessable"}],
        "status": "422",
    }
    mock_get.return_value = mock_response

    stderr = io.StringIO()
    with mock.patch.object(sys, "stderr", stderr):
      label = relnotes.get_label("31056", auth="fake-token")

    self.assertIsNone(label)
    self.assertIn("HTTP 422", stderr.getvalue())
    self.assertIn("#31056", stderr.getvalue())

  @mock.patch.object(requests, "get", autospec=True)
  def test_get_label_returns_none_on_request_exception(self, mock_get):
    mock_get.side_effect = requests.RequestException("connection reset")

    stderr = io.StringIO()
    with mock.patch.object(sys, "stderr", stderr):
      label = relnotes.get_label("12345", auth="fake-token")

    self.assertIsNone(label)
    self.assertIn("Failed to fetch labels for #12345", stderr.getvalue())

  @mock.patch.object(requests, "get", autospec=True)
  def test_get_label_returns_none_on_invalid_json(self, mock_get):
    mock_response = mock.Mock()
    mock_response.ok = True
    mock_response.status_code = 200
    mock_response.json.side_effect = ValueError("No JSON object")
    mock_get.return_value = mock_response

    stderr = io.StringIO()
    with mock.patch.object(sys, "stderr", stderr):
      label = relnotes.get_label("12345", auth="fake-token")

    self.assertIsNone(label)
    self.assertIn("Invalid JSON", stderr.getvalue())

  @mock.patch.object(relnotes, "get_label", autospec=True)
  @mock.patch.object(relnotes, "get_github_token", autospec=True)
  def test_get_categorized_relnotes_fetches_token_once_and_falls_back(
      self, mock_get_token, mock_get_label
  ):
    mock_get_token.return_value = "secret-token"
    mock_get_label.side_effect = ["team-Core", None]
    notes = [
        "Fix bazel query (#101)",
        "Fix deleted author PR (#31056)",
        "Manual note without PR",
        "   ",
    ]

    categorized = relnotes.get_categorized_relnotes(notes)

    mock_get_token.assert_called_once_with()
    self.assertEqual(
        mock_get_label.call_args_list,
        [
            mock.call("101", auth="secret-token"),
            mock.call("31056", auth="secret-token"),
        ],
    )
    self.assertEqual(
        categorized,
        {
            "Core": ["Fix bazel query (#101)"],
            "General": [
                "Fix deleted author PR (#31056)",
                "Manual note without PR",
            ],
        },
    )

  @mock.patch.object(relnotes, "get_github_token", autospec=True)
  def test_get_categorized_relnotes_skips_token_when_no_issues(
      self, mock_get_token
  ):
    categorized = relnotes.get_categorized_relnotes(
        ["Note without issue reference"]
    )

    mock_get_token.assert_not_called()
    self.assertEqual(
        categorized, {"General": ["Note without issue reference"]}
    )


if __name__ == "__main__":
  unittest.main()
