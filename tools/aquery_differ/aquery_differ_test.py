# Copyright 2018 The Bazel Authors. All rights reserved.
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

import unittest
from src.main.protobuf import analysis_pb2
from tools.aquery_differ import aquery_differ
from third_party.py import mock
try:
  # Python 2
  from cStringIO import StringIO
except ImportError:
  # Python 3
  from io import StringIO


def make_aquery_output(actions, artifact_paths):
  action_graph = analysis_pb2.ActionGraphContainer()

  for artifact_path in artifact_paths:
    next_id = len(action_graph.artifacts)
    artifact = action_graph.artifacts.add()
    artifact.id = str(next_id)
    artifact.exec_path = artifact_path

  for next_action in actions:
    action = action_graph.actions.add()
    action.output_ids.extend(next_action["output_ids"])
    action.arguments.extend(next_action["arguments"])

    if "input_dep_set_ids" in next_action:
      action.input_dep_set_ids.extend(next_action["input_dep_set_ids"])

  return action_graph


def make_aquery_output_with_dep_set(actions, artifact_paths, dep_sets):
  action_graph = make_aquery_output(actions, artifact_paths)

  for ds in dep_sets:
    next_id = len(action_graph.dep_set_of_files)
    dep_set = action_graph.dep_set_of_files.add()
    dep_set.id = str(next_id)
    dep_set.direct_artifact_ids.extend(ds["direct_artifact_ids"])
    dep_set.transitive_dep_set_ids.extend(ds["transitive_dep_set_ids"])

  return action_graph


class CmdLineDifferTest(unittest.TestCase):

  def test_no_difference(self):
    action_graph = make_aquery_output(
        actions=[{
            "arguments": ["-a", "-b"],
            "output_ids": ["0", "1"]
        }, {
            "arguments": ["-c"],
            "output_ids": ["2"]
        }],
        artifact_paths=["exec/path/zero", "exec/path/one", "exec/path/two"])
    mock_stdout = StringIO()
    attrs = ["cmdline"]
    with mock.patch("sys.stdout", mock_stdout):
      aquery_differ._aquery_diff(action_graph, action_graph, attrs, "before",
                                 "after")
      self.assertEqual(mock_stdout.getvalue(), "No difference\n")

  def test_no_difference_different_output_files_order(self):
    first = make_aquery_output(
        actions=[
            {
                "arguments": ["-a", "-b"],
                "output_ids": ["0", "1"]
            },
        ],
        artifact_paths=["exec/path/zero", "exec/path/one"])
    second = make_aquery_output(
        actions=[
            {
                "arguments": ["-a", "-b"],
                "output_ids": ["1", "0"]
            },
        ],
        artifact_paths=["exec/path/zero", "exec/path/one"])

    mock_stdout = StringIO()
    attrs = ["cmdline"]
    with mock.patch("sys.stdout", mock_stdout):
      aquery_differ._aquery_diff(first, second, attrs, "before", "after")
      self.assertEqual(mock_stdout.getvalue(), "No difference\n")

  def test_first_has_extra_output_files(self):
    first = make_aquery_output(
        actions=[
            {
                "arguments": ["-a", "-b"],
                "output_ids": ["0", "1"]
            },
            {
                "arguments": ["-c"],
                "output_ids": ["2"]
            },
        ],
        artifact_paths=["exec/path/zero", "exec/path/one", "exec/path/two"],
    )
    second = make_aquery_output(
        actions=[
            {
                "arguments": ["-a", "-b"],
                "output_ids": ["1", "0"]
            },
        ],
        artifact_paths=["exec/path/zero", "exec/path/one", "exec/path/two"],
    )

    expected_error = ("Aquery output 'before' change contains an action "
                      "that generates the following outputs that aquery "
                      "output 'after' change doesn't:\nexec/path/two\n\n")
    mock_stdout = StringIO()
    attrs = ["cmdline"]
    with mock.patch("sys.stdout", mock_stdout):
      aquery_differ._aquery_diff(first, second, attrs, "before", "after")
      self.assertEqual(mock_stdout.getvalue(), expected_error)

  def test_second_has_extra_output_files(self):
    first = make_aquery_output(
        actions=[
            {
                "arguments": ["-a", "-b"],
                "output_ids": ["0", "1"]
            },
        ],
        artifact_paths=["exec/path/zero", "exec/path/one", "exec/path/two"],
    )
    second = make_aquery_output(
        actions=[
            {
                "arguments": ["-a", "-b"],
                "output_ids": ["0", "1"]
            },
            {
                "arguments": ["-c"],
                "output_ids": ["2"]
            },
        ],
        artifact_paths=["exec/path/zero", "exec/path/one", "exec/path/two"],
    )

    expected_error = ("Aquery output 'after' change contains an action that"
                      " generates the following outputs that aquery"
                      " output 'before' change doesn't:\nexec/path/two\n\n")
    mock_stdout = StringIO()
    attrs = ["cmdline"]
    with mock.patch("sys.stdout", mock_stdout):
      aquery_differ._aquery_diff(first, second, attrs, "before", "after")
      self.assertEqual(mock_stdout.getvalue(), expected_error)

  def test_different_command_lines(self):
    first = make_aquery_output(
        actions=[
            {
                "arguments": ["-a", "-d"],
                "output_ids": ["0", "1"]
            },
            {
                "arguments": ["-c"],
                "output_ids": ["2"]
            },
        ],
        artifact_paths=["exec/path/zero", "exec/path/one", "exec/path/two"],
    )
    second = make_aquery_output(
        actions=[
            {
                "arguments": ["-a", "-b"],
                "output_ids": ["0", "1"]
            },
            {
                "arguments": ["-c", "-d"],
                "output_ids": ["2"]
            },
        ],
        artifact_paths=["exec/path/zero", "exec/path/one", "exec/path/two"],
    )

    expected_error_one = "\n".join([
        "Difference in the action that generates the following output(s):",
        "\texec/path/two", "--- before", "+++ after", "@@ -1 +1,2 @@", " -c",
        "+-d", "\n"
    ])
    expected_error_two = "\n".join([
        "Difference in the action that generates the following output(s):",
        "\texec/path/one", "\texec/path/zero", "--- before", "+++ after",
        "@@ -1,2 +1,2 @@", " -a", "--d", "+-b", "\n"
    ])
    attrs = ["cmdline"]

    mock_stdout = StringIO()
    with mock.patch("sys.stdout", mock_stdout):
      aquery_differ._aquery_diff(first, second, attrs, "before", "after")
      self.assertIn(expected_error_one, mock_stdout.getvalue())
      self.assertIn(expected_error_two, mock_stdout.getvalue())

  def test_different_inputs(self):
    first = make_aquery_output_with_dep_set(
        actions=[{
            "arguments": [],
            "output_ids": ["0", "1"],
            "input_dep_set_ids": ["1"]
        }],
        artifact_paths=["exec/path/zero", "exec/path/one"],
        dep_sets=[{
            "transitive_dep_set_ids": [],
            "direct_artifact_ids": ["0"]
        }, {
            "transitive_dep_set_ids": ["0"],
            "direct_artifact_ids": ["1"]
        }])
    second = make_aquery_output_with_dep_set(
        actions=[
            {
                "arguments": [],
                "output_ids": ["0", "1"],
                "input_dep_set_ids": ["0"]
            },
        ],
        artifact_paths=["exec/path/zero", "exec/path/one"],
        dep_sets=[{
            "transitive_dep_set_ids": [],
            "direct_artifact_ids": ["0"]
        }])

    expected_error_one = "\n".join([
        "Difference in the action that generates the following output(s):",
        "\texec/path/one", "\texec/path/zero", "--- before", "+++ after",
        "@@ -1,2 +1 @@", "-exec/path/one", " exec/path/zero", "\n"
    ])
    attrs = ["inputs"]

    mock_stdout = StringIO()
    with mock.patch("sys.stdout", mock_stdout):
      aquery_differ._aquery_diff(first, second, attrs, "before", "after")
      self.assertIn(expected_error_one, mock_stdout.getvalue())


if __name__ == "__main__":
  unittest.main()
