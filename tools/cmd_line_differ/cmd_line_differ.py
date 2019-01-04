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
r"""Command line diffing tool that compares two bazel aquery invocations.

This script compares the proto output of two bazel aquery invocations. For
each set of output files of an action, it compares the command lines that
generated the files.

Example usage:
bazel aquery //path/to:target_one --output=textproto > \
    /path/to/output_one.textproto
bazel aquery //path/to:target_two --output=textproto > \
    /path/to/output_two.textproto

From a bazel repo:
bazel run //tools/cmd_line_differ:cmd_line_differ -- \
--before=/path/to/output_one.textproto \
--after=/path/to/output_two.textproto
--input_type=textproto
"""

import os
from absl import app
from absl import flags
from google.protobuf import text_format
from src.main.protobuf import analysis_pb2

flags.DEFINE_string("before", None, "Aquery output before the change")
flags.DEFINE_string("after", None, "Aquery output after the change")
flags.DEFINE_enum(
    "input_type", "proto", ["proto", "textproto"],
    "The format of the aquery proto input. One of 'proto' and 'textproto.")
flags.mark_flag_as_required("before")
flags.mark_flag_as_required("after")


def _map_artifact_id_to_path(artifacts):
  return {artifact.id: artifact.exec_path for artifact in artifacts}


def _map_output_files_to_command_line(actions, artifacts):
  output_files_to_command_line = {}
  for action in actions:
    output_files = " ".join(
        sorted([artifacts[output_id] for output_id in action.output_ids]))
    output_files_to_command_line[output_files] = action.arguments
  return output_files_to_command_line


def _aquery_diff(before, after):
  """Returns differences between command lines that generate same outputs."""
  # TODO(bazel-team): Currently we compare only command lines of actions that
  # generate the same output files. Expand the differ to compare other values as
  # well (e.g. mnemonic, inputs, execution tags...).

  found_difference = False
  artifacts_before = _map_artifact_id_to_path(before.artifacts)
  artifacts_after = _map_artifact_id_to_path(after.artifacts)

  output_to_command_line_before = _map_output_files_to_command_line(
      before.actions, artifacts_before)
  output_to_command_line_after = _map_output_files_to_command_line(
      after.actions, artifacts_after)

  output_files_before = set(output_to_command_line_before.keys())
  output_files_after = set(output_to_command_line_after.keys())

  before_after_diff = output_files_before - output_files_after
  after_before_diff = output_files_after - output_files_before

  if before_after_diff:
    print(("Aquery output before change contains an action that generates "
           "the following outputs that aquery output after change doesn't:"
           "\n%s\n") % "\n".join(before_after_diff))
    found_difference = True
  if after_before_diff:
    print(("Aquery output after change contains an action that generates "
           "the following outputs that aquery output before change doesn't:"
           "\n%s\n") % "\n".join(after_before_diff))
    found_difference = True

  for output_files in output_to_command_line_before:
    arguments = output_to_command_line_before[output_files]
    after_arguments = output_to_command_line_after.get(output_files, None)
    if after_arguments and arguments != after_arguments:
      print(("Difference in action that generates the following outputs:\n%s\n"
             "Aquery output before change has the following command line:\n%s\n"
             "Aquery output after change has the following command line:\n%s\n")
            % ("\n".join(output_files.split()), "\n".join(arguments),
               "\n".join(after_arguments)))
      found_difference = True

  if not found_difference:
    print("No difference")


def to_absolute_path(path):
  path = os.path.expanduser(path)
  if os.path.isabs(path):
    return path
  else:
    if "BUILD_WORKING_DIRECTORY" in os.environ:
      return os.path.join(os.environ["BUILD_WORKING_DIRECTORY"], path)
    else:
      return path


def main(unused_argv):

  before_file = to_absolute_path(flags.FLAGS.before)
  after_file = to_absolute_path(flags.FLAGS.after)
  input_type = to_absolute_path(flags.FLAGS.input_type)

  before_proto = analysis_pb2.ActionGraphContainer()
  after_proto = analysis_pb2.ActionGraphContainer()
  if input_type == "proto":
    with open(before_file, "rb") as f:
      before_proto.ParseFromString(f.read())
    with open(after_file, "rb") as f:
      after_proto.ParseFromString(f.read())
  else:
    with open(before_file, "r") as f:
      before_text = f.read()
      text_format.Merge(before_text, before_proto)
    with open(after_file, "r") as f:
      after_text = f.read()
      text_format.Merge(after_text, after_proto)

  _aquery_diff(before_proto, after_proto)


if __name__ == "__main__":
  app.run(main)
