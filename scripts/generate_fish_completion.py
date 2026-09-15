# Copyright 2020 The Bazel Authors. All rights reserved.
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
"""Generates a fish completion script for Bazel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import subprocess
import tempfile

from absl import app
from absl import flags

flags.DEFINE_string('bazel', None, 'Path to the bazel binary')
flags.DEFINE_string('output', None, 'Where to put the generated fish script')

flags.mark_flag_as_required('bazel')
flags.mark_flag_as_required('output')

FLAGS = flags.FLAGS
_BAZEL = 'bazel'
_FISH_SEEN_SUBCOMMAND_FROM = '__fish_seen_subcommand_from'
_FISH_BAZEL_HEADER = """#!/usr/bin/env fish
# Copyright 2020 The Bazel Authors. All rights reserved.
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

# Fish completion for Bazel commands and options.
#
# This script was generated from a specific Bazel build distribution. See
# https://github.com/bazelbuild/bazel/blob/master/scripts/generate_fish_completion.py
# for details and implementation.
"""
_FISH_BAZEL_COMMAND_LIST_VAR = 'BAZEL_COMMAND_LIST'
_FISH_BAZEL_SEEN_SUBCOMMAND = '__bazel_seen_subcommand'
_FISH_BAZEL_SEEN_SUBCOMMAND_DEF = """
function {} -d "Checks whether the current command line contains a bazel subcommand."
    {} ${}
end
""".format(_FISH_BAZEL_SEEN_SUBCOMMAND, _FISH_SEEN_SUBCOMMAND_FROM,
           _FISH_BAZEL_COMMAND_LIST_VAR)


class BazelCompletionWriter(object):
  """Constructs a Fish completion script for Bazel."""

  def __init__(self, bazel, output_user_root):
    """Initializes writer state.

    Args:
        bazel: String containing a path to the bazel binary to run.
        output_user_root: String path to user root directory used for
          running bazel commands.
    """
    self._bazel = bazel
    self._output_user_root = output_user_root
    self._startup_options = self._get_options_from_bazel(
        ('help', 'startup_options'))
    self._bazel_help_completion_text = self._get_bazel_output(
        ('help', 'completion'))
    self._param_types_by_subcommand = self._get_param_types()
    self._subcommands = self._get_subcommands()

  def write_completion(self, output_file):
    """Writes a Fish completion script for Bazel to an output file.

    Args:
        output_file: File object opened in a writable mode.
    """
    output_file.write(_FISH_BAZEL_HEADER)
    output_file.write('set {} {}\n'.format(
        _FISH_BAZEL_COMMAND_LIST_VAR,
        ' '.join(c.name for c in self._subcommands)))
    output_file.write(_FISH_BAZEL_SEEN_SUBCOMMAND_DEF)
    for opt in self._startup_options:
      opt.write_completion(output_file)
    for sub in self._subcommands:
      sub.write_completion(output_file)

  def _get_bazel_output(self, args):
    return subprocess.check_output(
        (
            self._bazel,
            '--batch',
            '--output_user_root={}'.format(self._output_user_root),
        )
        + tuple(args),
        universal_newlines=True,
    )

  def _get_options_from_bazel(self, bazel_args, **kwargs):
    output = self._get_bazel_output(bazel_args)
    return list(
        Arg.generate_from_help(
            r'^\s*--(\[no\])?(?P<name>\w+)\s+\((?P<desc>.*)\)$', output,
            **kwargs))

  def _get_param_types(self):
    param_types = {}
    for match in re.finditer(
        r'^BAZEL_COMMAND_(?P<subcommand>.*)_ARGUMENT="(?P<type>.*)"$',
        self._bazel_help_completion_text, re.MULTILINE):
      sub = self._normalize_subcommand_name(match.group('subcommand'))
      param_types[sub] = match.group('type')
    return param_types

  def _get_subcommands(self):
    """Runs `bazel help` and parses its output to derive Bazel commands.

    Returns:
        (:obj:`list` of :obj:`Arg`): List of Bazel commands.
    """
    subs = []
    output = self._get_bazel_output(('help',))
    block = re.search(r'Available commands:(.*\n\n)', output, re.DOTALL)
    for sub in Arg.generate_from_help(
        r'^\s*(?P<name>\S+)\s*(?P<desc>\S+.*\.)\s*$',
        block.group(1),
        is_subcommand=True):
      sub.sub_opts = self._get_options_from_bazel(('help', sub.name),
                                                  expected_subcommand=sub.name)
      sub.sub_params = self._get_params(sub.name)
      subs.append(sub)
    return subs

  _BAZEL_QUERY_BY_LABEL = {
      'label': r'//...',
      'label-bin': r'kind(".*_binary", //...)',
      'label-test': r'tests(//...)',
  }

  def _get_params(self, subcommand):
    """Produces a list of param completions for a given Bazel command.

    Uses a previously generated mapping of Bazel commands to parameter types
    to determine how to complete params following a given command. For
    example, `bazel build` expects `label` type params, whereas `bazel info`
    expects an `info-key` type. The param type is finally translated into a
    list of completion strings.

    Args:
        subcommand: Bazel command string.

    Returns:
        (:obj:`list` of :obj:`str`): List of completions based on the param
            type for the given Bazel command.
    """
    name = self._normalize_subcommand_name(subcommand)
    if name not in self._param_types_by_subcommand:
      return []
    params = []
    param_type = self._param_types_by_subcommand[name]
    if param_type.startswith('label'):
      query = self._BAZEL_QUERY_BY_LABEL[param_type]
      params.append("({} query -k '{}' 2>/dev/null)".format(_BAZEL, query))
    elif param_type.startswith('command'):
      match = re.match(r'command\|\{(?P<commands>.*)\}', param_type)
      params.extend(match.group('commands').split(','))
    elif param_type == 'info-key':
      match = re.search(r'BAZEL_INFO_KEYS="(?P<keys>[^"]*)"',
                        self._bazel_help_completion_text)
      params.extend(match.group('keys').split())
    return params

  @staticmethod
  def _normalize_subcommand_name(subcommand):
    return subcommand.strip().lower().replace('_', '-')


class Arg(object):
  """Represents a Bazel argument and its metadata.

    Attributes:
        name: String containing the name of the argument.
        desc: String describing the argument usage.
        is_subcommand: True if this arg represents a Bazel subcommand. Defaults
          to False, indicating that this arg is an option flag.
        expected_subcommand: Nullable string containing a subcommand that this
          option must follow. Defaults to None, indicating that this option or
          subcommand must not follow another subcommand.
        sub_opts: List of Args representing options of a subcommand. Used only
          if is_subcommand is True.
        sub_params: List of Args representing parameters of a subcommand. Used
          only if is_subcommand is True.
  """

  def __init__(self,
               name,
               desc=None,
               is_subcommand=False,
               expected_subcommand=None):
    self.name = name
    self.desc = desc
    self.is_subcommand = is_subcommand
    self.expected_subcommand = expected_subcommand
    self.sub_opts = []
    self.sub_params = []
    self._is_boolean = (self.desc and self.desc.startswith('a boolean'))

  @classmethod
  def generate_from_help(cls, line_regex, text, **kwargs):
    """Generates Arg objects using a line regex on a block of help text.

    Args:
        line_regex: Regular expression string to match a line of text.
        text: String of help text to parse.
        **kwargs: Extra keywords to pass into the Arg constructor.

    Yields:
        Arg objects parsed from the help text.
    """
    for match in re.finditer(line_regex, text, re.MULTILINE):
      kwargs.update(match.groupdict())
      yield cls(**kwargs)

  def write_completion(self, output_file, command=_BAZEL):
    """Writes Fish completion commands to a file.

        Uses the metadata stored in this class to write Fish shell commands
        that enable completion for this Bazel argument.

    Args:
        output_file: File object to write completions into. Must be open in
          a writable mode.
        command: String containing the command name (i.e. "bazel").
    """
    args = self._get_complete_args_base(
        command=command, subcommand=self.expected_subcommand)

    # Argument can be subcommand or option flag.
    if self.is_subcommand:
      args.append('-xa')  # Exclusive subcommand argument.
    else:
      args.append('-l')  # Long option.
    args.append('"{}"'.format(self.name))
    name_index = len(args) - 1

    if self.desc:
      args.extend(('-d', '"{}"'.format(self._escape(self.desc))))

    if not self._is_boolean:
      args.append('-r')  # Require a subsequent parameter.

    # Write completion commands to the file.
    output_file.write(self._complete(args))
    if self._is_boolean:
      # Include the "false" version of a boolean option.
      args[name_index] = '"no{}"'.format(self.name)
      output_file.write(self._complete(args))
    if self.is_subcommand:
      for opt in self.sub_opts:
        opt.write_completion(output_file, command=command)
      self._write_params_completion(output_file, command=command)
      output_file.write('\n')

  def _write_params_completion(self, output_file, command=_BAZEL):
    args = self._get_complete_args_base(command, subcommand=self.name)
    if self.sub_params:
      args.extend(
          ('-fa', '"{}"'.format(self._escape(' '.join(self.sub_params)))))
    output_file.write(self._complete(args))

  @staticmethod
  def _get_complete_args_base(command, subcommand=None):
    """Provides basic arguments for all fish `complete` invocations.

    Args:
        command: Name of the Bazel executable (i.e. "bazel").
        subcommand: Optional Bazel command like "build".

    Returns:
        (:obj:`list` of :obj:`str`): List of args for `complete`.
    """
    args = ['-c', command]

    # Completion pre-condition.
    args.append('-n')
    if subcommand:
      args.append('"{} {}"'.format(_FISH_SEEN_SUBCOMMAND_FROM, subcommand))
    else:
      args.append('"not {}"'.format(_FISH_BAZEL_SEEN_SUBCOMMAND))

    return args

  @staticmethod
  def _complete(args):
    return 'complete {}\n'.format(' '.join(args))

  @staticmethod
  def _escape(text):
    return text.replace('"', r'\"')


def main(argv):
  """Generates fish completion using provided flags."""
  del argv  # Unused.
  with tempfile.TemporaryDirectory() as output_user_root:
    writer = BazelCompletionWriter(FLAGS.bazel, output_user_root)
    with open(FLAGS.output, mode='w') as output:
      writer.write_completion(output)


if __name__ == '__main__':
  app.run(main)
