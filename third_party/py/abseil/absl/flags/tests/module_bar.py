# Copyright 2017 The Abseil Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Auxiliary module for testing flags.py.

The purpose of this module is to define a few flags.  We want to make
sure the unit tests for flags.py involve more than one module.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from absl.flags import _helpers

FLAGS = flags.FLAGS


def define_flags(flag_values=FLAGS):
  """Defines some flags.

  Args:
    flag_values: The FlagValues object we want to register the flags
      with.
  """
  # The 'tmod_bar_' prefix (short for 'test_module_bar') ensures there
  # is no name clash with the existing flags.
  flags.DEFINE_boolean('tmod_bar_x', True, 'Boolean flag.',
                       flag_values=flag_values)
  flags.DEFINE_string('tmod_bar_y', 'default', 'String flag.',
                      flag_values=flag_values)
  flags.DEFINE_boolean('tmod_bar_z', False,
                       'Another boolean flag from module bar.',
                       flag_values=flag_values)
  flags.DEFINE_integer('tmod_bar_t', 4, 'Sample int flag.',
                       flag_values=flag_values)
  flags.DEFINE_integer('tmod_bar_u', 5, 'Sample int flag.',
                       flag_values=flag_values)
  flags.DEFINE_integer('tmod_bar_v', 6, 'Sample int flag.',
                       flag_values=flag_values)


def remove_one_flag(flag_name, flag_values=FLAGS):
  """Removes the definition of one flag from flags.FLAGS.

  Note: if the flag is not defined in flags.FLAGS, this function does
  not do anything (in particular, it does not raise any exception).

  Motivation: We use this function for cleanup *after* a test: if
  there was a failure during a test and not all flags were declared,
  we do not want the cleanup code to crash.

  Args:
    flag_name: A string, the name of the flag to delete.
    flag_values: The FlagValues object we remove the flag from.
  """
  if flag_name in flag_values:
    flag_values.__delattr__(flag_name)


def names_of_defined_flags():
  """Returns: List of names of the flags declared in this module."""
  return ['tmod_bar_x',
          'tmod_bar_y',
          'tmod_bar_z',
          'tmod_bar_t',
          'tmod_bar_u',
          'tmod_bar_v']


def remove_flags(flag_values=FLAGS):
  """Deletes the flag definitions done by the above define_flags().

  Args:
    flag_values: The FlagValues object we remove the flags from.
  """
  for flag_name in names_of_defined_flags():
    remove_one_flag(flag_name, flag_values=flag_values)


def get_module_name():
  """Uses get_calling_module() to return the name of this module.

  For checking that get_calling_module works as expected.

  Returns:
    A string, the name of this module.
  """
  return _helpers.get_calling_module()


def execute_code(code, global_dict):
  """Executes some code in a given global environment.

  For testing of get_calling_module.

  Args:
    code: A string, the code to be executed.
    global_dict: A dictionary, the global environment that code should
      be executed in.
  """
  # Indeed, using exec generates a lint warning.  But some user code
  # actually uses exec, and we have to test for it ...
  exec(code, global_dict)  # pylint: disable=exec-used


def disclaim_key_flags():
  """Disclaims flags declared in this module."""
  flags.disclaim_key_flags()
