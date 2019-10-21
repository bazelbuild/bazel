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

The purpose of this module is to define a few flags, and declare some
other flags as being important.  We want to make sure the unit tests
for flags.py involve more than one module.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from absl.flags import _helpers
from absl.flags.tests import module_bar

FLAGS = flags.FLAGS


DECLARED_KEY_FLAGS = ['tmod_bar_x', 'tmod_bar_z', 'tmod_bar_t',
                      # Special (not user-defined) flag:
                      'flagfile']


def define_flags(flag_values=FLAGS):
  """Defines a few flags."""
  module_bar.define_flags(flag_values=flag_values)
  # The 'tmod_foo_' prefix (short for 'test_module_foo') ensures that we
  # have no name clash with existing flags.
  flags.DEFINE_boolean('tmod_foo_bool', True, 'Boolean flag from module foo.',
                       flag_values=flag_values)
  flags.DEFINE_string('tmod_foo_str', 'default', 'String flag.',
                      flag_values=flag_values)
  flags.DEFINE_integer('tmod_foo_int', 3, 'Sample int flag.',
                       flag_values=flag_values)


def declare_key_flags(flag_values=FLAGS):
  """Declares a few key flags."""
  for flag_name in DECLARED_KEY_FLAGS:
    flags.declare_key_flag(flag_name, flag_values=flag_values)


def declare_extra_key_flags(flag_values=FLAGS):
  """Declares some extra key flags."""
  flags.adopt_module_key_flags(module_bar, flag_values=flag_values)


def names_of_defined_flags():
  """Returns: list of names of flags defined by this module."""
  return ['tmod_foo_bool', 'tmod_foo_str', 'tmod_foo_int']


def names_of_declared_key_flags():
  """Returns: list of names of key flags for this module."""
  return names_of_defined_flags() + DECLARED_KEY_FLAGS


def names_of_declared_extra_key_flags():
  """Returns the list of names of additional key flags for this module.

  These are the flags that became key for this module only as a result
  of a call to declare_extra_key_flags() above.  I.e., the flags declared
  by module_bar, that were not already declared as key for this
  module.

  Returns:
    The list of names of additional key flags for this module.
  """
  names_of_extra_key_flags = list(module_bar.names_of_defined_flags())
  for flag_name in names_of_declared_key_flags():
    while flag_name in names_of_extra_key_flags:
      names_of_extra_key_flags.remove(flag_name)
  return names_of_extra_key_flags


def remove_flags(flag_values=FLAGS):
  """Deletes the flag definitions done by the above define_flags()."""
  for flag_name in names_of_defined_flags():
    module_bar.remove_one_flag(flag_name, flag_values=flag_values)
  module_bar.remove_flags(flag_values=flag_values)


def get_module_name():
  """Uses get_calling_module() to return the name of this module.

  For checking that _get_calling_module works as expected.

  Returns:
    A string, the name of this module.
  """
  return _helpers.get_calling_module()


def duplicate_flags(flagnames=None):
  """Returns a new FlagValues object with the requested flagnames.

  Used to test DuplicateFlagError detection.

  Args:
    flagnames: str, A list of flag names to create.

  Returns:
    A FlagValues object with one boolean flag for each name in flagnames.
  """
  flag_values = flags.FlagValues()
  for name in flagnames:
    flags.DEFINE_boolean(name, False, 'Flag named %s' % (name,),
                         flag_values=flag_values)
  return flag_values


def define_bar_flags(flag_values=FLAGS):
  """Defines flags from module_bar."""
  module_bar.define_flags(flag_values)
