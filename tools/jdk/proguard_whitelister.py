# Copyright 2015 The Bazel Authors. All rights reserved.
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

"""Checks for proguard configuration rules that cannot be combined across libs.

The only valid proguard arguments for a library are -keep, -assumenosideeffects,
and -dontnote and -dontwarn when they are provided with arguments.
Limiting libraries to using these flags prevents drastic, sweeping effects
(such as obfuscation being disabled) from being inadvertently applied to a
binary through a library dependency.
"""

import re
import sys

from third_party.py import gflags

gflags.DEFINE_string('path', None, 'Path to the proguard config to validate')
gflags.DEFINE_string('output', None, 'Where to put the validated config')

FLAGS = gflags.FLAGS
PROGUARD_COMMENTS_PATTERN = '#.*(\n|$)'


class ProguardConfigValidator(object):
  """Validates a proguard config."""

  # Must be a tuple for str.startswith()
  _VALID_ARGS = ('keep', 'assumenosideeffects')

  def __init__(self, config_path, outconfig_path):
    self._config_path = config_path
    self._outconfig_path = outconfig_path

  def ValidateAndWriteOutput(self):
    with open(self._config_path) as config:
      config_string = config.read()
      invalid_configs = self._Validate(config_string)
      if invalid_configs:
        raise RuntimeError(
            'Invalid proguard config parameters: ' + str(invalid_configs))
    with open(self._outconfig_path, 'w+') as outconfig:
      config_string = '# Merged from %s \n%s' % (
          self._config_path, config_string)
      outconfig.write(config_string)

  def _Validate(self, config):
    """Checks the config for illegal arguments."""
    config = re.sub(PROGUARD_COMMENTS_PATTERN, '', config)
    args = config.split('-')
    invalid_configs = []
    for arg in args:
      arg = arg.strip()
      if not arg or self._ValidateArg(arg):
        continue
      invalid_configs.append('-' + arg.split()[0])

    return invalid_configs

  def _ValidateArg(self, arg):
    if arg.startswith(ProguardConfigValidator._VALID_ARGS):
      return True
    elif arg.split()[0] in ['dontnote', 'dontwarn']:
      if len(arg.split()) > 1:
        return True
    return False


def main():
  validator = ProguardConfigValidator(FLAGS.path, FLAGS.output)
  validator.ValidateAndWriteOutput()


if __name__ == '__main__':
  FLAGS(sys.argv)
  main()
