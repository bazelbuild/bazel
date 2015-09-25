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
and -dontnote and -dontwarn and -checkdiscard when they are provided with
arguments.
"""

import re
import sys

from third_party.py import gflags

gflags.DEFINE_string('path', None, 'Path to the proguard config to validate')
gflags.DEFINE_string('output', None, 'Where to put the validated config')

FLAGS = gflags.FLAGS
PROGUARD_COMMENTS_PATTERN = '#.*(\n|$)'


def main():
  with open(FLAGS.path) as config:
    config_string = config.read()
    invalid_configs = Validate(config_string)
    if invalid_configs:
      raise RuntimeError('Invalid proguard config parameters: '
                         + str(invalid_configs))
  with open(FLAGS.output, 'w+') as outconfig:
    config_string = ('# Merged from %s \n' % FLAGS.path) + config_string
    outconfig.write(config_string)


def Validate(config):
  """Checks the config for illegal arguments."""
  config = re.sub(PROGUARD_COMMENTS_PATTERN, '', config)
  args = config.split('-')
  invalid_configs = []
  for arg in args:
    arg = arg.strip()
    if not arg:
      continue
    elif arg.startswith('checkdiscard'):
      continue
    elif arg.startswith('keep'):
      continue
    elif arg.startswith('assumenosideeffects'):
      continue
    elif arg.split()[0] == 'dontnote':
      if len(arg.split()) > 1:
        continue
    elif arg.split()[0] == 'dontwarn':
      if len(arg.split()) > 1:
        continue
    invalid_configs.append('-' + arg.split()[0])

  return invalid_configs

if __name__ == '__main__':
  FLAGS(sys.argv)
  main()
