#!/usr/bin/env python3
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Proof of concept license checker.

This is only a demonstration. It will be replaced with other tools.
"""

import argparse
import codecs
import json

# Conditions allowed for all applications
_ALWAYS_ALLOWED_CONDITIONS = frozenset(['notice', 'permissive', 'unencumberd'])


def _get_licenses(licenses_info):
  with codecs.open(licenses_info, encoding='utf-8') as licenses_file:
    return json.loads(licenses_file.read())


def _do_report(out, licenses):
  """Produce a report showing the set of licenses being used.

  Args:
    out: file object to write to
    licenses: list of LicenseInfo objects

  Returns:
    0 for no restricted licenses.
  """
  for lic in licenses:  # using strange name lic because license is built-in
    rule = lic['rule']
    for kind in lic['license_kinds']:
      out.write('= %s\n  kind: %s\n' % (rule, kind['target']))
      out.write('  conditions: %s\n' % kind['conditions'])


def _check_conditions(out, licenses, allowed_conditions):
  """Check that the application does not use any disallowed licenses.

  Args:
    out: file object to write to
    licenses: list of LicenseInfo objects
    allowed_conditions: list of allowed condition names

  Returns:
    0 for no licenses from outside allowed_conditions.
  """
  err = 0
  for lic in licenses:  # using strange name lic because license is built-in
    rule = lic['rule']
    for kind in lic['license_kinds']:
      disallowed = []
      for condition in kind['conditions']:
        if condition not in allowed_conditions:
          disallowed.append(condition)
      if disallowed:
        out.write('ERROR: %s\n' % rule)
        out.write('   kind: %s\n' % kind['target'])
        out.write('   conditions: %s\n' % kind['conditions'])
        out.write('   disallowed condition: %s\n' % ','.join(disallowed))
        err += 1
  return err


def _do_copyright_notices(out, licenses):
  for l in licenses:
    # IGNORE_COPYRIGHT: Not a copyright notice. It is a variable holding one.
    out.write('package(%s), copyright(%s)\n' % (l.get('package_name') or '<unknown>',
                                                l['copyright_notice']))


def _do_licenses(out, licenses):
  for lic in licenses:
    path = lic['license_text']
    with codecs.open(path, encoding='utf-8') as license_file:
      out.write('= %s\n' % path)
      out.write(license_file.read())


def main():
  parser = argparse.ArgumentParser(
      description='Demonstraton license compliance checker')

  parser.add_argument('--licenses_info',
                      help='path to JSON file containing all license data')
  parser.add_argument('--report', default='report', help='Summary report')
  parser.add_argument('--copyright_notices', 
                      help='output file of all copyright notices')
  parser.add_argument('--license_texts', help='output file of all license files')
  parser.add_argument('--check_conditions', action='store_true',
                      help='check that the dep only includes allowed license conditions')
  args = parser.parse_args()

  licenses = _get_licenses(args.licenses_info)
  err = 0
  with codecs.open(args.report, mode='w', encoding='utf-8') as rpt:
    _do_report(rpt, licenses)
    if args.check_conditions:
      # TODO(aiuto): Read conditions from a file of allowed conditions for
      # a specified application deployment environment.
      err = _check_conditions(rpt, licenses, _ALWAYS_ALLOWED_CONDITIONS)
  if args.copyright_notices:
    with codecs.open(
        args.copyright_notices, mode='w', encoding='utf-8') as out:
      _do_copyright_notices(out, licenses)
  if args.license_texts:
    with codecs.open(args.license_texts, mode='w', encoding='utf-8') as out:
      _do_licenses(out, licenses)
  return err


if __name__ == '__main__':
  main()
