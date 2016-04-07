# Copyright (c) 2015, Google Inc.
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
# SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION
# OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN
# CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

"""Enumerates the BoringSSL source in src/ and either generates two gypi files
  (boringssl.gypi and boringssl_tests.gypi) for Chromium, or generates
  source-list files for Android."""

import os
import subprocess
import sys
import json


# OS_ARCH_COMBOS maps from OS and platform to the OpenSSL assembly "style" for
# that platform and the extension used by asm files.
OS_ARCH_COMBOS = [
    ('linux', 'arm', 'linux32', [], 'S'),
    ('linux', 'aarch64', 'linux64', [], 'S'),
    ('linux', 'x86', 'elf', ['-fPIC', '-DOPENSSL_IA32_SSE2'], 'S'),
    ('linux', 'x86_64', 'elf', [], 'S'),
    ('mac', 'x86', 'macosx', ['-fPIC', '-DOPENSSL_IA32_SSE2'], 'S'),
    ('mac', 'x86_64', 'macosx', [], 'S'),
    ('win', 'x86', 'win32n', ['-DOPENSSL_IA32_SSE2'], 'asm'),
    ('win', 'x86_64', 'nasm', [], 'asm'),
]

# NON_PERL_FILES enumerates assembly files that are not processed by the
# perlasm system.
NON_PERL_FILES = {
    ('linux', 'arm'): [
        'src/crypto/poly1305/poly1305_arm_asm.S',
        'src/crypto/chacha/chacha_vec_arm.S',
        'src/crypto/cpu-arm-asm.S',
    ],
}


class Chromium(object):

  def __init__(self):
    self.header = \
"""# Copyright (c) 2014 The Chromium Authors. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

# This file is created by generate_build_files.py. Do not edit manually.

"""

  def PrintVariableSection(self, out, name, files):
    out.write('    \'%s\': [\n' % name)
    for f in sorted(files):
      out.write('      \'%s\',\n' % f)
    out.write('    ],\n')

  def WriteFiles(self, files, asm_outputs):
    with open('boringssl.gypi', 'w+') as gypi:
      gypi.write(self.header + '{\n  \'variables\': {\n')

      self.PrintVariableSection(
          gypi, 'boringssl_ssl_sources', files['ssl'])
      self.PrintVariableSection(
          gypi, 'boringssl_crypto_sources', files['crypto'])

      for ((osname, arch), asm_files) in asm_outputs:
        self.PrintVariableSection(gypi, 'boringssl_%s_%s_sources' %
                                  (osname, arch), asm_files)

      gypi.write('  }\n}\n')

    with open('boringssl_tests.gypi', 'w+') as test_gypi:
      test_gypi.write(self.header + '{\n  \'targets\': [\n')

      test_names = []
      for test in sorted(files['test']):
        test_name = 'boringssl_%s' % os.path.splitext(os.path.basename(test))[0]
        test_gypi.write("""    {
      'target_name': '%s',
      'type': 'executable',
      'dependencies': [
        'boringssl.gyp:boringssl',
      ],
      'sources': [
        '%s',
        '<@(boringssl_test_support_sources)',
      ],
      # TODO(davidben): Fix size_t truncations in BoringSSL.
      # https://crbug.com/429039
      'msvs_disabled_warnings': [ 4267, ],
    },\n""" % (test_name, test))
        test_names.append(test_name)

      test_names.sort()

      test_gypi.write('  ],\n  \'variables\': {\n')

      self.PrintVariableSection(
          test_gypi, 'boringssl_test_support_sources', files['test_support'])

      test_gypi.write('    \'boringssl_test_targets\': [\n')

      for test in test_names:
        test_gypi.write("""      '%s',\n""" % test)

      test_gypi.write('    ],\n  }\n}\n')


class Android(object):

  def __init__(self):
    self.header = \
"""# Copyright (C) 2015 The Android Open Source Project
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

"""

  def ExtraFiles(self):
    return ['android_compat_hacks.c', 'android_compat_keywrap.c']

  def PrintVariableSection(self, out, name, files):
    out.write('%s := \\\n' % name)
    for f in sorted(files):
      out.write('  %s\\\n' % f)
    out.write('\n')

  def WriteFiles(self, files, asm_outputs):
    with open('sources.mk', 'w+') as makefile:
      makefile.write(self.header)

      files['crypto'].extend(self.ExtraFiles())
      self.PrintVariableSection(makefile, 'crypto_sources', files['crypto'])
      self.PrintVariableSection(makefile, 'ssl_sources', files['ssl'])
      self.PrintVariableSection(makefile, 'tool_sources', files['tool'])

      for ((osname, arch), asm_files) in asm_outputs:
        self.PrintVariableSection(
            makefile, '%s_%s_sources' % (osname, arch), asm_files)


class AndroidStandalone(Android):
  """AndroidStandalone is for Android builds outside of the Android-system, i.e.

  for applications that wish wish to ship BoringSSL.
  """

  def ExtraFiles(self):
    return []


class Bazel(object):
  """Bazel outputs files suitable for including in Bazel files."""

  def __init__(self):
    self.firstSection = True
    self.header = \
"""# This file is created by generate_build_files.py. Do not edit manually.

"""

  def PrintVariableSection(self, out, name, files):
    if not self.firstSection:
      out.write('\n')
    self.firstSection = False

    out.write('%s = [\n' % name)
    for f in sorted(files):
      out.write('    "%s",\n' % f)
    out.write(']\n')

  def WriteFiles(self, files, asm_outputs):
    with open('BUILD.generated.bzl', 'w+') as out:
      out.write(self.header)

      self.PrintVariableSection(out, 'ssl_headers', files['ssl_headers'])
      self.PrintVariableSection(
          out, 'ssl_internal_headers', files['ssl_internal_headers'])
      self.PrintVariableSection(out, 'ssl_sources', files['ssl'])
      self.PrintVariableSection(out, 'crypto_headers', files['crypto_headers'])
      self.PrintVariableSection(
          out, 'crypto_internal_headers', files['crypto_internal_headers'])
      self.PrintVariableSection(out, 'crypto_sources', files['crypto'])
      self.PrintVariableSection(out, 'tool_sources', files['tool'])

      for ((osname, arch), asm_files) in asm_outputs:
        if osname is not 'linux':
          continue
        self.PrintVariableSection(
            out, 'crypto_sources_%s' % arch, asm_files)

    with open('BUILD.generated_tests.bzl', 'w+') as out:
      out.write(self.header)

      out.write('test_support_sources = [\n')
      for filename in files['test_support']:
        if os.path.basename(filename) == 'malloc.cc':
          continue
        out.write('    "%s",\n' % filename)

      out.write(']\n\n')

      out.write('def create_tests(copts):\n')
      out.write('  test_support_sources_complete = test_support_sources + \\\n')
      out.write('      native.glob(["src/crypto/test/*.h"])\n')
      name_counts = {}
      for test in files['tests']:
        name = os.path.basename(test[0])
        name_counts[name] = name_counts.get(name, 0) + 1

      first = True
      for test in files['tests']:
        name = os.path.basename(test[0])
        if name_counts[name] > 1:
          if '/' in test[1]:
            name += '_' + os.path.splitext(os.path.basename(test[1]))[0]
          else:
            name += '_' + test[1].replace('-', '_')

        if not first:
          out.write('\n')
        first = False

        src_prefix = 'src/' + test[0]
        for src in files['test']:
          if src.startswith(src_prefix):
            src = src
            break
        else:
          raise ValueError("Can't find source for %s" % test[0])

        out.write('  native.cc_test(\n')
        out.write('      name = "%s",\n' % name)
        out.write('      size = "small",\n')
        out.write('      srcs = ["%s"] + test_support_sources_complete,\n' % src)

        data_files = []
        if len(test) > 1:

          out.write('      args = [\n')
          for arg in test[1:]:
            if '/' in arg:
              out.write('          "$(location src/%s)",\n' % arg)
              data_files.append('src/%s' % arg)
            else:
              out.write('          "%s",\n' % arg)
          out.write('      ],\n')

        out.write('      copts = copts,\n')

        if len(data_files) > 0:
          out.write('      data = [\n')
          for filename in data_files:
            out.write('          "%s",\n' % filename)
          out.write('      ],\n')

        if 'ssl/' in test[0]:
          out.write('      deps = [\n')
          out.write('          ":crypto",\n')
          out.write('          ":ssl",\n')
          out.write('      ],\n')
        else:
          out.write('      deps = [":crypto"],\n')
        out.write('  )\n')


def FindCMakeFiles(directory):
  """Returns list of all CMakeLists.txt files recursively in directory."""
  cmakefiles = []

  for (path, _, filenames) in os.walk(directory):
    for filename in filenames:
      if filename == 'CMakeLists.txt':
        cmakefiles.append(os.path.join(path, filename))

  return cmakefiles


def NoTests(dent, is_dir):
  """Filter function that can be passed to FindCFiles in order to remove test
  sources."""
  if is_dir:
    return dent != 'test'
  return 'test.' not in dent and not dent.startswith('example_')


def OnlyTests(dent, is_dir):
  """Filter function that can be passed to FindCFiles in order to remove
  non-test sources."""
  if is_dir:
    return dent != 'test'
  return '_test.' in dent or dent.startswith('example_')


def AllFiles(dent, is_dir):
  """Filter function that can be passed to FindCFiles in order to include all
  sources."""
  return True


def SSLHeaderFiles(dent, is_dir):
  return dent in ['ssl.h', 'tls1.h', 'ssl23.h', 'ssl3.h', 'dtls1.h']


def FindCFiles(directory, filter_func):
  """Recurses through directory and returns a list of paths to all the C source
  files that pass filter_func."""
  cfiles = []

  for (path, dirnames, filenames) in os.walk(directory):
    for filename in filenames:
      if not filename.endswith('.c') and not filename.endswith('.cc'):
        continue
      if not filter_func(filename, False):
        continue
      cfiles.append(os.path.join(path, filename))

    for (i, dirname) in enumerate(dirnames):
      if not filter_func(dirname, True):
        del dirnames[i]

  return cfiles


def FindHeaderFiles(directory, filter_func):
  """Recurses through directory and returns a list of paths to all the header files that pass filter_func."""
  hfiles = []

  for (path, dirnames, filenames) in os.walk(directory):
    for filename in filenames:
      if not filename.endswith('.h'):
        continue
      if not filter_func(filename, False):
        continue
      hfiles.append(os.path.join(path, filename))

  return hfiles


def ExtractPerlAsmFromCMakeFile(cmakefile):
  """Parses the contents of the CMakeLists.txt file passed as an argument and
  returns a list of all the perlasm() directives found in the file."""
  perlasms = []
  with open(cmakefile) as f:
    for line in f:
      line = line.strip()
      if not line.startswith('perlasm('):
        continue
      if not line.endswith(')'):
        raise ValueError('Bad perlasm line in %s' % cmakefile)
      # Remove "perlasm(" from start and ")" from end
      params = line[8:-1].split()
      if len(params) < 2:
        raise ValueError('Bad perlasm line in %s' % cmakefile)
      perlasms.append({
          'extra_args': params[2:],
          'input': os.path.join(os.path.dirname(cmakefile), params[1]),
          'output': os.path.join(os.path.dirname(cmakefile), params[0]),
      })

  return perlasms


def ReadPerlAsmOperations():
  """Returns a list of all perlasm() directives found in CMake config files in
  src/."""
  perlasms = []
  cmakefiles = FindCMakeFiles('src')

  for cmakefile in cmakefiles:
    perlasms.extend(ExtractPerlAsmFromCMakeFile(cmakefile))

  return perlasms


def PerlAsm(output_filename, input_filename, perlasm_style, extra_args):
  """Runs the a perlasm script and puts the output into output_filename."""
  base_dir = os.path.dirname(output_filename)
  if not os.path.isdir(base_dir):
    os.makedirs(base_dir)
  output = subprocess.check_output(
      ['perl', input_filename, perlasm_style] + extra_args)
  with open(output_filename, 'w+') as out_file:
    out_file.write(output)


def ArchForAsmFilename(filename):
  """Returns the architectures that a given asm file should be compiled for
  based on substrings in the filename."""

  if 'x86_64' in filename or 'avx2' in filename:
    return ['x86_64']
  elif ('x86' in filename and 'x86_64' not in filename) or '586' in filename:
    return ['x86']
  elif 'armx' in filename:
    return ['arm', 'aarch64']
  elif 'armv8' in filename:
    return ['aarch64']
  elif 'arm' in filename:
    return ['arm']
  else:
    raise ValueError('Unknown arch for asm filename: ' + filename)


def WriteAsmFiles(perlasms):
  """Generates asm files from perlasm directives for each supported OS x
  platform combination."""
  asmfiles = {}

  for osarch in OS_ARCH_COMBOS:
    (osname, arch, perlasm_style, extra_args, asm_ext) = osarch
    key = (osname, arch)
    outDir = '%s-%s' % key

    for perlasm in perlasms:
      filename = os.path.basename(perlasm['input'])
      output = perlasm['output']
      if not output.startswith('src'):
        raise ValueError('output missing src: %s' % output)
      output = os.path.join(outDir, output[4:])
      if output.endswith('-armx.${ASM_EXT}'):
        output = output.replace('-armx',
                                '-armx64' if arch == 'aarch64' else '-armx32')
      output = output.replace('${ASM_EXT}', asm_ext)

      if arch in ArchForAsmFilename(filename):
        PerlAsm(output, perlasm['input'], perlasm_style,
                perlasm['extra_args'] + extra_args)
        asmfiles.setdefault(key, []).append(output)

  for (key, non_perl_asm_files) in NON_PERL_FILES.iteritems():
    asmfiles.setdefault(key, []).extend(non_perl_asm_files)

  return asmfiles


def main(platforms):
  crypto_c_files = FindCFiles(os.path.join('src', 'crypto'), NoTests)
  ssl_c_files = FindCFiles(os.path.join('src', 'ssl'), NoTests)
  tool_cc_files = FindCFiles(os.path.join('src', 'tool'), NoTests)

  # Generate err_data.c
  with open('err_data.c', 'w+') as err_data:
    subprocess.check_call(['go', 'run', 'err_data_generate.go'],
                          cwd=os.path.join('src', 'crypto', 'err'),
                          stdout=err_data)
  crypto_c_files.append('err_data.c')

  test_support_cc_files = FindCFiles(os.path.join('src', 'crypto', 'test'),
                                     AllFiles)

  test_c_files = FindCFiles(os.path.join('src', 'crypto'), OnlyTests)
  test_c_files += FindCFiles(os.path.join('src', 'ssl'), OnlyTests)

  ssl_h_files = (
      FindHeaderFiles(
          os.path.join('src', 'include', 'openssl'),
          SSLHeaderFiles))

  def NotSSLHeaderFiles(filename, is_dir):
    return not SSLHeaderFiles(filename, is_dir)
  crypto_h_files = (
      FindHeaderFiles(
          os.path.join('src', 'include', 'openssl'),
          NotSSLHeaderFiles))

  ssl_internal_h_files = FindHeaderFiles(os.path.join('src', 'ssl'), NoTests)
  crypto_internal_h_files = FindHeaderFiles(
      os.path.join('src', 'crypto'), NoTests)

  with open('src/util/all_tests.json', 'r') as f:
    tests = json.load(f)
  test_binaries = set([test[0] for test in tests])
  test_sources = set([
      test.replace('.cc', '').replace('.c', '').replace(
          'src/',
          '')
      for test in test_c_files])
  if test_binaries != test_sources:
    print 'Test sources and configured tests do not match'
    a = test_binaries.difference(test_sources)
    if len(a) > 0:
      print 'These tests are configured without sources: ' + str(a)
    b = test_sources.difference(test_binaries)
    if len(b) > 0:
      print 'These test sources are not configured: ' + str(b)

  files = {
      'crypto': crypto_c_files,
      'crypto_headers': crypto_h_files,
      'crypto_internal_headers': crypto_internal_h_files,
      'ssl': ssl_c_files,
      'ssl_headers': ssl_h_files,
      'ssl_internal_headers': ssl_internal_h_files,
      'tool': tool_cc_files,
      'test': test_c_files,
      'test_support': test_support_cc_files,
      'tests': tests,
  }

  asm_outputs = sorted(WriteAsmFiles(ReadPerlAsmOperations()).iteritems())

  for platform in platforms:
    platform.WriteFiles(files, asm_outputs)

  return 0


def Usage():
  print 'Usage: python %s [chromium|android|android-standalone|bazel]' % sys.argv[0]
  sys.exit(1)


if __name__ == '__main__':
  if len(sys.argv) < 2:
    Usage()

  platforms = []
  for s in sys.argv[1:]:
    if s == 'chromium' or s == 'gyp':
      platforms.append(Chromium())
    elif s == 'android':
      platforms.append(Android())
    elif s == 'android-standalone':
      platforms.append(AndroidStandalone())
    elif s == 'bazel':
      platforms.append(Bazel())
    else:
      Usage()

  sys.exit(main(platforms))
