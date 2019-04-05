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

import os
import unittest
import zipfile

from src.test.py.bazel import test_base


class TestWrapperTest(test_base.TestBase):

  @staticmethod
  def _ReadFile(path):
    # Read the runfiles manifest.
    contents = []
    with open(path, 'rt') as f:
      contents = [line.strip() for line in f.readlines()]
    return contents

  def _FailWithOutput(self, output):
    self.fail('FAIL:\n | %s\n---' % '\n | '.join(output))

  def _CreateMockWorkspace(self):
    self.ScratchFile('WORKSPACE')
    # All test targets are called <something>.bat, for the benefit of Windows.
    # This makes test execution faster on Windows for the following reason:
    #
    # When building a sh_test rule, the main output's name is the same as the
    # rule. On Unixes, this output is a symlink to the main script (the first
    # entry in `srcs`), on Windows it's a copy of the file. In fact the main
    # "script" does not have to be a script, it may be any executable file.
    #
    # On Unixes anything with the +x permission can be executed; the file's
    # shebang line specifies the interpreter. On Windows, there's no such
    # mechanism; Bazel runs the main script (which is typically a ".sh" file)
    # through Bash. However, if the main file is a native executable, it's
    # faster to run it directly than through Bash (plus it removes the need for
    # Bash).
    #
    # Therefore on Windows, if the main script is a native executable (such as a
    # ".bat" file) and has the same extension as the main file, Bazel (in case
    # of sh_test) makes a copy of the file and runs it directly, rather than
    # through Bash.
    self.ScratchFile('foo/BUILD', [
        'sh_test(',
        '    name = "passing_test.bat",',
        '    srcs = ["passing.bat"],',
        ')',
        'sh_test(',
        '    name = "failing_test.bat",',
        '    srcs = ["failing.bat"],',
        ')',
        'sh_test(',
        '    name = "printing_test.bat",',
        '    srcs = ["printing.bat"],',
        ')',
        'sh_test(',
        '    name = "runfiles_test.bat",',
        '    srcs = ["runfiles.bat"],',
        '    data = ["passing.bat"],',
        ')',
        'sh_test(',
        '    name = "sharded_test.bat",',
        '    srcs = ["sharded.bat"],',
        '    shard_count = 2,',
        ')',
        'sh_test(',
        '    name = "unexported_test.bat",',
        '    srcs = ["unexported.bat"],',
        ')',
        'sh_test(',
        '    name = "testargs_test.exe",',
        '    srcs = ["testargs.exe"],',
        r'    args = ["foo", "a b", "", "\"c d\"", "\"\"", "bar"],',
        ')',
        'py_test(',
        '    name = "undecl_test",',
        '    srcs = ["undecl_test.py"],',
        '    data = ["dummy.ico", "dummy.dat"],',
        '    deps = ["@bazel_tools//tools/python/runfiles"],',
        ')',
        'py_test(',
        '    name = "annot_test",',
        '    srcs = ["annot_test.py"],',
        ')',
        'py_test(',
        '    name = "xml_test",',
        '    srcs = ["xml_test.py"],',
        ')',
        'py_test(',
        '    name = "xml2_test",',
        '    srcs = ["xml2_test.py"],',
        ')',
    ])
    self.ScratchFile('foo/passing.bat', ['@exit /B 0'], executable=True)
    self.ScratchFile('foo/failing.bat', ['@exit /B 1'], executable=True)
    self.ScratchFile(
        'foo/printing.bat', [
            '@echo lorem ipsum',
            '@echo HOME=%HOME%',
            '@echo TEST_SRCDIR=%TEST_SRCDIR%',
            '@echo TEST_TMPDIR=%TEST_TMPDIR%',
            '@echo USER=%USER%',
        ],
        executable=True)
    self.ScratchFile(
        'foo/runfiles.bat', [
            '@echo MF=%RUNFILES_MANIFEST_FILE%',
            '@echo ONLY=%RUNFILES_MANIFEST_ONLY%',
            '@echo DIR=%RUNFILES_DIR%',
        ],
        executable=True)
    self.ScratchFile(
        'foo/sharded.bat', [
            '@echo STATUS=%TEST_SHARD_STATUS_FILE%',
            '@echo INDEX=%TEST_SHARD_INDEX% TOTAL=%TEST_TOTAL_SHARDS%',
        ],
        executable=True)
    self.ScratchFile(
        'foo/unexported.bat', [
            '@echo GOOD=%HOME%',
            '@echo BAD=%TEST_UNDECLARED_OUTPUTS_MANIFEST%',
        ],
        executable=True)

    self.CopyFile(
        src_path = self.Rlocation("io_bazel/src/test/py/bazel/printargs.exe"),
        dst_path = "foo/testargs.exe",
        executable = True)

    # A single white pixel as an ".ico" file. /usr/bin/file should identify this
    # as "image/x-icon".
    # The MIME type lookup logic of the test wrapper only looks at file names,
    # but the test-setup.sh calls /usr/bin/file which inspects file contents, so
    # we need a valid ".ico" file.
    ico_file = bytearray([
        0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01, 0x00,
        0x18, 0x00, 0x30, 0x00, 0x00, 0x00, 0x16, 0x00, 0x00, 0x00, 0x28, 0x00,
        0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x01, 0x00,
        0x18, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00
    ])
    # 16 bytes of random data. /usr/bin/file should identify this as
    # "application/octet-stream".
    # The MIME type lookup logic of the test wrapper only looks at file names,
    # but the test-setup.sh calls /usr/bin/file which inspects file contents, so
    # we need a valid ".ico" file.
    dat_file = bytearray([
        0x40, 0x5a, 0x2e, 0x7e, 0x53, 0x86, 0x98, 0x0e, 0x12, 0xc4, 0x92, 0x38,
        0x27, 0xcd, 0x09, 0xf9
    ])

    ico_file_path = self.ScratchFile('foo/dummy.ico').replace('/', '\\')
    dat_file_path = self.ScratchFile('foo/dummy.dat').replace('/', '\\')

    with open(ico_file_path, 'wb') as f:
      f.write(ico_file)

    with open(dat_file_path, 'wb') as f:
      f.write(dat_file)

    self.ScratchFile(
        'foo/undecl_test.py', [
            'from bazel_tools.tools.python.runfiles import runfiles',
            'import os',
            'import shutil',
            '',
            'root = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")',
            'os.mkdir(os.path.join(root, "out1"))',
            'os.mkdir(os.path.join(root, "out2"))',
            'os.makedirs(os.path.join(root, "empty/sub"))',
            'r = runfiles.Create()',
            'shutil.copyfile(r.Rlocation("__main__/foo/dummy.ico"),',
            '                os.path.join(root, "out1", "data1.ico"))',
            'shutil.copyfile(r.Rlocation("__main__/foo/dummy.dat"),',
            '                os.path.join(root, "out2", "data2.dat"))',
        ],
        executable=True)

    self.ScratchFile(
        'foo/annot_test.py', [
            'import os',
            'root = os.environ.get("TEST_UNDECLARED_OUTPUTS_ANNOTATIONS_DIR")',
            'dir1 = os.path.join(root, "out1")',
            'dir2 = os.path.join(root, "out2.part")',
            'os.mkdir(dir1)',
            'os.mkdir(dir2)',
            'with open(os.path.join(root, "a.part"), "wt") as f:',
            '  f.write("Hello a")',
            'with open(os.path.join(root, "b.txt"), "wt") as f:',
            '  f.write("Hello b")',
            'with open(os.path.join(root, "c.part"), "wt") as f:',
            '  f.write("Hello c")',
            'with open(os.path.join(dir1, "d.part"), "wt") as f:',
            '  f.write("Hello d")',
            'with open(os.path.join(dir2, "e.part"), "wt") as f:',
            '  f.write("Hello e")',
        ],
        executable=True)

    self.ScratchFile(
        'foo/xml_test.py', [
            'from __future__ import print_function',
            'import time',
            'import sys',
            'print("stdout_line_1")',
            'print("stdout_line_2")',
            'time.sleep(2)',
            'print("stderr_line_1", file=sys.stderr)',
            'print("stderr_line_2", file=sys.stderr)',
        ],
        executable=True)

    self.ScratchFile(
        'foo/xml2_test.py', [
            'import os',
            'with open(os.environ.get("XML_OUTPUT_FILE"), "wt") as f:',
            '  f.write("leave this")'
        ],
        executable=True)

  def _AssertPassingTest(self, flag):
    exit_code, _, stderr = self.RunBazel([
        'test',
        '//foo:passing_test.bat',
        '-t-',
        flag,
    ])
    self.AssertExitCode(exit_code, 0, stderr)

  def _AssertFailingTest(self, flag):
    exit_code, _, stderr = self.RunBazel([
        'test',
        '//foo:failing_test.bat',
        '-t-',
        flag,
    ])
    self.AssertExitCode(exit_code, 3, stderr)

  def _AssertPrintingTest(self, flag):
    exit_code, stdout, stderr = self.RunBazel([
        'test',
        '//foo:printing_test.bat',
        '-t-',
        '--test_output=all',
        flag,
    ])
    self.AssertExitCode(exit_code, 0, stderr)
    lorem = False
    for line in stderr + stdout:
      if line.startswith('lorem ipsum'):
        lorem = True
      elif line.startswith('HOME='):
        home = line[len('HOME='):]
      elif line.startswith('TEST_SRCDIR='):
        srcdir = line[len('TEST_SRCDIR='):]
      elif line.startswith('TEST_TMPDIR='):
        tmpdir = line[len('TEST_TMPDIR='):]
      elif line.startswith('USER='):
        user = line[len('USER='):]
    if not lorem:
      self._FailWithOutput(stderr + stdout)
    if not home:
      self._FailWithOutput(stderr + stdout)
    if not os.path.isabs(home):
      self._FailWithOutput(stderr + stdout)
    if not os.path.isdir(srcdir):
      self._FailWithOutput(stderr + stdout)
    if not os.path.isfile(os.path.join(srcdir, 'MANIFEST')):
      self._FailWithOutput(stderr + stdout)
    if not os.path.isabs(srcdir):
      self._FailWithOutput(stderr + stdout)
    if not os.path.isdir(tmpdir):
      self._FailWithOutput(stderr + stdout)
    if not os.path.isabs(tmpdir):
      self._FailWithOutput(stderr + stdout)
    if not user:
      self._FailWithOutput(stderr + stdout)

  def _AssertRunfiles(self, flag):
    exit_code, stdout, stderr = self.RunBazel([
        'test',
        '//foo:runfiles_test.bat',
        '-t-',
        '--test_output=all',
        # Ensure Bazel does not create a runfiles tree.
        '--enable_runfiles=no',
        flag,
    ])
    self.AssertExitCode(exit_code, 0, stderr)
    mf = mf_only = rf_dir = None
    for line in stderr + stdout:
      if line.startswith('MF='):
        mf = line[len('MF='):]
      elif line.startswith('ONLY='):
        mf_only = line[len('ONLY='):]
      elif line.startswith('DIR='):
        rf_dir = line[len('DIR='):]

    if mf_only != '1':
      self._FailWithOutput(stderr + stdout)

    if not os.path.isfile(mf):
      self._FailWithOutput(stderr + stdout)
    mf_contents = TestWrapperTest._ReadFile(mf)
    # Assert that the data dependency is listed in the runfiles manifest.
    if not any(
        line.split(' ', 1)[0].endswith('foo/passing.bat')
        for line in mf_contents):
      self._FailWithOutput(mf_contents)

    if not os.path.isdir(rf_dir):
      self._FailWithOutput(stderr + stdout)

  def _AssertShardedTest(self, flag):
    exit_code, stdout, stderr = self.RunBazel([
        'test',
        '//foo:sharded_test.bat',
        '-t-',
        '--test_output=all',
        flag,
    ])
    self.AssertExitCode(exit_code, 0, stderr)
    status = None
    index_lines = []
    for line in stderr + stdout:
      if line.startswith('STATUS='):
        status = line[len('STATUS='):]
      elif line.startswith('INDEX='):
        index_lines.append(line)
    if not status:
      self._FailWithOutput(stderr + stdout)
    # Test test-setup.sh / test wrapper only ensure that the directory of the
    # shard status file exist, not that the file itself does too.
    if not os.path.isdir(os.path.dirname(status)):
      self._FailWithOutput(stderr + stdout)
    if sorted(index_lines) != ['INDEX=0 TOTAL=2', 'INDEX=1 TOTAL=2']:
      self._FailWithOutput(stderr + stdout)

  def _AssertUnexportsEnvvars(self, flag):
    exit_code, stdout, stderr = self.RunBazel([
        'test',
        '//foo:unexported_test.bat',
        '-t-',
        '--test_output=all',
        flag,
    ])
    self.AssertExitCode(exit_code, 0, stderr)
    good = bad = None
    for line in stderr + stdout:
      if line.startswith('GOOD='):
        good = line[len('GOOD='):]
      elif line.startswith('BAD='):
        bad = line[len('BAD='):]
    if not good or bad:
      self._FailWithOutput(stderr + stdout)

  def _AssertTestArgs(self, flag, expected):
    exit_code, bazel_bin, stderr = self.RunBazel(['info', 'bazel-bin'])
    self.AssertExitCode(exit_code, 0, stderr)
    bazel_bin = bazel_bin[0]

    exit_code, stdout, stderr = self.RunBazel([
        'test',
        '//foo:testargs_test.exe',
        '-t-',
        '--test_output=all',
        '--test_arg=baz',
        '--test_arg="x y"',
        '--test_arg=""',
        '--test_arg=qux',
        flag,
    ])
    self.AssertExitCode(exit_code, 0, stderr)

    actual = []
    for line in stderr + stdout:
      if line.startswith('arg='):
        actual.append(str(line[len('arg='):]))
    self.assertListEqual(expected, actual)

  def _AssertUndeclaredOutputs(self, flag):
    exit_code, bazel_testlogs, stderr = self.RunBazel(
        ['info', 'bazel-testlogs'])
    self.AssertExitCode(exit_code, 0, stderr)
    bazel_testlogs = bazel_testlogs[0]

    exit_code, _, stderr = self.RunBazel([
        'test',
        '//foo:undecl_test',
        '-t-',
        '--test_output=errors',
        flag,
    ])
    self.AssertExitCode(exit_code, 0, stderr)

    undecl_zip = os.path.join(bazel_testlogs, 'foo', 'undecl_test',
                              'test.outputs', 'outputs.zip')
    self.assertTrue(os.path.exists(undecl_zip))
    zip_content = {}
    with zipfile.ZipFile(undecl_zip, 'r') as z:
      zip_content = {f: z.getinfo(f).file_size for f in z.namelist()}
    self.assertDictEqual(
        zip_content, {
            'out1/': 0,
            'out2/': 0,
            'empty/': 0,
            'empty/sub/': 0,
            'out1/data1.ico': 70,
            'out2/data2.dat': 16
        })

    undecl_mf = os.path.join(bazel_testlogs, 'foo', 'undecl_test',
                             'test.outputs_manifest', 'MANIFEST')
    self.assertTrue(os.path.exists(undecl_mf))
    mf_content = []
    with open(undecl_mf, 'rt') as f:
      mf_content = [line.strip() for line in f.readlines()]
    # Using an ".ico" file as example, because as of 2018-11-09 Bazel's CI
    # machines run Windows Server 2016 core which recognizes fewer MIME types
    # than desktop Windows versions, and one of the recognized types is ".ico"
    # files.
    # Update(2019-03-05): apparently this MIME type is now recognized on CI as
    # as "image/vnd.microsoft.icon". The standard MIME type is "image/x-icon",
    # but Wikipedia lists a few alterantive ones, so the test will accept all of
    # them.
    if len(mf_content) != 2:
      self._FailWithOutput(mf_content)
    tokens = mf_content[0].split('\t')
    if (len(tokens) != 3 or tokens[0] != 'out1/data1.ico' or
        tokens[1] != '70' or tokens[2] not in [
            'image/x-icon', 'image/vnd.microsoft.icon', 'image/ico',
            'image/icon', 'text/ico', 'application/ico'
        ]):
      self._FailWithOutput(mf_content)
    if mf_content[1] != 'out2/data2.dat\t16\tapplication/octet-stream':
      self._FailWithOutput(mf_content)

  def _AssertUndeclaredOutputsAnnotations(self, flag):
    exit_code, bazel_testlogs, stderr = self.RunBazel(
        ['info', 'bazel-testlogs'])
    self.AssertExitCode(exit_code, 0, stderr)
    bazel_testlogs = bazel_testlogs[0]

    exit_code, _, stderr = self.RunBazel([
        'test',
        '//foo:annot_test',
        '-t-',
        '--test_output=errors',
        flag,
    ])
    self.AssertExitCode(exit_code, 0, stderr)

    undecl_annot = os.path.join(bazel_testlogs, 'foo', 'annot_test',
                                'test.outputs_manifest', 'ANNOTATIONS')
    self.assertTrue(os.path.exists(undecl_annot))
    annot_content = []
    with open(undecl_annot, 'rt') as f:
      annot_content = [line.strip() for line in f.readlines()]

    self.assertListEqual(annot_content, ['Hello aHello c'])

  def _AssertXmlGeneration(self, flag, split_xml=False):
    exit_code, bazel_testlogs, stderr = self.RunBazel(
        ['info', 'bazel-testlogs'])
    self.AssertExitCode(exit_code, 0, stderr)
    bazel_testlogs = bazel_testlogs[0]

    exit_code, _, stderr = self.RunBazel([
        'test',
        '//foo:xml_test',
        '-t-',
        '--test_output=errors',
        '--%sexperimental_split_xml_generation' % ('' if split_xml else 'no'),
        flag,
    ])
    self.AssertExitCode(exit_code, 0, stderr)

    test_xml = os.path.join(bazel_testlogs, 'foo', 'xml_test', 'test.xml')
    self.assertTrue(os.path.exists(test_xml))
    duration = 0
    xml_contents = []
    stdout_lines = []
    stderr_lines = []
    with open(test_xml, 'rt') as f:
      xml_contents = [line.strip() for line in f]
    for line in xml_contents:
      if 'duration=' in line:
        line = line[line.find('duration="') + len('duration="'):]
        line = line[:line.find('"')]
        duration = int(line)
      elif 'stdout_line' in line:
        stdout_lines.append(line)
      elif 'stderr_line' in line:
        stderr_lines.append(line)
    # Since stdout and stderr of the test are redirected to the same file, it's
    # possible that a line L1 written to stdout before a line L2 written to
    # stderr is dumped to the file later, i.e. the file will have lines L2 then
    # L1. It is however true that lines printed to the same stream (stdout or
    # stderr) have to preserve their ordering, i.e. if line L3 is printed to
    # stdout after L1, then it must be strictly ordered after L1 (but not
    # necessarily after L2).
    # Therefore we only assert partial ordering of lines.
    if duration <= 1:
      self._FailWithOutput(xml_contents)
    if (len(stdout_lines) != 2 or 'stdout_line_1' not in stdout_lines[0] or
        'stdout_line_2' not in stdout_lines[1]):
      self._FailWithOutput(xml_contents)
    if (len(stderr_lines) != 2 or 'stderr_line_1' not in stderr_lines[0] or
        'stderr_line_2' not in stderr_lines[1]):
      self._FailWithOutput(xml_contents)

  def _AssertXmlGeneratedByTestIsRetained(self, flag, split_xml=False):
    exit_code, bazel_testlogs, stderr = self.RunBazel(
        ['info', 'bazel-testlogs'])
    self.AssertExitCode(exit_code, 0, stderr)
    bazel_testlogs = bazel_testlogs[0]

    exit_code, _, stderr = self.RunBazel([
        'test',
        '//foo:xml2_test',
        '-t-',
        '--test_output=errors',
        '--%sexperimental_split_xml_generation' % ('' if split_xml else 'no'),
        flag,
    ])
    self.AssertExitCode(exit_code, 0, stderr)

    test_xml = os.path.join(bazel_testlogs, 'foo', 'xml2_test', 'test.xml')
    self.assertTrue(os.path.exists(test_xml))
    xml_contents = []
    with open(test_xml, 'rt') as f:
      xml_contents = [line.strip() for line in f.readlines()]
    self.assertListEqual(xml_contents, ['leave this'])

  def testTestExecutionWithTestSetupSh(self):
    self._CreateMockWorkspace()
    flag = '--noincompatible_windows_native_test_wrapper'
    self._AssertPassingTest(flag)
    self._AssertFailingTest(flag)
    self._AssertPrintingTest(flag)
    self._AssertRunfiles(flag)
    self._AssertShardedTest(flag)
    self._AssertUnexportsEnvvars(flag)
    self._AssertTestArgs(
        flag,
        [
            '(foo)',
            '(a)',
            '(b)',
            '(c d)',
            '()',
            '(bar)',
            '(baz)',
            '("x y")',
            # I (laszlocsomor@) don't know the exact reason (as of 2019-04-05)
            # why () and (qux) are mangled as they are, but since I'm planning
            # to phase out test-setup.sh on Windows in favor of the native test
            # wrapper, I don't intend to debug this further.  The test is here
            # merely to guard against unwanted future change of behavior.
            '(\\" qux)'
        ])
    self._AssertUndeclaredOutputs(flag)
    self._AssertUndeclaredOutputsAnnotations(flag)
    self._AssertXmlGeneration(flag, split_xml=False)
    self._AssertXmlGeneration(flag, split_xml=True)
    self._AssertXmlGeneratedByTestIsRetained(flag, split_xml=False)
    self._AssertXmlGeneratedByTestIsRetained(flag, split_xml=True)

  def testTestExecutionWithTestWrapperExe(self):
    self._CreateMockWorkspace()
    flag = '--incompatible_windows_native_test_wrapper'
    self._AssertPassingTest(flag)
    self._AssertFailingTest(flag)
    self._AssertPrintingTest(flag)
    self._AssertRunfiles(flag)
    self._AssertShardedTest(flag)
    self._AssertUnexportsEnvvars(flag)
    self._AssertTestArgs(
        flag,
        [
            '(foo)',
            # TODO(laszlocsomor): assert that "a b" is passed as one argument,
            # not two, after https://github.com/bazelbuild/bazel/issues/6277
            # is fixed.
            '(a)',
            '(b)',
            # TODO(laszlocsomor): assert that the empty string argument is
            # passed, after https://github.com/bazelbuild/bazel/issues/6276
            # is fixed.
            '(c d)',
            '()',
            '(bar)',
            '(baz)',
            '("x y")',
            '("")',
            '(qux)',
        ])
    self._AssertUndeclaredOutputs(flag)
    self._AssertUndeclaredOutputsAnnotations(flag)
    self._AssertXmlGeneration(flag, split_xml=False)
    self._AssertXmlGeneration(flag, split_xml=True)
    self._AssertXmlGeneratedByTestIsRetained(flag, split_xml=False)
    self._AssertXmlGeneratedByTestIsRetained(flag, split_xml=True)


if __name__ == '__main__':
  unittest.main()
