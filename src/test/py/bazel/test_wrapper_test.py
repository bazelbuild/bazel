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
import zipfile

from absl.testing import absltest
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
    self.ScratchFile(
        'foo/BUILD',
        [
            'load(":native_test.bzl", "bat_test", "exe_test")',
            'bat_test(',
            '    name = "passing_test",',
            '    content = ["@exit /B 0"],',
            ')',
            'bat_test(',
            '    name = "failing_test",',
            '    content = ["@exit /B 1"],',
            ')',
            'bat_test(',
            '    name = "printing_test",',
            '    content = [',
            '        "@echo lorem ipsum",',
            '        "@echo HOME=%HOME%",',
            '        "@echo TEST_SRCDIR=%TEST_SRCDIR%",',
            '        "@echo TEST_TMPDIR=%TEST_TMPDIR%",',
            '        "@echo USER=%USER%",',
            '    ]',
            ')',
            'bat_test(',
            '    name = "runfiles_test",',
            '    content = [',
            '        "@echo off",',
            '        "echo MF=%RUNFILES_MANIFEST_FILE%",',
            '        "echo ONLY=%RUNFILES_MANIFEST_ONLY%",',
            '        "echo DIR=%RUNFILES_DIR%",',
            '        "echo data_path=%1",',
            (
                '        "if exist %1 (echo data_exists=1) else (echo'
                ' data_exists=0)",'
            ),
            '    ],',
            '    data = ["dummy.dat"],',
            '    args = ["$(location dummy.dat)"],',
            ')',
            'bat_test(',
            '    name = "sharded_test",',
            '    content = [',
            '        "type nul > %TEST_SHARD_STATUS_FILE%",',
            '        "@echo STATUS=%TEST_SHARD_STATUS_FILE%",',
            (
                '        "@echo INDEX=%TEST_SHARD_INDEX%'
                ' TOTAL=%TEST_TOTAL_SHARDS%",'
            ),
            '    ],',
            '    shard_count = 2,',
            ')',
            'bat_test(',
            '    name = "unexported_test",',
            '    content = [',
            '        "@echo GOOD=%HOME%",',
            '        "@echo BAD=%TEST_UNDECLARED_OUTPUTS_MANIFEST%",',
            '    ],',
            ')',
            'bat_test(',
            '    name = "print_arg0_test",',
            '    content = [',
            '        "@echo ARG0=%0",',
            '    ],',
            ')',
            'exe_test(',
            '    name = "testargs_test",',
            '    src = "testargs.exe",',
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
            'py_test(',
            '    name = "add_cur_dir_to_path_test",',
            '    srcs = ["add_cur_dir_to_path_test.py"],',
            ')',
        ],
    )

    self.CopyFile(
        src_path=self.Rlocation('io_bazel/src/test/py/bazel/printargs.exe'),
        dst_path='foo/testargs.exe',
        executable=True)

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

    self.CopyFile(
        src_path=self.Rlocation('io_bazel/src/test/py/bazel/native_test.bzl'),
        dst_path='foo/native_test.bzl')

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
            'shutil.copyfile(r.Rlocation("_main/foo/dummy.ico"),',
            '                os.path.join(root, "out1", "data1.ico"))',
            'shutil.copyfile(r.Rlocation("_main/foo/dummy.dat"),',
            '                os.path.join(root, "out2", "my data 2.dat"))',
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

    self.ScratchFile(
        'foo/add_cur_dir_to_path_test.py', [
            'import os',
            'path = os.getenv("PATH")',
            'if ".;" not in path:',
            '  exit(1)'
        ],
        executable=True)

  def _AssertPassingTest(self, flags):
    self.RunBazel(
        [
            'test',
            '//foo:passing_test',
            '-t-',
        ]
        + flags
    )

  def _AssertFailingTest(self, flags):
    exit_code, _, stderr = self.RunBazel(
        [
            'test',
            '//foo:failing_test',
            '-t-',
        ]
        + flags,
        allow_failure=True,
    )
    self.AssertExitCode(exit_code, 3, stderr)

  def _AssertPrintingTest(self, flags):
    _, stdout, stderr = self.RunBazel(
        [
            'test',
            '//foo:printing_test',
            '-t-',
            '--test_output=all',
        ]
        + flags
    )
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

  def _AssertRunfiles(self, flags):
    _, stdout, stderr = self.RunBazel(
        [
            'test',
            '//foo:runfiles_test',
            '-t-',
            '--test_output=all',
            # Ensure Bazel does not create a runfiles tree.
            '--enable_runfiles=no',
        ]
        + flags
    )
    mf = mf_only = rf_dir = path = exists = None
    for line in stderr + stdout:
      if line.startswith('MF='):
        mf = line[len('MF='):]
      elif line.startswith('ONLY='):
        mf_only = line[len('ONLY='):]
      elif line.startswith('DIR='):
        rf_dir = line[len('DIR='):]
      elif line.startswith('data_path='):
        path = line[len('data_path='):]
      elif line.startswith('data_exists='):
        exists = line[len('data_exists='):]

    if mf_only != '1':
      self._FailWithOutput(stderr + stdout)

    if not os.path.isfile(mf):
      self._FailWithOutput(stderr + stdout)
    mf_contents = TestWrapperTest._ReadFile(mf)
    # Assert that the data dependency is listed in the runfiles manifest.
    if not any(
        line.split(' ', 1)[0].endswith('foo/dummy.dat')
        for line in mf_contents):
      self._FailWithOutput(mf_contents)

    if not os.path.isdir(rf_dir):
      self._FailWithOutput(stderr + stdout)

    if not path:
      # Expect the $(location) expansion in 'args' worked
      self._FailWithOutput(stderr + stdout)

    if exists != '0':
      # Runfiles are disabled, expect the runfile symlink to be missing.
      self._FailWithOutput(stderr + stdout)

  def _AssertRunfilesSymlinks(self, flags):
    _, stdout, stderr = self.RunBazel(
        [
            'test',
            '//foo:runfiles_test',
            '-t-',
            '--test_output=all',
            # Ensure Bazel creates a runfiles tree.
            '--enable_runfiles=yes',
        ]
        + flags
    )
    mf_only = rf_dir = path = exists = None
    for line in stderr + stdout:
      if line.startswith('ONLY='):
        mf_only = line[len('ONLY='):]
      elif line.startswith('DIR='):
        rf_dir = line[len('DIR='):]
      elif line.startswith('data_path='):
        path = line[len('data_path='):]
      elif line.startswith('data_exists='):
        exists = line[len('data_exists='):]

    if mf_only == '1':
      self._FailWithOutput(stderr + stdout)

    if not rf_dir or not os.path.isdir(rf_dir):
      self._FailWithOutput(stderr + stdout)

    if not path:
      # Expect the $(location) expansion in 'args' worked
      self._FailWithOutput(stderr + stdout)

    if exists != '1':
      # Runfiles are enabled, expect the runfile symlink to exist.
      self._FailWithOutput(stderr + stdout)

  def _AssertShardedTest(self, flags):
    _, stdout, stderr = self.RunBazel(
        [
            'test',
            '//foo:sharded_test',
            '-t-',
            '--test_output=all',
        ]
        + flags
    )
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

  def _AssertUnexportsEnvvars(self, flags):
    _, stdout, stderr = self.RunBazel(
        [
            'test',
            '//foo:unexported_test',
            '-t-',
            '--test_output=all',
        ]
        + flags
    )
    good = bad = None
    for line in stderr + stdout:
      if line.startswith('GOOD='):
        good = line[len('GOOD='):]
      elif line.startswith('BAD='):
        bad = line[len('BAD='):]
    if not good or bad:
      self._FailWithOutput(stderr + stdout)

  def _AssertTestBinaryLocation(self, flags):
    _, bazel_bin, _ = self.RunBazel(['info', 'bazel-bin'])
    bazel_bin = bazel_bin[0].replace('/', '\\')

    _, stdout, stderr = self.RunBazel(
        [
            'test',
            '//foo:print_arg0_test',
            '--test_output=all',
        ]
        + flags
    )

    arg0 = None
    for line in stderr + stdout:
      if line.startswith('ARG0='):
        arg0 = line[len('ARG0='):]
    # Get rid of the quotes if there is any
    if arg0[0] == '"' and arg0[-1] == '"':
      arg0 = arg0[1:-1]
    # The test binary should located at the bazel bin folder
    self.assertEqual(arg0, os.path.join(bazel_bin, 'foo\\print_arg0_test.bat'))

    _, stdout, stderr = self.RunBazel(
        [
            'test',
            '//foo:print_arg0_test',
            '--test_output=all',
            '--enable_runfiles',
        ]
        + flags
    )

    arg0 = None
    for line in stderr + stdout:
      if line.startswith('ARG0='):
        arg0 = line[len('ARG0='):]
    # Get rid of the quotes if there is any
    if arg0[0] == '"' and arg0[-1] == '"':
      arg0 = arg0[1:-1]
    self.assertEqual(
        arg0,
        os.path.join(bazel_bin,
                     'foo\\print_arg0_test.bat.runfiles\\'
                     '_main\\foo\\print_arg0_test.bat')
    )

  def _AssertTestArgs(self, flags):
    _, bazel_bin, _ = self.RunBazel(['info', 'bazel-bin'])
    bazel_bin = bazel_bin[0]

    _, stdout, stderr = self.RunBazel(
        [
            'test',
            '//foo:testargs_test',
            '-t-',
            '--test_output=all',
            '--test_arg=baz',
            '--test_arg="x y"',
            '--test_arg=""',
            '--test_arg=qux',
        ]
        + flags
    )

    actual = []
    for line in stderr + stdout:
      if line.startswith('arg='):
        actual.append(str(line[len('arg='):]))
    self.assertListEqual(
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
        ],
        actual)

  def _AssertUndeclaredOutputs(self, flags):
    _, bazel_testlogs, _ = self.RunBazel(['info', 'bazel-testlogs'])
    bazel_testlogs = bazel_testlogs[0]

    self.RunBazel(
        [
            'test',
            '//foo:undecl_test',
            '-t-',
            '--test_output=errors',
            '--zip_undeclared_test_outputs',
        ]
        + flags
    )

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
            'out2/my data 2.dat': 16
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
    if mf_content[1] != 'out2/my data 2.dat\t16\tapplication/octet-stream':
      self._FailWithOutput(mf_content)

  def _AssertUndeclaredOutputsAnnotations(self, flags):
    _, bazel_testlogs, _ = self.RunBazel(['info', 'bazel-testlogs'])
    bazel_testlogs = bazel_testlogs[0]

    self.RunBazel(
        [
            'test',
            '//foo:annot_test',
            '-t-',
            '--test_output=errors',
        ]
        + flags
    )

    undecl_annot = os.path.join(bazel_testlogs, 'foo', 'annot_test',
                                'test.outputs_manifest', 'ANNOTATIONS')
    self.assertTrue(os.path.exists(undecl_annot))
    annot_content = []
    with open(undecl_annot, 'rt') as f:
      annot_content = [line.strip() for line in f.readlines()]

    self.assertListEqual(annot_content, ['Hello aHello c'])

  def _AssertXmlGeneration(self, flags, split_xml=False):
    _, bazel_testlogs, _ = self.RunBazel(['info', 'bazel-testlogs'])
    bazel_testlogs = bazel_testlogs[0]

    self.RunBazel(
        [
            'test',
            '//foo:xml_test',
            '-t-',
            '--test_output=errors',
            '--%sexperimental_split_xml_generation'
            % ('' if split_xml else 'no'),
        ]
        + flags
    )

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

  def _AssertXmlGeneratedByTestIsRetained(self, flags, split_xml=False):
    _, bazel_testlogs, _ = self.RunBazel(['info', 'bazel-testlogs'])
    bazel_testlogs = bazel_testlogs[0]

    self.RunBazel(
        [
            'test',
            '//foo:xml2_test',
            '-t-',
            '--test_output=errors',
            '--%sexperimental_split_xml_generation'
            % ('' if split_xml else 'no'),
        ]
        + flags
    )

    test_xml = os.path.join(bazel_testlogs, 'foo', 'xml2_test', 'test.xml')
    self.assertTrue(os.path.exists(test_xml))
    xml_contents = []
    with open(test_xml, 'rt') as f:
      xml_contents = [line.strip() for line in f.readlines()]
    self.assertListEqual(xml_contents, ['leave this'])

  # Test that we can run tests from external repositories.
  # See https://github.com/bazelbuild/bazel/issues/8088
  def testRunningTestFromExternalRepo(self):
    rule_definition = [
        (
            'local_repository ='
            ' use_repo_rule("@bazel_tools//tools/build_defs/repo:local.bzl",'
            ' "local_repository")'
        ),
        'local_repository(name = "a", path = "a")',
    ]
    self.ScratchFile('MODULE.bazel', rule_definition)
    self.ScratchFile('BUILD', ['py_test(name = "x", srcs = ["x.py"])'])
    self.ScratchFile('a/REPO.bazel')
    self.ScratchFile('a/BUILD', ['py_test(name = "x", srcs = ["x.py"])'])
    self.ScratchFile('x.py')
    self.ScratchFile('a/x.py')

    for flag in ['--legacy_external_runfiles', '--nolegacy_external_runfiles']:
      for layout in [
          '--experimental_sibling_repository_layout',
          '--noexperimental_sibling_repository_layout',
      ]:
        for target in ['//:x', '@a//:x']:
          exit_code, _, stderr = self.RunBazel([
              'test',
              '-t-',
              '--shell_executable=',
              '--test_output=errors',
              '--verbose_failures',
              flag,
              layout,
              target,
          ])
          self.AssertExitCode(exit_code, 0, [
              'flag=%s' % flag,
              'layout=%s' % layout,
              'target=%s' % target,
          ] + stderr)

  def _AssertAddCurrentDirectoryToPathTest(self, flags):
    self.RunBazel(
        [
            'test',
            '//foo:add_cur_dir_to_path_test',
            '--test_output=all',
        ]
        + flags
    )

  def testTestExecutionWithTestWrapperExe(self):
    self._CreateMockWorkspace()
    flags = ['--shell_executable=']
    self._AssertPassingTest(flags)
    self._AssertFailingTest(flags)
    self._AssertPrintingTest(flags)
    self._AssertRunfiles(flags)
    self._AssertRunfilesSymlinks(flags)
    self._AssertShardedTest(flags)
    self._AssertUnexportsEnvvars(flags)
    self._AssertTestBinaryLocation(flags)
    self._AssertTestArgs(flags)
    self._AssertUndeclaredOutputs(flags)
    self._AssertUndeclaredOutputsAnnotations(flags)
    self._AssertXmlGeneration(flags, split_xml=False)
    self._AssertXmlGeneration(flags, split_xml=True)
    self._AssertXmlGeneratedByTestIsRetained(flags, split_xml=False)
    self._AssertXmlGeneratedByTestIsRetained(flags, split_xml=True)
    self._AssertAddCurrentDirectoryToPathTest(flags)


if __name__ == '__main__':
  absltest.main()
