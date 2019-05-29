# Copyright 2017 The Bazel Authors. All rights reserved.
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
"""Tests for make_rpm."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import os
import unittest

from tools.build_defs.pkg import make_rpm


@contextlib.contextmanager
def PrependPath(dirs):
  with ReplacePath(dirs + [os.environ['PATH']]):
    yield


@contextlib.contextmanager
def ReplacePath(dirs):
  original_path = os.environ['PATH']
  try:
    os.environ['PATH'] = os.pathsep.join(dirs)
    yield
  finally:
    os.environ['PATH'] = original_path


def WriteFile(filename, *contents):
  with open(filename, 'w') as text_file:
    text_file.write('\n'.join(contents))


def DirExists(dirname):
  return os.path.exists(dirname) and os.path.isdir(dirname)


def FileExists(filename):
  return os.path.exists(filename) and not os.path.isdir(filename)


def FileContents(filename):
  with open(filename, 'r') as text_file:
    return [s.strip() for s in text_file.readlines()]


class MakeRpmTest(unittest.TestCase):

  # Python 2 alias
  if not hasattr(unittest.TestCase, 'assertCountEqual'):

    def assertCountEqual(self, a, b):
      # pylint: disable=g-deprecated-assert
      return self.assertItemsEqual(a, b)

  def testFindOutputFile(self):
    log = """
    Lots of data.
    Wrote: /path/to/file/here.rpm
    More data present.
    """

    result = make_rpm.FindOutputFile(log)
    self.assertEqual('/path/to/file/here.rpm', result)

  def testFindOutputFile_missing(self):
    log = """
    Lots of data.
    More data present.
    """

    result = make_rpm.FindOutputFile(log)
    self.assertEqual(None, result)

  def testCopyAndRewrite(self):
    with make_rpm.Tempdir():
      WriteFile('test.txt', 'Some: data1', 'Other: data2', 'More: data3')
      make_rpm.CopyAndRewrite('test.txt', 'out.txt', {
          'Some:': 'data1a',
          'More:': 'data3a',
      })

      self.assertTrue(FileExists('out.txt'))
      self.assertCountEqual(['Some: data1a', 'Other: data2', 'More: data3a'],
                            FileContents('out.txt'))

  def testFindRpmbuild_present(self):
    with make_rpm.Tempdir() as outer:
      dummy = os.sep.join([outer, 'rpmbuild'])
      WriteFile(dummy, 'dummy rpmbuild')
      os.chmod(dummy, 0o777)
      with PrependPath([outer]):
        path = make_rpm.FindRpmbuild('')
        self.assertEqual(dummy, path)

  def testFindRpmbuild_missing(self):
    with make_rpm.Tempdir() as outer:
      with ReplacePath([outer]):
        with self.assertRaises(make_rpm.NoRpmbuildFoundError) as context:
          make_rpm.FindRpmbuild('')
        self.assertIsNotNone(context)

  def testSetupWorkdir(self):
    with make_rpm.Tempdir() as outer:
      dummy = os.sep.join([outer, 'rpmbuild'])
      WriteFile(dummy, 'dummy rpmbuild')
      os.chmod(dummy, 0o777)

      with PrependPath([outer]):
        # Create the builder and exercise it.
        builder = make_rpm.RpmBuilder('test', '1.0', '0', 'x86', False, None)

        # Create spec_file, test files.
        WriteFile('test.spec', 'Name: test', 'Version: 0.1',
                  'Summary: test data')
        WriteFile('file1.txt', 'Hello')
        WriteFile('file2.txt', 'Goodbye')
        builder.AddFiles(['file1.txt', 'file2.txt'])

        with make_rpm.Tempdir():
          # Call RpmBuilder.
          builder.SetupWorkdir('test.spec', outer)

          # Make sure files exist.
          self.assertTrue(DirExists('SOURCES'))
          self.assertTrue(DirExists('BUILD'))
          self.assertTrue(DirExists('TMP'))
          self.assertTrue(FileExists('test.spec'))
          self.assertCountEqual(
              ['Name: test', 'Version: 1.0', 'Summary: test data'],
              FileContents('test.spec'))
          self.assertTrue(FileExists('BUILD/file1.txt'))
          self.assertCountEqual(['Hello'], FileContents('BUILD/file1.txt'))
          self.assertTrue(FileExists('BUILD/file2.txt'))
          self.assertCountEqual(['Goodbye'], FileContents('BUILD/file2.txt'))

  def testBuild(self):
    with make_rpm.Tempdir() as outer:
      dummy = os.sep.join([outer, 'rpmbuild'])
      WriteFile(
          dummy,
          '#!/bin/sh',
          'mkdir -p RPMS',
          'touch RPMS/test.rpm',
          'echo "Wrote: $PWD/RPMS/test.rpm"',
      )
      os.chmod(dummy, 0o777)

      with PrependPath([outer]):
        # Create the builder and exercise it.
        builder = make_rpm.RpmBuilder('test', '1.0', '0', 'x86', False, None)

        # Create spec_file, test files.
        WriteFile('test.spec', 'Name: test', 'Version: 0.1',
                  'Summary: test data')

        # Call RpmBuilder.
        builder.Build('test.spec', 'test.rpm')

        # Make sure files exist.
        self.assertTrue(FileExists('test.rpm'))


if __name__ == '__main__':
  unittest.main()
