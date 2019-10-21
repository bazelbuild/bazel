# -*- coding=utf-8 -*-
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

"""Tests for absl.command_name."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ctypes
import errno
import os
import unittest

from absl import command_name
from absl.testing import absltest
import mock


def _get_kernel_process_name():
  """Returns the Kernel's name for our process or an empty string."""
  try:
    with open('/proc/self/status', 'rt') as status_file:
      for line in status_file:
        if line.startswith('Name:'):
          return line.split(':', 2)[1].strip().encode('ascii', 'replace')
      return b''
  except IOError:
    return b''


def _is_prctl_syscall_available():
  try:
    libc = ctypes.CDLL('libc.so.6', use_errno=True)
  except OSError:
    return False
  zero = ctypes.c_ulong(0)
  try:
    status = libc.prctl(zero, zero, zero, zero, zero)
  except AttributeError:
    return False
  if status < 0 and errno.ENOSYS == ctypes.get_errno():
    return False
  return True


@unittest.skipIf(not _get_kernel_process_name(),
                 '_get_kernel_process_name() fails.')
class CommandNameTest(absltest.TestCase):

  def assertProcessNameSimilarTo(self, new_name):
    if not isinstance(new_name, bytes):
      new_name = new_name.encode('ascii', 'replace')
    actual_name = _get_kernel_process_name()
    self.assertTrue(actual_name)
    self.assertTrue(new_name.startswith(actual_name),
                    msg='set {!r} vs found {!r}'.format(new_name, actual_name))

  @unittest.skipIf(not os.access('/proc/self/comm', os.W_OK),
                   '/proc/self/comm is not writeable.')
  def test_set_kernel_process_name(self):
    new_name = u'ProcessNam0123456789abcdefghijklmnöp'
    command_name.set_kernel_process_name(new_name)
    self.assertProcessNameSimilarTo(new_name)

  @unittest.skipIf(not _is_prctl_syscall_available(),
                   'prctl() system call missing from libc.so.6.')
  def test_set_kernel_process_name_no_proc_file(self):
    new_name = b'NoProcFile0123456789abcdefghijklmnop'
    mock_open = mock.mock_open()
    with mock.patch.object(command_name, 'open', mock_open, create=True):
      mock_open.side_effect = IOError('mock open that raises.')
      command_name.set_kernel_process_name(new_name)
    mock_open.assert_called_with('/proc/self/comm', mock.ANY)
    self.assertProcessNameSimilarTo(new_name)

  def test_set_kernel_process_name_failure(self):
    starting_name = _get_kernel_process_name()
    new_name = b'NameTest'
    mock_open = mock.mock_open()
    mock_ctypes_cdll = mock.patch('ctypes.CDLL')
    with mock.patch.object(command_name, 'open', mock_open, create=True):
      with mock.patch('ctypes.CDLL') as mock_ctypes_cdll:
        mock_open.side_effect = IOError('mock open that raises.')
        mock_libc = mock.Mock(['prctl'])
        mock_ctypes_cdll.return_value = mock_libc
        command_name.set_kernel_process_name(new_name)
    mock_open.assert_called_with('/proc/self/comm', mock.ANY)
    self.assertEqual(1, mock_libc.prctl.call_count)
    self.assertEqual(starting_name, _get_kernel_process_name())  # No change.

  def test_make_process_name_useful(self):
    test_name = 'hello.from.test'
    with mock.patch('sys.argv', [test_name]):
      command_name.make_process_name_useful()
    self.assertProcessNameSimilarTo(test_name)


if __name__ == '__main__':
  absltest.main()
