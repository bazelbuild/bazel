"""A tiny example binary for the native Python rules of Bazel."""

import unittest

from examples.py_native.fibonacci.fib import Fib
from examples.py_native.lib import GetNumber


class TestGetNumber(unittest.TestCase):

  def test_ok(self):
    self.assertEqual(GetNumber(), 42)

  def test_fib(self):
    self.assertEqual(Fib(5), 8)

if __name__ == '__main__':
  unittest.main()
