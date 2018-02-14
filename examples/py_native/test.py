"""A tiny example binary for the native Python rules of Bazel."""

import unittest
from fib import Fib
from lib import GetNumber


class TestGetNumber(unittest.TestCase):

  def test_ok(self):
    self.assertEquals(GetNumber(), 42)

  def test_fib(self):
    self.assertEquals(Fib(5), 8)

if __name__ == '__main__':
  unittest.main()
