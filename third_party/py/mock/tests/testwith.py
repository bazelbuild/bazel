import sys

if sys.version_info[:2] >= (2, 5):
    from tests._testwith import *
else:
    from tests.support import unittest2

    class TestWith(unittest2.TestCase):

        @unittest2.skip('tests using with statement skipped on Python 2.4')
        def testWith(self):
            pass


if __name__ == '__main__':
    unittest2.main()
