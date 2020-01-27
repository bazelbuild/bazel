# Copyright (C) 2007-2012 Michael Foord & the mock team
# E-mail: fuzzyman AT voidspace DOT org DOT uk
# http://www.voidspace.org.uk/python/mock/

from tests.support import unittest2

from mock import sentinel, DEFAULT


class SentinelTest(unittest2.TestCase):

    def testSentinels(self):
        self.assertEqual(sentinel.whatever, sentinel.whatever,
                         'sentinel not stored')
        self.assertNotEqual(sentinel.whatever, sentinel.whateverelse,
                            'sentinel should be unique')


    def testSentinelName(self):
        self.assertEqual(str(sentinel.whatever), 'sentinel.whatever',
                         'sentinel name incorrect')


    def testDEFAULT(self):
        self.assertTrue(DEFAULT is sentinel.DEFAULT)

    def testBases(self):
        # If this doesn't raise an AttributeError then help(mock) is broken
        self.assertRaises(AttributeError, lambda: sentinel.__bases__)


if __name__ == '__main__':
    unittest2.main()
