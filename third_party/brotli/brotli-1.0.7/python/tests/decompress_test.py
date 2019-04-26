# Copyright 2016 The Brotli Authors. All rights reserved.
#
# Distributed under MIT license.
# See file LICENSE for detail or copy at https://opensource.org/licenses/MIT

import unittest

from . import _test_utils
import brotli


def _get_original_name(test_data):
    return test_data.split('.compressed')[0]


class TestDecompress(_test_utils.TestCase):

    def _check_decompression(self, test_data):
        # Verify decompression matches the original.
        temp_uncompressed = _test_utils.get_temp_uncompressed_name(test_data)
        original = _get_original_name(test_data)
        self.assertFilesMatch(temp_uncompressed, original)

    def _decompress(self, test_data):
        temp_uncompressed = _test_utils.get_temp_uncompressed_name(test_data)
        with open(temp_uncompressed, 'wb') as out_file:
            with open(test_data, 'rb') as in_file:
                out_file.write(brotli.decompress(in_file.read()))

    def _test_decompress(self, test_data):
        self._decompress(test_data)
        self._check_decompression(test_data)

    def test_garbage_appended(self):
        with self.assertRaises(brotli.error):
            brotli.decompress(brotli.compress(b'a') + b'a')


_test_utils.generate_test_methods(TestDecompress, for_decompression=True)

if __name__ == '__main__':
    unittest.main()
