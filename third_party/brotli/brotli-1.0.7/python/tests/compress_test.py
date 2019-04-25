# Copyright 2016 The Brotli Authors. All rights reserved.
#
# Distributed under MIT license.
# See file LICENSE for detail or copy at https://opensource.org/licenses/MIT

import unittest

from . import _test_utils
import brotli


class TestCompress(_test_utils.TestCase):

    VARIANTS = {'quality': (1, 6, 9, 11), 'lgwin': (10, 15, 20, 24)}

    def _check_decompression(self, test_data, **kwargs):
        kwargs = {}
        # Write decompression to temp file and verify it matches the original.
        temp_uncompressed = _test_utils.get_temp_uncompressed_name(test_data)
        temp_compressed = _test_utils.get_temp_compressed_name(test_data)
        original = test_data
        with open(temp_uncompressed, 'wb') as out_file:
            with open(temp_compressed, 'rb') as in_file:
                out_file.write(brotli.decompress(in_file.read(), **kwargs))
        self.assertFilesMatch(temp_uncompressed, original)

    def _compress(self, test_data, **kwargs):
        temp_compressed = _test_utils.get_temp_compressed_name(test_data)
        with open(temp_compressed, 'wb') as out_file:
            with open(test_data, 'rb') as in_file:
                out_file.write(brotli.compress(in_file.read(), **kwargs))

    def _test_compress(self, test_data, **kwargs):
        self._compress(test_data, **kwargs)
        self._check_decompression(test_data, **kwargs)


_test_utils.generate_test_methods(TestCompress, variants=TestCompress.VARIANTS)

if __name__ == '__main__':
    unittest.main()
