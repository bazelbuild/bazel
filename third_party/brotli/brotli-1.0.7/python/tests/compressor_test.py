# Copyright 2016 The Brotli Authors. All rights reserved.
#
# Distributed under MIT license.
# See file LICENSE for detail or copy at https://opensource.org/licenses/MIT

import functools
import unittest

from . import _test_utils
import brotli


# Do not inherit from TestCase here to ensure that test methods
# are not run automatically and instead are run as part of a specific
# configuration below.
class _TestCompressor(object):

    CHUNK_SIZE = 2048

    def tearDown(self):
        self.compressor = None

    def _check_decompression(self, test_data):
        # Write decompression to temp file and verify it matches the original.
        temp_uncompressed = _test_utils.get_temp_uncompressed_name(test_data)
        temp_compressed = _test_utils.get_temp_compressed_name(test_data)
        original = test_data
        with open(temp_uncompressed, 'wb') as out_file:
            with open(temp_compressed, 'rb') as in_file:
                out_file.write(brotli.decompress(in_file.read()))
        self.assertFilesMatch(temp_uncompressed, original)

    def _test_single_process(self, test_data):
        # Write single-shot compression to temp file.
        temp_compressed = _test_utils.get_temp_compressed_name(test_data)
        with open(temp_compressed, 'wb') as out_file:
            with open(test_data, 'rb') as in_file:
                out_file.write(self.compressor.process(in_file.read()))
            out_file.write(self.compressor.finish())
        self._check_decompression(test_data)

    def _test_multiple_process(self, test_data):
        # Write chunked compression to temp file.
        temp_compressed = _test_utils.get_temp_compressed_name(test_data)
        with open(temp_compressed, 'wb') as out_file:
            with open(test_data, 'rb') as in_file:
                read_chunk = functools.partial(in_file.read, self.CHUNK_SIZE)
                for data in iter(read_chunk, b''):
                    out_file.write(self.compressor.process(data))
            out_file.write(self.compressor.finish())
        self._check_decompression(test_data)

    def _test_multiple_process_and_flush(self, test_data):
        # Write chunked and flushed compression to temp file.
        temp_compressed = _test_utils.get_temp_compressed_name(test_data)
        with open(temp_compressed, 'wb') as out_file:
            with open(test_data, 'rb') as in_file:
                read_chunk = functools.partial(in_file.read, self.CHUNK_SIZE)
                for data in iter(read_chunk, b''):
                    out_file.write(self.compressor.process(data))
                    out_file.write(self.compressor.flush())
            out_file.write(self.compressor.finish())
        self._check_decompression(test_data)


_test_utils.generate_test_methods(_TestCompressor)


class TestCompressorQuality1(_TestCompressor, _test_utils.TestCase):

    def setUp(self):
        self.compressor = brotli.Compressor(quality=1)


class TestCompressorQuality6(_TestCompressor, _test_utils.TestCase):

    def setUp(self):
        self.compressor = brotli.Compressor(quality=6)


class TestCompressorQuality9(_TestCompressor, _test_utils.TestCase):

    def setUp(self):
        self.compressor = brotli.Compressor(quality=9)


class TestCompressorQuality11(_TestCompressor, _test_utils.TestCase):

    def setUp(self):
        self.compressor = brotli.Compressor(quality=11)


if __name__ == '__main__':
    unittest.main()
