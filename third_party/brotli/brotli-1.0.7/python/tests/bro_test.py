# Copyright 2016 The Brotli Authors. All rights reserved.
#
# Distributed under MIT license.
# See file LICENSE for detail or copy at https://opensource.org/licenses/MIT

import subprocess
import unittest

from . import _test_utils
import brotli

BRO_ARGS = _test_utils.BRO_ARGS
TEST_ENV = _test_utils.TEST_ENV


def _get_original_name(test_data):
    return test_data.split('.compressed')[0]


class TestBroDecompress(_test_utils.TestCase):

    def _check_decompression(self, test_data):
        # Verify decompression matches the original.
        temp_uncompressed = _test_utils.get_temp_uncompressed_name(test_data)
        original = _get_original_name(test_data)
        self.assertFilesMatch(temp_uncompressed, original)

    def _decompress_file(self, test_data):
        temp_uncompressed = _test_utils.get_temp_uncompressed_name(test_data)
        args = BRO_ARGS + ['-f', '-d', '-i', test_data, '-o', temp_uncompressed]
        subprocess.check_call(args, env=TEST_ENV)

    def _decompress_pipe(self, test_data):
        temp_uncompressed = _test_utils.get_temp_uncompressed_name(test_data)
        args = BRO_ARGS + ['-d']
        with open(temp_uncompressed, 'wb') as out_file:
            with open(test_data, 'rb') as in_file:
                subprocess.check_call(
                    args, stdin=in_file, stdout=out_file, env=TEST_ENV)

    def _test_decompress_file(self, test_data):
        self._decompress_file(test_data)
        self._check_decompression(test_data)

    def _test_decompress_pipe(self, test_data):
        self._decompress_pipe(test_data)
        self._check_decompression(test_data)


_test_utils.generate_test_methods(TestBroDecompress, for_decompression=True)


class TestBroCompress(_test_utils.TestCase):

    VARIANTS = {'quality': (1, 6, 9, 11), 'lgwin': (10, 15, 20, 24)}

    def _check_decompression(self, test_data, **kwargs):
        # Write decompression to temp file and verify it matches the original.
        temp_uncompressed = _test_utils.get_temp_uncompressed_name(test_data)
        temp_compressed = _test_utils.get_temp_compressed_name(test_data)
        original = test_data
        args = BRO_ARGS + ['-f', '-d']
        args.extend(['-i', temp_compressed, '-o', temp_uncompressed])
        subprocess.check_call(args, env=TEST_ENV)
        self.assertFilesMatch(temp_uncompressed, original)

    def _compress_file(self, test_data, **kwargs):
        temp_compressed = _test_utils.get_temp_compressed_name(test_data)
        args = BRO_ARGS + ['-f']
        if 'quality' in kwargs:
            args.extend(['-q', str(kwargs['quality'])])
        if 'lgwin' in kwargs:
            args.extend(['--lgwin', str(kwargs['lgwin'])])
        args.extend(['-i', test_data, '-o', temp_compressed])
        subprocess.check_call(args, env=TEST_ENV)

    def _compress_pipe(self, test_data, **kwargs):
        temp_compressed = _test_utils.get_temp_compressed_name(test_data)
        args = BRO_ARGS
        if 'quality' in kwargs:
            args.extend(['-q', str(kwargs['quality'])])
        if 'lgwin' in kwargs:
            args.extend(['--lgwin', str(kwargs['lgwin'])])
        with open(temp_compressed, 'wb') as out_file:
            with open(test_data, 'rb') as in_file:
                subprocess.check_call(
                    args, stdin=in_file, stdout=out_file, env=TEST_ENV)

    def _test_compress_file(self, test_data, **kwargs):
        self._compress_file(test_data, **kwargs)
        self._check_decompression(test_data)

    def _test_compress_pipe(self, test_data, **kwargs):
        self._compress_pipe(test_data, **kwargs)
        self._check_decompression(test_data)


_test_utils.generate_test_methods(
    TestBroCompress, variants=TestBroCompress.VARIANTS)

if __name__ == '__main__':
    unittest.main()
