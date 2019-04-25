#!/usr/bin/env bash
#
# Test that the brotli command-line tool can decompress old brotli-compressed
# files.
#
# The first argument may be a wrapper for brotli, such as 'qemu-arm'.

set -o errexit

BROTLI_WRAPPER=$1
BROTLI="${BROTLI_WRAPPER} bin/brotli"
TMP_DIR=bin/tmp

for file in tests/testdata/*.compressed*; do
  echo "Testing decompression of file $file"
  expected=${file%.compressed*}
  uncompressed=${TMP_DIR}/${expected##*/}.uncompressed
  echo $uncompressed
  $BROTLI $file -fdo $uncompressed
  diff -q $uncompressed $expected
  # Test the streaming version
  cat $file | $BROTLI -dc > $uncompressed
  diff -q $uncompressed $expected
  rm -f $uncompressed
done
