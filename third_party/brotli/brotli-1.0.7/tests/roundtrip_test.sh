#!/usr/bin/env bash
#
# Roundtrip test for the brotli command-line tool.
#
# The first argument may be a wrapper for brotli, such as 'qemu-arm'.

set -o errexit

BROTLI_WRAPPER=$1
BROTLI="${BROTLI_WRAPPER} bin/brotli"
TMP_DIR=bin/tmp
INPUTS="""
tests/testdata/alice29.txt
tests/testdata/asyoulik.txt
tests/testdata/lcet10.txt
tests/testdata/plrabn12.txt
c/enc/encode.c
c/common/dictionary.h
c/dec/decode.c
"""

for file in $INPUTS; do
  for quality in 1 6 9 11; do
    echo "Roundtrip testing $file at quality $quality"
    compressed=${TMP_DIR}/${file##*/}.br
    uncompressed=${TMP_DIR}/${file##*/}.unbr
    $BROTLI -fq $quality $file -o $compressed
    $BROTLI $compressed -fdo $uncompressed
    diff -q $file $uncompressed
    # Test the streaming version
    cat $file | $BROTLI -cq $quality | $BROTLI -cd >$uncompressed
    diff -q $file $uncompressed
  done
done
