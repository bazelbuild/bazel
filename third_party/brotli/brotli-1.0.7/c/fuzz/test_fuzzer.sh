#!/usr/bin/env bash
set -e

export CC=${CC:-cc}

BROTLI="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"
SRC=$BROTLI/c

cd $BROTLI

rm -rf bin
mkdir bin
cd bin

cmake $BROTLI -DCMAKE_C_COMPILER="$CC" \
    -DBUILD_TESTING=OFF -DENABLE_SANITIZER=address
make -j$(nproc) brotlidec-static

${CC} -o run_decode_fuzzer -std=c99 -fsanitize=address -I$SRC/include \
    $SRC/fuzz/decode_fuzzer.c $SRC/fuzz/run_decode_fuzzer.c \
    ./libbrotlidec-static.a ./libbrotlicommon-static.a

mkdir decode_corpora
unzip $BROTLI/java/org/brotli/integration/fuzz_data.zip -d decode_corpora

for f in `ls decode_corpora`
do
 echo "Testing $f"
 ./run_decode_fuzzer decode_corpora/$f
done

cd $BROTLI
rm -rf bin
