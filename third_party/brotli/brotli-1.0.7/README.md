<p align="center"><img src="https://brotli.org/brotli.svg" alt="Brotli" width="64"></p>

### Introduction

Brotli is a generic-purpose lossless compression algorithm that compresses data
using a combination of a modern variant of the LZ77 algorithm, Huffman coding
and 2nd order context modeling, with a compression ratio comparable to the best
currently available general-purpose compression methods. It is similar in speed
with deflate but offers more dense compression.

The specification of the Brotli Compressed Data Format is defined in [RFC 7932](https://tools.ietf.org/html/rfc7932).

Brotli is open-sourced under the MIT License, see the LICENSE file.

Brotli mailing list:
https://groups.google.com/forum/#!forum/brotli

[![TravisCI Build Status](https://travis-ci.org/google/brotli.svg?branch=master)](https://travis-ci.org/google/brotli)
[![AppVeyor Build Status](https://ci.appveyor.com/api/projects/status/github/google/brotli?branch=master&svg=true)](https://ci.appveyor.com/project/szabadka/brotli)

### Build instructions

#### Autotools-style CMake

[configure-cmake](https://github.com/nemequ/configure-cmake) is an
autotools-style configure script for CMake-based projects (not supported on Windows).

The basic commands to build, test and install brotli are:

    $ mkdir out && cd out
    $ ../configure-cmake
    $ make
    $ make test
    $ make install
  
By default, debug binaries are built. To generate "release" `Makefile` specify `--disable-debug` option to `configure-cmake`.

#### Bazel

See [Bazel](http://www.bazel.build/)

#### CMake

The basic commands to build and install brotli are:

    $ mkdir out && cd out
    $ cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=./installed ..
    $ cmake --build . --config Release --target install

You can use other [CMake](https://cmake.org/) configuration.

#### Premake5

See [Premake5](https://premake.github.io/)

#### Python

To install the latest release of the Python module, run the following:

    $ pip install brotli

To install the tip-of-the-tree version, run:

    $ pip install --upgrade git+https://github.com/google/brotli

See the [Python readme](python/README.md) for more details on installing
from source, development, and testing.

### Benchmarks
* [Squash Compression Benchmark](https://quixdb.github.io/squash-benchmark/) / [Unstable Squash Compression Benchmark](https://quixdb.github.io/squash-benchmark/unstable/)
* [Large Text Compression Benchmark](http://mattmahoney.net/dc/text.html)
* [Lzturbo Benchmark](https://sites.google.com/site/powturbo/home/benchmark)

### Related projects
> **Disclaimer:** Brotli authors take no responsibility for the third party projects mentioned in this section.

Independent [decoder](https://github.com/madler/brotli) implementation by Mark Adler, based entirely on format specification.

JavaScript port of brotli [decoder](https://github.com/devongovett/brotli.js). Could be used directly via `npm install brotli`

Hand ported [decoder / encoder](https://github.com/dominikhlbg/BrotliHaxe) in haxe by Dominik Homberger. Output source code: JavaScript, PHP, Python, Java and C#

7Zip [plugin](https://github.com/mcmilk/7-Zip-Zstd)

Dart [native bindings](https://github.com/thosakwe/brotli)
