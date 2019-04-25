#! /usr/bin/env python
"""Compression/decompression utility using the Brotli algorithm."""

from __future__ import print_function
import argparse
import sys
import os
import platform

import brotli

# default values of encoder parameters
DEFAULT_PARAMS = {
    'mode': brotli.MODE_GENERIC,
    'quality': 11,
    'lgwin': 22,
    'lgblock': 0,
}


def get_binary_stdio(stream):
    """ Return the specified standard input, output or errors stream as a
    'raw' buffer object suitable for reading/writing binary data from/to it.
    """
    assert stream in ['stdin', 'stdout', 'stderr'], 'invalid stream name'
    stdio = getattr(sys, stream)
    if sys.version_info[0] < 3:
        if sys.platform == 'win32':
            # set I/O stream binary flag on python2.x (Windows)
            runtime = platform.python_implementation()
            if runtime == 'PyPy':
                # the msvcrt trick doesn't work in pypy, so I use fdopen
                mode = 'rb' if stream == 'stdin' else 'wb'
                stdio = os.fdopen(stdio.fileno(), mode, 0)
            else:
                # this works with CPython -- untested on other implementations
                import msvcrt
                msvcrt.setmode(stdio.fileno(), os.O_BINARY)
        return stdio
    else:
        # get 'buffer' attribute to read/write binary data on python3.x
        if hasattr(stdio, 'buffer'):
            return stdio.buffer
        else:
            orig_stdio = getattr(sys, '__%s__' % stream)
            return orig_stdio.buffer


def main(args=None):

    parser = argparse.ArgumentParser(
        prog=os.path.basename(__file__), description=__doc__)
    parser.add_argument(
        '--version', action='version', version=brotli.__version__)
    parser.add_argument(
        '-i',
        '--input',
        metavar='FILE',
        type=str,
        dest='infile',
        help='Input file',
        default=None)
    parser.add_argument(
        '-o',
        '--output',
        metavar='FILE',
        type=str,
        dest='outfile',
        help='Output file',
        default=None)
    parser.add_argument(
        '-f',
        '--force',
        action='store_true',
        help='Overwrite existing output file',
        default=False)
    parser.add_argument(
        '-d',
        '--decompress',
        action='store_true',
        help='Decompress input file',
        default=False)
    params = parser.add_argument_group('optional encoder parameters')
    params.add_argument(
        '-m',
        '--mode',
        metavar='MODE',
        type=int,
        choices=[0, 1, 2],
        help='The compression mode can be 0 for generic input, '
        '1 for UTF-8 encoded text, or 2 for WOFF 2.0 font data. '
        'Defaults to 0.')
    params.add_argument(
        '-q',
        '--quality',
        metavar='QUALITY',
        type=int,
        choices=list(range(0, 12)),
        help='Controls the compression-speed vs compression-density '
        'tradeoff. The higher the quality, the slower the '
        'compression. Range is 0 to 11. Defaults to 11.')
    params.add_argument(
        '--lgwin',
        metavar='LGWIN',
        type=int,
        choices=list(range(10, 25)),
        help='Base 2 logarithm of the sliding window size. Range is '
        '10 to 24. Defaults to 22.')
    params.add_argument(
        '--lgblock',
        metavar='LGBLOCK',
        type=int,
        choices=[0] + list(range(16, 25)),
        help='Base 2 logarithm of the maximum input block size. '
        'Range is 16 to 24. If set to 0, the value will be set based '
        'on the quality. Defaults to 0.')
    # set default values using global DEFAULT_PARAMS dictionary
    parser.set_defaults(**DEFAULT_PARAMS)

    options = parser.parse_args(args=args)

    if options.infile:
        if not os.path.isfile(options.infile):
            parser.error('file "%s" not found' % options.infile)
        with open(options.infile, 'rb') as infile:
            data = infile.read()
    else:
        if sys.stdin.isatty():
            # interactive console, just quit
            parser.error('no input')
        infile = get_binary_stdio('stdin')
        data = infile.read()

    if options.outfile:
        if os.path.isfile(options.outfile) and not options.force:
            parser.error('output file exists')
        outfile = open(options.outfile, 'wb')
    else:
        outfile = get_binary_stdio('stdout')

    try:
        if options.decompress:
            data = brotli.decompress(data)
        else:
            data = brotli.compress(
                data,
                mode=options.mode,
                quality=options.quality,
                lgwin=options.lgwin,
                lgblock=options.lgblock)
    except brotli.error as e:
        parser.exit(1,
                    'bro: error: %s: %s' % (e, options.infile or 'sys.stdin'))

    outfile.write(data)
    outfile.close()


if __name__ == '__main__':
    main()
