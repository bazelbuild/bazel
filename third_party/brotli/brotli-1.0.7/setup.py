# Copyright 2015 The Brotli Authors. All rights reserved.
#
# Distributed under MIT license.
# See file LICENSE for detail or copy at https://opensource.org/licenses/MIT

import os
import platform
import re
import unittest

try:
    from setuptools import Extension
    from setuptools import setup
except:
    from distutils.core import Extension
    from distutils.core import setup
from distutils.command.build_ext import build_ext
from distutils import errors
from distutils import dep_util
from distutils import log


CURR_DIR = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))


def get_version():
    """ Return BROTLI_VERSION string as defined in 'common/version.h' file. """
    version_file_path = os.path.join(CURR_DIR, 'c', 'common', 'version.h')
    version = 0
    with open(version_file_path, 'r') as f:
        for line in f:
            m = re.match(r'#define\sBROTLI_VERSION\s+0x([0-9a-fA-F]+)', line)
            if m:
                version = int(m.group(1), 16)
    if version == 0:
        return ''
    # Semantic version is calculated as (MAJOR << 24) | (MINOR << 12) | PATCH.
    major = version >> 24
    minor = (version >> 12) & 0xFFF
    patch = version & 0xFFF
    return '{0}.{1}.{2}'.format(major, minor, patch)


def get_test_suite():
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('python', pattern='*_test.py')
    return test_suite


class BuildExt(build_ext):

    def get_source_files(self):
        filenames = build_ext.get_source_files(self)
        for ext in self.extensions:
            filenames.extend(ext.depends)
        return filenames

    def build_extension(self, ext):
        if ext.sources is None or not isinstance(ext.sources, (list, tuple)):
            raise errors.DistutilsSetupError(
                "in 'ext_modules' option (extension '%s'), "
                "'sources' must be present and must be "
                "a list of source filenames" % ext.name)

        ext_path = self.get_ext_fullpath(ext.name)
        depends = ext.sources + ext.depends
        if not (self.force or dep_util.newer_group(depends, ext_path, 'newer')):
            log.debug("skipping '%s' extension (up-to-date)", ext.name)
            return
        else:
            log.info("building '%s' extension", ext.name)

        c_sources = []
        cxx_sources = []
        for source in ext.sources:
            if source.endswith('.c'):
                c_sources.append(source)
            else:
                cxx_sources.append(source)
        extra_args = ext.extra_compile_args or []

        objects = []
        for lang, sources in (('c', c_sources), ('c++', cxx_sources)):
            if lang == 'c++':
                if self.compiler.compiler_type == 'msvc':
                    extra_args.append('/EHsc')

            macros = ext.define_macros[:]
            if platform.system() == 'Darwin':
                macros.append(('OS_MACOSX', '1'))
            elif self.compiler.compiler_type == 'mingw32':
                # On Windows Python 2.7, pyconfig.h defines "hypot" as "_hypot",
                # This clashes with GCC's cmath, and causes compilation errors when
                # building under MinGW: http://bugs.python.org/issue11566
                macros.append(('_hypot', 'hypot'))
            for undef in ext.undef_macros:
                macros.append((undef,))

            objs = self.compiler.compile(
                sources,
                output_dir=self.build_temp,
                macros=macros,
                include_dirs=ext.include_dirs,
                debug=self.debug,
                extra_postargs=extra_args,
                depends=ext.depends)
            objects.extend(objs)

        self._built_objects = objects[:]
        if ext.extra_objects:
            objects.extend(ext.extra_objects)
        extra_args = ext.extra_link_args or []
        # when using GCC on Windows, we statically link libgcc and libstdc++,
        # so that we don't need to package extra DLLs
        if self.compiler.compiler_type == 'mingw32':
            extra_args.extend(['-static-libgcc', '-static-libstdc++'])

        ext_path = self.get_ext_fullpath(ext.name)
        # Detect target language, if not provided
        language = ext.language or self.compiler.detect_language(sources)

        self.compiler.link_shared_object(
            objects,
            ext_path,
            libraries=self.get_libraries(ext),
            library_dirs=ext.library_dirs,
            runtime_library_dirs=ext.runtime_library_dirs,
            extra_postargs=extra_args,
            export_symbols=self.get_export_symbols(ext),
            debug=self.debug,
            build_temp=self.build_temp,
            target_lang=language)


NAME = 'Brotli'

VERSION = get_version()

URL = 'https://github.com/google/brotli'

DESCRIPTION = 'Python bindings for the Brotli compression library'

AUTHOR = 'The Brotli Authors'

LICENSE = 'Apache 2.0'

PLATFORMS = ['Posix', 'MacOS X', 'Windows']

CLASSIFIERS = [
    'Development Status :: 4 - Beta',
    'Environment :: Console',
    'Intended Audience :: Developers',
    'License :: OSI Approved :: Apache Software License',
    'Operating System :: MacOS :: MacOS X',
    'Operating System :: Microsoft :: Windows',
    'Operating System :: POSIX :: Linux',
    'Programming Language :: C',
    'Programming Language :: C++',
    'Programming Language :: Python',
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Unix Shell',
    'Topic :: Software Development :: Libraries',
    'Topic :: Software Development :: Libraries :: Python Modules',
    'Topic :: System :: Archiving',
    'Topic :: System :: Archiving :: Compression',
    'Topic :: Text Processing :: Fonts',
    'Topic :: Utilities',
]

PACKAGE_DIR = {'': 'python'}

PY_MODULES = ['brotli']

EXT_MODULES = [
    Extension(
        '_brotli',
        sources=[
            'python/_brotli.cc',
            'c/common/dictionary.c',
            'c/common/transform.c',
            'c/dec/bit_reader.c',
            'c/dec/decode.c',
            'c/dec/huffman.c',
            'c/dec/state.c',
            'c/enc/backward_references.c',
            'c/enc/backward_references_hq.c',
            'c/enc/bit_cost.c',
            'c/enc/block_splitter.c',
            'c/enc/brotli_bit_stream.c',
            'c/enc/cluster.c',
            'c/enc/compress_fragment.c',
            'c/enc/compress_fragment_two_pass.c',
            'c/enc/dictionary_hash.c',
            'c/enc/encode.c',
            'c/enc/encoder_dict.c',
            'c/enc/entropy_encode.c',
            'c/enc/histogram.c',
            'c/enc/literal_cost.c',
            'c/enc/memory.c',
            'c/enc/metablock.c',
            'c/enc/static_dict.c',
            'c/enc/utf8_util.c',
        ],
        depends=[
            'c/common/constants.h',
            'c/common/context.h',
            'c/common/dictionary.h',
            'c/common/platform.h',
            'c/common/transform.h',
            'c/common/version.h',
            'c/dec/bit_reader.h',
            'c/dec/huffman.h',
            'c/dec/prefix.h',
            'c/dec/state.h',
            'c/enc/backward_references.h',
            'c/enc/backward_references_hq.h',
            'c/enc/backward_references_inc.h',
            'c/enc/bit_cost.h',
            'c/enc/bit_cost_inc.h',
            'c/enc/block_encoder_inc.h',
            'c/enc/block_splitter.h',
            'c/enc/block_splitter_inc.h',
            'c/enc/brotli_bit_stream.h',
            'c/enc/cluster.h',
            'c/enc/cluster_inc.h',
            'c/enc/command.h',
            'c/enc/compress_fragment.h',
            'c/enc/compress_fragment_two_pass.h',
            'c/enc/dictionary_hash.h',
            'c/enc/encoder_dict.h',
            'c/enc/entropy_encode.h',
            'c/enc/entropy_encode_static.h',
            'c/enc/fast_log.h',
            'c/enc/find_match_length.h',
            'c/enc/hash.h',
            'c/enc/hash_composite_inc.h',
            'c/enc/hash_forgetful_chain_inc.h',
            'c/enc/hash_longest_match64_inc.h',
            'c/enc/hash_longest_match_inc.h',
            'c/enc/hash_longest_match_quickly_inc.h',
            'c/enc/hash_rolling_inc.h',
            'c/enc/hash_to_binary_tree_inc.h',
            'c/enc/histogram.h',
            'c/enc/histogram_inc.h',
            'c/enc/literal_cost.h',
            'c/enc/memory.h',
            'c/enc/metablock.h',
            'c/enc/metablock_inc.h',
            'c/enc/params.h',
            'c/enc/prefix.h',
            'c/enc/quality.h',
            'c/enc/ringbuffer.h',
            'c/enc/static_dict.h',
            'c/enc/static_dict_lut.h',
            'c/enc/utf8_util.h',
            'c/enc/write_bits.h',
        ],
        include_dirs=[
            'c/include',
        ],
        language='c++'),
]

TEST_SUITE = 'setup.get_test_suite'

CMD_CLASS = {
    'build_ext': BuildExt,
}

setup(
    name=NAME,
    description=DESCRIPTION,
    version=VERSION,
    url=URL,
    author=AUTHOR,
    license=LICENSE,
    platforms=PLATFORMS,
    classifiers=CLASSIFIERS,
    package_dir=PACKAGE_DIR,
    py_modules=PY_MODULES,
    ext_modules=EXT_MODULES,
    test_suite=TEST_SUITE,
    cmdclass=CMD_CLASS)
