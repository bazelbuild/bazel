# pylint: disable=g-bad-file-header
# Copyright 2016 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Wrapper script for executing the Microsoft Compiler."""
import os
import sys
import msvc_link
import msvc_tools

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

GCCPATTERNS = [
    ('-m(32|64)', ['$TARGET_ARCH']),
    ('-Xcompilation-mode=(dbg|fastbuild|opt)', ['$COMPILATION_MODE']),
    ('-msse', ['/arch:SSE']),
    ('-msse2', ['/arch:SSE2']),
    ('-D(.+)', ['/D$0']),
    ('-U(.+)', ['/U$0']),
    ('-E', ['/E']),
    ('-O0', ['/Od']),
    ('-Os', ['/O1']),
    ('-O2', ['/O2']),
    ('-g0', []),
    ('-g', ['/MTd']),
    ('-fexceptions', ['/U_HAS_EXCEPTIONS', '/D_HAS_EXCEPTIONS=1', '/EHsc']),
    ('-fomit-frame-pointer', ['/Oy']),
    ('-fno-rtti', ['/GR-']),
    ('-frtti', ['/GR']),
    ('-fPIC', []),

    # This is unneeded for Windows.
    (('-include', '(.+)'), ['/FI$PATH0']),
    (('/DEPENDENCY_FILE', '(.+)'), ['$GENERATE_DEPS0']),
    ('-w', ['/w']),
    ('-Wall', ['/Wall']),
    ('-Wsign-compare', ['/we4018']),
    ('-Wno-sign-compare', ['/wd4018']),
    ('-Wconversion', ['/we4244', '/we4267']),
    ('-Wno-conversion', ['/wd4244', '/wd4267']),
    ('-Wno-sign-conversion', []),
    ('-Wno-implicit-fallthrough', []),
    ('-Wno-implicit-function-declaration', []),
    ('-Wcovered-switch-default', ['/we4062']),
    ('-Wno-covered-switch-default', ['/wd4062']),
    ('-Wno-error', []),
    ('-Wno-invalid-offsetof', []),
    ('-Wno-overloaded-virtual', []),
    ('-Wno-reorder', []),
    ('-Wno-string-plus-int', []),
    ('-Wl,S', []),  # Stripping is unnecessary since msvc uses pdb files.
    ('-Wl,-rpath(.+)', []),
    ('-B(.+)', []),
    ('-static', []),
    ('-shared', ['/DLL']),
    ('-std=(.+)', []),
]


def _IsLink(args):
  """Determines whether we need to link rather than compile.

  A set of arguments is for linking if they contain -static, -shared, are adding
  adding library search paths through -L, or libraries via -l.

  Args:
    args: List of arguments

  Returns:
    Boolean whether this is a link operation or not.
  """
  for arg in args:
    # Certain flags indicate we are linking.
    if (arg in ['-shared', '-static'] or arg[:2] in ['-l', '-L'] or
        arg[:3] == '-Wl'):
      return True
  return False


class MsvcCompiler(msvc_tools.WindowsRunner):
  """Driver for the Microsoft compiler."""

  def Run(self, argv):
    """Runs the compiler using the passed clang/gcc style argument list.

    Args:
      argv: List of arguments

    Returns:
      The return code of the compilation.

    Raises:
      ValueError: if target architecture isn't specified
    """
    parser = msvc_tools.ArgParser(self, argv, GCCPATTERNS)
    if not parser.target_arch:
      raise ValueError('Must specify target architecture (-m32 or -m64)')

    return self.RunBinary('cl', parser.options, parser.target_arch, parser)


def main(argv):
  # If we are supposed to link create a static library.
  if _IsLink(argv[1:]):
    return msvc_link.main(argv)
  else:
    return MsvcCompiler().Run(argv[1:])


if __name__ == '__main__':
  sys.exit(main(sys.argv[1:]))  # need to skip the first argument
