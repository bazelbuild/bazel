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
    ('-I(.+)', ['/I$0']),
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
    ('-g', ['$DEBUG_RT']),
    ('-fexceptions', ['/U_HAS_EXCEPTIONS', '/D_HAS_EXCEPTIONS=1', '/EHsc']),
    ('-fomit-frame-pointer', ['/Oy']),
    ('-fno-rtti', ['/GR-']),
    ('-frtti', ['/GR']),
    ('-fPIC', []),

    # This is unneeded for Windows.
    (('-include', '(.+)'), ['/FI$PATH0']),
    ('-w', ['/w']),
    ('-Wall', ['/Wall']),
    ('-Wsign-compare', ['/we4018']),
    ('-Wno-sign-compare', ['/wd4018']),
    ('-Wconversion', ['/we4244', '/we4267']),
    ('-Wno-conversion', ['/wd4244', '/wd4267']),
    ('-Wno-sign-conversion', []),
    ('-Wno-implicit-fallthrough', []),
    ('-Wno-implicit-function-declaration', ['/wd4013']),
    ('-Wimplicit-function-declaration', ['/we4013']),
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


def _IsLink():
  """Determines whether we need to link rather than compile.

  Returns:
    True if USE_LINKER is set to 1.
  """
  return 'USE_LINKER' in os.environ and os.environ['USE_LINKER'] == '1'


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
    parser.ApplyUndefines()

    # Select runtime option
    # Find the last runtime option passed
    rt = None
    rt_idx = -1
    for i, opt in enumerate(reversed(parser.options)):
      if opt in ['/MT', '/MTd', '/MD', '/MDd']:
        if opt[-1] == 'd':
          parser.enforce_debug_rt = True
        rt = opt[:3]
        rt_idx = len(parser.options) - i - 1
        break
    rt = rt or '/MT'  # Default to static runtime
    # Add debug if necessary
    if parser.enforce_debug_rt:
      rt += 'd'
    # Include runtime option
    if rt_idx >= 0:
      parser.options[rt_idx] = rt
    else:
      if parser.is_cuda_compilation:
        parser.options.append('--compiler-options="%s"' % rt)
      else:
        parser.options.append(rt)

    compiler = 'cl'
    if parser.is_cuda_compilation:
      compiler = 'nvcc'
    return self.RunBinary(compiler, parser.options, parser)


def main(argv):
  # If we are supposed to link create a static library.
  if _IsLink():
    return msvc_link.main(argv)
  else:
    return MsvcCompiler().Run(argv[1:])


if __name__ == '__main__':
  sys.exit(main(sys.argv[1:]))  # need to skip the first argument
