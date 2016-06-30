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
"""Wrapper script for executing the Microsoft Linker."""

import os
import sys
import msvc_tools

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append(SCRIPT_DIR)

LINKPATTERNS = [
    ('-m(32|64)', ['$TARGET_ARCH']),
    ('-Xcompilation-mode=(dbg|fastbuild|opt)', ['$COMPILATION_MODE']),
    (('rcs.*', '(.+)'), ['/OUT:$PATH0']),
    (('-o', '(.+)'), ['/OUT:$PATH0']),
    ('-B(.+)', []),
    ('-lpthread', []),
    ('-l(.+)', ['lib$0.so']),
    ('-L(.+)', ['/LIBPATH:$PATH0']),
    ('-static', []),
    ('-shared', ['/DLL']),
    ('-whole-archive', []),
    ('-no-whole-archive', []),
    ('-rdynamic', []),
    (r'-Wl,(.+)\.lib', ['$0.lib']),
    ('-Wl,@(.+)', ['$LOAD_PARAMS0']),
    ('@(.+)', ['$LOAD_PARAMS0']),
    ('-Wl,-rpath(.+)', []),
    ('-Wl,-S', []),  # Debug symbols are in pdb files.
    ('-Wl,/SUBSYSTEM:(WINDOWS|CONSOLE)', ['/SUBSYSTEM:$0']),
]


class MsvcLinker(msvc_tools.WindowsRunner):
  """Driver for the Microsoft linker."""

  def Run(self, argv):
    """Runs the linker using the passed clang/gcc style argument list.

    Args:
      argv: List of arguments

    Returns:
      The return code of the link.

    Raises:
      ValueError: if target architecture or compile mode isn't specified
    """
    # For now assume we are building a library.
    tool = 'lib'
    default_args = ['/nologo']

    # Build argument list.
    parser = msvc_tools.ArgParser(self, argv, LINKPATTERNS)

    # Find the output file name.
    name = ''
    for arg in parser.options:
      if '/OUT:' in arg:
        name = arg[5:]
    if not name:
      raise msvc_tools.Error('No output file name specified!')
    # Check if the library is empty, which is what happens when we create header
    # or intermediate link-only libraries.
    if (len(parser.options) == 2 and parser.options[0].startswith('/OUT') and
        parser.options[1].startswith('/M')):
      # Just "touch" the library to create the file.
      with open(name, 'w'):
        os.utime(name, None)
    else:
      # If the output name ends in .lo, or .a, it is a library, otherwise
      # we need to use link to create an executable.
      if os.path.splitext(name)[1] not in ['.a', '.lo']:
        tool = 'link'

        if not parser.target_arch:
          raise ValueError('Must specify target architecture (-m32 or -m64)')

        # Append explicit machine type.
        if parser.target_arch == 'x64':
          default_args.append('/MACHINE:X64')
        else:
          default_args.append('/MACHINE:X86')

        # Args for buildng a console application. These must appear here since
        # blaze will not properly pass them to the linker.
        # /SUBSYSTEM:CONSOLE: Build a console application.
        default_args += ['/SUBSYSTEM:CONSOLE']
        # If there is no .o on the command line, then we need to add the
        # run-time library for the target. Without this the linker gives a
        # LNK4001 error and cannot find an entry point.
        for arg in parser.options:
          if arg.endswith('.o'):
            break
        else:
          if not parser.compilation_mode:
            raise ValueError('Must specify compilation mode '
                             '(-Xcompilation-mode={dbg,fastbuild,opt})')

          if parser.compilation_mode == 'dbg':
            default_args.insert(0, 'libcmtd.lib')
          else:
            default_args.insert(0, 'libcmt.lib')

      return self.RunBinary(tool, default_args + parser.options,
                            parser.target_arch, parser)


def main(argv):
  return MsvcLinker().Run(argv[1:])


if __name__ == '__main__':
  sys.exit(main(sys.argv[1:]))  # need to skip the first argument
