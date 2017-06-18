# pylint: disable=g-bad-file-header
# pylint: disable=cell-var-from-loop
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
"""Tools for working with the Microsoft Visual C++ toolchain."""

from argparse import ArgumentParser
import ntpath
import os
import re
import subprocess
import sys

MAX_PATH = 260  # The maximum number of characters in a Windows path.
MAX_OPTION_LENGTH = 10  # The maximum length of a compiler/linker option.
MAX_DRIVE_LENGTH = 3  # The maximum length of a drive.
MAX_PATH_ADJUSTED = MAX_PATH - MAX_OPTION_LENGTH - MAX_DRIVE_LENGTH
ASSEMBLY_AS_C_SOURCE = '/Tc'
LIB_SUFFIX = '.lib'
LIB_TOOL = "%{lib_tool}"
supported_cuda_compute_capabilities = [ %{cuda_compute_capabilities} ]

class Error(Exception):
  """Base class for all script-specific errors."""
  pass

def Log(s):
  """Print log messages."""
  print('msvc_tools.py: {0}'.format(s))

class ArgParser(object):
  """Class that parses gcc/clang-style options for a Windows.

  The particular substitutions that are needed are passed to the object.
  """

  def __init__(self, driver, argv, substitutions):
    self.driver = driver
    self.substitutions = substitutions
    self.options = []
    self.leftover = []
    self.target_arch = None
    self.compilation_mode = None
    self.deps_file = None
    self.output_file = None
    self.params_file = None
    self.support_whole_archive = %{support_whole_archive}
    self.global_whole_archive = None
    self.is_cuda_compilation = None
    self.cuda_log = False
    self.enforce_debug_rt = False
    self._ParseArgs(argv)

  def ReplaceLibrary(self, arg):
    """Do the actual replacement if necessary."""
    if arg == "/WHOLEARCHIVE":
      return []
    if arg.startswith("/OUT:") or os.path.splitext(arg)[1] not in ['.a', '.lo']:
      return [arg]
    if self.global_whole_archive or arg.startswith("/WHOLEARCHIVE:"):
      if arg.startswith("/WHOLEARCHIVE:"):
        arg = arg[len("/WHOLEARCHIVE:"):]
      output = subprocess.check_output([LIB_TOOL, "/list", arg]).decode("utf-8")
      object_files = []
      for line in output.split("\n"):
        line = line.strip()
        if line.endswith(".o"):
          object_files.append(line)
      return object_files
    return [arg]

  def WholeArchivePreprocess(self):
    """Replace library file with object files if /WHOLEARCHIVE is not supported."""
    if self.support_whole_archive:
      return
    options = []
    self.global_whole_archive = "/WHOLEARCHIVE" in self.options
    for arg in self.options:
      options.extend(self.ReplaceLibrary(arg))
    self.options = options

  def IsCudaCompilation(self):
    """Check if it's a cuda compilation."""
    parser = ArgumentParser()
    parser.add_argument('-x', nargs=1)
    parser.add_argument('--cuda_log', action='store_true')
    args, leftover = parser.parse_known_args(self.leftover)
    if args.x and args.x[0] == 'cuda':
      if args.cuda_log:
        Log('Using nvcc')
        self.cuda_log = True
      self.leftover = leftover
      return True
    return False

  def GetNvccOptions(self):
    """Collect the -nvcc_options values from self.leftover.

    Returns:
      The list of options that can be passed directly to nvcc.
    """

    parser = ArgumentParser()
    parser.add_argument('-nvcc_options', nargs='*', action='append')

    args, leftover = parser.parse_known_args(self.leftover)

    if args.nvcc_options:
      self.leftover = leftover
      return ['--'+a for a in sum(args.nvcc_options, [])]
    return []

  def GetOptionValue(self, option):
    """Extract the list of values for option from self.options.

    Args:
      option: The option whose value to extract, without the leading '/'.

    Returns:
      A list of values, either directly following the option,
      (eg., /opt val1 val2) or values collected from multiple occurrences of
      the option (eg., /opt val1 /opt val2).
    """

    parser = ArgumentParser(prefix_chars='/')
    parser.add_argument('/' + option, nargs='*', action='append')
    args, leftover = parser.parse_known_args(self.options)
    if args and vars(args)[option]:
      self.options = leftover
      return sum(vars(args)[option], [])
    return []

  def GetOptionsForCudaCompilation(self):
    """Get nvcc options with arguments assembled from self.options."""

    src_files = [f for f in self.options if
                 re.search('\.cpp$|\.cc$|\.c$|\.cxx$|\.C$', f)]
    if len(src_files) == 0:
      raise Error('No source files found for cuda compilation.')

    out_file = [ f for f in self.options if f.startswith('/Fo') ]
    if len(out_file) != 1:
      raise Error('Please sepecify exactly one output file for cuda compilation.')
    out = ['-o', out_file[0][len('/Fo'):]]

    nvcc_compiler_options = self.GetNvccOptions()

    opt_option = self.GetOptionValue('O')
    opt = ['-g', '-G']
    if (len(opt_option) > 0 and opt_option[0] != 'd'):
      opt = ['-O2']

    include_options = self.GetOptionValue('I')
    includes = ["-I " + include for include in include_options]

    defines = self.GetOptionValue('D')
    defines = ['-D' + define for define in defines]

    undefines = self.GetOptionValue('U')
    undefines = ['-U' + define for define in undefines]

    # The rest of the unrecongized options should be passed to host compiler
    host_compiler_options = [option for option in self.options if option not in (src_files + out_file)]

    m_options = ["-m64"]

    nvccopts = ['-D_FORCE_INLINES']
    for capability in supported_cuda_compute_capabilities:
      capability = capability.replace('.', '')
      nvccopts += [r'-gencode=arch=compute_%s,"code=sm_%s,compute_%s"' % (
          capability, capability, capability)]
    nvccopts += nvcc_compiler_options
    nvccopts += undefines
    nvccopts += defines
    nvccopts += m_options
    nvccopts += ['--compiler-options="' + " ".join(host_compiler_options) + '"']
    nvccopts += ['-x', 'cu'] + opt + includes + out + ['-c'] + src_files

    if self.cuda_log:
      Log("Running: " + " ".join(["nvcc"] + nvccopts))

    self.options = nvccopts

  def _MatchOneArg(self, args):
    """Finds a pattern which matches the beginning elements of args.

    Args:
      args: A list of arguments to replace.

    Returns:
       A tuple of (number of arguments parsed, action, match groups).
    """
    for (regex, action) in self.substitutions:
      if isinstance(regex, str):
        regex = [regex]
      j = 0
      matches = []
      for r in regex:
        if j < len(args):
          match = re.compile('^' + r + '$').match(args[j])
        else:
          match = None
        matches.append(match)
        j += 1
      if None in matches:
        continue
      groups = []
      for m in matches:
        groups.extend(m.groups())
      return (len(regex), action, groups)
    return (0, '', [])

  def _ParseArgs(self, argv):
    """Parses argv and replaces its elements using special tokens.

    The following is a list of supported tokens. The format is $TOKEN%d, where
    %d is the 0-based group number from regex matches of the pattern.
      $CREATE_PATH%d: Touches a file at the path in the matching token.
      $LOAD_PARAMS%d: Loads an ld-style params file and appends all arguments to
                      the current argument list by recursively calling
                      _ParseArgs.
      $%d           : Numeric token that just replaces the match group with
                      the value specified in the match list.
      $PATH%d       : Replaces the match with a Windows-normalized version of
                      the match; assumes that the match is a path.
      $PATH%d_NO_EXT: Same as $PATH but strips out any file extension.
      $TARGET_ARCH  : Set self.target_arch to 'x86' or 'x64' for '-m32' and
                      '-m64', respectively.
      $DEBUG_RT     : Enforce linkage to debug runtime.
      $COMPILE_OUTPUT%d: Sets the output name of a compilation step.
      $COMPILATION_MODE: Sets self.compilation_mode from the value of a
                      '-Xcompilation-mode=' flag.
      $CREATE_PRECOMPILED_HEADER: Informs the system that we are generating a
                      precompiled header rather than an object file.
      $GENERATE_DEPS%d: Generates a gcc-style .d file containing dependencies.

    Args:
      argv: A list of arguments to replace.

    Returns:
      A list of replaced arguments to pass to the target command.

    Raises:
      Error: if wrong arguments found
    """
    i = 0
    matched = []
    unmatched = []
    files = []
    while i < len(argv):
      num_matched, action, groups = self._MatchOneArg(argv[i:])
      arg = argv[i]
      if arg.startswith('/Fo') or arg.startswith('/Fa') or arg.startswith(
          '/Fi'):
        self.output_file = arg[3:]
      if num_matched == 0:
        # Strip out any .a's that have 0 size, they are header or intermediate
        # dependency libraries and don't contain any code. 0-length files are
        # considered corrupt by the linker (error LNK1136).
        if (os.path.isfile(arg) and os.path.splitext(arg)[1] == '.a' and
            os.path.getsize(arg) == 0):
          i += 1
          continue

        # If the argument is an absolute path, then add it directly.
        if arg[0] == '/':
          self.AddOpt(arg)
        elif os.path.isfile(arg):
          path = self.NormPath(arg)
          ext = os.path.splitext(arg)[1].lower()
          if ext in ['.s']:
            # Treat assembly files as C source files using a special option.
            path = ASSEMBLY_AS_C_SOURCE + path
          # If this is an actual file on disk then just pass it to the tool.
          files.append(path)
        elif not arg.endswith(LIB_SUFFIX):
          # Ignore .lib files.
          unmatched.append(arg)
        i += 1
        continue
      matched += argv[i:i + num_matched]
      # Handle special options.
      for entry in action:
        if entry == '$TARGET_ARCH':
          if arg == '-m32':
            self.target_arch = 'x86'
          elif arg == '-m64':
            self.target_arch = 'x64'
          else:
            raise Error('Unknown target arch flag: %r' % arg)
          continue

        if entry == '$COMPILATION_MODE':
          empty, prefix, mode = arg.partition('-Xcompilation-mode=')
          if empty or not prefix or mode not in ['dbg', 'fastbuild', 'opt']:
            raise Error('Invalid compilation mode flag: %r' % arg)
          self.compilation_mode = mode
          continue

        if entry == '$DEBUG_RT':
          self.enforce_debug_rt = True
          continue

        if not groups:
          self.options.append(entry)
        else:
          # Substitute special tokens.
          for g in range(0, len(groups)):
            value = groups[g]

            # Check for special tokens.
            if entry == ('$CREATE_PATH%d' % g):
              with open(value, 'a'):
                os.utime(value, None)
              continue

            if entry == ('$LOAD_PARAMS%d' % g):
              try:
                # The arguments in the params file need to be processed as
                # regular command-line arguments.
                params = [line.rstrip() for line in open(value, 'r')]
                self._ParseArgs(params)
                # Because we have no write permission to orginal params file,
                # create a new params file with addtional suffix
                self.params_file = value + '.msvc'
              except (IOError, e):
                print('Could not open', value, 'for reading:', str(e))
                exit(-1)
              continue

            # Regular substitution.
            patterns = {
                '$%d' % g: value,
                '$PATH%d_NO_EXT' % g: self.NormPath(os.path.splitext(value)[0]),
                '$PATH%d' % g: self.NormPath(value),
            }
            pattern = re.compile('(%s)' %
                                 '|'.join(map(re.escape, patterns.keys())))
            result = pattern.sub(lambda x: patterns[x.group(0)], entry)
            self.options.append(result)
      i += num_matched
    self.leftover = unmatched

    # Add in any parsed files
    self.options += files

    # Suppress all warning messages if /w is specified
    is_warning_off = '/w' in self.options
    if is_warning_off:
      self.options = [option for option in self.options
                      if option not in ['/W2', '/W3', '/W4', '/Wall']]

    self.is_cuda_compilation = self.IsCudaCompilation()
    if self.is_cuda_compilation:
      self.GetOptionsForCudaCompilation()

    if self.leftover and not is_warning_off:
      print('Warning: Unmatched arguments: ' + ' '.join(self.leftover))

  def NormPath(self, path):
    """Uses the current WindowsRunner to normalize the passed path.

    Args:
      path: the path to normalize.

    Returns:
      A normalized string representing a path suitable for Windows.
    """
    return self.driver.NormPath(path)

  def AddOpt(self, option):
    """Adds a single option.

    Args:
      option: the option to add.
    """
    self.options.append(option)


class WindowsRunner(object):
  """Base class that encapsulates the details of running a binary."""

  def NormPath(self, path):
    """Normalizes an input unix style path to a < MAX_PATH char Windows format.

    Windows paths cannot be greater than MAX_PATH characters.

    Args:
      path: A path in unix format.

    Returns:
      An absolute path in Windows format, rooted from some
      directory.

    Raises:
      Error: if path is too long
    """
    abspath = os.path.abspath(path)
    # We must allow for the drive letter as well, which is three characters, and
    # the length of any compiler option ahead of the path,
    if len(abspath) >= MAX_PATH_ADJUSTED:
      print(
          'Warning: path "%s" is >= %d characters (%d); programs may crash '
          'with long arguments'
          % (str(abspath), MAX_PATH_ADJUSTED, len(abspath)))
    return abspath

  def RunBinary(self, binary, args, parser):
    """Runs binary on Windows with the passed args.

    Args:
      binary: The binary to run.
      args: The arguments to pass to binary.
      parser: An ArgParser that contains parsed arguments.

    Returns:
      The return code from executing binary.
    """

    # Run the command.
    if parser.params_file:
      try:
        # Using parameter file as input when linking static libraries.
        params_file = open(parser.params_file, 'w')
        for arg in args:
          params_file.write(('"%s"' % arg) if os.path.isfile(arg) else arg)
          params_file.write('\n')
        params_file.close()
      except (IOError, e):
        print('Could not open', parser.params_file, 'for writing:', str(e))
        exit(-1)
      cmd = [binary] + [('@' + os.path.normpath(parser.params_file))]
    else:
      cmd = [binary] + args
    # Save stderr output to a temporary in case we need it.
    # Unconmment the following line to see what exact command is executed.
    # print("Running: " + " ".join(cmd))
    proc = subprocess.Popen(cmd,
                            stdout=sys.stdout,
                            stderr=sys.stderr,
                            env=os.environ.copy(),
                            shell=True)
    proc.wait()
    return proc.returncode
