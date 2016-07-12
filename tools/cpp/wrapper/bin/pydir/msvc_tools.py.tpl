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

import ntpath
import os
import re
import subprocess

MAX_PATH = 260  # The maximum number of characters in a Windows path.
MAX_OPTION_LENGTH = 10  # The maximum length of a compiler/linker option.
MAX_DRIVE_LENGTH = 3  # The maximum length of a drive.
ASSEMBLY_AS_C_SOURCE = '/Tc'
LIB_SUFFIX = '.lib'

TMP_PATH = '%{tmp}'

PATH = "%{path}"
INCLUDE = "%{include}"
LIB = "%{lib}"

class Error(Exception):
  """Base class for all script-specific errors."""
  pass


class ArgParser(object):
  """Class that parses gcc/clang-style options for a Windows.

  The particular substitutions that are needed are passed to the object.
  """

  def __init__(self, driver, argv, substitutions):
    self.driver = driver
    self.substitutions = substitutions
    self.options = []
    self.target_arch = None
    self.compilation_mode = None
    self.deps_file = None
    self.output_file = None
    self.params_file = None
    self._ParseArgs(argv)

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
        self.options.append(
            '/Fd%s.pdb' % self.NormPath(os.path.splitext(self.output_file)[0]))
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

        if not groups:
          self.options.append(entry)
        else:
          # Substitute special tokens.
          for g in xrange(0, len(groups)):
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
              except IOError, e:
                print 'Could not open', value, 'for reading:', str(e)
                exit(-1)
              continue

            if entry == ('$GENERATE_DEPS%d' % g):
              self.options.append('/showIncludes')
              self.deps_file = value
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
    if unmatched:
      print 'Warning: Unmatched arguments: ' + ' '.join(unmatched)

    # Use the proper runtime flag depending on compilation mode. If the
    # compilation is happening in debug mode, this flag already exists. If not,
    # then we must add it.
    if '/MT' not in self.options and '/MTd' not in self.options:
      self.AddOpt('/MT')
    # Add in any parsed files
    self.options += files

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
    """Normalizes an input unix style path to a < 260 char Windows format.

    Windows paths cannot be greater than 260 characters.

    Args:
      path: A path in unix format.

    Returns:
      An absolute path in Windows format, rooted from some
      directory.

    Raises:
      Error: if path is too long
    """
    abspath = os.path.abspath(path)
    long_path = abspath.replace('\\', '\\\\')
    # We must allow for the drive letter as well, which is three characters, and
    # the length of any compiler option ahead of the path,

    if len(long_path) + MAX_DRIVE_LENGTH + MAX_OPTION_LENGTH < MAX_PATH:
      return long_path
    else:
      # TODO(pcloudy):
      # Find a new way to deal with long path in real windows
      print 'Error: path is too long:' + long_path
      raise Error('path is too long: ' + long_path)
    return None

  def SetupEnvironment(self):
    """Setup proper path for running.

    Returns:
      An environment suitable for running on Windows.
    """

    build_env = os.environ.copy()
    build_env['PATH'] = PATH
    build_env['INCLUDE'] = INCLUDE
    build_env['LIB'] = LIB
    build_env['TEMP'] = TMP_PATH
    build_env['TMP'] = TMP_PATH
    return build_env

  def RunBinary(self, binary, args, build_arch, parser):
    """Runs binary on Windows with the passed args.

    Args:
      binary: The binary to run.
      args: The arguments to pass to binary.
      build_arch: Either 'x64' or 'x86', which binary architecture to build for.
      parser: An ArgParser that contains parsed arguments.

    Returns:
      The return code from executing binary.
    """
    # Filter out some not-so-useful cl windows messages.
    filters = [
        '.*warning LNK4006: __NULL_IMPORT_DESCRIPTOR already defined.*\n',
        '.*warning LNK4044: unrecognized option \'/MT\'; ignored.*\n',
        '.*warning LNK4044: unrecognized option \'/link\'; ignored.*\n',
        '.*warning LNK4221: This object file does not define any '
        'previously.*\n',
        # Comment the following line if you want to see warning messages
        '.*warning C.*\n',
        '\r\n',
        '\n\r',
    ]

    # Check again the arguments are within MAX_PATH.
    for arg in args:
      if len(arg) > MAX_PATH:
        print('Warning: arg "' + arg + '" is > than 260 characters (' +
              str(len(arg)) + '); programs may crash with long arguments')
      if os.path.splitext(arg)[1].lower() in ['.c', '.cc', '.cpp', '.s']:
        # cl.exe prints out the file name it is compiling; add that to the
        # filter.
        name = arg.rpartition(ntpath.sep)[2]
        filters.append(name)

    if '/w' in args:
      args = [arg for arg in args if arg not in ['/W2', '/W3', '/W4']]

    # Setup the Windows paths and the build environment.
    build_env = self.SetupEnvironment()

    # Construct a large regular expression for all filters.
    output_filter = re.compile('(' + ')|('.join(filters) + ')')
    includes_filter = re.compile(r'Note: including file:\s+(.*)')
    # Run the command.
    if parser.params_file:
      try:
        # Using parameter file as input when linking static libraries.
        params_file = open(parser.params_file, 'w')
        for arg in args:
          params_file.write(arg + '\n')
        params_file.close()
      except IOError, e:
        print 'Could not open', parser.params_file, 'for writing:', str(e)
        exit(-1)
      cmd = [binary] + [('@' + os.path.normpath(parser.params_file))]
    else:
      cmd = [binary] + args
    # Save stderr output to a temporary in case we need it.
    # Unconmment the following line to see what exact command is executed.
    # print "Running: " + " ".join(cmd)
    proc = subprocess.Popen(cmd,
                            env=build_env,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            shell=True)
    deps = []
    for line in proc.stdout:
      if not output_filter.match(line):
        includes = includes_filter.match(line)
        if includes:
          filename = includes.group(1).rstrip()
          deps += [filename]
        else:
          print line.rstrip()
    proc.wait()

    # Generate deps file if requested.
    if parser.deps_file:
      with open(parser.deps_file, 'w') as deps_file:
        # Start with the name of the output file.
        deps_file.write(parser.output_file + ': \\\n')
        for i, dep in enumerate(deps):
          dep = dep.replace('\\', '/').replace(' ', '\\ ')
          deps_file.write('  ' + dep)
          if i < len(deps) - 1:
            deps_file.write(' \\')
          deps_file.write('\n')

    return proc.returncode
