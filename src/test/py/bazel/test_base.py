# pylint: disable=g-bad-file-header
# Copyright 2017 The Bazel Authors. All rights reserved.
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

import locale
import os
import subprocess
import sys
import tempfile
import unittest


class Error(Exception):
  """Base class for errors in this module."""
  pass


class ArgumentError(Error):
  """A function received a bad argument."""
  pass


class EnvVarUndefinedError(Error):
  """An expected environment variable is not defined."""

  def __init__(self, name):
    Error.__init__(self, 'Environment variable "%s" is not defined' % name)


class TestBase(unittest.TestCase):

  _runfiles = None
  _temp = None
  _tests_root = None
  _test_cwd = None

  def setUp(self):
    unittest.TestCase.setUp(self)
    if self._runfiles is None:
      self._runfiles = TestBase._LoadRunfiles()
    test_tmpdir = TestBase._CreateDirs(TestBase.GetEnv('TEST_TMPDIR'))
    self._tests_root = TestBase._CreateDirs(
        os.path.join(test_tmpdir, 'tests_root'))
    self._temp = TestBase._CreateDirs(os.path.join(test_tmpdir, 'tmp'))
    self._test_cwd = tempfile.mkdtemp(dir=self._tests_root)
    os.chdir(self._test_cwd)

  def AssertExitCode(self, actual_exit_code, expected_exit_code, stderr_lines):
    """Assert that `actual_exit_code` == `expected_exit_code`."""
    if actual_exit_code != expected_exit_code:
      self.fail('\n'.join([
          'Bazel exited with %d (expected %d), stderr:' % (actual_exit_code,
                                                           expected_exit_code),
          '(start stderr)----------------------------------------',
      ] + (stderr_lines or []) + [
          '(end stderr)------------------------------------------',
      ]))

  @staticmethod
  def GetEnv(name, default=None):
    """Returns environment variable `name`.

    Args:
      name: string; name of the environment variable
      default: anything; return this value if the envvar is not defined
    Returns:
      string, the envvar's value if defined, or `default` if the envvar is not
      defined but `default` is
    Raises:
      EnvVarUndefinedError: if `name` is not a defined envvar and `default` is
        None
    """
    value = os.getenv(name, '__undefined_envvar__')
    if value == '__undefined_envvar__':
      if default:
        return default
      raise EnvVarUndefinedError(name)
    return value

  @staticmethod
  def IsWindows():
    """Returns true if the current platform is Windows."""
    return os.name == 'nt'

  def Path(self, path):
    """Returns the absolute path of `path` relative to self._test_cwd.

    Args:
      path: string; a path, relative to self._test_cwd,
        self._test_cwd is different for each test case.
        e.g. "foo/bar/BUILD"
    Returns:
      an absolute path
    Raises:
      ArgumentError: if `path` is absolute or contains uplevel references
    """
    if os.path.isabs(path) or '..' in path:
      raise ArgumentError(('path="%s" may not be absolute and may not contain '
                           'uplevel references') % path)
    return os.path.join(self._test_cwd, path)

  def Rlocation(self, runfile):
    """Returns the absolute path to a runfile."""
    if TestBase.IsWindows():
      return self._runfiles.get(runfile)
    else:
      return os.path.join(self._runfiles, runfile)

  def ScratchDir(self, path):
    """Creates directories under the test's scratch directory.

    Args:
      path: string; a path, relative to the test's scratch directory,
        e.g. "foo/bar"
    Raises:
      ArgumentError: if `path` is absolute or contains uplevel references
      IOError: if an I/O error occurs
    """
    if not path:
      return
    abspath = self.Path(path)
    if os.path.exists(abspath):
      if os.path.isdir(abspath):
        return
      raise IOError('"%s" (%s) exists and is not a directory' % (path, abspath))
    os.makedirs(abspath)

  def ScratchFile(self, path, lines=None):
    """Creates a file under the test's scratch directory.

    Args:
      path: string; a path, relative to the test's scratch directory,
        e.g. "foo/bar/BUILD"
      lines: [string]; the contents of the file (newlines are added
        automatically)
    Returns:
      The absolute path of the scratch file.
    Raises:
      ArgumentError: if `path` is absolute or contains uplevel references
      IOError: if an I/O error occurs
    """
    if not path:
      return
    abspath = self.Path(path)
    if os.path.exists(abspath) and not os.path.isfile(abspath):
      raise IOError('"%s" (%s) exists and is not a file' % (path, abspath))
    self.ScratchDir(os.path.dirname(path))
    with open(abspath, 'w') as f:
      if lines:
        for l in lines:
          f.write(l)
          f.write('\n')
    return abspath

  def RunBazel(self, args, env_remove=None, env_add=None):
    """Runs "bazel <args>", waits for it to exit.

    Args:
      args: [string]; flags to pass to bazel (e.g. ['--batch', 'build', '//x'])
      env_remove: set(string); optional; environment variables to NOT pass to
        Bazel
      env_add: set(string); optional; environment variables to pass to
        Bazel, won't be removed by env_remove.
    Returns:
      (int, [string], [string]) tuple: exit code, stdout lines, stderr lines
    """
    return self.RunProgram([
        self.Rlocation('io_bazel/src/bazel'),
        '--bazelrc=/dev/null',
        '--nomaster_bazelrc',
    ] + args, env_remove, env_add)

  def RunProgram(self, args, env_remove=None, env_add=None):
    """Runs a program (args[0]), waits for it to exit.

    Args:
      args: [string]; the args to run; args[0] should be the program itself
      env_remove: set(string); optional; environment variables to NOT pass to
        the program
      env_add: set(string); optional; environment variables to pass to
        the program, won't be removed by env_remove.
    Returns:
      (int, [string], [string]) tuple: exit code, stdout lines, stderr lines
    """
    with tempfile.TemporaryFile(dir=self._test_cwd) as stdout:
      with tempfile.TemporaryFile(dir=self._test_cwd) as stderr:
        proc = subprocess.Popen(
            args,
            stdout=stdout,
            stderr=stderr,
            cwd=self._test_cwd,
            env=self._EnvMap(env_remove, env_add))
        exit_code = proc.wait()

        stdout.seek(0)
        stdout_lines = [
            l.decode(locale.getpreferredencoding()).strip()
            for l in stdout.readlines()
        ]

        stderr.seek(0)
        stderr_lines = [
            l.decode(locale.getpreferredencoding()).strip()
            for l in stderr.readlines()
        ]

        return exit_code, stdout_lines, stderr_lines

  def _EnvMap(self, env_remove=None, env_add=None):
    """Returns the environment variable map to run Bazel or other programs."""
    if TestBase.IsWindows():
      result = []
      if sys.version_info.major == 3:
        # Python 3.2 has os.listdir
        result = [
            n for n in os.listdir('c:\\program files\\java')
            if n.startswith('jdk')
        ]
      else:
        # Python 2.7 has os.path.walk
        def _Visit(result, _, names):
          result.extend(n for n in names if n.startswith('jdk'))
          while names:
            names.pop()

        os.path.walk('c:\\program files\\java\\', _Visit, result)

      env = {
          'SYSTEMROOT': TestBase.GetEnv('SYSTEMROOT'),
          # TODO(laszlocsomor): Let Bazel pass BAZEL_SH and JAVA_HOME to tests
          # and use those here instead of hardcoding paths.
          'JAVA_HOME': 'c:\\program files\\java\\' + sorted(result)[-1],
          'BAZEL_SH': 'c:\\tools\\msys64\\usr\\bin\\bash.exe',
          # TODO(pcloudy): Remove this after no longer need to debug
          # https://github.com/bazelbuild/bazel/issues/3273
          'CC_CONFIGURE_DEBUG': '1'
      }

      # TODO(pcloudy): Remove these hardcoded paths after resolving
      # https://github.com/bazelbuild/bazel/issues/3273
      env['BAZEL_VC'] = 'visual-studio-not-found'
      for p in [
          (r'C:\Program Files (x86)\Microsoft Visual Studio\2017\Professional'
           r'\VC'),
          r'C:\Program Files (x86)\Microsoft Visual Studio\2017\BuildTools\VC',
          r'C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC'
      ]:
        if os.path.exists(p):
          env['BAZEL_VC'] = p
          break
    else:
      env = {'HOME': os.path.join(self._temp, 'home')}

    env['PATH'] = TestBase.GetEnv('PATH')
    # The inner Bazel must know that it's running as part of a test (so that it
    # uses --max_idle_secs=15 by default instead of 3 hours, etc.), and it knows
    # that by checking for TEST_TMPDIR.
    env['TEST_TMPDIR'] = TestBase.GetEnv('TEST_TMPDIR')
    env['TMP'] = self._temp
    if env_remove:
      for e in env_remove:
        del env[e]
    if env_add:
      for e in env_add:
        env[e] = env_add[e]
    return env

  @staticmethod
  def _LoadRunfiles():
    """Loads the runfiles manifest from ${TEST_SRCDIR}/MANIFEST.

    Only necessary to use on Windows, where runfiles are not symlinked in to the
    runfiles directory, but are written to a MANIFEST file instead.

    Returns:
      on Windows: {string: string} dictionary, keys are runfiles-relative paths,
        values are absolute paths that the runfiles entry is mapped to;
      on other platforms: string; value of $TEST_SRCDIR
    """
    test_srcdir = TestBase.GetEnv('TEST_SRCDIR')
    if not TestBase.IsWindows():
      return test_srcdir

    result = {}
    with open(os.path.join(test_srcdir, 'MANIFEST'), 'r') as f:
      for l in f:
        tokens = l.strip().split(' ')
        if len(tokens) == 2:
          result[tokens[0]] = tokens[1]
    return result

  @staticmethod
  def _CreateDirs(path):
    if not os.path.exists(path):
      os.makedirs(path)
    elif not os.path.isdir(path):
      os.remove(path)
      os.makedirs(path)
    return path
