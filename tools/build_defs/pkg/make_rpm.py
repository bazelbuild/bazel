# Lint as: python2, python3
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
"""A simple cross-platform helper to create an RPM package."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import fileinput
import os
import re
import shutil
import subprocess
import tempfile

# Do not edit this line. Copybara replaces it with PY2 migration helper.
from absl import app
from absl import flags
import six

flags.DEFINE_string('rpmbuild', '', 'Path to rpmbuild executable')
flags.DEFINE_string('name', '', 'The name of the software being packaged.')
flags.DEFINE_string('version', '',
                    'The version of the software being packaged.')
flags.DEFINE_string('release', '',
                    'The release of the software being packaged.')
flags.DEFINE_string('arch', '',
                    'The CPU architecture of the software being packaged.')

flags.DEFINE_string('spec_file', '',
                    'The file containing the RPM specification.')
flags.DEFINE_string('out_file', '',
                    'The destination to save the resulting RPM file to.')
flags.DEFINE_boolean('debug', False, 'Print debug messages.')


# Setup to safely create a temporary directory and clean it up when done.
@contextlib.contextmanager
def Cd(newdir, cleanup=lambda: True):
  """Change the current working directory.

  This will run the provided cleanup function when the context exits and the
  previous working directory is restored.

  Args:
    newdir: The directory to change to. This must already exist.
    cleanup: An optional cleanup function to be executed when the context exits.

  Yields:
    Nothing.
  """

  prevdir = os.getcwd()
  os.chdir(os.path.expanduser(newdir))
  try:
    yield
  finally:
    os.chdir(prevdir)
    cleanup()


@contextlib.contextmanager
def Tempdir():
  """Create a new temporary directory and change to it.

  The temporary directory will be removed when the context exits.

  Yields:
    The full path of the temporary directory.
  """

  dirpath = tempfile.mkdtemp()

  def Cleanup():
    shutil.rmtree(dirpath)

  with Cd(dirpath, Cleanup):
    yield dirpath


def GetFlagValue(flagvalue, strip=True):
  if flagvalue:
    if flagvalue[0] == '@':
      with open(flagvalue[1:], 'r') as f:
        flagvalue = f.read()
    if strip:
      return flagvalue.strip()
  return flagvalue


WROTE_FILE_RE = re.compile(r'Wrote: (?P<rpm_path>.+)', re.MULTILINE)


def FindOutputFile(log):
  """Find the written file from the log information."""

  m = WROTE_FILE_RE.search(log)
  if m:
    return m.group('rpm_path')
  return None


def CopyAndRewrite(input_file, output_file, replacements=None):
  """Copies the given file and optionally rewrites with replacements.

  Args:
    input_file: The file to copy.
    output_file: The file to write to.
    replacements: A dictionary of replacements.
      Keys are prefixes scan for, values are the replacements to write after
      the prefix.
  """
  with open(output_file, 'w') as output:
    for line in fileinput.input(input_file):
      if replacements:
        for prefix, text in replacements.items():
          if six.ensure_str(line).startswith(prefix):
            line = prefix + ' ' + six.ensure_str(text) + '\n'
            break
      output.write(six.ensure_str(line))


def IsExe(fpath):
  return os.path.isfile(fpath) and os.access(fpath, os.X_OK)


def Which(program):
  """Search for the given program in the PATH.

  Args:
    program: The program to search for.

  Returns:
    The full path to the program.
  """

  for path in os.environ['PATH'].split(os.pathsep):
    filename = os.path.join(path, program)
    if IsExe(filename):
      return filename

  return None


class NoRpmbuildFoundError(Exception):
  pass


class InvalidRpmbuildError(Exception):
  pass


def FindRpmbuild(rpmbuild_path):
  if rpmbuild_path:
    if not IsExe(rpmbuild_path):
      raise InvalidRpmbuildError('{} is not executable'.format(rpmbuild_path))
    return rpmbuild_path
  path = Which('rpmbuild')
  if path:
    return path
  raise NoRpmbuildFoundError()


class RpmBuilder(object):
  """A helper class to manage building the RPM file."""

  SOURCE_DIR = 'SOURCES'
  BUILD_DIR = 'BUILD'
  TEMP_DIR = 'TMP'
  DIRS = [SOURCE_DIR, BUILD_DIR, TEMP_DIR]

  def __init__(self, name, version, release, arch, debug, rpmbuild_path):
    self.name = name
    self.version = GetFlagValue(version)
    self.release = GetFlagValue(release)
    self.arch = arch
    self.debug = debug
    self.files = []
    self.rpmbuild_path = FindRpmbuild(rpmbuild_path)
    self.rpm_path = None

  def AddFiles(self, paths, root=''):
    """Add a set of files to the current RPM.

    If an item in paths is a directory, its files are recursively added.

    Args:
      paths: The files to add.
      root: The root of the filesystem to search for files. Defaults to ''.
    """
    for path in paths:
      full_path = os.path.join(root, path)
      if os.path.isdir(full_path):
        self.AddFiles(os.listdir(full_path), full_path)
      else:
        self.files.append(full_path)

  def SetupWorkdir(self, spec_file, original_dir):
    """Create the needed structure in the workdir."""

    # Create directory structure.
    for name in RpmBuilder.DIRS:
      if not os.path.exists(name):
        os.makedirs(name, 0o777)

    # Copy the files.
    for f in self.files:
      dst_dir = os.path.join(RpmBuilder.BUILD_DIR, os.path.dirname(f))
      if not os.path.exists(dst_dir):
        os.makedirs(dst_dir, 0o777)
      shutil.copy(os.path.join(original_dir, f), dst_dir)

    # Copy the spec file, updating with the correct version.
    spec_origin = os.path.join(original_dir, spec_file)
    self.spec_file = os.path.basename(spec_file)
    replacements = {}
    if self.version:
      replacements['Version:'] = self.version
    if self.release:
      replacements['Release:'] = self.release
    CopyAndRewrite(spec_origin, self.spec_file, replacements)

  def CallRpmBuild(self, dirname):
    """Call rpmbuild with the correct arguments."""

    args = [
        self.rpmbuild_path,
        '--define',
        '_topdir %s' % dirname,
        '--define',
        '_tmppath %s/TMP' % dirname,
        '--bb',
        '--buildroot',
        os.path.join(dirname, 'BUILDROOT'),
        self.spec_file,
    ]
    p = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={'LANG': 'C'})
    output = p.communicate()[0].decode()

    if p.returncode == 0:
      # Find the created file.
      self.rpm_path = FindOutputFile(output)

    if p.returncode != 0 or not self.rpm_path:
      print('Error calling rpmbuild:')
      print(output)

    # Return the status.
    return p.returncode

  def SaveResult(self, out_file):
    """Save the result RPM out of the temporary working directory."""

    if self.rpm_path:
      shutil.copy(self.rpm_path, out_file)
      if self.debug:
        print('Saved RPM file to %s' % out_file)
    else:
      print('No RPM file created.')

  def Build(self, spec_file, out_file):
    """Build the RPM described by the spec_file."""
    if self.debug:
      print('Building RPM for %s at %s' % (self.name, out_file))

    original_dir = os.getcwd()
    spec_file = os.path.join(original_dir, spec_file)
    out_file = os.path.join(original_dir, out_file)
    with Tempdir() as dirname:
      self.SetupWorkdir(spec_file, original_dir)
      status = self.CallRpmBuild(dirname)
      self.SaveResult(out_file)

    return status


def main(argv=()):
  try:
    builder = RpmBuilder(FLAGS.name, FLAGS.version, FLAGS.release, FLAGS.arch,
                         FLAGS.debug, FLAGS.rpmbuild)
    builder.AddFiles(argv[1:])
    return builder.Build(FLAGS.spec_file, FLAGS.out_file)
  except NoRpmbuildFoundError:
    print('ERROR: rpmbuild is required but is not present in PATH')
    return 1


if __name__ == '__main__':
  FLAGS = flags.FLAGS
  app.run(main)
