# Lint as: python3
# Copyright 2022 The Bazel Authors. All rights reserved.
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
"""This tool build tar files from a list of inputs."""

import argparse
import os
import tarfile

# Use a deterministic mtime that doesn't confuse other programs.
# See: https://github.com/bazelbuild/bazel/issues/1299
PORTABLE_MTIME = 946684800  # 2000-01-01 00:00:00.000 UTC


class TarFileWriter(object):
  """A wrapper to write tar files."""

  class Error(Exception):
    pass

  def __init__(self,
               name,
               root_directory='',
               default_uid=0,
               default_gid=0,
               default_mtime=None):
    """TarFileWriter wraps tarfile.open().

    Args:
      name: the tar file name.
      root_directory: virtual root to prepend to elements in the archive.
      default_uid: uid to assign to files in the archive.
      default_gid: gid to assign to files in the archive.
      default_mtime: default mtime to use for elements in the archive. May be an
        integer or the value 'portable' to use the date 2000-01-01, which is
        compatible with non *nix OSes'.
    """
    mode = 'w:'
    self.name = name
    self.root_directory = root_directory.strip('/')
    self.default_gid = default_gid
    self.default_uid = default_uid
    if default_mtime is None:
      self.default_mtime = 0
    elif default_mtime == 'portable':
      self.default_mtime = PORTABLE_MTIME
    else:
      self.default_mtime = int(default_mtime)
    self.tar = tarfile.open(name=name, mode=mode)
    self.members = set()
    self.directories = set(['.'])

  def __enter__(self):
    return self

  def __exit__(self, t, v, traceback):
    self.close()

  def close(self):
    """Close the output tar file."""
    self.tar.close()

  def _addfile(self, info, fileobj=None):
    """Add a file in the tar file if there is no conflict."""
    if info.type == tarfile.DIRTYPE:
      # Enforce the ending / for directories so we correctly deduplicate.
      if not info.name.endswith('/'):
        info.name += '/'
    if info.name not in self.members:
      self.tar.addfile(info, fileobj)
      self.members.add(info.name)
    elif info.type != tarfile.DIRTYPE:
      print(('Duplicate file in archive: %s, '
             'picking first occurrence' % info.name))

  def add_parents(self, path, mode=0o755):
    """Add the parents of this path to the archive.

    Args:
      path: destination path in archive.
      mode: unix permission mode of the dir, default 0o755.
    """

    def add_dirs(path):
      """Helper to add dirs."""
      path = path.strip('/')
      if not path:
        return
      if path in self.directories:
        return
      components = path.rsplit('/', 1)
      if len(components) > 1:
        add_dirs(components[0])
      self.directories.add(path)
      tarinfo = tarfile.TarInfo(path + '/')
      tarinfo.mtime = self.default_mtime
      tarinfo.uid = self.default_uid
      tarinfo.gid = self.default_gid
      tarinfo.type = tarfile.DIRTYPE
      tarinfo.mode = mode or 0o755
      self.tar.addfile(tarinfo, fileobj=None)

    components = path.rsplit('/', 1)
    if len(components) > 1:
      add_dirs(components[0])

  def add_tree(self, input_path, dest_path, mode=None):
    """Recursively add a tree of files.

    Args:
      input_path: the path of the directory to add.
      dest_path: the destination path of the directory to add.
      mode: unix permission mode of the file, default 0644 (0755).
    """
    # Add the x bit to directories to prevent non-traversable directories.
    # The x bit is set only to if the read bit is set.
    dirmode = (mode | ((0o444 & mode) >> 2)) if mode else mode
    self.add_parents(dest_path, mode=dirmode)

    if os.path.isdir(input_path):
      dest_path = dest_path.rstrip('/') + '/'
      # Iterate over the sorted list of file so we get a deterministic result.
      filelist = os.listdir(input_path)
      filelist.sort()
      for f in filelist:
        self.add_tree(
            input_path=input_path + '/' + f, dest_path=dest_path + f, mode=mode)
    else:
      self.add_file_and_parents(
          dest_path, tarfile.REGTYPE, file_content=input_path, mode=mode)

  def add_file_and_parents(self,
                           name,
                           kind=tarfile.REGTYPE,
                           link=None,
                           file_content=None,
                           mode=None):
    """Add a file to the current tar.

    Creates parent directories if needed.

    Args:
      name: the name of the file to add.
      kind: the type of the file to add, see tarfile.*TYPE.
      link: if the file is a link, the destination of the link.
      file_content: file to read the content from. Provide either this one or
        `content` to specifies a content for the file.
      mode: unix permission mode of the file, default 0644 (0755).
    """
    if self.root_directory and (
        not (name == self.root_directory or name.startswith('/') or
             name.startswith(self.root_directory + '/'))):
      name = self.root_directory + '/' + name
    self.add_parents(name, mode=0o755)

    if kind == tarfile.DIRTYPE:
      name = name.rstrip('/')
      if name in self.directories:
        return

    if file_content and os.path.isdir(file_content):
      self.add_tree(input_path=file_content, dest_path=name, mode=mode)
      return

    tarinfo = tarfile.TarInfo(name)
    tarinfo.mtime = self.default_mtime
    tarinfo.uid = self.default_uid
    tarinfo.gid = self.default_gid
    tarinfo.type = kind
    if mode is None:
      tarinfo.mode = 0o644 if kind == tarfile.REGTYPE else 0o755
    else:
      tarinfo.mode = mode
    if link:
      tarinfo.linkname = link
    if file_content:
      with open(file_content, 'rb') as f:
        tarinfo.size = os.fstat(f.fileno()).st_size
        self._addfile(tarinfo, fileobj=f)
    else:
      self._addfile(tarinfo, fileobj=None)

  def add_file_at_dest(self, in_path, dest_path, mode=None):
    """Add a file to the tar file.

    Args:
       in_path: the path of the file to add to the artifact
       dest_path: the name of the file in the artifact
       mode: force to the specified mode. Default is mode from the file.
    """
    # Make a clean, '/' deliminted destination path
    dest = os.path.normpath(dest_path.strip('/')).replace(os.path.sep, '/')
    # If mode is unspecified, derive the mode from the file's mode.
    if mode is None:
      mode = 0o755 if os.access(dest, os.X_OK) else 0o644
    self.add_file_and_parents(dest, file_content=in_path, mode=mode)


def unquote_and_split(arg, c):
  """Split a string at the first unquoted occurrence of a character.

  Split the string arg at the first unquoted occurrence of the character c.
  Here, in the first part of arg, the backslash is considered the
  quoting character indicating that the next character is to be
  added literally to the first part, even if it is the split character.

  Args:
    arg: the string to be split
    c: the character at which to split

  Returns:
    The unquoted string before the separator and the string after the
    separator.
  """
  head = ''
  i = 0
  while i < len(arg):
    if arg[i] == c:
      return (head, arg[i + 1:])
    elif arg[i] == '\\':
      i += 1
      if i == len(arg):
        # dangling quotation symbol
        return (head, '')
      else:
        head += arg[i]
    else:
      head += arg[i]
    i += 1
  # if we leave the loop, the character c was not found unquoted
  return (head, '')


def main():
  parser = argparse.ArgumentParser(
      description='Helper for building tar packages', fromfile_prefix_chars='@')
  parser.add_argument(
      '--output', required=True, help='The output file, mandatory.')
  parser.add_argument(
      '--mode', help='Force the mode on the added files (in octal).')
  parser.add_argument(
      '--directory',
      help='Directory in which to store the file inside the layer')
  parser.add_argument('--file', action='append', help='input_paty=dest_path')
  parser.add_argument(
      '--owner',
      default='0.0',
      help='Specify the numeric default owner of all files. E.g. 0.0')
  options = parser.parse_args()

  # Parse modes arguments
  default_mode = None
  if options.mode:
    # Convert from octal
    default_mode = int(options.mode, 8)

  uid = gid = 0
  if options.owner:
    ids = options.owner.split('.', 1)
    uid = int(ids[0])
    gid = int(ids[1])

  # Add objects to the tar file
  with TarFileWriter(
      name=options.output,
      root_directory=options.directory or '',
      default_uid=uid,
      default_gid=gid,
      default_mtime=PORTABLE_MTIME) as output:
    for f in options.file:
      (input_path, dest) = unquote_and_split(f, '=')
      output.add_file_at_dest(input_path, dest, mode=default_mode)


if __name__ == '__main__':
  main()
