# Lint as: python2, python3
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
import gzip
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
               default_mtime=None,
               preserve_tar_mtimes=False):
    """TarFileWriter wraps tarfile.open().

    Args:
      name: the tar file name.
      root_directory: virtual root to prepend to elements in the archive.
      default_mtime: default mtime to use for elements in the archive.
          May be an integer or the value 'portable' to use the date
          2000-01-01, which is compatible with non *nix OSes'.
      preserve_tar_mtimes: if true, keep file mtimes from input tar file.
    """
    mode = 'w:'
    self.gz = False
    self.name = name
    self.root_directory = root_directory.rstrip('/')

    self.preserve_mtime = preserve_tar_mtimes

    if default_mtime is None:
      self.default_mtime = 0
    elif default_mtime == 'portable':
      self.default_mtime = PORTABLE_MTIME
    else:
      self.default_mtime = int(default_mtime)

    self.fileobj = None
    if self.gz:
      # The Tarfile class doesn't allow us to specify gzip's mtime attribute.
      # Instead, we manually re-implement gzopen from tarfile.py and set mtime.
      self.fileobj = gzip.GzipFile(
          filename=name, mode='w', compresslevel=9, mtime=self.default_mtime)
    self.tar = tarfile.open(name=name, mode=mode, fileobj=self.fileobj)
    self.members = set([])
    self.directories = set(['.'])

  def __enter__(self):
    return self

  def __exit__(self, t, v, traceback):
    self.close()

  def add_dir(self,
              name,
              path,
              uid=0,
              gid=0,
              uname='',
              gname='',
              mtime=None,
              mode=None,
              depth=100):
    """Recursively add a directory.

    Args:
      name: the destination path of the directory to add.
      path: the path of the directory to add.
      uid: owner user identifier.
      gid: owner group identifier.
      uname: owner user names.
      gname: owner group names.
      mtime: modification time to put in the archive.
      mode: unix permission mode of the file, default 0644 (0755).
      depth: maximum depth to recurse in to avoid infinite loops
             with cyclic mounts.

    Raises:
      TarFileWriter.Error: when the recursion depth has exceeded the
                           `depth` argument.
    """
    if not (name == self.root_directory or name.startswith('/')
            or name.startswith(self.root_directory + '/')):
      name = os.path.join(self.root_directory, name)
    if mtime is None:
      mtime = self.default_mtime
    if os.path.isdir(path):
      # Remove trailing '/' (index -1 => last character)
      if name[-1] == '/':
        name = name[:-1]
      # Add the x bit to directories to prevent non-traversable directories.
      # The x bit is set only to if the read bit is set.
      dirmode = (mode | ((0o444 & mode) >> 2)) if mode else mode
      self.add_file(
          name + '/',
          tarfile.DIRTYPE,
          uid=uid,
          gid=gid,
          uname=uname,
          gname=gname,
          mtime=mtime,
          mode=dirmode)
      if depth <= 0:
        raise self.Error('Recursion depth exceeded, probably in '
                         'an infinite directory loop.')
      # Iterate over the sorted list of file so we get a deterministic result.
      filelist = os.listdir(path)
      filelist.sort()
      for f in filelist:
        new_name = os.path.join(name, f)
        new_path = os.path.join(path, f)
        self.add_dir(new_name, new_path, uid, gid, uname, gname, mtime, mode,
                     depth - 1)
    else:
      self.add_file(name,
                    tarfile.REGTYPE,
                    file_content=path,
                    uid=uid,
                    gid=gid,
                    uname=uname,
                    gname=gname,
                    mtime=mtime,
                    mode=mode)

  def _addfile(self, info, fileobj=None):
    """Add a file in the tar file if there is no conflict."""
    if not info.name.endswith('/') and info.type == tarfile.DIRTYPE:
      # Enforce the ending / for directories so we correctly deduplicate.
      info.name += '/'
    if info.name not in self.members:
      self.tar.addfile(info, fileobj)
      self.members.add(info.name)
    elif info.type != tarfile.DIRTYPE:
      print(('Duplicate file in archive: %s, '
             'picking first occurrence' % info.name))

  def add_file(self,
               name,
               kind=tarfile.REGTYPE,
               link=None,
               file_content=None,
               uid=0,
               gid=0,
               uname='',
               gname='',
               mtime=None,
               mode=None):
    """Add a file to the current tar.

    Args:
      name: the name of the file to add.
      kind: the type of the file to add, see tarfile.*TYPE.
      link: if the file is a link, the destination of the link.
      file_content: file to read the content from. Provide either this
          one or `content` to specifies a content for the file.
      uid: owner user identifier.
      gid: owner group identifier.
      uname: owner user names.
      gname: owner group names.
      mtime: modification time to put in the archive.
      mode: unix permission mode of the file, default 0644 (0755).
    """
    if file_content and os.path.isdir(file_content):
      # Recurse into directory
      self.add_dir(name, file_content, uid, gid, uname, gname, mtime, mode)
      return
    if not (name == self.root_directory or name.startswith('/') or
            name.startswith(self.root_directory + '/')):
      name = os.path.join(self.root_directory, name)
    if kind == tarfile.DIRTYPE:
      name = name.rstrip('/')
      if name in self.directories:
        return
    if mtime is None:
      mtime = self.default_mtime

    components = name.rsplit('/', 1)
    if len(components) > 1:
      d = components[0]
      self.add_file(d,
                    tarfile.DIRTYPE,
                    uid=uid,
                    gid=gid,
                    uname=uname,
                    gname=gname,
                    mtime=mtime,
                    mode=0o755)
    tarinfo = tarfile.TarInfo(name)
    tarinfo.mtime = mtime
    tarinfo.uid = uid
    tarinfo.gid = gid
    tarinfo.uname = uname
    tarinfo.gname = gname
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
        self._addfile(tarinfo, f)
    else:
      if kind == tarfile.DIRTYPE:
        self.directories.add(name)
      self._addfile(tarinfo)

  def add_tar(self,
              tar,
              rootuid=None,
              rootgid=None,
              numeric=False,
              root=None):
    """Merge a tar content into the current tar, stripping timestamp.

    Args:
      tar: the name of tar to extract and put content into the current tar.
      rootuid: user id that we will pretend is root (replaced by uid 0).
      rootgid: group id that we will pretend is root (replaced by gid 0).
      numeric: set to true to strip out name of owners (and just use the
          numeric values).
      root: place all non-absolute content under given root directory, if not
          None.

    Raises:
      TarFileWriter.Error: if an error happens when uncompressing the tar file.
    """
    if root and root[0] not in ['/', '.']:
      # Root prefix should start with a '/', adds it if missing
      root = '/' + root
    compression = os.path.splitext(tar)[-1][1:]
    if compression == 'tgz':
      compression = 'gz'
    elif compression == 'bzip2':
      compression = 'bz2'
    elif compression not in ['gz', 'bz2']:
      compression = ''
    if compression in ['gz', 'bz2']:
      # prevent performance issues due to accidentally-introduced seeks
      # during intar traversal by opening in "streaming" mode. gz, bz2
      # are supported natively by python 2.7 and 3.x
      inmode = 'r|' + compression
    else:
      inmode = 'r:' + compression
    intar = tarfile.open(name=tar, mode=inmode)
    for tarinfo in intar:
      if not self.preserve_mtime:
        tarinfo.mtime = self.default_mtime
      if rootuid is not None and tarinfo.uid == rootuid:
        tarinfo.uid = 0
        tarinfo.uname = 'root'
      if rootgid is not None and tarinfo.gid == rootgid:
        tarinfo.gid = 0
        tarinfo.gname = 'root'
      if numeric:
        tarinfo.uname = ''
        tarinfo.gname = ''

      name = tarinfo.name
      if (not name.startswith('/') and
          not name.startswith(self.root_directory)):
        name = os.path.join(self.root_directory, name)
      if root is not None:
        if name.startswith('.'):
          name = '.' + root + name.lstrip('.')
          # Add root dir with same permissions if missing. Note that
          # add_file deduplicates directories and is safe to call here.
          self.add_file('.' + root,
                        tarfile.DIRTYPE,
                        uid=tarinfo.uid,
                        gid=tarinfo.gid,
                        uname=tarinfo.uname,
                        gname=tarinfo.gname,
                        mtime=tarinfo.mtime,
                        mode=0o755)
        # Relocate internal hardlinks as well to avoid breaking them.
        link = tarinfo.linkname
        if link.startswith('.') and tarinfo.type == tarfile.LNKTYPE:
          tarinfo.linkname = '.' + root + link.lstrip('.')
      tarinfo.name = name

      if tarinfo.isfile():
        # use extractfile(tarinfo) instead of tarinfo.name to preserve
        # seek position in intar
        self._addfile(tarinfo, intar.extractfile(tarinfo))
      else:
        self._addfile(tarinfo)
    intar.close()

  def close(self):
    """Close the output tar file.

    This class should not be used anymore after calling that method.

    Raises:
      TarFileWriter.Error: if an error happens when compressing the output file.
    """
    self.tar.close()
    # Close the gzip file object if necessary.
    if self.fileobj:
      self.fileobj.close()


class TarFile(object):
  """A class to generates a TAR file."""

  def __init__(self, output, root_directory, default_mtime):
    self.root_directory = root_directory
    self.output = output
    self.default_mtime = default_mtime

  def __enter__(self):
    self.tarfile = TarFileWriter(
        self.output,
        self.root_directory,
        default_mtime=self.default_mtime)
    return self

  def __exit__(self, t, v, traceback):
    self.tarfile.close()

  def add_file(self, f, destfile, mode=None, ids=None, names=None):
    """Add a file to the tar file.

    Args:
       f: the file to add to the layer
       destfile: the name of the file in the layer
       mode: force to set the specified mode, by default the value from the
         source is taken.
       ids: (uid, gid) for the file to set ownership
       names: (username, groupname) for the file to set ownership. `f` will be
         copied to `self.directory/destfile` in the layer.
    """
    dest = destfile.lstrip('/')  # Remove leading slashes
    #if self.directory and self.directory != '/':
    #  dest = self.directory.lstrip('/') + '/' + dest
    # If mode is unspecified, derive the mode from the file's mode.
    if mode is None:
      mode = 0o755 if os.access(f, os.X_OK) else 0o644
    if ids is None:
      ids = (0, 0)
    if names is None:
      names = ('', '')
    dest = os.path.normpath(dest).replace(os.path.sep, '/')
    self.tarfile.add_file(
        dest,
        file_content=f,
        mode=mode,
        uid=ids[0],
        gid=ids[1],
        uname=names[0],
        gname=names[1])

  def add_link(self, symlink, destination):
    """Add a symbolic link pointing to `destination`.

    Args:
      symlink: the name of the symbolic link to add.
      destination: where the symbolic link point to.
    """
    symlink = os.path.normpath(symlink).replace(os.path.sep, '/')
    self.tarfile.add_file(symlink, tarfile.SYMTYPE, link=destination)


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
      description='Helper for building tar packages',
      fromfile_prefix_chars='@')
  parser.add_argument('--output', required=True,
                      help='The output file, mandatory.')
  parser.add_argument('--mode',
                      help='Force the mode on the added files (in octal).')
  #parser.add_argument(
  #    '--mtime',
  #    help='Set mtime on tar file entries. May be an integer or the'
  #         ' value "portable", to get the value 2000-01-01, which is'
  #         ' is usable with non *nix OSes.')
  parser.add_argument(
      '--directory',
      help='Directory in which to store the file inside the layer')

  parser.add_argument(
      '--file', action='append',
      help='input_paty=dest_path')
  parser.add_argument(
      '--owner', default='0.0',
      help='Specify the numeric default owner of all files,'
           ' e.g., 0.0')
  parser.add_argument(
      '--owner_name',
      help='Specify the owner name of all files, e.g. root.root.')
  options = parser.parse_args()

  # Parse modes arguments
  default_mode = None
  if options.mode:
    # Convert from octal
    default_mode = int(options.mode, 8)

  default_ownername = ('', '')
  if options.owner_name:
    default_ownername = options.owner_name.split('.', 1)

  # Add objects to the tar file
  with TarFile(options.output, options.directory, PORTABLE_MTIME) as output:
    for f in options.file:
      (inf, tof) = unquote_and_split(f, '=')
      output.add_file(inf, tof, mode=default_mode, names=default_ownername)
    #for link in options.link:
    #  l = unquote_and_split(link, ':')
    #  output.add_link(l[0], l[1])


if __name__ == '__main__':
  main()
