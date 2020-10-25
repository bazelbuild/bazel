# Lint as: python2, python3
# Copyright 2015 The Bazel Authors. All rights reserved.
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
"""Archive manipulation library for the Docker rules."""

# pylint: disable=g-import-not-at-top
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import gzip
import io
import os
import subprocess
import tarfile
import six

# Use a deterministic mtime that doesn't confuse other programs.
# See: https://github.com/bazelbuild/bazel/issues/1299
PORTABLE_MTIME = 946684800  # 2000-01-01 00:00:00.000 UTC


class SimpleArFile(object):
  """A simple AR file reader.

  This enable to read AR file (System V variant) as described
  in https://en.wikipedia.org/wiki/Ar_(Unix).

  The standard usage of this class is:

  with SimpleArFile(filename) as ar:
    nextFile = ar.next()
    while nextFile:
      print(nextFile.filename)
      nextFile = ar.next()

  Upon error, this class will raise a ArError exception.
  """

  # TODO(dmarting): We should use a standard library instead but python 2.7
  #   does not have AR reading library.

  class ArError(Exception):
    pass

  class SimpleArFileEntry(object):
    """Represent one entry in a AR archive.

    Attributes:
      filename: the filename of the entry, as described in the archive.
      timestamp: the timestamp of the file entry.
      owner_id, group_id: numeric id of the user and group owning the file.
      mode: unix permission mode of the file
      size: size of the file
      data: the content of the file.
    """

    def __init__(self, f):
      self.filename = f.read(16).decode('utf-8').strip()
      if self.filename.endswith('/'):  # SysV variant
        self.filename = self.filename[:-1]
      self.timestamp = int(f.read(12).strip())
      self.owner_id = int(f.read(6).strip())
      self.group_id = int(f.read(6).strip())
      self.mode = int(f.read(8).strip(), 8)
      self.size = int(f.read(10).strip())
      pad = f.read(2)
      if pad != b'\x60\x0a':
        raise SimpleArFile.ArError('Invalid AR file header')
      self.data = f.read(self.size)

  MAGIC_STRING = b'!<arch>\n'

  def __init__(self, filename):
    self.filename = filename

  def __enter__(self):
    self.f = open(self.filename, 'rb')
    if self.f.read(len(self.MAGIC_STRING)) != self.MAGIC_STRING:
      raise self.ArError('Not a ar file: ' + six.ensure_str(self.filename))
    return self

  def __exit__(self, t, v, traceback):
    self.f.close()

  def next(self):
    """Read the next file. Returns None when reaching the end of file."""
    # AR sections are two bit aligned using new lines.
    if self.f.tell() % 2 != 0:
      self.f.read(1)
    # An AR sections is at least 60 bytes. Some file might contains garbage
    # bytes at the end of the archive, ignore them.
    if self.f.tell() > os.fstat(self.f.fileno()).st_size - 60:
      return None
    return self.SimpleArFileEntry(self.f)


class TarFileWriter(object):
  """A wrapper to write tar files."""

  class Error(Exception):
    pass

  def __init__(self,
               name,
               compression='',
               root_directory='./',
               default_mtime=None,
               preserve_tar_mtimes=False):
    """TarFileWriter wraps tarfile.open().

    Args:
      name: the tar file name.
      compression: compression type: bzip2, bz2, gz, tgz, xz, lzma.
      root_directory: virtual root to prepend to elements in the archive.
      default_mtime: default mtime to use for elements in the archive.
          May be an integer or the value 'portable' to use the date
          2000-01-01, which is compatible with non *nix OSes'.
      preserve_tar_mtimes: if true, keep file mtimes from input tar file.
    """
    if compression in ['bzip2', 'bz2']:
      mode = 'w:bz2'
    else:
      mode = 'w:'
    self.gz = compression in ['tgz', 'gz']
    # Support xz compression through xz... until we can use Py3
    self.xz = compression in ['xz', 'lzma']
    self.name = name
    self.root_directory = six.ensure_str(root_directory).rstrip('/')

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
    self.directories = set([])

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
    if not (name == self.root_directory or six.ensure_str(name).startswith('/')
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
          six.ensure_str(name) + '/',
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
    if not six.ensure_str(
        info.name).endswith('/') and info.type == tarfile.DIRTYPE:
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
               content=None,
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
      content: a textual content to put in the file.
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
    if content:
      content_bytes = six.ensure_binary(content, 'utf-8')
      tarinfo.size = len(content_bytes)
      self._addfile(tarinfo, io.BytesIO(content_bytes))
    elif file_content:
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
              name_filter=None,
              root=None):
    """Merge a tar content into the current tar, stripping timestamp.

    Args:
      tar: the name of tar to extract and put content into the current tar.
      rootuid: user id that we will pretend is root (replaced by uid 0).
      rootgid: group id that we will pretend is root (replaced by gid 0).
      numeric: set to true to strip out name of owners (and just use the
          numeric values).
      name_filter: filter out file by names. If not none, this method will be
          called for each file to add, given the name and should return true if
          the file is to be added to the final tar and false otherwise.
      root: place all non-absolute content under given root directory, if not
          None.

    Raises:
      TarFileWriter.Error: if an error happens when uncompressing the tar file.
    """
    if root and root[0] not in ['/', '.']:
      # Root prefix should start with a '/', adds it if missing
      root = '/' + six.ensure_str(root)
    compression = os.path.splitext(tar)[-1][1:]
    if compression == 'tgz':
      compression = 'gz'
    elif compression == 'bzip2':
      compression = 'bz2'
    elif compression == 'lzma':
      compression = 'xz'
    elif compression not in ['gz', 'bz2', 'xz']:
      compression = ''
    if compression == 'xz':
      # Python 2 does not support lzma, our py3 support is terrible so let's
      # just hack around.
      # Note that we buffer the file in memory and it can have an important
      # memory footprint but it's probably fine as we don't use them for really
      # large files.
      # TODO(dmarting): once our py3 support gets better, compile this tools
      # with py3 for proper lzma support.
      if subprocess.call('which xzcat', shell=True, stdout=subprocess.PIPE):
        raise self.Error('Cannot handle .xz and .lzma compression: '
                         'xzcat not found.')
      p = subprocess.Popen('cat %s | xzcat' % tar,
                           shell=True,
                           stdout=subprocess.PIPE)
      f = io.BytesIO(p.stdout.read())
      p.wait()
      intar = tarfile.open(fileobj=f, mode='r:')
    else:
      if compression in ['gz', 'bz2']:
        # prevent performance issues due to accidentally-introduced seeks
        # during intar traversal by opening in "streaming" mode. gz, bz2
        # are supported natively by python 2.7 and 3.x
        inmode = 'r|' + six.ensure_str(compression)
      else:
        inmode = 'r:' + six.ensure_str(compression)
      intar = tarfile.open(name=tar, mode=inmode)
    for tarinfo in intar:
      if name_filter is None or name_filter(tarinfo.name):
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
    if self.xz:
      # Support xz compression through xz... until we can use Py3
      if subprocess.call('which xz', shell=True, stdout=subprocess.PIPE):
        raise self.Error('Cannot handle .xz and .lzma compression: '
                         'xz not found.')
      subprocess.call(
          'mv {0} {0}.d && xz -z {0}.d && mv {0}.d.xz {0}'.format(self.name),
          shell=True,
          stdout=subprocess.PIPE)
