# Copyright 2015 Google Inc. All rights reserved.
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
"""This tool build docker layer tar file from a list of inputs."""

import os
import os.path
import sys
import tarfile
import tempfile

from tools.build_defs.docker import archive
from third_party.py import gflags

gflags.DEFINE_string(
    'output', None,
    'The output file, mandatory')
gflags.MarkFlagAsRequired('output')

gflags.DEFINE_multistring(
    'file', [],
    'A file to add to the layer')

gflags.DEFINE_multistring(
    'tar', [],
    'A tar file to add to the layer')

gflags.DEFINE_multistring(
    'deb', [],
    'A debian package to add to the layer')

gflags.DEFINE_multistring(
    'link', [],
    'Add a symlink a inside the layer ponting to b if a:b is specified')
gflags.RegisterValidator(
    'link',
    lambda l: all(value.find(':') > 0 for value in l),
    message='--link value should contains a : separator')

gflags.DEFINE_string(
    'directory', None,
    'Directory in which to store the file inside the layer')

FLAGS = gflags.FLAGS


class DockerLayer(object):
  """A class to generates a Docker layer."""

  class DebError(Exception):
    pass

  def __init__(self, output, directory):
    self.directory = directory
    self.output = output

  def __enter__(self):
    self.tarfile = archive.TarFileWriter(self.output)
    return self

  def __exit__(self, t, v, traceback):
    self.tarfile.close()

  def add_file(self, f, destfile):
    """Add a file to the layer.

    Args:
       f: the file to add to the layer
       destfile: the name of the file in the layer

    `f` will be copied to `self.directory/destfile` in the layer.
    """
    dest = destfile.lstrip('/')  # Remove leading slashes
    # TODO(mattmoor): Consider applying the working directory to all four
    # options, not just files...
    if self.directory and self.directory != '/':
      dest = self.directory.lstrip('/') + '/' + dest
    self.tarfile.add_file(dest, file_content=f)

  def add_tar(self, tar):
    """Add a tar file to the layer.

    All files presents in that tar will be added to the layer under
    the same paths. No user name nor group name will be added to
    the layer.

    Args:
      tar: the tar file to add to the layer
    """
    self.tarfile.add_tar(tar, numeric=True)

  def add_link(self, symlink, destination):
    """Add a symbolic link pointing to `destination` in the layer.

    Args:
      symlink: the name of the symbolic link to add.
      destination: where the symbolic link point to.
    """
    self.tarfile.add_file(symlink, tarfile.SYMTYPE, link=destination)

  def add_deb(self, deb):
    """Extract a debian package in the layer.

    All files presents in that debian package will be added to the
    layer under the same paths. No user name nor group names will
    be added to the layer.

    Args:
      deb: the tar file to add to the layer

    Raises:
      DebError: if the format of the deb archive is incorrect.

    This method does not support LZMA (data.tar.xz or data.tar.lzma)
    for the data in the deb package. Using Python 3 would fix it.
    """
    with archive.SimpleArFile(deb) as arfile:
      current = arfile.next()
      while current and not current.filename.startswith('data.'):
        current = arfile.next()
      if not current:
        raise self.DebError(deb + ' does not contains a data file!')
      tmpfile = tempfile.mkstemp(suffix=os.path.splitext(current.filename)[-1])
      with open(tmpfile[1], 'wb') as f:
        f.write(current.data)
      self.add_tar(tmpfile[1])
      os.remove(tmpfile[1])


def main(unused_argv):
  with DockerLayer(FLAGS.output, FLAGS.directory) as layer:
    for f in FLAGS.file:
      (inf, tof) = f.split('=', 1)
      layer.add_file(inf, tof)
    for tar in FLAGS.tar:
      layer.add_tar(tar)
    for deb in FLAGS.deb:
      layer.add_deb(deb)
    for link in FLAGS.link:
      l = link.split(':', 1)
      layer.add_link(l[0], l[1])

if __name__ == '__main__':
  main(FLAGS(sys.argv))
