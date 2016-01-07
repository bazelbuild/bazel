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
"""This tool creates a docker image from a layer and the various metadata."""

import sys
import tarfile

from tools.build_defs.pkg import archive
from third_party.py import gflags

# Hardcoded docker versions that we are claiming to be.
DATA_FORMAT_VERSION = '1.0'

gflags.DEFINE_string(
    'output', None,
    'The output file, mandatory')
gflags.MarkFlagAsRequired('output')

gflags.DEFINE_string(
    'metadata', None,
    'The JSON metadata file for this image, mandatory.')
gflags.MarkFlagAsRequired('metadata')

gflags.DEFINE_string(
    'layer', None,
    'The tar file for the top layer of this image, mandatory.')
gflags.MarkFlagAsRequired('layer')

gflags.DEFINE_string(
    'id', None,
    'The hex identifier of this image (hexstring or @filename), mandatory.')
gflags.MarkFlagAsRequired('id')

gflags.DEFINE_string(
    'base', None,
    'The base image file for this image.')

gflags.DEFINE_string(
    'repository', None,
    'The name of the repository to add this image.')

gflags.DEFINE_string(
    'name', None,
    'The symbolic name of this image.')

FLAGS = gflags.FLAGS


def _base_name_filter(name):
  """Do not add multiple times 'top' and 'repositories' when merging images."""
  filter_names = ['top', 'repositories']
  return all([not name.endswith(s) for s in filter_names])


def create_image(output, identifier,
                 base=None, layer=None, metadata=None,
                 name=None, repository=None):
  """Creates a Docker image.

  Args:
    output: the name of the docker image file to create.
    identifier: the identifier of the top layer for this image.
    base: a base layer (optional) to merge to current layer.
    layer: the layer content (a tar file).
    metadata: the json metadata file for the top layer.
    name: symbolic name for this docker image.
    repository: repository name for this docker image.
  """
  tar = archive.TarFileWriter(output)
  # Write our id to 'top' as we are now the topmost layer.
  tar.add_file('top', content=identifier)
  # Each layer is encoded as a directory in the larger tarball of the form:
  #  {id}\
  #    layer.tar
  #    VERSION
  #    json
  # Create the directory for us to now fill in.
  tar.add_file(identifier + '/', tarfile.DIRTYPE)
  # VERSION generally seems to contain 1.0, not entirely sure
  # what the point of this is.
  tar.add_file(identifier + '/VERSION', content=DATA_FORMAT_VERSION)
  # Add the layer file
  tar.add_file(identifier + '/layer.tar', file_content=layer)
  # Now the json metadata
  tar.add_file(identifier + '/json', file_content=metadata)
  # Merge the base if any
  if base:
    tar.add_tar(base, name_filter=_base_name_filter)
  # In addition to N layers of the form described above, there is
  # a single file at the top of the image called repositories.
  # This file contains a JSON blob of the form:
  # {
  #   'repo':{
  #     'tag-name': 'top-most layer hex',
  #     ...
  #   },
  #   ...
  # }
  if repository:
    tar.add_file('repositories', content='\n'.join([
        '{',
        '  "%s": {' % repository,
        '    "%s": "%s"' % (name, identifier),
        '  }',
        '}']))


# Main program to create a docker image. It expect to be run with:
# create_image --output=output_file \
#              --id=@identifier \
#              [--base=base] \
#              --layer=layer.tar \
#              --metadata=metadata.json \
#              --name=myname --repository=repositoryName
# See the gflags declaration about the flags argument details.
def main(unused_argv):
  identifier = FLAGS.id
  if identifier.startswith('@'):
    with open(identifier[1:], 'r') as f:
      identifier = f.read()
  create_image(FLAGS.output, identifier, FLAGS.base,
               FLAGS.layer, FLAGS.metadata,
               FLAGS.name, FLAGS.repository)

if __name__ == '__main__':
  main(FLAGS(sys.argv))
