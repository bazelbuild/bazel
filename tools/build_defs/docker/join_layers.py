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
"""This tool creates a docker image from a list of layers."""
# This is the main program to create a docker image. It expect to be run with:
# join_layers --output=output_file \
#             --layer=layer1 [--layer=layer2 ... --layer=layerN] \
#             --id=@identifier \
#             --name=myname --repository=repositoryName
# See the gflags declaration about the flags argument details.

import json
import os.path
import sys

from tools.build_defs.docker import utils
from tools.build_defs.pkg import archive
from third_party.py import gflags

gflags.DEFINE_string('output', None, 'The output file, mandatory')
gflags.MarkFlagAsRequired('output')

gflags.DEFINE_multistring('layer', [], 'The tar files for layers to join.')

gflags.DEFINE_string(
    'id', None, 'The hex identifier of the top layer (hexstring or @filename).')

gflags.DEFINE_string(
    'repository', None,
    'The name of the repository to add this image (use with --id and --name).')

gflags.DEFINE_string(
    'name', None,
    'The symbolic name of this image (use with --id and --repository).')

FLAGS = gflags.FLAGS


def _layer_filter(name):
  basename = os.path.basename(name)
  return basename not in ('manifest.json', 'top', 'repositories')


def create_image(output, layers, identifier=None,
                 name=None, repository=None):
  """Creates a Docker image from a list of layers.

  Args:
    output: the name of the docker image file to create.
    layers: the layers (tar files) to join to the image.
    identifier: the identifier of the top layer for this image.
    name: symbolic name for this docker image.
    repository: repository name for this docker image.
  """
  manifest = []

  tar = archive.TarFileWriter(output)
  for layer in layers:
    tar.add_tar(layer, name_filter=_layer_filter)
    manifest += utils.GetManifestFromTar(layer)

  manifest_content = json.dumps(manifest, sort_keys=True)
  tar.add_file('manifest.json', content=manifest_content)

  # In addition to N layers of the form described above, there might be
  # a single file at the top of the image called repositories.
  # This file contains a JSON blob of the form:
  # {
  #   'repo':{
  #     'tag-name': 'top-most layer hex',
  #     ...
  #   },
  #   ...
  # }
  if identifier:
    # If the identifier is not provided, then the resulted layer will be
    # created without a 'top' file. Docker doesn't needs that file nor
    # the repository to load the image and for intermediate layer,
    # docker_build store the name of the layer in a separate artifact so
    # this 'top' file is not needed.
    tar.add_file('top', content=identifier)
    if repository and name:
      tar.add_file('repositories',
                   content='\n'.join([
                       '{', '  "%s": {' % repository, '    "%s": "%s"' % (
                           name, identifier), '  }', '}'
                   ]))


def main(unused_argv):
  identifier = FLAGS.id
  if identifier and identifier.startswith('@'):
    with open(identifier[1:], 'r') as f:
      identifier = f.read()
  create_image(FLAGS.output, FLAGS.layer, identifier, FLAGS.name,
               FLAGS.repository)


if __name__ == '__main__':
  main(FLAGS(sys.argv))
