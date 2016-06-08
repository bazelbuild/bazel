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

import json
import re
import sys
import tarfile

from tools.build_defs.docker import utils
from tools.build_defs.pkg import archive
from third_party.py import gflags

# Hardcoded docker versions that we are claiming to be.
DATA_FORMAT_VERSION = '1.0'

gflags.DEFINE_string(
    'output', None,
    'The output file, mandatory')
gflags.MarkFlagAsRequired('output')

gflags.DEFINE_multistring(
    'layer', [],
    'Layer tar files and their identifiers that make up this image')

gflags.DEFINE_string(
    'id', None,
    'The hex identifier of this image (hexstring or @filename), mandatory.')
gflags.MarkFlagAsRequired('id')

gflags.DEFINE_string('config', None,
                     'The JSON configuration file for this image, mandatory.')
gflags.MarkFlagAsRequired('config')

gflags.DEFINE_string('base', None, 'The base image file for this image.')

gflags.DEFINE_string(
    'legacy_id', None,
    'The legacy hex identifier of this layer (hexstring or @filename).')

gflags.DEFINE_string('metadata', None,
                     'The legacy JSON metadata file for this layer.')

gflags.DEFINE_string('legacy_base', None,
                     'The legacy base image file for this image.')

gflags.DEFINE_string(
    'repository', None,
    'The name of the repository to add this image.')

gflags.DEFINE_string(
    'name', None,
    'The symbolic name of this image.')

gflags.DEFINE_multistring('tag', None,
                          'The repository tags to apply to the image')

FLAGS = gflags.FLAGS


def _base_name_filter(name):
  """Do not add multiple times 'top' and 'repositories' when merging images."""
  filter_names = ['top', 'repositories', 'manifest.json']
  return all([not name.endswith(s) for s in filter_names])


def create_image(output,
                 identifier,
                 layers,
                 config,
                 tags=None,
                 base=None,
                 legacy_base=None,
                 metadata_id=None,
                 metadata=None,
                 name=None,
                 repository=None):
  """Creates a Docker image.

  Args:
    output: the name of the docker image file to create.
    identifier: the identifier for this image (sha256 of the metadata).
    layers: the layer content (a sha256 and a tar file).
    config: the configuration file for the image.
    tags: tags that apply to this image.
    base: a base layer (optional) to build on top of.
    legacy_base: a base layer (optional) to build on top of.
    metadata_id: the identifier of the top layer for this image.
    metadata: the json metadata file for the top layer.
    name: symbolic name for this docker image.
    repository: repository name for this docker image.
  """
  tar = archive.TarFileWriter(output)

  # add the image config referenced by the Config section in the manifest
  # the name can be anything but docker uses the format below
  config_file_name = identifier + '.json'
  tar.add_file(config_file_name, file_content=config)

  layer_file_names = []

  if metadata_id:
    # Write our id to 'top' as we are now the topmost layer.
    tar.add_file('top', content=metadata_id)

    # Each layer is encoded as a directory in the larger tarball of the form:
    #  {id}\
    #    layer.tar
    #    VERSION
    #    json
    # Create the directory for us to now fill in.
    tar.add_file(metadata_id + '/', tarfile.DIRTYPE)
    # VERSION generally seems to contain 1.0, not entirely sure
    # what the point of this is.
    tar.add_file(metadata_id + '/VERSION', content=DATA_FORMAT_VERSION)
    # Add the layer file
    layer_file_name = metadata_id + '/layer.tar'
    layer_file_names.append(layer_file_name)
    tar.add_file(layer_file_name, file_content=layers[0]['layer'])
    # Now the json metadata
    tar.add_file(metadata_id + '/json', file_content=metadata)

    # Merge the base if any
    if legacy_base:
      tar.add_tar(legacy_base, name_filter=_base_name_filter)
  else:
    for layer in layers:
      # layers can be called anything, so just name them by their sha256
      layer_file_name = identifier + '/' + layer['name'] + '.tar'
      layer_file_names.append(layer_file_name)
      tar.add_file(layer_file_name, file_content=layer['layer'])

  base_layer_file_names = []
  parent = None
  if base:
    latest_item = utils.GetLatestManifestFromTar(base)
    if latest_item:
      base_layer_file_names = latest_item.get('Layers', [])
      config_file = latest_item['Config']
      parent_search = re.search('^(.+)\\.json$', config_file)
      if parent_search:
        parent = parent_search.group(1)

  manifest_item = {
      'Config': config_file_name,
      'Layers': base_layer_file_names + layer_file_names,
      'RepoTags': tags or []
  }
  if parent:
    manifest_item['Parent'] = 'sha256:' + parent

  manifest = [manifest_item]

  manifest_content = json.dumps(manifest, sort_keys=True)
  tar.add_file('manifest.json', content=manifest_content)

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
#              --layer=@identifier=layer.tar \
#              --metadata=metadata.json \
#              --name=myname --repository=repositoryName \
#              --tag=repo/image:tag
# See the gflags declaration about the flags argument details.
def main(unused_argv):
  identifier = utils.ExtractValue(FLAGS.id)
  legacy_id = utils.ExtractValue(FLAGS.legacy_id)

  layers = []
  for kv in FLAGS.layer:
    (k, v) = kv.split('=', 1)
    layers.append({
        'name': utils.ExtractValue(k),
        'layer': v,
    })

  create_image(FLAGS.output, identifier, layers, FLAGS.config, FLAGS.tag,
               FLAGS.base, FLAGS.legacy_base, legacy_id, FLAGS.metadata,
               FLAGS.name, FLAGS.repository)

if __name__ == '__main__':
  main(FLAGS(sys.argv))
