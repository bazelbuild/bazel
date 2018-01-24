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

gflags.DEFINE_multistring(
    'tags', [],
    'An associative list of fully qualified tag names and the layer they tag. '
    'e.g. ubuntu=deadbeef,gcr.io/blah/debian=baadf00d')

FLAGS = gflags.FLAGS


def _layer_filter(name):
  basename = os.path.basename(name)
  return basename not in ('manifest.json', 'top', 'repositories')


def _add_top(tar, repositories):
  # Don't add 'top' if there are multiple images in this bundle.
  if len(repositories) != 1:
    return

  # Walk the single-item dictionary, and if there is a single tag
  # for the single repository, then emit a 'top' file pointing to
  # the single image in this bundle.
  for (unused_x, tags) in repositories.items():
    if len(tags) != 1:
      continue
    for (unused_y, layer_id) in tags.items():
      tar.add_file('top', content=layer_id)


def create_image(output, layers, repositories=None):
  """Creates a Docker image from a list of layers.

  Args:
    output: the name of the docker image file to create.
    layers: the layers (tar files) to join to the image.
    repositories: the repositories two-level dictionary, which is keyed by
                  repo names at the top-level, and tag names at the second
                  level pointing to layer ids.
  """
  # Compute a map from layer tarball names to the tags that should apply to them
  layers_to_tag = {}
  for repo in repositories:
    tags = repositories[repo]
    for tag in tags:
      layer_name = tags[tag] + '/layer.tar'
      fq_name = '%s:%s' % (repo, tag)
      layer_tags = layers_to_tag.get(layer_name, [])
      layer_tags.append(fq_name)
      layers_to_tag[layer_name] = layer_tags

  manifests = []
  tar = archive.TarFileWriter(output)
  for layer in layers:
    tar.add_tar(layer, name_filter=_layer_filter)
    layer_manifests = utils.GetManifestFromTar(layer)

    # Augment each manifest with any tags that should apply to their top layer.
    for manifest in layer_manifests:
      top_layer = manifest['Layers'][-1]
      manifest['RepoTags'] = list(sorted(set(manifest['RepoTags'] +
                                             layers_to_tag.get(top_layer, []))))

    manifests += layer_manifests

  manifest_content = json.dumps(manifests, sort_keys=True)
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
  # This is the exact structure we expect repositories to have.
  if repositories:
    # If the identifier is not provided, then the resulted layer will be
    # created without a 'top' file.  Docker doesn't needs that file nor
    # the repository to load the image and for intermediate layer,
    # docker_build store the name of the layer in a separate artifact so
    # this 'top' file is not needed.
    _add_top(tar, repositories)
    tar.add_file('repositories',
                 content=json.dumps(repositories, sort_keys=True))


def resolve_layer(identifier):
  if not identifier:
    # TODO(mattmoor): This should not happen.
    return None

  if not identifier.startswith('@'):
    return identifier

  with open(identifier[1:], 'r') as f:
    return f.read()


def main(unused_argv):
  repositories = {}
  for entry in FLAGS.tags:
    elts = entry.split('=')
    if len(elts) != 2:
      raise Exception('Expected associative list key=value, got: %s' % entry)
    (fq_tag, layer_id) = elts

    tag_parts = fq_tag.rsplit(':', 2)
    if len(tag_parts) != 2:
      raise Exception('Expected fully-qualified tag name (e.g. ubuntu:latest), '
                      'got: %s' % fq_tag)
    (repository, tag) = tag_parts

    others = repositories.get(repository, {})
    others[tag] = resolve_layer(layer_id)
    repositories[repository] = others

  create_image(FLAGS.output, FLAGS.layer, repositories)


if __name__ == '__main__':
  main(FLAGS(sys.argv))
