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
"""This package manipulates Docker image layer metadata."""
from collections import namedtuple
import copy
import json
import os
import os.path
import sys
import tarfile

from third_party.py import gflags

gflags.DEFINE_string(
    'name', None, 'The name of the current layer')

gflags.DEFINE_string(
    'base', None, 'The parent image')

gflags.DEFINE_string(
    'output', None, 'The output file to generate')

gflags.DEFINE_string(
    'layer', None, 'The current layer tar')

gflags.DEFINE_list(
    'entrypoint', None,
    'Override the "Entrypoint" of the previous layer')

gflags.DEFINE_list(
    'command', None,
    'Override the "Cmd" of the previous layer')

gflags.DEFINE_list(
    'ports', None,
    'Augment the "ExposedPorts" of the previous layer')

gflags.DEFINE_list(
    'volumes', None,
    'Augment the "Volumes" of the previous layer')

gflags.DEFINE_list(
    'env', None,
    'Augment the "Env" of the previous layer')

FLAGS = gflags.FLAGS

_MetadataOptionsT = namedtuple(
    'MetadataOptionsT',
    ['name', 'parent', 'size', 'entrypoint', 'cmd', 'env', 'ports', 'volumes'])


class MetadataOptions(_MetadataOptionsT):
  """Docker image layer metadata options."""

  def __new__(cls, name=None, parent=None, size=None,
              entrypoint=None, cmd=None, env=None,
              ports=None, volumes=None):
    """Constructor."""
    return super(MetadataOptions, cls).__new__(
        cls, name=name, parent=parent, size=size,
        entrypoint=entrypoint, cmd=cmd, env=env,
        ports=ports, volumes=volumes)


_DOCKER_VERSION = '1.5.0'

_PROCESSOR_ARCHITECTURE = 'amd64'

_OPERATING_SYSTEM = 'linux'


def Resolve(value, environment):
  """Resolves environment variables embedded in the given value."""
  outer_env = os.environ
  try:
    os.environ = environment
    return os.path.expandvars(value)
  finally:
    os.environ = outer_env


def RewriteMetadata(data, options):
  """Rewrite and return a copy of the input data according to options.

  Args:
    data: The dict of Docker image layer metadata we're copying and rewriting.
    options: The changes this layer makes to the overall image's metadata, which
             first appears in this layer's version of the metadata

  Returns:
    A deep copy of data, which has been updated to reflect the metadata
    additions of this layer.

  Raises:
    Exception: a required option was missing.
  """
  output = copy.deepcopy(data)

  if not options.name:
    raise Exception('Missing required option: name')
  output['id'] = options.name

  if options.parent:
    output['parent'] = options.parent
  elif data:
    raise Exception('Expected empty input object when parent is omitted')

  if options.size:
    output['Size'] = options.size
  elif 'Size' in output:
    del output['Size']

  if 'config' not in output:
    output['config'] = {}

  if options.entrypoint:
    output['config']['Entrypoint'] = options.entrypoint
  if options.cmd:
    output['config']['Cmd'] = options.cmd

  output['docker_version'] = _DOCKER_VERSION
  output['architecture'] = _PROCESSOR_ARCHITECTURE
  output['os'] = _OPERATING_SYSTEM

  if options.env:
    environ_dict = {}
    # Build a dictionary of existing environment variables (used by Resolve).
    for kv in output['config'].get('Env', []):
      (k, v) = kv.split('=', 1)
      environ_dict[k] = v
    # Merge in new environment variables, resolving references.
    for kv in options.env:
      (k, v) = kv.split('=', 1)
      # Resolve handles scenarios like "PATH=$PATH:...".
      v = Resolve(v, environ_dict)
      environ_dict[k] = v
    output['config']['Env'] = [
        '%s=%s' % (k, environ_dict[k]) for k in sorted(environ_dict.keys())]

  if options.ports:
    if 'ExposedPorts' not in output['config']:
      output['config']['ExposedPorts'] = {}
    for p in options.ports:
      if '/' in p:
        # The port spec has the form 80/tcp, 1234/udp
        # so we simply use it as the key.
        output['config']['ExposedPorts'][p] = {}
      else:
        # Assume tcp
        output['config']['ExposedPorts'][p + '/tcp'] = {}

  if options.volumes:
    if 'Volumes' not in output['config']:
      output['config']['Volumes'] = {}
    for p in options.volumes:
      output['config']['Volumes'][p] = {}

  # TODO(mattmoor): comment, created, container_config

  # container_config contains information about the container
  # that was used to create this layer, so it shouldn't
  # propagate from the parent to child.  This is where we would
  # annotate information that can be extract by tools like Blubber
  # or Quay.io's UI to gain insight into the source that generated
  # the layer.  A Dockerfile might produce something like:
  #   # (nop) /bin/sh -c "apt-get update"
  # We might consider encoding the fully-qualified bazel build target:
  #  //tools/build_defs/docker:image
  # However, we should be sensitive to leaking data through this field.
  if 'container_config' in output:
    del output['container_config']

  return output


def GetTarFile(f, name):
  """Return the content of a file inside a tar file."""
  with tarfile.open(f, 'r') as tar:
    tarinfo = tar.getmember(name)
    if not tarinfo:
      return ''
    return tar.extractfile(tarinfo).read()


def main(unused_argv):
  parent = ''
  base_json = '{}'
  if FLAGS.base:
    parent = GetTarFile(FLAGS.base, './top')
    base_json = GetTarFile(FLAGS.base, './%s/json' % parent)
  data = json.loads(base_json)

  name = FLAGS.name
  if name.startswith('@'):
    with open(name[1:], 'r') as f:
      name = f.read()

  output = RewriteMetadata(data, MetadataOptions(
      name=name,
      parent=parent,
      size=os.path.getsize(FLAGS.layer),
      entrypoint=FLAGS.entrypoint,
      cmd=FLAGS.command,
      env=FLAGS.env,
      ports=FLAGS.ports,
      volumes=FLAGS.volumes))

  with open(FLAGS.output, 'w') as fp:
    json.dump(output, fp, sort_keys=True)
    fp.write('\n')

if __name__ == '__main__':
  main(FLAGS(sys.argv))
