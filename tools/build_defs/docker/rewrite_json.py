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
"""This package manipulates Docker image layer metadata."""
from collections import namedtuple
import copy
import json
import os
import os.path
import sys

from tools.build_defs.docker import utils
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

gflags.DEFINE_string(
    'user', None, 'The username to run commands under')

gflags.DEFINE_list('labels', None, 'Augment the "Label" of the previous layer')

gflags.DEFINE_list(
    'ports', None,
    'Augment the "ExposedPorts" of the previous layer')

gflags.DEFINE_list(
    'volumes', None,
    'Augment the "Volumes" of the previous layer')

gflags.DEFINE_string(
    'workdir', None,
    'Set the working directory for the layer')

gflags.DEFINE_list(
    'env', None,
    'Augment the "Env" of the previous layer')

FLAGS = gflags.FLAGS

_MetadataOptionsT = namedtuple('MetadataOptionsT',
                               ['name', 'parent', 'size', 'entrypoint', 'cmd',
                                'env', 'labels', 'ports', 'volumes', 'workdir',
                                'user'])


class MetadataOptions(_MetadataOptionsT):
  """Docker image layer metadata options."""

  def __new__(cls,
              name=None,
              parent=None,
              size=None,
              entrypoint=None,
              cmd=None,
              user=None,
              labels=None,
              env=None,
              ports=None,
              volumes=None,
              workdir=None):
    """Constructor."""
    return super(MetadataOptions, cls).__new__(cls,
                                               name=name,
                                               parent=parent,
                                               size=size,
                                               entrypoint=entrypoint,
                                               cmd=cmd,
                                               user=user,
                                               labels=labels,
                                               env=env,
                                               ports=ports,
                                               volumes=volumes,
                                               workdir=workdir)


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


def DeepCopySkipNull(data):
  """Do a deep copy, skipping null entry."""
  if type(data) == type(dict()):
    return dict((DeepCopySkipNull(k), DeepCopySkipNull(v))
                for k, v in data.iteritems() if v is not None)
  return copy.deepcopy(data)


def KeyValueToDict(pair):
  """Converts an iterable object of key=value pairs to dictionary."""
  d = dict()
  for kv in pair:
    (k, v) = kv.split('=', 1)
    d[k] = v
  return d


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
  output = DeepCopySkipNull(data)

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
  if options.user:
    output['config']['User'] = options.user

  output['docker_version'] = _DOCKER_VERSION
  output['architecture'] = _PROCESSOR_ARCHITECTURE
  output['os'] = _OPERATING_SYSTEM

  def Dict2ConfigValue(d):
    return ['%s=%s' % (k, d[k]) for k in sorted(d.keys())]

  if options.env:
    # Build a dictionary of existing environment variables (used by Resolve).
    environ_dict = KeyValueToDict(output['config'].get('Env', []))
    # Merge in new environment variables, resolving references.
    for k, v in options.env.iteritems():
      # Resolve handles scenarios like "PATH=$PATH:...".
      environ_dict[k] = Resolve(v, environ_dict)
    output['config']['Env'] = Dict2ConfigValue(environ_dict)

  if options.labels:
    label_dict = KeyValueToDict(output['config'].get('Label', []))
    for k, v in options.labels.iteritems():
      label_dict[k] = v
    output['config']['Label'] = Dict2ConfigValue(label_dict)

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

  if options.workdir:
    output['config']['WorkingDir'] = options.workdir

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


def GetParentIdentifier(f):
  """Try to look at the parent identifier from a docker image.

  The identifier is expected to be in the 'top' file for our rule so we look at
  it first ('./top', 'top'). If it's not found, then we use the 'repositories'
  file and tries to parse it to get the first declared repository (so we can
  actually parse a file generated by 'docker save').

  Args:
    f: the input tar file.
  Returns:
    The identifier of the docker image, or None if no identifier was found.
  """
  # TODO(dmarting): Maybe we could drop the 'top' file all together?
  top = utils.GetTarFile(f, 'top')
  if top:
    return top.strip()
  repositories = utils.GetTarFile(f, 'repositories')
  if repositories:
    data = json.loads(repositories)
    for k1 in data:
      for k2 in data[k1]:
        # Returns the first found key
        return data[k1][k2].strip()
  return None


def main(unused_argv):
  parent = ''
  base_json = '{}'
  if FLAGS.base:
    parent = GetParentIdentifier(FLAGS.base)
    if parent:
      base_json = utils.GetTarFile(FLAGS.base, '%s/json' % parent)
  data = json.loads(base_json)

  name = FLAGS.name
  if name.startswith('@'):
    with open(name[1:], 'r') as f:
      name = f.read()

  labels = KeyValueToDict(FLAGS.labels)
  for label, value in labels.iteritems():
    if value.startswith('@'):
      with open(value[1:], 'r') as f:
        labels[label] = f.read()

  output = RewriteMetadata(data,
                           MetadataOptions(name=name,
                                           parent=parent,
                                           size=os.path.getsize(FLAGS.layer),
                                           entrypoint=FLAGS.entrypoint,
                                           cmd=FLAGS.command,
                                           user=FLAGS.user,
                                           labels=labels,
                                           env=KeyValueToDict(FLAGS.env),
                                           ports=FLAGS.ports,
                                           volumes=FLAGS.volumes,
                                           workdir=FLAGS.workdir))

  with open(FLAGS.output, 'w') as fp:
    json.dump(output, fp, sort_keys=True)
    fp.write('\n')

if __name__ == '__main__':
  main(FLAGS(sys.argv))
