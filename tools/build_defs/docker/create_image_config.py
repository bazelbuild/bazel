# Copyright 2016 The Bazel Authors. All rights reserved.
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
"""This package manipulates OCI image configuration metadata."""
from collections import namedtuple
import copy
import json
import os
import os.path
import sys

from tools.build_defs.docker import utils
from third_party.py import gflags

gflags.DEFINE_string('base', None, 'The parent image')

gflags.DEFINE_string('output', None, 'The output file to generate')
gflags.MarkFlagAsRequired('output')

gflags.DEFINE_multistring('layer', [],
                          'Layer sha256 hashes that make up this image')

gflags.DEFINE_list('entrypoint', None,
                   'Override the "Entrypoint" of the previous image')

gflags.DEFINE_list('command', None, 'Override the "Cmd" of the previous image')

gflags.DEFINE_string('user', None, 'The username to run commands under')

gflags.DEFINE_list('labels', None, 'Augment the "Label" of the previous image')

gflags.DEFINE_list('ports', None,
                   'Augment the "ExposedPorts" of the previous image')

gflags.DEFINE_list('volumes', None,
                   'Augment the "Volumes" of the previous image')

gflags.DEFINE_string('workdir', None, 'Set the working directory for the image')

gflags.DEFINE_list('env', None, 'Augment the "Env" of the previous image')

FLAGS = gflags.FLAGS

_ConfigOptionsT = namedtuple('ConfigOptionsT', ['layers', 'entrypoint', 'cmd',
                                                'env', 'labels', 'ports',
                                                'volumes', 'workdir', 'user'])


class ConfigOptions(_ConfigOptionsT):
  """Docker image configuration options."""

  def __new__(cls,
              layers=None,
              entrypoint=None,
              cmd=None,
              user=None,
              labels=None,
              env=None,
              ports=None,
              volumes=None,
              workdir=None):
    """Constructor."""
    return super(ConfigOptions, cls).__new__(cls,
                                             layers=layers,
                                             entrypoint=entrypoint,
                                             cmd=cmd,
                                             user=user,
                                             labels=labels,
                                             env=env,
                                             ports=ports,
                                             volumes=volumes,
                                             workdir=workdir)

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
  if isinstance(data, dict):
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


def CreateImageConfig(data, options):
  """Create an image config possibly based on an existing one.

  Args:
    data: A dict of Docker image config to base on top of.
    options: Options specific to this image which will be merged with any
             existing data

  Returns:
    Image config for the new image
  """
  defaults = DeepCopySkipNull(data)

  # dont propagate non-spec keys
  output = dict()
  output['created'] = '0001-01-01T00:00:00Z'
  output['author'] = 'Bazel'
  output['architecture'] = _PROCESSOR_ARCHITECTURE
  output['os'] = _OPERATING_SYSTEM

  output['config'] = defaults.get('config', {})

  if options.entrypoint:
    output['config']['Entrypoint'] = options.entrypoint
  if options.cmd:
    output['config']['Cmd'] = options.cmd
  if options.user:
    output['config']['User'] = options.user

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

  # TODO(babel-team) Label is currently docker specific
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

  # diff_ids are ordered from bottom-most to top-most
  diff_ids = defaults.get('rootfs', {}).get('diff_ids', [])
  layers = options.layers if options.layers else []
  diff_ids += ['sha256:%s' % l for l in layers]
  output['rootfs'] = {
      'type': 'layers',
      'diff_ids': diff_ids,
  }

  # history is ordered from bottom-most layer to top-most layer
  history = defaults.get('history', [])
  # docker only allows the child to have one more history entry than the parent
  history += [{
      'created': '0001-01-01T00:00:00Z',
      'created_by': 'bazel build ...',
      'author': 'Bazel'}]
  output['history'] = history

  return output


def main(unused_argv):
  base_json = '{}'
  manifest = utils.GetLatestManifestFromTar(FLAGS.base)
  if manifest:
    config_file = manifest['Config']
    base_json = utils.GetTarFile(FLAGS.base, config_file)
  data = json.loads(base_json)

  layers = []
  for layer in FLAGS.layer:
    layers.append(utils.ExtractValue(layer))

  labels = KeyValueToDict(FLAGS.labels)
  for label, value in labels.iteritems():
    if value.startswith('@'):
      with open(value[1:], 'r') as f:
        labels[label] = f.read()

  output = CreateImageConfig(data,
                             ConfigOptions(layers=layers,
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
