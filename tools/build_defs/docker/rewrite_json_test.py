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
"""Testing for rewrite_json."""

import unittest

from tools.build_defs.docker.rewrite_json import _DOCKER_VERSION
from tools.build_defs.docker.rewrite_json import _OPERATING_SYSTEM
from tools.build_defs.docker.rewrite_json import _PROCESSOR_ARCHITECTURE
from tools.build_defs.docker.rewrite_json import MetadataOptions
from tools.build_defs.docker.rewrite_json import RewriteMetadata


class RewriteJsonTest(unittest.TestCase):
  """Testing for rewrite_json."""

  def testNewEntrypoint(self):
    in_data = {
        'config': {
            'User': 'mattmoor',
            'WorkingDir': '/usr/home/mattmoor'
        }
    }
    name = 'deadbeef'
    parent = 'blah'
    entrypoint = ['/bin/bash']
    expected = {
        'id': name,
        'parent': parent,
        'config': {
            'User': 'mattmoor',
            'WorkingDir': '/usr/home/mattmoor',
            'Entrypoint': entrypoint
        },
        'docker_version': _DOCKER_VERSION,
        'architecture': _PROCESSOR_ARCHITECTURE,
        'os': _OPERATING_SYSTEM,
    }

    actual = RewriteMetadata(in_data, MetadataOptions(
        name=name, entrypoint=entrypoint, parent=parent))
    self.assertEquals(expected, actual)

  def testOverrideEntrypoint(self):
    in_data = {
        'config': {
            'User': 'mattmoor',
            'WorkingDir': '/usr/home/mattmoor',
            'Entrypoint': ['/bin/sh', 'does', 'not', 'matter'],
        }
    }
    name = 'deadbeef'
    parent = 'blah'
    entrypoint = ['/bin/bash']
    expected = {
        'id': name,
        'parent': parent,
        'config': {
            'User': 'mattmoor',
            'WorkingDir': '/usr/home/mattmoor',
            'Entrypoint': entrypoint
        },
        'docker_version': _DOCKER_VERSION,
        'architecture': _PROCESSOR_ARCHITECTURE,
        'os': _OPERATING_SYSTEM,
    }

    actual = RewriteMetadata(in_data, MetadataOptions(
        name=name, entrypoint=entrypoint, parent=parent))
    self.assertEquals(expected, actual)

  def testNewCmd(self):
    in_data = {
        'config': {
            'User': 'mattmoor',
            'WorkingDir': '/usr/home/mattmoor',
            'Entrypoint': ['/bin/bash'],
        }
    }
    name = 'deadbeef'
    parent = 'blah'
    cmd = ['/bin/bash']
    expected = {
        'id': name,
        'parent': parent,
        'config': {
            'User': 'mattmoor',
            'WorkingDir': '/usr/home/mattmoor',
            'Entrypoint': ['/bin/bash'],
            'Cmd': cmd
        },
        'docker_version': _DOCKER_VERSION,
        'architecture': _PROCESSOR_ARCHITECTURE,
        'os': _OPERATING_SYSTEM,
    }

    actual = RewriteMetadata(in_data, MetadataOptions(
        name=name, cmd=cmd, parent=parent))
    self.assertEquals(expected, actual)

  def testOverrideCmd(self):
    in_data = {
        'config': {
            'User': 'mattmoor',
            'WorkingDir': '/usr/home/mattmoor',
            'Entrypoint': ['/bin/bash'],
            'Cmd': ['does', 'not', 'matter'],
        }
    }
    name = 'deadbeef'
    parent = 'blah'
    cmd = ['does', 'matter']
    expected = {
        'id': name,
        'parent': parent,
        'config': {
            'User': 'mattmoor',
            'WorkingDir': '/usr/home/mattmoor',
            'Entrypoint': ['/bin/bash'],
            'Cmd': cmd
        },
        'docker_version': _DOCKER_VERSION,
        'architecture': _PROCESSOR_ARCHITECTURE,
        'os': _OPERATING_SYSTEM,
    }

    actual = RewriteMetadata(in_data, MetadataOptions(
        name=name, cmd=cmd, parent=parent))
    self.assertEquals(expected, actual)

  def testOverrideBoth(self):
    in_data = {
        'config': {
            'User': 'mattmoor',
            'WorkingDir': '/usr/home/mattmoor',
            'Entrypoint': ['/bin/sh'],
            'Cmd': ['does', 'not', 'matter'],
        }
    }
    name = 'deadbeef'
    parent = 'blah'
    entrypoint = ['/bin/bash', '-c']
    cmd = ['my-command', 'my-arg1', 'my-arg2']
    expected = {
        'id': name,
        'parent': parent,
        'config': {
            'User': 'mattmoor',
            'WorkingDir': '/usr/home/mattmoor',
            'Entrypoint': entrypoint,
            'Cmd': cmd
        },
        'docker_version': _DOCKER_VERSION,
        'architecture': _PROCESSOR_ARCHITECTURE,
        'os': _OPERATING_SYSTEM,
    }

    actual = RewriteMetadata(in_data, MetadataOptions(
        name=name, entrypoint=entrypoint, cmd=cmd, parent=parent))
    self.assertEquals(expected, actual)

  def testOverrideParent(self):
    name = 'me!'
    parent = 'parent'
    # In the typical case, we expect the parent to
    # come in as the 'id', and our grandparent to
    # be its 'parent'.
    in_data = {
        'id': parent,
        'parent': 'grandparent',
    }
    expected = {
        'id': name,
        'parent': parent,
        'config': {},
        'docker_version': _DOCKER_VERSION,
        'architecture': _PROCESSOR_ARCHITECTURE,
        'os': _OPERATING_SYSTEM,
    }

    actual = RewriteMetadata(in_data, MetadataOptions(
        name=name, parent=parent))
    self.assertEquals(expected, actual)

  def testNewSize(self):
    # Size is one of the few fields that, when omitted,
    # should be removed.
    in_data = {
        'id': 'you',
        'Size': '124',
    }
    name = 'me'
    parent = 'blah'
    size = '4321'
    expected = {
        'id': name,
        'parent': parent,
        'Size': size,
        'config': {},
        'docker_version': _DOCKER_VERSION,
        'architecture': _PROCESSOR_ARCHITECTURE,
        'os': _OPERATING_SYSTEM,
    }

    actual = RewriteMetadata(in_data, MetadataOptions(
        name=name, size=size, parent=parent))
    self.assertEquals(expected, actual)

  def testOmitSize(self):
    # Size is one of the few fields that, when omitted,
    # should be removed.
    in_data = {
        'id': 'you',
        'Size': '124',
    }
    name = 'me'
    parent = 'blah'
    expected = {
        'id': name,
        'parent': parent,
        'config': {},
        'docker_version': _DOCKER_VERSION,
        'architecture': _PROCESSOR_ARCHITECTURE,
        'os': _OPERATING_SYSTEM,
    }

    actual = RewriteMetadata(in_data, MetadataOptions(
        name=name, parent=parent))
    self.assertEquals(expected, actual)

  def testOmitName(self):
    # Name is required.
    with self.assertRaises(Exception):
      RewriteMetadata({}, MetadataOptions(name=None))

  def testStripContainerConfig(self):
    # Size is one of the few fields that, when omitted,
    # should be removed.
    in_data = {
        'id': 'you',
        'container_config': {},
    }
    name = 'me'
    parent = 'blah'
    expected = {
        'id': name,
        'parent': parent,
        'config': {},
        'docker_version': _DOCKER_VERSION,
        'architecture': _PROCESSOR_ARCHITECTURE,
        'os': _OPERATING_SYSTEM,
    }

    actual = RewriteMetadata(in_data, MetadataOptions(
        name=name, parent=parent))
    self.assertEquals(expected, actual)

  def testEmptyBase(self):
    in_data = {}
    name = 'deadbeef'
    entrypoint = ['/bin/bash', '-c']
    cmd = ['my-command', 'my-arg1', 'my-arg2']
    size = '999'
    expected = {
        'id': name,
        'config': {
            'Entrypoint': entrypoint,
            'Cmd': cmd,
            'ExposedPorts': {
                '80/tcp': {}
            }
        },
        'docker_version': _DOCKER_VERSION,
        'architecture': _PROCESSOR_ARCHITECTURE,
        'os': _OPERATING_SYSTEM,
        'Size': size,
    }

    actual = RewriteMetadata(in_data, MetadataOptions(
        name=name, entrypoint=entrypoint, cmd=cmd, size=size,
        ports=['80']))
    self.assertEquals(expected, actual)

  def testOmitParentWithBase(self):
    # Our input data should be empty when parent is omitted
    in_data = {
        'id': 'you',
    }
    with self.assertRaises(Exception):
      RewriteMetadata(in_data, MetadataOptions(name='me'))

  def testNewPort(self):
    in_data = {
        'config': {
            'User': 'mattmoor',
            'WorkingDir': '/usr/home/mattmoor'
        }
    }
    name = 'deadbeef'
    parent = 'blah'
    port = '80'
    expected = {
        'id': name,
        'parent': parent,
        'config': {
            'User': 'mattmoor',
            'WorkingDir': '/usr/home/mattmoor',
            'ExposedPorts': {
                port + '/tcp': {}
            }
        },
        'docker_version': _DOCKER_VERSION,
        'architecture': _PROCESSOR_ARCHITECTURE,
        'os': _OPERATING_SYSTEM,
    }

    actual = RewriteMetadata(in_data, MetadataOptions(
        name=name, parent=parent, ports=[port]))
    self.assertEquals(expected, actual)

  def testAugmentPort(self):
    in_data = {
        'config': {
            'User': 'mattmoor',
            'WorkingDir': '/usr/home/mattmoor',
            'ExposedPorts': {
                '443/tcp': {}
            }
        }
    }
    name = 'deadbeef'
    parent = 'blah'
    port = '80'
    expected = {
        'id': name,
        'parent': parent,
        'config': {
            'User': 'mattmoor',
            'WorkingDir': '/usr/home/mattmoor',
            'ExposedPorts': {
                '443/tcp': {},
                port + '/tcp': {}
            }
        },
        'docker_version': _DOCKER_VERSION,
        'architecture': _PROCESSOR_ARCHITECTURE,
        'os': _OPERATING_SYSTEM,
    }

    actual = RewriteMetadata(in_data, MetadataOptions(
        name=name, parent=parent, ports=[port]))
    self.assertEquals(expected, actual)

  def testMultiplePorts(self):
    in_data = {
        'config': {
            'User': 'mattmoor',
            'WorkingDir': '/usr/home/mattmoor'
        }
    }
    name = 'deadbeef'
    parent = 'blah'
    port1 = '80'
    port2 = '8080'
    expected = {
        'id': name,
        'parent': parent,
        'config': {
            'User': 'mattmoor',
            'WorkingDir': '/usr/home/mattmoor',
            'ExposedPorts': {
                port1 + '/tcp': {},
                port2 + '/tcp': {}
            }
        },
        'docker_version': _DOCKER_VERSION,
        'architecture': _PROCESSOR_ARCHITECTURE,
        'os': _OPERATING_SYSTEM,
    }

    actual = RewriteMetadata(in_data, MetadataOptions(
        name=name, parent=parent, ports=[port1, port2]))
    self.assertEquals(expected, actual)

  def testPortCollision(self):
    port = '80'
    in_data = {
        'config': {
            'User': 'mattmoor',
            'WorkingDir': '/usr/home/mattmoor',
            'ExposedPorts': {
                port + '/tcp': {}
            }
        }
    }
    name = 'deadbeef'
    parent = 'blah'
    expected = {
        'id': name,
        'parent': parent,
        'config': {
            'User': 'mattmoor',
            'WorkingDir': '/usr/home/mattmoor',
            'ExposedPorts': {
                port + '/tcp': {}
            }
        },
        'docker_version': _DOCKER_VERSION,
        'architecture': _PROCESSOR_ARCHITECTURE,
        'os': _OPERATING_SYSTEM,
    }

    actual = RewriteMetadata(in_data, MetadataOptions(
        name=name, parent=parent, ports=[port]))
    self.assertEquals(expected, actual)

  def testPortWithProtocol(self):
    in_data = {
        'config': {
            'User': 'mattmoor',
            'WorkingDir': '/usr/home/mattmoor'
        }
    }
    name = 'deadbeef'
    parent = 'blah'
    port = '80/tcp'
    expected = {
        'id': name,
        'parent': parent,
        'config': {
            'User': 'mattmoor',
            'WorkingDir': '/usr/home/mattmoor',
            'ExposedPorts': {
                port: {}
            }
        },
        'docker_version': _DOCKER_VERSION,
        'architecture': _PROCESSOR_ARCHITECTURE,
        'os': _OPERATING_SYSTEM,
    }

    actual = RewriteMetadata(in_data, MetadataOptions(
        name=name, parent=parent, ports=[port]))
    self.assertEquals(expected, actual)

  def testNewVolume(self):
    in_data = {
        'config': {
            'User': 'mattmoor',
            'WorkingDir': '/usr/home/mattmoor'
        }
    }
    name = 'deadbeef'
    parent = 'blah'
    volume = '/logs'
    expected = {
        'id': name,
        'parent': parent,
        'config': {
            'User': 'mattmoor',
            'WorkingDir': '/usr/home/mattmoor',
            'Volumes': {
                volume: {}
            }
        },
        'docker_version': _DOCKER_VERSION,
        'architecture': _PROCESSOR_ARCHITECTURE,
        'os': _OPERATING_SYSTEM,
    }

    actual = RewriteMetadata(in_data, MetadataOptions(
        name=name, parent=parent, volumes=[volume]))
    self.assertEquals(expected, actual)

  def testAugmentVolume(self):
    in_data = {
        'config': {
            'User': 'mattmoor',
            'WorkingDir': '/usr/home/mattmoor',
            'Volumes': {
                '/original': {}
            }
        }
    }
    name = 'deadbeef'
    parent = 'blah'
    volume = '/data'
    expected = {
        'id': name,
        'parent': parent,
        'config': {
            'User': 'mattmoor',
            'WorkingDir': '/usr/home/mattmoor',
            'Volumes': {
                '/original': {},
                volume: {}
            }
        },
        'docker_version': _DOCKER_VERSION,
        'architecture': _PROCESSOR_ARCHITECTURE,
        'os': _OPERATING_SYSTEM,
    }

    actual = RewriteMetadata(in_data, MetadataOptions(
        name=name, parent=parent, volumes=[volume]))
    self.assertEquals(expected, actual)

  def testMultipleVolumes(self):
    in_data = {
        'config': {
            'User': 'mattmoor',
            'WorkingDir': '/usr/home/mattmoor'
        }
    }
    name = 'deadbeef'
    parent = 'blah'
    volume1 = '/input'
    volume2 = '/output'
    expected = {
        'id': name,
        'parent': parent,
        'config': {
            'User': 'mattmoor',
            'WorkingDir': '/usr/home/mattmoor',
            'Volumes': {
                volume1: {},
                volume2: {}
            }
        },
        'docker_version': _DOCKER_VERSION,
        'architecture': _PROCESSOR_ARCHITECTURE,
        'os': _OPERATING_SYSTEM,
    }

    actual = RewriteMetadata(in_data, MetadataOptions(
        name=name, parent=parent, volumes=[volume1, volume2]))
    self.assertEquals(expected, actual)

  def testEnv(self):
    in_data = {
        'config': {
            'User': 'mattmoor',
            'WorkingDir': '/usr/home/mattmoor'
        }
    }
    name = 'deadbeef'
    parent = 'blah'
    env = {'baz': 'blah', 'foo': 'bar',}
    expected = {
        'id': name,
        'parent': parent,
        'config': {
            'User': 'mattmoor',
            'WorkingDir': '/usr/home/mattmoor',
            'Env': [
                'baz=blah',
                'foo=bar',
            ],
        },
        'docker_version': _DOCKER_VERSION,
        'architecture': _PROCESSOR_ARCHITECTURE,
        'os': _OPERATING_SYSTEM,
    }

    actual = RewriteMetadata(in_data, MetadataOptions(
        name=name, env=env, parent=parent))
    self.assertEquals(expected, actual)

  def testEnvResolveReplace(self):
    in_data = {
        'config': {
            'User': 'mattmoor',
            'WorkingDir': '/usr/home/mattmoor',
            'Env': [
                'foo=bar',
                'baz=blah',
                'blah=still around',
            ],
        }
    }
    name = 'deadbeef'
    parent = 'blah'
    env = {'baz': 'replacement', 'foo': '$foo:asdf',}
    expected = {
        'id': name,
        'parent': parent,
        'config': {
            'User': 'mattmoor',
            'WorkingDir': '/usr/home/mattmoor',
            'Env': [
                'baz=replacement',
                'blah=still around',
                'foo=bar:asdf',
            ],
        },
        'docker_version': _DOCKER_VERSION,
        'architecture': _PROCESSOR_ARCHITECTURE,
        'os': _OPERATING_SYSTEM,
    }

    actual = RewriteMetadata(in_data, MetadataOptions(
        name=name, env=env, parent=parent))
    self.assertEquals(expected, actual)

  def testLabel(self):
    in_data = {
        'config': {
            'User': 'mattmoor',
            'WorkingDir': '/usr/home/mattmoor'
        }
    }
    name = 'deadbeef'
    parent = 'blah'
    labels = {'baz': 'blah', 'foo': 'bar',}
    expected = {
        'id': name,
        'parent': parent,
        'config': {
            'User': 'mattmoor',
            'WorkingDir': '/usr/home/mattmoor',
            'Label': [
                'baz=blah',
                'foo=bar',
            ],
        },
        'docker_version': _DOCKER_VERSION,
        'architecture': _PROCESSOR_ARCHITECTURE,
        'os': _OPERATING_SYSTEM,
    }

    actual = RewriteMetadata(in_data,
                             MetadataOptions(name=name,
                                             labels=labels,
                                             parent=parent))
    self.assertEquals(expected, actual)

  def testAugmentLabel(self):
    in_data = {
        'config': {
            'User': 'mattmoor',
            'WorkingDir': '/usr/home/mattmoor',
            'Label': [
                'baz=blah',
                'blah=still around',
            ],
        }
    }
    name = 'deadbeef'
    parent = 'blah'
    labels = {'baz': 'replacement', 'foo': 'bar',}
    expected = {
        'id': name,
        'parent': parent,
        'config': {
            'User': 'mattmoor',
            'WorkingDir': '/usr/home/mattmoor',
            'Label': [
                'baz=replacement',
                'blah=still around',
                'foo=bar',
            ],
        },
        'docker_version': _DOCKER_VERSION,
        'architecture': _PROCESSOR_ARCHITECTURE,
        'os': _OPERATING_SYSTEM,
    }

    actual = RewriteMetadata(in_data,
                             MetadataOptions(name=name,
                                             labels=labels,
                                             parent=parent))
    self.assertEquals(expected, actual)

  def testAugmentVolumeWithNullInput(self):
    in_data = {
        'config': {
            'User': 'mattmoor',
            'WorkingDir': '/usr/home/mattmoor',
            'Volumes': None,
        }
    }
    name = 'deadbeef'
    parent = 'blah'
    volume = '/data'
    expected = {
        'id': name,
        'parent': parent,
        'config': {
            'User': 'mattmoor',
            'WorkingDir': '/usr/home/mattmoor',
            'Volumes': {
                volume: {}
            }
        },
        'docker_version': _DOCKER_VERSION,
        'architecture': _PROCESSOR_ARCHITECTURE,
        'os': _OPERATING_SYSTEM,
    }

    actual = RewriteMetadata(in_data, MetadataOptions(
        name=name, parent=parent, volumes=[volume]))
    self.assertEquals(expected, actual)

  def testSetWorkingDir(self):
    in_data = {
        'config': {
            'User': 'bleh',
            'WorkingDir': '/home/bleh',
            'Volumes': {
            }
        }
    }
    name = 'deadbeef'
    parent = 'blah'
    workdir = '/some/path'
    expected = {
        'id': name,
        'parent': parent,
        'config': {
            'User': 'bleh',
            'WorkingDir': '/some/path',
            'Volumes': {
            }
        },
        'docker_version': _DOCKER_VERSION,
        'architecture': _PROCESSOR_ARCHITECTURE,
        'os': _OPERATING_SYSTEM,
    }

    actual = RewriteMetadata(in_data, MetadataOptions(
        name=name, parent=parent, workdir=workdir))
    self.assertEquals(expected, actual)

if __name__ == '__main__':
  unittest.main()
