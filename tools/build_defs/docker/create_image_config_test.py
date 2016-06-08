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
"""Testing for create_image_config."""

import unittest

from tools.build_defs.docker.create_image_config import _OPERATING_SYSTEM
from tools.build_defs.docker.create_image_config import _PROCESSOR_ARCHITECTURE
from tools.build_defs.docker.create_image_config import ConfigOptions
from tools.build_defs.docker.create_image_config import CreateImageConfig


class CreateImageConfigTest(unittest.TestCase):
  """Testing for create_image_config."""

  base_expected = {
      'created': '0001-01-01T00:00:00Z',
      'author': 'Bazel',
      'architecture': _PROCESSOR_ARCHITECTURE,
      'os': _OPERATING_SYSTEM,
      'config': {},
      'rootfs': {'diff_ids': [],
                 'type': 'layers'},
      'history': [{'author': 'Bazel',
                   'created': '0001-01-01T00:00:00Z',
                   'created_by': 'bazel build ...'}],
  }

  def testNewUser(self):
    in_data = {'config': {'WorkingDir': '/usr/home/mattmoor'}}
    user = 'mattmoor'
    expected = self.base_expected.copy()
    expected.update({
        'config': {
            'User': 'mattmoor',
            'WorkingDir': '/usr/home/mattmoor',
        },
    })

    actual = CreateImageConfig(in_data, ConfigOptions(user=user))
    self.assertEquals(expected, actual)

  def testOverrideUser(self):
    in_data = {
        'config': {
            'User': 'mattmoor',
            'WorkingDir': '/usr/home/mattmoor',
        }
    }
    user = 'mattmoor2'
    expected = self.base_expected.copy()
    expected.update({
        'config': {
            'User': 'mattmoor2',
            'WorkingDir': '/usr/home/mattmoor',
        },
    })

    actual = CreateImageConfig(in_data, ConfigOptions(user=user))
    self.assertEquals(expected, actual)

  def testNewEntrypoint(self):
    in_data = {
        'config': {
            'User': 'mattmoor',
            'WorkingDir': '/usr/home/mattmoor'
        }
    }
    entrypoint = ['/bin/bash']
    expected = self.base_expected.copy()
    expected.update({
        'config': {
            'User': 'mattmoor',
            'WorkingDir': '/usr/home/mattmoor',
            'Entrypoint': entrypoint
        },
    })

    actual = CreateImageConfig(in_data, ConfigOptions(entrypoint=entrypoint))
    self.assertEquals(expected, actual)

  def testOverrideEntrypoint(self):
    in_data = {
        'config': {
            'User': 'mattmoor',
            'WorkingDir': '/usr/home/mattmoor',
            'Entrypoint': ['/bin/sh', 'does', 'not', 'matter'],
        }
    }
    entrypoint = ['/bin/bash']
    expected = self.base_expected.copy()
    expected.update({
        'config': {
            'User': 'mattmoor',
            'WorkingDir': '/usr/home/mattmoor',
            'Entrypoint': entrypoint
        },
    })

    actual = CreateImageConfig(in_data, ConfigOptions(entrypoint=entrypoint))
    self.assertEquals(expected, actual)

  def testNewCmd(self):
    in_data = {
        'config': {
            'User': 'mattmoor',
            'WorkingDir': '/usr/home/mattmoor',
            'Entrypoint': ['/bin/bash'],
        }
    }
    cmd = ['/bin/bash']
    expected = self.base_expected.copy()
    expected.update({
        'config': {
            'User': 'mattmoor',
            'WorkingDir': '/usr/home/mattmoor',
            'Entrypoint': ['/bin/bash'],
            'Cmd': cmd
        },
    })

    actual = CreateImageConfig(in_data, ConfigOptions(cmd=cmd))
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
    cmd = ['does', 'matter']
    expected = self.base_expected.copy()
    expected.update({
        'config': {
            'User': 'mattmoor',
            'WorkingDir': '/usr/home/mattmoor',
            'Entrypoint': ['/bin/bash'],
            'Cmd': cmd
        },
    })

    actual = CreateImageConfig(in_data, ConfigOptions(cmd=cmd))
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
    entrypoint = ['/bin/bash', '-c']
    cmd = ['my-command', 'my-arg1', 'my-arg2']
    expected = self.base_expected.copy()
    expected.update({
        'config': {
            'User': 'mattmoor',
            'WorkingDir': '/usr/home/mattmoor',
            'Entrypoint': entrypoint,
            'Cmd': cmd
        },
    })

    actual = CreateImageConfig(in_data,
                               ConfigOptions(entrypoint=entrypoint,
                                             cmd=cmd))
    self.assertEquals(expected, actual)

  def testStripContainerConfig(self):
    in_data = {'container_config': {},}
    expected = self.base_expected.copy()

    actual = CreateImageConfig(in_data, ConfigOptions())
    self.assertEquals(expected, actual)

  def testEmptyBase(self):
    in_data = {}
    entrypoint = ['/bin/bash', '-c']
    cmd = ['my-command', 'my-arg1', 'my-arg2']
    expected = self.base_expected.copy()
    expected.update({
        'config': {
            'Entrypoint': entrypoint,
            'Cmd': cmd,
            'ExposedPorts': {
                '80/tcp': {}
            }
        },
    })

    actual = CreateImageConfig(in_data,
                               ConfigOptions(entrypoint=entrypoint,
                                             cmd=cmd,
                                             ports=['80']))
    self.assertEquals(expected, actual)

  def testNewPort(self):
    in_data = {
        'config': {
            'User': 'mattmoor',
            'WorkingDir': '/usr/home/mattmoor'
        }
    }
    port = '80'
    expected = self.base_expected.copy()
    expected.update({
        'config': {
            'User': 'mattmoor',
            'WorkingDir': '/usr/home/mattmoor',
            'ExposedPorts': {
                port + '/tcp': {}
            }
        },
    })

    actual = CreateImageConfig(in_data, ConfigOptions(ports=[port]))
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
    port = '80'
    expected = self.base_expected.copy()
    expected.update({
        'config': {
            'User': 'mattmoor',
            'WorkingDir': '/usr/home/mattmoor',
            'ExposedPorts': {
                '443/tcp': {},
                port + '/tcp': {}
            }
        },
    })

    actual = CreateImageConfig(in_data, ConfigOptions(ports=[port]))
    self.assertEquals(expected, actual)

  def testMultiplePorts(self):
    in_data = {
        'config': {
            'User': 'mattmoor',
            'WorkingDir': '/usr/home/mattmoor'
        }
    }
    port1 = '80'
    port2 = '8080'
    expected = self.base_expected.copy()
    expected.update({
        'config': {
            'User': 'mattmoor',
            'WorkingDir': '/usr/home/mattmoor',
            'ExposedPorts': {
                port1 + '/tcp': {},
                port2 + '/tcp': {}
            }
        },
    })

    actual = CreateImageConfig(in_data, ConfigOptions(ports=[port1, port2]))
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
    expected = self.base_expected.copy()
    expected.update({
        'config': {
            'User': 'mattmoor',
            'WorkingDir': '/usr/home/mattmoor',
            'ExposedPorts': {
                port + '/tcp': {}
            }
        },
    })

    actual = CreateImageConfig(in_data, ConfigOptions(ports=[port]))
    self.assertEquals(expected, actual)

  def testPortWithProtocol(self):
    in_data = {
        'config': {
            'User': 'mattmoor',
            'WorkingDir': '/usr/home/mattmoor'
        }
    }
    port = '80/tcp'
    expected = self.base_expected.copy()
    expected.update({
        'config': {
            'User': 'mattmoor',
            'WorkingDir': '/usr/home/mattmoor',
            'ExposedPorts': {
                port: {}
            }
        },
    })

    actual = CreateImageConfig(in_data, ConfigOptions(ports=[port]))
    self.assertEquals(expected, actual)

  def testNewVolume(self):
    in_data = {
        'config': {
            'User': 'mattmoor',
            'WorkingDir': '/usr/home/mattmoor'
        }
    }
    volume = '/logs'
    expected = self.base_expected.copy()
    expected.update({
        'config': {
            'User': 'mattmoor',
            'WorkingDir': '/usr/home/mattmoor',
            'Volumes': {
                volume: {}
            }
        },
    })

    actual = CreateImageConfig(in_data, ConfigOptions(volumes=[volume]))
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
    volume = '/data'
    expected = self.base_expected.copy()
    expected.update({
        'config': {
            'User': 'mattmoor',
            'WorkingDir': '/usr/home/mattmoor',
            'Volumes': {
                '/original': {},
                volume: {}
            }
        },
    })

    actual = CreateImageConfig(in_data, ConfigOptions(volumes=[volume]))
    self.assertEquals(expected, actual)

  def testMultipleVolumes(self):
    in_data = {
        'config': {
            'User': 'mattmoor',
            'WorkingDir': '/usr/home/mattmoor'
        }
    }
    volume1 = '/input'
    volume2 = '/output'
    expected = self.base_expected.copy()
    expected.update({
        'config': {
            'User': 'mattmoor',
            'WorkingDir': '/usr/home/mattmoor',
            'Volumes': {
                volume1: {},
                volume2: {}
            }
        },
    })

    actual = CreateImageConfig(in_data,
                               ConfigOptions(volumes=[volume1, volume2]))
    self.assertEquals(expected, actual)

  def testEnv(self):
    in_data = {
        'config': {
            'User': 'mattmoor',
            'WorkingDir': '/usr/home/mattmoor'
        }
    }
    env = {'baz': 'blah',
           'foo': 'bar',}
    expected = self.base_expected.copy()
    expected.update({
        'config': {
            'User': 'mattmoor',
            'WorkingDir': '/usr/home/mattmoor',
            'Env': [
                'baz=blah',
                'foo=bar',
            ],
        },
    })

    actual = CreateImageConfig(in_data, ConfigOptions(env=env))
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
    env = {'baz': 'replacement',
           'foo': '$foo:asdf',}
    expected = self.base_expected.copy()
    expected.update({
        'config': {
            'User': 'mattmoor',
            'WorkingDir': '/usr/home/mattmoor',
            'Env': [
                'baz=replacement',
                'blah=still around',
                'foo=bar:asdf',
            ],
        },
    })

    actual = CreateImageConfig(in_data, ConfigOptions(env=env))
    self.assertEquals(expected, actual)

  def testLabel(self):
    in_data = {
        'config': {
            'User': 'mattmoor',
            'WorkingDir': '/usr/home/mattmoor'
        }
    }
    labels = {'baz': 'blah',
              'foo': 'bar',}
    expected = self.base_expected.copy()
    expected.update({
        'config': {
            'User': 'mattmoor',
            'WorkingDir': '/usr/home/mattmoor',
            'Label': [
                'baz=blah',
                'foo=bar',
            ],
        },
    })

    actual = CreateImageConfig(in_data, ConfigOptions(labels=labels))
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
    labels = {'baz': 'replacement',
              'foo': 'bar',}
    expected = self.base_expected.copy()
    expected.update({
        'config': {
            'User': 'mattmoor',
            'WorkingDir': '/usr/home/mattmoor',
            'Label': [
                'baz=replacement',
                'blah=still around',
                'foo=bar',
            ],
        },
    })

    actual = CreateImageConfig(in_data, ConfigOptions(labels=labels))
    self.assertEquals(expected, actual)

  def testAugmentVolumeWithNullInput(self):
    in_data = {
        'config': {
            'User': 'mattmoor',
            'WorkingDir': '/usr/home/mattmoor',
            'Volumes': None,
        }
    }
    volume = '/data'
    expected = self.base_expected.copy()
    expected.update({
        'config': {
            'User': 'mattmoor',
            'WorkingDir': '/usr/home/mattmoor',
            'Volumes': {
                volume: {}
            }
        },
    })

    actual = CreateImageConfig(in_data, ConfigOptions(volumes=[volume]))
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
    workdir = '/some/path'
    expected = self.base_expected.copy()
    expected.update({
        'config': {
            'User': 'bleh',
            'WorkingDir': '/some/path',
            'Volumes': {
            }
        },
    })

    actual = CreateImageConfig(in_data, ConfigOptions(workdir=workdir))
    self.assertEquals(expected, actual)

  def testLayersAddedToDiffIds(self):
    initial_diff_ids = [
        'sha256:1',
        'sha256:2',
    ]
    in_data = {
        'rootfs': {
            'type': 'layers',
            'diff_ids': initial_diff_ids,
        }
    }
    layers = ['3', '4']
    expected = self.base_expected.copy()
    expected.update({
        'rootfs': {
            'type': 'layers',
            'diff_ids': initial_diff_ids + ['sha256:%s' % l for l in layers],
        }
    })

    actual = CreateImageConfig(in_data, ConfigOptions(layers=layers))
    self.assertEquals(expected, actual)

  def testHistoryAdded(self):
    in_data = self.base_expected.copy()
    expected = self.base_expected.copy()
    expected.update({
        'history': [
            {
                'author': 'Bazel',
                'created': '0001-01-01T00:00:00Z',
                'created_by': 'bazel build ...'
            }, {
                'author': 'Bazel',
                'created': '0001-01-01T00:00:00Z',
                'created_by': 'bazel build ...'
            }
        ]
    })

    actual = CreateImageConfig(in_data, ConfigOptions())
    self.assertEquals(expected, actual)


if __name__ == '__main__':
  unittest.main()
