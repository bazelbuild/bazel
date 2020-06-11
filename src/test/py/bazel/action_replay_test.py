# Copyright 2020 The Bazel Authors. All rights reserved.
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

import unittest
from src.test.py.bazel import test_base


class ActionReplayTest(test_base.TestBase):

  def RunBazel(self, args, env_remove=None, env_add=None):
    extra_flags = []
    if self.IsWindows():
      # This ensures we don't run Bash on Windows,
      # otherwise an error will be thrown.
      extra_flags.append('--shell_executable=')
    return super(ActionReplayTest, self).RunBazel(args + extra_flags,
                                                  env_remove, env_add)

  def testReplayInMemoryCache(self):
    self.ScratchFile('WORKSPACE')
    self.ScratchFile('foo/BUILD', [
        'genrule(',
        '  name = "gen",',
        '  outs = ["gen.x"],',
        '  cmd_bash = "echo GEN_X 1>&2 && touch $@",',
        '  cmd_bat = "echo GEN_X 1>&2 && echo. 2> $@",',
        ')',
    ])

    _, _, stderr = self.RunBazel(
        ['build', '--experimental_replay_action_out_err', '//foo:gen'])
    self.assertRegex('\n'.join(stderr), 'GEN_X')

    _, _, stderr = self.RunBazel(
        ['build', '--experimental_replay_action_out_err', '//foo:gen'])
    self.assertRegex('\n'.join(stderr), 'GEN_X')

  def testNoReplay(self):
    self.ScratchFile('WORKSPACE')
    self.ScratchFile('foo/BUILD', [
        'genrule(',
        '  name = "gen",',
        '  outs = ["gen.x"],',
        '  cmd_bash = "echo GEN_X 1>&2 && touch $@",',
        '  cmd_bat = "echo GEN_X 1>&2 && echo. 2> $@",',
        ')',
    ])

    _, _, stderr = self.RunBazel(['build', '//foo:gen'])
    self.assertRegex('\n'.join(stderr), 'GEN_X')

    _, _, stderr = self.RunBazel(['build', '//foo:gen'])
    self.assertNotRegex('\n'.join(stderr), 'GEN_X')

  def testReplayOnDiskCache(self):
    self.ScratchFile('WORKSPACE')
    self.ScratchFile('foo/BUILD', [
        'genrule(',
        '  name = "gen",',
        '  outs = ["gen.x"],',
        '  cmd_bash = "echo GEN_X 1>&2 && touch $@",',
        '  cmd_bat = "echo GEN_X 1>&2 && echo. 2> $@",',
        ')',
    ])

    _, _, stderr = self.RunBazel(
        ['build', '--experimental_replay_action_out_err', '//foo:gen'])
    self.assertRegex('\n'.join(stderr), 'GEN_X')

    self.RunBazel(['shutdown'])

    _, _, stderr = self.RunBazel(
        ['build', '--experimental_replay_action_out_err', '//foo:gen'])
    self.assertRegex('\n'.join(stderr), 'GEN_X')

  def testReplayWithOutputFilter(self):
    self.ScratchFile('WORKSPACE')
    self.ScratchFile('foo/BUILD', [
        'genrule(',
        '  name = "gen",',
        '  outs = ["gen.x"],',
        '  cmd_bash = "echo GEN_X 1>&2 && touch $@",',
        '  cmd_bat = "echo GEN_X 1>&2 && echo. 2> $@",',
        ')',
    ])

    # Check that the state of the output filter on the first build is not
    # cached with the action result.
    _, _, stderr = self.RunBazel([
        'build', '--experimental_replay_action_out_err',
        '--auto_output_filter=all', '//foo:gen'
    ])
    self.assertNotRegex('\n'.join(stderr), 'GEN_X')

    # Action is cached, but the output filter no longer filters the message.
    _, _, stderr = self.RunBazel(
        ['build', '--experimental_replay_action_out_err', '//foo:gen'])
    self.assertRegex('\n'.join(stderr), 'GEN_X')


if __name__ == '__main__':
  unittest.main()
