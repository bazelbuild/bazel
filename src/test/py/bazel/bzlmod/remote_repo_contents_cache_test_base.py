# Copyright 2026 The Bazel Authors. All rights reserved.
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

import json
from src.test.py.bazel import test_base


class RemoteRepoContentsCacheTestBase(test_base.TestBase):
  """Common setup for tests of the remote repo contents cache."""

  def setUp(self):
    test_base.TestBase.setUp(self)
    # The remote repo contents cache has to cope with caches that serve action
    # results without verifying that the blobs they reference are still
    # present, which is what makes a repo's cached Tree outlive its file
    # contents in the first place.
    self._worker_port = self.StartRemoteWorker(
        ['--noaction_cache_integrity_check']
    )
    self.ScratchFile('.bazelrc', self.BazelrcLines())

  def tearDown(self):
    test_base.TestBase.tearDown(self)
    self.StopRemoteWorker()

  def BazelrcLines(self):
    """Returns the lines of the .bazelrc shared by all tests."""
    return [
        'startup --experimental_remote_repo_contents_cache',
        # Only use the remote repo contents cache.
        'common --repo_contents_cache=',
        'common --remote_cache=grpc://localhost:' + str(self._worker_port),
        'common --auth_enabled=false',
        'common --remote_timeout=3600s',
        'common --verbose_failures',
    ]

  def RepoDir(self, repo_name, cwd=None):
    _, stdout, _ = self.RunBazel(['info', 'output_base'], cwd=cwd)
    self.assertLen(stdout, 1)
    output_base = stdout[0].strip()

    _, stdout, _ = self.RunBazel(['mod', 'dump_repo_mapping', ''], cwd=cwd)
    self.assertLen(stdout, 1)
    mapping = json.loads(stdout[0])
    canonical_repo_name = mapping[repo_name]

    return output_base + '/external/' + canonical_repo_name
