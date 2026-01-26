# pylint: disable=g-bad-file-header
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

import os
from src.test.py.bazel import test_base


class BazelProtoTest(test_base.TestBase):

  def testProtoLibraryInBazelTools(self):
    self.AddBazelDep('protobuf')
    self.ScratchFile('BUILD', [
        'load("@protobuf//bazel:proto_library.bzl", "proto_library")',
        'proto_library(',
        '    name = "my_proto",',
        '    srcs = ["my.proto"],',
        '    deps = ["@bazel_tools//src/main/protobuf:action_cache_proto"],',
        ')',
    ])
    self.ScratchFile('my.proto', [
        'syntax = "proto3";',
        'import "src/main/protobuf/action_cache.proto";',
        'package my;',
        'message MyMessage {',
        '  blaze.ActionCacheStatistics stats = 1;',
        '}',
    ])

    # This should succeed because we fixed the visibility/load issues in @bazel_tools//src/main/protobuf:BUILD
    self.RunBazel(['build', '//:my_proto'])

  def testProtocAliasInBazelTools(self):
    self.AddBazelDep('protobuf')
    self.ScratchFile('BUILD', [
        'alias(',
        '    name = "my_protoc",',
        '    actual = "@bazel_tools//tools/proto:protoc",',
        ')',
    ])

    # This should succeed because @bazel_tools//tools/proto:protoc aliases to @com_google_protobuf//:protoc
    self.RunBazel(['build', '//:my_protoc'])

  def testUnsupportedLanguageSpecificProtoTargets(self):
    self.AddBazelDep('protobuf')
    self.ScratchFile('BUILD', [
        'alias(',
        '    name = "bad_alias",',
        '    actual = "@bazel_tools//src/main/protobuf:action_cache_java_proto",',
        ')',
    ])

    exit_code, _, stderr = self.RunBazel(['build', '//:bad_alias'], allow_failure=True)
    self.AssertExitCode(exit_code, 1, stderr)
    self.assertIn('Language specific proto targets are not available in @bazel_tools', ''.join(stderr))
    self.assertIn('Please use the proto_library and generate your own code', ''.join(stderr))


if __name__ == "__main__":
  test_base.absltest.main()
