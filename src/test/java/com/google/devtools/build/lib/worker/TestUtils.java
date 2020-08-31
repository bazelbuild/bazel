// Copyright 2020 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.worker;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.hash.HashCode;
import com.google.devtools.build.lib.actions.ExecutionRequirements.WorkerProtocolFormat;
import com.google.devtools.build.lib.vfs.FileSystem;

/** Utilities that come in handy when unit-testing the worker code. */
public class TestUtils {

  private TestUtils() {}

  static WorkerKey createWorkerKey(
      FileSystem fileSystem, String mnemonic, boolean proxied, String... args) {
    return new WorkerKey(
        /* args= */ ImmutableList.copyOf(args),
        /* env= */ ImmutableMap.of("env1", "foo", "env2", "bar"),
        /* execRoot= */ fileSystem.getPath("/fake"),
        /* mnemonic= */ mnemonic,
        /* workerFilesCombinedHash= */ HashCode.fromInt(0),
        /* workerFilesWithHashes= */ ImmutableSortedMap.of(),
        /* mustBeSandboxed= */ false,
        /* proxied= */ proxied,
        WorkerProtocolFormat.PROTO);
  }
}
