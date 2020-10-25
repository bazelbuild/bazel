// Copyright 2019 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.worker;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.hash.HashCode;
import com.google.devtools.build.lib.actions.ExecutionRequirements.WorkerProtocolFormat;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link WorkerKey}. */
@RunWith(JUnit4.class)
public class WorkerKeyTest {
  final FileSystem fs = new InMemoryFileSystem(DigestHashFunction.SHA256);

  Path workerBaseDir = fs.getPath("/outputbase/bazel-workers");
  WorkerKey workerKey =
      new WorkerKey(
          /* args= */ ImmutableList.of("arg1", "arg2", "arg3"),
          /* env= */ ImmutableMap.of("env1", "foo", "env2", "bar"),
          /* execRoot= */ fs.getPath("/outputbase/execroot/workspace"),
          /* mnemonic= */ "dummy",
          /* workerFilesCombinedHash= */ HashCode.fromInt(0),
          /* workerFilesWithHashes= */ ImmutableSortedMap.of(),
          /* mustBeSandboxed= */ true,
          /* proxied= */ true,
          WorkerProtocolFormat.PROTO);

  @Test
  public void testWorkerKeyGetter() {
    assertThat(workerKey.mustBeSandboxed()).isEqualTo(true);
    assertThat(workerKey.getProxied()).isEqualTo(true);
    assertThat(WorkerKey.makeWorkerTypeName(false)).isEqualTo("worker");
    assertThat(WorkerKey.makeWorkerTypeName(true)).isEqualTo("multiplex-worker");
    // Hash code contains args, env, execRoot, proxied, and mnemonic.
    assertThat(workerKey.hashCode()).isEqualTo(1605714200);
    assertThat(workerKey.getProtocolFormat()).isEqualTo(WorkerProtocolFormat.PROTO);
  }

  @Test
  public void testWorkerKeyEquality() {
    WorkerKey workerKeyWithSameFields =
        new WorkerKey(
            workerKey.getArgs(),
            workerKey.getEnv(),
            workerKey.getExecRoot(),
            workerKey.getMnemonic(),
            workerKey.getWorkerFilesCombinedHash(),
            workerKey.getWorkerFilesWithHashes(),
            workerKey.mustBeSandboxed(),
            workerKey.getProxied(),
            workerKey.getProtocolFormat());
    assertThat(workerKey).isEqualTo(workerKeyWithSameFields);
  }

  @Test
  public void testWorkerKeyInequality_protocol() {
    WorkerKey workerKeyWithDifferentProtocol =
        new WorkerKey(
            workerKey.getArgs(),
            workerKey.getEnv(),
            workerKey.getExecRoot(),
            workerKey.getMnemonic(),
            workerKey.getWorkerFilesCombinedHash(),
            workerKey.getWorkerFilesWithHashes(),
            workerKey.mustBeSandboxed(),
            workerKey.getProxied(),
            WorkerProtocolFormat.JSON);
    assertThat(workerKey).isNotEqualTo(workerKeyWithDifferentProtocol);
  }
}
