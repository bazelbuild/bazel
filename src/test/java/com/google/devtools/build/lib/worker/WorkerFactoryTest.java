// Copyright 2017 The Bazel Authors. All rights reserved.
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

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.hash.HashCode;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link WorkerFactory}. */
@RunWith(JUnit4.class)
public class WorkerFactoryTest {

  final FileSystem fs = new InMemoryFileSystem();

  /**
   * Regression test for b/64689608: The execroot of the sandboxed worker process must end with the
   * workspace name, just like the normal execroot does.
   */
  @Test
  public void sandboxedWorkerPathEndsWithWorkspaceName() throws Exception {
    Path workerBaseDir = fs.getPath("/outputbase/bazel-workers");
    WorkerFactory workerFactory = new WorkerFactory(new WorkerOptions(), workerBaseDir);
    WorkerKey workerKey =
        new WorkerKey(
            /* args= */ ImmutableList.of(),
            /* env= */ ImmutableMap.of(),
            /* execRoot= */ fs.getPath("/outputbase/execroot/workspace"),
            /* mnemonic= */ "dummy",
            /* workerFilesCombinedHash= */ HashCode.fromInt(0),
            /* workerFilesWithHashes= */ ImmutableSortedMap.of(),
            /* mustBeSandboxed= */ true,
            /* proxied= */ false);
    Path sandboxedWorkerPath = workerFactory.getSandboxedWorkerPath(workerKey, 1);

    assertThat(sandboxedWorkerPath.getBaseName()).isEqualTo("workspace");
  }

  /** WorkerFactory should create correct worker type based on WorkerKey. */
  @Test
  public void workerCreationTypeCheck() throws Exception {
    Path workerBaseDir = fs.getPath("/outputbase/bazel-workers");
    WorkerFactory workerFactory = new WorkerFactory(new WorkerOptions(), workerBaseDir);
    WorkerKey sandboxedWorkerKey =
        new WorkerKey(
            /* args= */ ImmutableList.of(),
            /* env= */ ImmutableMap.of(),
            /* execRoot= */ fs.getPath("/outputbase/execroot/workspace"),
            /* mnemonic= */ "dummy",
            /* workerFilesCombinedHash= */ HashCode.fromInt(0),
            /* workerFilesWithHashes= */ ImmutableSortedMap.of(),
            /* mustBeSandboxed= */ true,
            /* proxied= */ false);
    Worker sandboxedWorker = workerFactory.create(sandboxedWorkerKey);
    assertThat(sandboxedWorker.getClass()).isEqualTo(SandboxedWorker.class);

    WorkerKey nonProxiedWorkerKey =
        new WorkerKey(
            /* args= */ ImmutableList.of(),
            /* env= */ ImmutableMap.of(),
            /* execRoot= */ fs.getPath("/outputbase/execroot/workspace"),
            /* mnemonic= */ "dummy",
            /* workerFilesCombinedHash= */ HashCode.fromInt(0),
            /* workerFilesWithHashes= */ ImmutableSortedMap.of(),
            /* mustBeSandboxed= */ false,
            /* proxied= */ false);
    Worker nonProxiedWorker = workerFactory.create(nonProxiedWorkerKey);
    assertThat(nonProxiedWorker.getClass()).isEqualTo(Worker.class);

    WorkerKey proxiedWorkerKey =
        new WorkerKey(
            /* args= */ ImmutableList.of(),
            /* env= */ ImmutableMap.of(),
            /* execRoot= */ fs.getPath("/outputbase/execroot/workspace"),
            /* mnemonic= */ "dummy",
            /* workerFilesCombinedHash= */ HashCode.fromInt(0),
            /* workerFilesWithHashes= */ ImmutableSortedMap.of(),
            /* mustBeSandboxed= */ false,
            /* proxied= */ true);
    Worker proxiedWorker = workerFactory.create(proxiedWorkerKey);
    // If proxied = true, WorkerProxy is created along with a WorkerMultiplexer.
    // Destroy WorkerMultiplexer to avoid unexpected behavior in WorkerMultiplexerManagerTest.
    WorkerMultiplexerManager.removeInstance(proxiedWorkerKey.hashCode());
    assertThat(proxiedWorker.getClass()).isEqualTo(WorkerProxy.class);
  }
}
