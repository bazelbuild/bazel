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
import com.google.common.collect.ImmutableSet;
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
            ImmutableList.of(),
            ImmutableMap.of(),
            fs.getPath("/outputbase/execroot/workspace"),
            "dummy",
            HashCode.fromInt(0),
            ImmutableSortedMap.of(),
            ImmutableMap.of(),
            ImmutableSet.of(),
            true);
    Path sandboxedWorkerPath = workerFactory.getSandboxedWorkerPath(workerKey, 1);

    assertThat(sandboxedWorkerPath.getBaseName()).isEqualTo("workspace");
  }
}
