// Copyright 2026 The Bazel Authors. All rights reserved.
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
import com.google.devtools.build.lib.actions.ExecutionRequirements.WorkerProtocolFormat;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link SandboxedWorker}. */
@RunWith(JUnit4.class)
public class SandboxedWorkerTest {

  @Test
  public void testWritableDirs_withoutDevShm() throws Exception {
    FileSystem fs = new InMemoryFileSystem(DigestHashFunction.SHA256);
    Path workDir = fs.getPath("/base/workDir");
    workDir.createDirectoryAndParents();

    // /dev/shm DOES NOT exist on this InMemoryFileSystem.

    WorkerKey workerKey = createWorkerKey(fs);
    SandboxedWorker.WorkerSandboxOptions sandboxOptions = createSandboxOptions(fs);

    SandboxedWorker worker =
        new SandboxedWorker(
            workerKey,
            1,
            workDir,
            fs.getPath("/logFile"),
            WorkerOptions.DEFAULTS,
            sandboxOptions,
            /* treeDeleter= */ null,
            false,
            /* cgroupFactory= */ null);

    ImmutableSet<Path> writableDirs = worker.getWritableDirs(workDir);

    assertThat(writableDirs).contains(fs.getPath("/tmp"));
    assertThat(writableDirs).doesNotContain(fs.getPath("/dev/shm"));
  }

  @Test
  public void testWritableDirs_withDevShm() throws Exception {
    FileSystem fs = new InMemoryFileSystem(DigestHashFunction.SHA256);
    Path workDir = fs.getPath("/base/workDir");
    workDir.createDirectoryAndParents();

    // Create /dev/shm
    fs.getPath("/dev/shm").createDirectoryAndParents();

    WorkerKey workerKey = createWorkerKey(fs);
    SandboxedWorker.WorkerSandboxOptions sandboxOptions = createSandboxOptions(fs);

    SandboxedWorker worker =
        new SandboxedWorker(
            workerKey,
            1,
            workDir,
            fs.getPath("/logFile"),
            WorkerOptions.DEFAULTS,
            sandboxOptions,
            /* treeDeleter= */ null,
            false,
            /* cgroupFactory= */ null);

    ImmutableSet<Path> writableDirs = worker.getWritableDirs(workDir);

    assertThat(writableDirs).contains(fs.getPath("/tmp"));
    assertThat(writableDirs).contains(fs.getPath("/dev/shm"));
  }

  private WorkerKey createWorkerKey(FileSystem fs) {
    return new WorkerKey(
        ImmutableList.of(),
        ImmutableMap.of(),
        fs.getPath("/execRoot"),
        "dummy",
        HashCode.fromInt(0),
        ImmutableSortedMap.of(),
        true,
        false,
        false,
        false,
        WorkerProtocolFormat.PROTO);
  }

  private SandboxedWorker.WorkerSandboxOptions createSandboxOptions(FileSystem fs) {
    return new SandboxedWorker.WorkerSandboxOptions(
        fs.getPath("/sandboxBinary"),
        false,
        false,
        false,
        ImmutableSet.of(),
        ImmutableSet.of(),
        0,
        ImmutableSet.of(),
        ImmutableMap.of());
  }
}
