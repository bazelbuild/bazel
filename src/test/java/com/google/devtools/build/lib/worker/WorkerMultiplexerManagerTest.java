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
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.hash.HashCode;
import com.google.devtools.build.lib.actions.ExecutionRequirements.WorkerProtocolFormat;
import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link WorkerMultiplexerManager}. */
@RunWith(JUnit4.class)
public class WorkerMultiplexerManagerTest {

  private FileSystem fileSystem;

  @Before
  public void setUp() {
    fileSystem = new InMemoryFileSystem(BlazeClock.instance(), DigestHashFunction.SHA256);
    WorkerMultiplexerManager.reset();
  }

  @Test
  public void instanceCreationRemovalTest() throws Exception {
    Path logFile = fileSystem.getPath("/tmp/logFilePath");
    // Create a WorkerProxy hash and request for a WorkerMultiplexer.
    WorkerKey workerKey1 =
        new WorkerKey(
            ImmutableList.of(),
            ImmutableMap.of(),
            fileSystem.getPath("/execRoot"),
            "mnemonic1",
            HashCode.fromInt(1),
            ImmutableSortedMap.of(),
            false,
            false,
            WorkerProtocolFormat.PROTO);
    WorkerMultiplexer wm1 =
        WorkerMultiplexerManager.getInstance(workerKey1, logFile, /* reporter */ null);

    assertThat(WorkerMultiplexerManager.getMultiplexer(workerKey1)).isEqualTo(wm1);
    assertThat(WorkerMultiplexerManager.getRefCount(workerKey1)).isEqualTo(1);
    assertThat(WorkerMultiplexerManager.getInstanceCount()).isEqualTo(1);

    // Create another WorkerProxy hash and request for a WorkerMultiplexer.
    WorkerKey workerKey2 =
        new WorkerKey(
            ImmutableList.of(),
            ImmutableMap.of(),
            fileSystem.getPath("/execRoot"),
            "mnemonic2",
            HashCode.fromInt(1),
            ImmutableSortedMap.of(),
            false,
            false,
            WorkerProtocolFormat.PROTO);
    WorkerMultiplexer wm2 =
        WorkerMultiplexerManager.getInstance(workerKey2, logFile, /* reporter */ null);

    assertThat(WorkerMultiplexerManager.getMultiplexer(workerKey2)).isEqualTo(wm2);
    assertThat(WorkerMultiplexerManager.getRefCount(workerKey2)).isEqualTo(1);
    assertThat(WorkerMultiplexerManager.getInstanceCount()).isEqualTo(2);

    // Use the same WorkerProxy hash, it shouldn't instantiate a new WorkerMultiplexer.
    WorkerMultiplexer wm2Annex =
        WorkerMultiplexerManager.getInstance(workerKey2, logFile, /* reporter */ null);

    assertThat(wm2).isEqualTo(wm2Annex);
    assertThat(WorkerMultiplexerManager.getRefCount(workerKey2)).isEqualTo(2);
    assertThat(WorkerMultiplexerManager.getInstanceCount()).isEqualTo(2);

    // Remove an instance. If reference count is larger than 0, instance shouldn't be destroyed.
    WorkerMultiplexerManager.removeInstance(workerKey2);

    assertThat(WorkerMultiplexerManager.getRefCount(workerKey2)).isEqualTo(1);
    assertThat(WorkerMultiplexerManager.getInstanceCount()).isEqualTo(2);

    // Remove an instance. Reference count is down to 0, instance should be destroyed.
    WorkerMultiplexerManager.removeInstance(workerKey2);

    assertThrows(
        UserExecException.class, () -> WorkerMultiplexerManager.getMultiplexer(workerKey2));
    assertThat(WorkerMultiplexerManager.getInstanceCount()).isEqualTo(1);

    // WorkerProxy hash not found.
    assertThrows(
        UserExecException.class, () -> WorkerMultiplexerManager.removeInstance(workerKey2));

    // Remove all the instances.
    WorkerMultiplexerManager.removeInstance(workerKey1);

    assertThat(WorkerMultiplexerManager.getInstanceCount()).isEqualTo(0);
  }
}
