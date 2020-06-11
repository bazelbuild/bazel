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

import com.google.devtools.build.lib.actions.UserExecException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link WorkerMultiplexerManager}. */
@RunWith(JUnit4.class)
public class WorkerMultiplexerManagerTest {

  @Test
  public void instanceCreationRemovalTest() throws Exception {
    // Create a WorkerProxy hash and request for a WorkerMultiplexer.
    Integer worker1Hash = "worker1".hashCode();
    WorkerMultiplexer wm1 = WorkerMultiplexerManager.getInstance(worker1Hash);

    assertThat(WorkerMultiplexerManager.getMultiplexer(worker1Hash)).isEqualTo(wm1);
    assertThat(WorkerMultiplexerManager.getRefCount(worker1Hash)).isEqualTo(1);
    assertThat(WorkerMultiplexerManager.getInstanceCount()).isEqualTo(1);

    // Create another WorkerProxy hash and request for a WorkerMultiplexer.
    Integer worker2Hash = "worker2".hashCode();
    WorkerMultiplexer wm2 = WorkerMultiplexerManager.getInstance(worker2Hash);

    assertThat(WorkerMultiplexerManager.getMultiplexer(worker2Hash)).isEqualTo(wm2);
    assertThat(WorkerMultiplexerManager.getRefCount(worker2Hash)).isEqualTo(1);
    assertThat(WorkerMultiplexerManager.getInstanceCount()).isEqualTo(2);

    // Use the same WorkerProxy hash, it shouldn't instantiate a new WorkerMultiplexer.
    WorkerMultiplexer wm2Annex = WorkerMultiplexerManager.getInstance(worker2Hash);

    assertThat(wm2).isEqualTo(wm2Annex);
    assertThat(WorkerMultiplexerManager.getRefCount(worker2Hash)).isEqualTo(2);
    assertThat(WorkerMultiplexerManager.getInstanceCount()).isEqualTo(2);

    // Remove an instance. If reference count is larger than 0, instance shouldn't be destroyed.
    WorkerMultiplexerManager.removeInstance(worker2Hash);

    assertThat(WorkerMultiplexerManager.getRefCount(worker2Hash)).isEqualTo(1);
    assertThat(WorkerMultiplexerManager.getInstanceCount()).isEqualTo(2);

    // Remove an instance. Reference count is down to 0, instance should be destroyed.
    WorkerMultiplexerManager.removeInstance(worker2Hash);

    assertThrows(
        UserExecException.class, () -> WorkerMultiplexerManager.getMultiplexer(worker2Hash));
    assertThat(WorkerMultiplexerManager.getInstanceCount()).isEqualTo(1);

    // WorkerProxy hash not found.
    assertThrows(
        UserExecException.class, () -> WorkerMultiplexerManager.removeInstance(worker2Hash));

    // Remove all the instances.
    WorkerMultiplexerManager.removeInstance(worker1Hash);

    assertThat(WorkerMultiplexerManager.getInstanceCount()).isEqualTo(0);
  }
}
