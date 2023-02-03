// Copyright 2023 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.concurrent;

import static com.google.common.truth.Truth.assertThat;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class AvailableStateTest {
  private static final int MAX_VALUE = 0x7FFF;

  @Test
  public void fromMaxValues_cancelDoesNotOverflow() {
    AvailableState state = new AvailableState(MAX_VALUE, MAX_VALUE);
    AvailableState out = new AvailableState();

    assertThat(state.isCancelled()).isFalse();

    state.cancel(MAX_VALUE, MAX_VALUE, out);
    assertThat(out.isCancelled()).isTrue();
    assertThat(out.threads()).isEqualTo(-1);
    assertThat(out.cpuPermits()).isEqualTo(-1);
    assertThat(out.tasksForTesting()).isEqualTo(Integer.MIN_VALUE);
    assertThat(out.cpuHeavyTasksForTesting()).isEqualTo(Integer.MIN_VALUE);

    assertThat(out.isQuiescent(MAX_VALUE)).isTrue();
  }

  @Test
  public void fromFullyDrained_cancelDoesNotOverflow() {
    // This is the minimum possible naturally reachable value for threads and cpu permits.
    AvailableState state = new AvailableState(0, 0);
    AvailableState out = new AvailableState();

    // Cancelling subtracts a (MAX_VALUE + 1) from the values. Verifies that this does not overflow.
    state.cancel(MAX_VALUE, MAX_VALUE, out);

    assertThat(out.isCancelled()).isTrue();
    assertThat(out.threads()).isEqualTo(-0x8000);
    assertThat(out.cpuPermits()).isEqualTo(-0x8000);
    assertThat(out.tasksForTesting()).isEqualTo(Integer.MIN_VALUE);
    assertThat(out.cpuHeavyTasksForTesting()).isEqualTo(Integer.MIN_VALUE);

    assertThat(out.isQuiescent(MAX_VALUE)).isFalse();

    for (int i = 0; i < MAX_VALUE; ++i) {
      AvailableState temp = new AvailableState();
      out.releaseThreadAndCpuPermit(temp);
      out = temp;
    }

    assertThat(out.isQuiescent(MAX_VALUE)).isTrue();
  }
}
