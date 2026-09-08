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
package com.google.devtools.build.lib.skyframe.serialization.analysis;

import static com.google.common.truth.Truth.assertThat;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class SkycacheChannelStateAdvisorTest {

  @Test
  public void disabledAdvisor_isNeverSaturated() {
    SkycacheChannelStateAdvisor advisor = SkycacheChannelStateAdvisor.DISABLED;
    assertThat(advisor.isSaturated()).isFalse();

    for (int i = 0; i < 100_000; i++) {
      advisor.incrementInFlightRequests();
    }
    assertThat(advisor.getInFlightRequests()).isEqualTo(100_000);
    assertThat(advisor.isSaturated()).isFalse();

    // Restore counter to zero
    advisor.decrementInFlightRequests(100_000);
    assertThat(advisor.getInFlightRequests()).isEqualTo(0);
  }

  @Test
  public void zeroMaxCapacity_isNeverSaturated() {
    SkycacheChannelStateAdvisor advisor = new SkycacheChannelStateAdvisor(0);
    assertThat(advisor.isSaturated()).isFalse();

    for (int i = 0; i < 500; i++) {
      advisor.incrementInFlightRequests();
    }
    assertThat(advisor.isSaturated()).isFalse();
  }

  @Test
  public void capacityThreshold_crossings() {
    SkycacheChannelStateAdvisor advisor = new SkycacheChannelStateAdvisor(100);

    assertThat(advisor.isSaturated()).isFalse();
    assertThat(advisor.getInFlightRequests()).isEqualTo(0);

    // Below limit
    for (int i = 0; i < 99; i++) {
      advisor.incrementInFlightRequests();
    }
    assertThat(advisor.isSaturated()).isFalse();
    assertThat(advisor.getInFlightRequests()).isEqualTo(99);

    // Increment to 100 (at limit)
    advisor.incrementInFlightRequests();
    assertThat(advisor.isSaturated()).isTrue();
    assertThat(advisor.getInFlightRequests()).isEqualTo(100);

    // Above limit
    for (int i = 0; i < 50; i++) {
      advisor.incrementInFlightRequests();
    }
    assertThat(advisor.isSaturated()).isTrue();
    assertThat(advisor.getInFlightRequests()).isEqualTo(150);

    // Decrement by delta back below limit
    advisor.decrementInFlightRequests(51);
    assertThat(advisor.isSaturated()).isFalse();
    assertThat(advisor.getInFlightRequests()).isEqualTo(99);

    // Decrement to zero
    advisor.decrementInFlightRequests(99);
    assertThat(advisor.isSaturated()).isFalse();
    assertThat(advisor.getInFlightRequests()).isEqualTo(0);
  }

  @Test
  public void concurrentIncrementsAndDecrements_isThreadSafe() throws Exception {
    SkycacheChannelStateAdvisor advisor = new SkycacheChannelStateAdvisor(10_000);
    int threadCount = 10;
    int iterationsPerThread = 1_000;

    ExecutorService executor = Executors.newFixedThreadPool(threadCount);
    try {
      List<Callable<Void>> tasks = new ArrayList<>();
      for (int i = 0; i < threadCount; i++) {
        tasks.add(
            () -> {
              for (int j = 0; j < iterationsPerThread; j++) {
                advisor.incrementInFlightRequests();
              }
              return null;
            });
      }

      List<Future<Void>> futures = executor.invokeAll(tasks);
      for (Future<Void> future : futures) {
        future.get();
      }

      assertThat(advisor.getInFlightRequests()).isEqualTo((long) threadCount * iterationsPerThread);
      assertThat(advisor.isSaturated()).isTrue();

      // Decrement back concurrently using batch decrements
      tasks.clear();
      for (int i = 0; i < threadCount; i++) {
        tasks.add(
            () -> {
              advisor.decrementInFlightRequests(iterationsPerThread);
              return null;
            });
      }

      futures = executor.invokeAll(tasks);
      for (Future<Void> future : futures) {
        future.get();
      }

      assertThat(advisor.getInFlightRequests()).isEqualTo(0);
      assertThat(advisor.isSaturated()).isFalse();
    } finally {
      executor.shutdown();
    }
  }
}
