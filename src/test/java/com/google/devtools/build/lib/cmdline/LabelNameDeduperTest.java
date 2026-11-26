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

package com.google.devtools.build.lib.cmdline;

import static com.google.common.truth.Truth.assertThat;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link LabelNameDeduper}. */
@RunWith(JUnit4.class)
public final class LabelNameDeduperTest {

  @Test
  public void deduplicateTargetName_returnsCachedInstance() {
    LabelNameDeduper.clear();
    String first = new StringBuilder("common_target").toString();
    String second = new StringBuilder("common_target").toString();
    assertThat(first).isNotSameInstanceAs(second);

    String firstResult = LabelNameDeduper.deduplicateTargetName(first);
    String secondResult = LabelNameDeduper.deduplicateTargetName(second);

    assertThat(firstResult).isSameInstanceAs(first);
    assertThat(secondResult).isSameInstanceAs(first);
  }

  @Test
  public void clear_evictsCachedEntries() {
    LabelNameDeduper.clear();
    String first = new StringBuilder("evicted_target").toString();
    assertThat(LabelNameDeduper.deduplicateTargetName(first)).isSameInstanceAs(first);

    LabelNameDeduper.clear();
    String second = new StringBuilder("evicted_target").toString();
    String result = LabelNameDeduper.deduplicateTargetName(second);

    assertThat(result).isSameInstanceAs(second);
  }

  @Test
  public void deduplicateTargetName_concurrentAccessAndClear() throws Exception {
    int workerCount = 4;
    int iterations = 1000;
    CountDownLatch startLatch = new CountDownLatch(1);
    ExecutorService executor = Executors.newFixedThreadPool(workerCount + 1);
    try {
      List<Future<?>> futures = new ArrayList<>();
      for (int t = 0; t < workerCount; t++) {
        futures.add(
            executor.submit(() -> {
              startLatch.await();
              for (int i = 0; i < iterations; i++) {
                String name = new StringBuilder("target_").append(i % 16).toString();
                String deduplicated = LabelNameDeduper.deduplicateTargetName(name);
                assertThat(deduplicated).isEqualTo(name);
              }
              return null;
            }));
      }
      futures.add(
          executor.submit(() -> {
            startLatch.await();
            LabelNameDeduper.clear();
            return null;
          }));
      startLatch.countDown();
      for (Future<?> future : futures) {
        future.get();
      }
    } finally {
      executor.shutdownNow();
    }
  }
}
