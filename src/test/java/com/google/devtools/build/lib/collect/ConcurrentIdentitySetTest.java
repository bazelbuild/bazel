// Copyright 2024 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.collect;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.testutil.TestUtils;
import java.util.ArrayList;
import java.util.Collections;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.BeforeClass;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class ConcurrentIdentitySetTest {
  private static final int PARALLELISM = 12;
  private static final int UNIQUE_ELEMENTS = 10000;
  private static final ImmutableList<Object> TEST_DATA = createTestObjects();

  /** Ensures that {@link #TEST_DATA} is populated before any test runs. */
  @BeforeClass
  public static void warmup() {
    for (Object obj : TEST_DATA) {
      assertThat(obj).isNotNull();
    }
  }

  @Test
  public void testDedupe() {
    ConcurrentIdentitySet deduper = new ConcurrentIdentitySet(/* sizeHint= */ UNIQUE_ELEMENTS);
    assertThat(runDedup(deduper::add)).isEqualTo(UNIQUE_ELEMENTS);
  }

  @Test
  public void testResize() {
    ConcurrentIdentitySet deduper = new ConcurrentIdentitySet(/* sizeHint= */ 0);
    assertThat(runDedup(deduper::add)).isEqualTo(UNIQUE_ELEMENTS);
  }

  private static int runDedup(BooleanFunction deduper) {
    ForkJoinPool pool = new ForkJoinPool(PARALLELISM);
    AtomicInteger nextUnused = new AtomicInteger(0);
    AtomicInteger uniqueCount = new AtomicInteger(0);
    for (int i = 0; i < PARALLELISM; ++i) {
      pool.execute(
          () -> {
            int next = 0;
            while ((next = nextUnused.getAndIncrement()) < TEST_DATA.size()) {
              if (deduper.apply(TEST_DATA.get(next))) {
                uniqueCount.incrementAndGet();
              }
            }
          });
    }
    assertThat(pool.awaitQuiescence(TestUtils.WAIT_TIMEOUT_SECONDS, TimeUnit.SECONDS)).isTrue();
    return uniqueCount.get();
  }

  private static ImmutableList<Object> createTestObjects() {
    ArrayList<Object> data = new ArrayList<>((UNIQUE_ELEMENTS * (UNIQUE_ELEMENTS + 1)) / 2);
    for (int i = 1; i <= UNIQUE_ELEMENTS; ++i) {
      Object next = new Object();
      for (int j = 0; j < i; ++j) {
        data.add(next);
      }
    }
    Collections.shuffle(data);
    return ImmutableList.copyOf(data);
  }

  private interface BooleanFunction {
    boolean apply(Object obj);
  }
}
