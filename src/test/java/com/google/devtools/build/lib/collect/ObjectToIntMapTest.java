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
package com.google.devtools.build.lib.collect;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicIntegerArray;
import java.util.function.IntFunction;
import org.junit.After;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link ObjectToIntMap}. */
@RunWith(JUnit4.class)
public final class ObjectToIntMapTest {
  private static final int THREADS = 16;

  private final ExecutorService executor = Executors.newFixedThreadPool(THREADS);

  @After
  public void shutDown() throws Exception {
    executor.shutdownNow();
    assertThat(executor.awaitTermination(1, TimeUnit.MINUTES)).isTrue();
  }

  /** A key whose hash code is that of {@code hash}, so that keys can be made to collide. */
  private record Key(int id, int hash) {
    @Override
    public int hashCode() {
      return hash;
    }
  }

  private static Key distinctKey(int id) {
    return new Key(id, Integer.hashCode(id) * 31 + 7);
  }

  private static Key collidingKey(int id) {
    return new Key(id, 17);
  }

  /** Keys with only 64 distinct hash codes, so that many of them share shards and slots. */
  private static Key clusteredKey(int id) {
    return new Key(id, id % 64);
  }

  @Test
  public void putGetAndGrow() {
    ObjectToIntMap<String> map = new ObjectToIntMap<>();
    int n = 100_000;
    for (int i = 0; i < n; i++) {
      map.put("key" + i, i);
    }
    map.put("key7", 70);

    assertThat(map.size()).isEqualTo(n);
    for (int i = 0; i < n; i++) {
      assertThat(map.get("key" + i)).isEqualTo(i == 7 ? 70 : i);
    }
    assertThat(map.get("absent")).isEqualTo(-1);
    // Load factor 0.75, growth 1.25: between 60% and 75% full, plus rounding.
    assertThat(map.slots()).isAtLeast((long) (n / ObjectToIntMap.LOAD_FACTOR));
    assertThat(map.slots())
        .isAtMost(
            (long) (n / (ObjectToIntMap.LOAD_FACTOR / ObjectToIntMap.GROWTH))
                + 16L * ObjectToIntMap.SHARDS);
  }

  @Test
  public void collidingHashCodes() {
    ObjectToIntMap<Key> map = new ObjectToIntMap<>();
    for (int i = 0; i < 1000; i++) {
      map.put(collidingKey(i), i);
    }

    for (int i = 0; i < 1000; i++) {
      assertThat(map.get(collidingKey(i))).isEqualTo(i);
    }
    assertThat(map.get(collidingKey(1000))).isEqualTo(-1);
    assertThat(map.size()).isEqualTo(1000);
  }

  @Test
  public void emptyMap() {
    ObjectToIntMap<String> map = new ObjectToIntMap<>();

    assertThat(map.get("absent")).isEqualTo(-1);
    assertThat(map.size()).isEqualTo(0);
    assertThat(map.slots()).isGreaterThan(0);
  }

  @Test
  public void getOrPut_callsFactoryOnlyForNewKeysAndReturnsComplement() {
    ObjectToIntMap<String> map = new ObjectToIntMap<>();
    AtomicInteger calls = new AtomicInteger();

    assertThat(map.getOrPut("a", () -> 5 + calls.getAndIncrement())).isEqualTo(~5);
    assertThat(map.getOrPut("a", () -> 100 + calls.getAndIncrement())).isEqualTo(5);
    assertThat(
            map.getOrPut(
                "b",
                () -> {
                  calls.incrementAndGet();
                  return 0;
                }))
        .isEqualTo(~0);
    assertThat(map.getOrPut("b", () -> 100)).isEqualTo(0);
    assertThat(calls.get()).isEqualTo(2);
    assertThat(map.get("a")).isEqualTo(5);
    assertThat(map.get("b")).isEqualTo(0);
    assertThat(map.size()).isEqualTo(2);
  }

  @Test
  public void getOrPut_existingKeyFromPut() {
    ObjectToIntMap<String> map = new ObjectToIntMap<>();
    map.put("a", 3);

    assertThat(map.getOrPut("a", () -> 4)).isEqualTo(3);
  }

  @Test
  public void negativeValuesAreRejected() {
    ObjectToIntMap<String> map = new ObjectToIntMap<>();

    assertThrows(IllegalArgumentException.class, () -> map.put("a", -1));
    assertThrows(IllegalArgumentException.class, () -> map.getOrPut("a", () -> -2));
    assertThat(map.get("a")).isEqualTo(-1);
    assertThat(map.size()).isEqualTo(0);
  }

  @Test
  public void getOrPut_singleThreadedWithGrowth() {
    ObjectToIntMap<Key> map = new ObjectToIntMap<>();
    AtomicInteger next = new AtomicInteger();
    int n = 50_000;
    for (int round = 0; round < 2; round++) {
      for (int i = 0; i < n; i++) {
        int result = map.getOrPut(distinctKey(i), next::getAndIncrement);
        assertThat(result).isEqualTo(round == 0 ? ~i : i);
      }
    }
    assertThat(next.get()).isEqualTo(n);
    assertThat(map.size()).isEqualTo(n);
  }

  @Test
  public void concurrentGetOrPut_distinctHashes() throws Exception {
    assertConcurrentGetOrPutAssignsDenseUniqueValues(200_000, ObjectToIntMapTest::distinctKey);
  }

  @Test
  public void concurrentGetOrPut_clusteredHashes() throws Exception {
    // Few distinct hash codes: long probe chains and many growths in few shards.
    assertConcurrentGetOrPutAssignsDenseUniqueValues(10_000, ObjectToIntMapTest::clusteredKey);
  }

  @Test
  public void concurrentGetOrPut_collidingHashes() throws Exception {
    // A single hash code: every thread contends on the same shard and probe chain.
    assertConcurrentGetOrPutAssignsDenseUniqueValues(2_000, ObjectToIntMapTest::collidingKey);
  }

  /**
   * Has {@link #THREADS} threads each call {@code getOrPut} for all of {@code n} keys, in different
   * orders, with values from a shared counter, and checks that every key gets exactly one value,
   * that every thread sees the same value for it, that exactly one thread sees it as new, and that
   * the values are 0..n-1.
   */
  private void assertConcurrentGetOrPutAssignsDenseUniqueValues(int n, IntFunction<Key> keys)
      throws Exception {
    for (int round = 0; round < 5; round++) {
      ObjectToIntMap<Key> map = new ObjectToIntMap<>();
      AtomicInteger next = new AtomicInteger();
      AtomicIntegerArray seenValue = new AtomicIntegerArray(n);
      AtomicIntegerArray newCount = new AtomicIntegerArray(n);
      for (int i = 0; i < n; i++) {
        seenValue.set(i, -1);
      }
      CyclicBarrier start = new CyclicBarrier(THREADS);
      List<Future<?>> futures = new ArrayList<>();
      for (int t = 0; t < THREADS; t++) {
        long seed = round * 1000L + t;
        futures.add(
            executor.submit(() -> {
              List<Integer> order = new ArrayList<>(n);
              for (int i = 0; i < n; i++) {
                order.add(i);
              }
              Collections.shuffle(order, new Random(seed));
              start.await();
              for (int i : order) {
                int result = map.getOrPut(keys.apply(i), next::getAndIncrement);
                int value = result < 0 ? ~result : result;
                if (result < 0) {
                  newCount.incrementAndGet(i);
                }
                if (!seenValue.compareAndSet(i, -1, value)) {
                  assertThat(seenValue.get(i)).isEqualTo(value);
                }
              }
              return null;
            }));
      }
      for (Future<?> future : futures) {
        future.get();
      }

      assertThat(next.get()).isEqualTo(n);
      assertThat(map.size()).isEqualTo(n);
      boolean[] valueUsed = new boolean[n];
      for (int i = 0; i < n; i++) {
        assertThat(newCount.get(i)).isEqualTo(1);
        int value = seenValue.get(i);
        assertThat(value).isAtLeast(0);
        assertThat(value).isLessThan(n);
        assertThat(valueUsed[value]).isFalse();
        valueUsed[value] = true;
        assertThat(map.get(keys.apply(i))).isEqualTo(value);
      }
    }
  }

  @Test
  public void concurrentGetDuringGrowth_neverMissesPublishedKeysOrSeesWrongValues()
      throws Exception {
    // Writers insert keys and then publish how far they got; readers concurrently look up keys,
    // which must be found with the right value once published and must otherwise be absent or
    // right, while shards keep growing.
    int writers = 4;
    int readers = THREADS - writers;
    int perWriter = 100_000;
    ObjectToIntMap<Key> map = new ObjectToIntMap<>();
    AtomicIntegerArray published = new AtomicIntegerArray(writers);
    AtomicBoolean done = new AtomicBoolean();
    AtomicInteger checks = new AtomicInteger();
    CyclicBarrier start = new CyclicBarrier(THREADS);
    List<Future<?>> futures = new ArrayList<>();
    for (int w = 0; w < writers; w++) {
      int writer = w;
      futures.add(
          executor.submit(() -> {
            start.await();
            for (int i = 0; i < perWriter; i++) {
              int id = writer * perWriter + i;
              if (i % 2 == 0) {
                map.put(distinctKey(id), id);
              } else {
                assertThat(map.getOrPut(distinctKey(id), () -> id)).isEqualTo(~id);
              }
              published.set(writer, i + 1);
            }
            return null;
          }));
    }
    for (int r = 0; r < readers; r++) {
      long seed = r;
      futures.add(
          executor.submit(() -> {
            Random random = new Random(seed);
            start.await();
            while (!done.get()) {
              int writer = random.nextInt(writers);
              int count = published.get(writer);
              if (count > 0) {
                int id = writer * perWriter + random.nextInt(count);
                assertThat(map.get(distinctKey(id))).isEqualTo(id);
              }
              // A key that may or may not have been inserted yet.
              int any = random.nextInt(writers * perWriter);
              int value = map.get(distinctKey(any));
              if (value != -1) {
                assertThat(value).isEqualTo(any);
              }
              checks.incrementAndGet();
            }
            return null;
          }));
    }
    for (int i = 0; i < writers; i++) {
      futures.get(i).get();
    }
    done.set(true);
    for (Future<?> future : futures) {
      future.get();
    }

    assertThat(checks.get()).isGreaterThan(0);
    assertThat(map.size()).isEqualTo(writers * perWriter);
    for (int id = 0; id < writers * perWriter; id++) {
      assertThat(map.get(distinctKey(id))).isEqualTo(id);
    }
  }

  @Test
  public void concurrentPutOfDistinctKeysIntoOneShard() throws Exception {
    // Every thread puts its own keys, all with the same hash code, so they all contend for the
    // same shard and its growth.
    int perThread = 300;
    ObjectToIntMap<Key> map = new ObjectToIntMap<>();
    CyclicBarrier start = new CyclicBarrier(THREADS);
    List<Future<?>> futures = new ArrayList<>();
    for (int t = 0; t < THREADS; t++) {
      int thread = t;
      futures.add(
          executor.submit(() -> {
            start.await();
            for (int i = 0; i < perThread; i++) {
              int id = thread * perThread + i;
              map.put(collidingKey(id), id);
            }
            return null;
          }));
    }
    for (Future<?> future : futures) {
      future.get();
    }

    assertThat(map.size()).isEqualTo(THREADS * perThread);
    for (int id = 0; id < THREADS * perThread; id++) {
      assertThat(map.get(collidingKey(id))).isEqualTo(id);
    }
  }
}
