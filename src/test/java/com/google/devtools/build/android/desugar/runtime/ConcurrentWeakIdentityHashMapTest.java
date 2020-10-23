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
package com.google.devtools.build.android.desugar.runtime;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.testing.GcFinalization;
import com.google.devtools.build.android.desugar.runtime.ThrowableExtension.ConcurrentWeakIdentityHashMap;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Test for {@link ConcurrentWeakIdentityHashMap}. This test uses multi-threading, and needs GC
 * sometime to assert weak references, so it could take long.
 */
@RunWith(JUnit4.class)
public class ConcurrentWeakIdentityHashMapTest {

  private final Random random = new Random();

  /**
   * This method makes sure that after return, all the exceptions in the map should be garbage
   * collected. .
   */
  private static ConcurrentWeakIdentityHashMap
      testConcurrentWeakIdentityHashMapSingleThreadedHelper(CountDownLatch latch)
          throws InterruptedException {
    ConcurrentWeakIdentityHashMap map = new ConcurrentWeakIdentityHashMap();
    Exception e1 = new ExceptionWithLatch("e1", latch);
    assertThat(map.get(e1, false)).isNull();
    assertThat(map.get(e1, true)).isNotNull();
    assertThat(map.get(e1, true)).isEmpty();
    assertThat(map.get(e1, false)).isNotNull();

    Exception suppressed1 = new ExceptionWithLatch("suppressed1", latch);
    map.get(e1, true).add(suppressed1);
    assertThat(map.get(e1, true)).containsExactly(suppressed1);

    Exception suppressed2 = new ExceptionWithLatch("suppressed2", latch);
    map.get(e1, true).add(suppressed2);
    assertThat(map.get(e1, true)).containsExactly(suppressed1, suppressed2);

    assertThat(map.get(suppressed1, false)).isNull();
    assertThat(map.get(suppressed2, false)).isNull();
    assertThat(map.size()).isEqualTo(1);
    assertThat(map.get(suppressed1, true)).isNotNull();
    assertThat(map.size()).isEqualTo(2);
    assertThat(map.get(suppressed1, true)).isNotNull();
    assertThat(map.size()).isEqualTo(2);

    Exception e2 = new ExceptionWithLatch("e2", latch);
    assertThat(map.get(e2, true)).isNotNull();
    Exception e3 = new ExceptionWithLatch("e3", latch);
    assertThat(map.get(e3, true)).isNotNull();
    assertThat(map.size()).isEqualTo(4);
    return map;
  }

  @Test
  public void testSingleThreadedUse() throws InterruptedException {
    CountDownLatch latch = new CountDownLatch(5);
    ConcurrentWeakIdentityHashMap map =
        testConcurrentWeakIdentityHashMapSingleThreadedHelper(latch);
    for (int i = 0; i < 5; i++) {
      map.deleteEmptyKeys();
      GcFinalization.awaitFullGc();
    }
    latch.await(); // wait for e1 to be garbage collected.
    map.deleteEmptyKeys();
    assertThat(map.size()).isEqualTo(0);
  }

  private static Map<Throwable, List<Throwable>> createExceptionWithSuppressed(
      int numMainExceptions, int numSuppressedPerMain, CountDownLatch latch) {
    Map<Throwable, List<Throwable>> map = new HashMap<>();
    for (int i = 0; i < numMainExceptions; ++i) {
      Exception main = new ExceptionWithLatch("main-" + i, latch);
      List<Throwable> suppressedList = new ArrayList<>();
      assertThat(map).doesNotContainKey(main);
      map.put(main, suppressedList);
      for (int j = 0; j < numSuppressedPerMain; ++j) {
        Exception suppressed = new ExceptionWithLatch("suppressed-" + j + "-main-" + i, latch);
        suppressedList.add(suppressed);
      }
    }
    return map;
  }

  private ConcurrentWeakIdentityHashMap testFunctionalCorrectnessForMultiThreadedUse(
      int numMainExceptions, int numSuppressedPerMain, CountDownLatch latch)
      throws InterruptedException {
    Map<Throwable, List<Throwable>> exceptionWithSuppressed =
        createExceptionWithSuppressed(numMainExceptions, numSuppressedPerMain, latch);
    assertThat(exceptionWithSuppressed).hasSize(numMainExceptions);
    List<Pair> allPairs =
        exceptionWithSuppressed.entrySet().stream()
            .flatMap(
                entry -> entry.getValue().stream().map(value -> new Pair(entry.getKey(), value)))
            .collect(Collectors.toList());
    Collections.shuffle(allPairs);
    ConcurrentWeakIdentityHashMap map = new ConcurrentWeakIdentityHashMap();
    List<Worker> workers =
        IntStream.range(1, 11) // ten threads.
            .mapToObj(i -> new Worker("worker-" + i, map))
            .collect(Collectors.toList());

    // Assign tasks to workers.
    Iterator<Worker> workIterator = workers.iterator();
    for (Pair pair : allPairs) {
      if (!workIterator.hasNext()) {
        workIterator = workers.iterator();
      }
      assertThat(workIterator.hasNext()).isTrue();
      workIterator.next().exceptionList.add(pair);
    }

    // Execute all the workers.
    ExecutorService executorService = Executors.newFixedThreadPool(workers.size());
    workers.forEach(executorService::execute);
    executorService.shutdown();
    executorService.awaitTermination(Long.MAX_VALUE, TimeUnit.DAYS); // wait for completion.
    exceptionWithSuppressed
        .entrySet()
        .forEach(
            entry -> {
              assertThat(map.get(entry.getKey(), false)).isNotNull();
              assertThat(map.get(entry.getKey(), false))
                  .containsExactlyElementsIn(exceptionWithSuppressed.get(entry.getKey()));
            });
    return map;
  }

  private void testMultiThreadedUse(int numMainExceptions, int numSuppressedPerMain)
      throws InterruptedException {
    CountDownLatch latch = new CountDownLatch(numMainExceptions * numSuppressedPerMain);
    ConcurrentWeakIdentityHashMap map =
        testFunctionalCorrectnessForMultiThreadedUse(
            numMainExceptions, numSuppressedPerMain, latch);
    /*
     * Calling the following methods multiple times to make sure the keys are garbage collected,
     * and their corresponding entries are removed from the map.
     */
    map.deleteEmptyKeys();
    GcFinalization.awaitFullGc();
    map.deleteEmptyKeys();
    GcFinalization.awaitFullGc();
    map.deleteEmptyKeys();

    assertThat(map.size()).isEqualTo(0);
  }

  @Test
  public void testMultiThreadedUseMedium() throws InterruptedException {
    for (int i = 0; i < 10; ++i) {
      testMultiThreadedUse(50, 100);
    }
  }

  @Test
  public void testMultiThreadedUseLarge() throws InterruptedException {
    for (int i = 0; i < 5; ++i) {
      testMultiThreadedUse(100, 100);
    }
  }

  @Test
  public void testMultiThreadedUseSmall() throws InterruptedException {
    for (int i = 0; i < 10; ++i) {
      testMultiThreadedUse(20, 100);
    }
  }

  private static class ExceptionWithLatch extends Exception {
    private final CountDownLatch latch;

    private ExceptionWithLatch(String message, CountDownLatch latch) {
      super(message);
      this.latch = latch;
    }

    @Override
    public String toString() {
      return this.getMessage();
    }

    @Override
    protected void finalize() throws Throwable {
      latch.countDown();
    }
  }

  private static class Pair {
    final Throwable throwable;
    final Throwable suppressed;

    public Pair(Throwable throwable, Throwable suppressed) {
      this.throwable = throwable;
      this.suppressed = suppressed;
    }
  }

  private class Worker implements Runnable {
    private final ConcurrentWeakIdentityHashMap map;
    private final List<Pair> exceptionList = new ArrayList<>();
    private final String name;

    private Worker(String name, ConcurrentWeakIdentityHashMap map) {
      this.name = name;
      this.map = map;
    }

    public String getName() {
      return name;
    }

    @Override
    public void run() {
      Iterator<Pair> iterator = exceptionList.iterator();
      while (iterator.hasNext()) {
        int timeToSleep = random.nextInt(3);
        if (random.nextBoolean() && timeToSleep > 0) {
          try {
            Thread.sleep(timeToSleep); // add randomness to the scheduler.
          } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
          }
        }
        Pair pair = iterator.next();
        List<Throwable> suppressed = map.get(pair.throwable, true);
        System.out.printf("add suppressed %s to %s\n", pair.suppressed, pair.throwable);
        suppressed.add(pair.suppressed);
      }
    }
  }
}
