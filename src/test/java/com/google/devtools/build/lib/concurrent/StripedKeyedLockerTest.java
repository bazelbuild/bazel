// Copyright 2015 The Bazel Authors. All rights reserved.
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
import static com.google.common.truth.Truth.assertWithMessage;
import static java.util.concurrent.TimeUnit.SECONDS;
import static org.junit.Assert.assertThrows;

import com.google.common.base.Preconditions;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.devtools.build.lib.concurrent.KeyedLocker.AutoUnlocker;
import com.google.devtools.build.lib.concurrent.KeyedLocker.AutoUnlocker.IllegalUnlockException;
import com.google.devtools.build.lib.testutil.TestUtils;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Supplier;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link StripedKeyedLocker}. */
@RunWith(JUnit4.class)
public final class StripedKeyedLockerTest {
  private static final int NUM_EXECUTOR_THREADS = 1000;
  private KeyedLocker<String> locker;
  private ExecutorService executorService;
  private final AtomicReference<Throwable> throwableFromRunnable = new AtomicReference<>();

  private KeyedLocker<String> makeFreshLocker() {
    return new StripedKeyedLocker<>(17);
  }

  @Before
  public final void setUp() {
    locker = makeFreshLocker();
    executorService = Executors.newFixedThreadPool(NUM_EXECUTOR_THREADS);
  }

  @After
  public final void shutdownExecutor() {
    locker = null;
    MoreExecutors.shutdownAndAwaitTermination(
        executorService, TestUtils.WAIT_TIMEOUT_SECONDS, SECONDS);
    if (throwableFromRunnable.get() != null) {
      throw new RuntimeException("Uncaught from thread", throwableFromRunnable.get());
    }
  }

  private Supplier<AutoUnlocker> makeLockInvoker(String key) {
    return () -> locker.writeLock(key);
  }

  private Supplier<AutoUnlocker> makeLockFn1() {
    return makeLockInvoker("1");
  }

  private Supplier<AutoUnlocker> makeLockFn2() {
    return makeLockInvoker("2");
  }

  @SuppressWarnings("ReturnValueIgnored")
  @Test
  public void simpleSingleThreaded_noUnlocks() {
    Supplier<AutoUnlocker> lockFn1 = makeLockFn1();
    Supplier<AutoUnlocker> lockFn2 = makeLockFn2();
    lockFn1.get();
    lockFn2.get();
    lockFn1.get();
    lockFn2.get();
  }

  @Test
  public void simpleSingleThreaded_withUnlocks() {
    Supplier<AutoUnlocker> lockFn1 = makeLockFn1();
    Supplier<AutoUnlocker> lockFn2 = makeLockFn2();
    try (AutoUnlocker unlockerCat1 = lockFn1.get();
        AutoUnlocker unlockerDog1 = lockFn2.get();
        AutoUnlocker unlockerCat2 = lockFn1.get();
        AutoUnlocker unlockerDog2 = lockFn2.get()) {
      // Do nothing.
    }
  }

  @Test
  public void doubleUnlockOnSameAutoUnlockerNotAllowed() {
    AutoUnlocker unlocker = makeLockFn1().get();
    unlocker.close();
    assertThrows(IllegalUnlockException.class, unlocker::close);
  }

  @Test
  public void unlockOnDifferentAutoUnlockersAllowed() {
    Supplier<AutoUnlocker> lockFn = makeLockFn1();
    AutoUnlocker unlocker1 = lockFn.get();
    AutoUnlocker unlocker2 = lockFn.get();
    unlocker1.close();
    unlocker2.close();
  }

  @Test
  public void threadLocksMultipleTimesBeforeUnlocking() {
    Supplier<AutoUnlocker> lockFn = makeLockFn1();
    AtomicReference<Long> currentThreadIdRef = new AtomicReference<>(-1L);
    AtomicInteger count = new AtomicInteger(0);
    Runnable runnable =
        () -> {
          Long currentThreadId = Thread.currentThread().getId();
          try (AutoUnlocker unlocker1 = lockFn.get()) {
            currentThreadIdRef.set(currentThreadId);
            try (AutoUnlocker unlocker2 = lockFn.get()) {
              assertThat(currentThreadIdRef.get()).isEqualTo(currentThreadId);
              try (AutoUnlocker unlocker3 = lockFn.get()) {
                assertThat(currentThreadIdRef.get()).isEqualTo(currentThreadId);
                try (AutoUnlocker unlocker4 = lockFn.get()) {
                  assertThat(currentThreadIdRef.get()).isEqualTo(currentThreadId);
                  try (AutoUnlocker unlocker5 = lockFn.get()) {
                    assertThat(currentThreadIdRef.get()).isEqualTo(currentThreadId);
                    count.incrementAndGet();
                  }
                }
              }
            }
          }
        };
    for (int i = 0; i < NUM_EXECUTOR_THREADS; i++) {
      executorService.execute(wrap(runnable));
    }
    assertThatExecutorShutsDown();
    assertThat(count.get()).isEqualTo(NUM_EXECUTOR_THREADS);
  }

  @Test
  public void unlockOnOtherThreadNotAllowed() {
    AtomicReference<AutoUnlocker> unlockerRef = new AtomicReference<>();
    CountDownLatch unlockerRefSetLatch = new CountDownLatch(1);
    AtomicBoolean runnable2Executed = new AtomicBoolean(false);
    Runnable runnable1 =
        () -> {
          unlockerRef.set(makeLockFn1().get());
          unlockerRefSetLatch.countDown();
        };
    Runnable runnable2 =
        () -> {
          try {
            unlockerRefSetLatch.await(TestUtils.WAIT_TIMEOUT_SECONDS, SECONDS);
          } catch (InterruptedException e) {
            throw new IllegalStateException(e);
          }
          assertThrows(
              IllegalMonitorStateException.class,
              () -> Preconditions.checkNotNull(unlockerRef.get()).close());
          runnable2Executed.set(true);
        };
    executorService.execute(wrap(runnable1));
    executorService.execute(wrap(runnable2));
    assertThatExecutorShutsDown();
    assertThat(runnable2Executed.get()).isTrue();
  }

  private void runRefCountingSanity(Supplier<AutoUnlocker> lockFn) {
    Set<AutoUnlocker> unlockers = new HashSet<>();
    for (int i = 0; i < 1000; i++) {
      try (AutoUnlocker unlocker = lockFn.get()) {
        assertThat(unlockers.add(unlocker)).isTrue();
      }
    }
  }

  @Test
  public void refCountingSanity() {
    runRefCountingSanity(makeLockFn1());
  }

  @Test
  public void simpleMultiThreaded_mutualExclusion() {
    CountDownLatch runnableLatch = new CountDownLatch(NUM_EXECUTOR_THREADS);
    AtomicInteger mutexCounter = new AtomicInteger(0);
    AtomicInteger runnableCounter = new AtomicInteger(0);
    Runnable runnable =
        () -> {
          runnableLatch.countDown();
          try {
            // Wait until all the Runnables are ready to try to acquire the lock all at once.
            runnableLatch.await(TestUtils.WAIT_TIMEOUT_SECONDS, SECONDS);
          } catch (InterruptedException e) {
            throw new IllegalStateException(e);
          }
          try (AutoUnlocker unlocker = makeLockFn1().get()) {
            runnableCounter.incrementAndGet();
            assertThat(mutexCounter.incrementAndGet()).isEqualTo(1);
            assertThat(mutexCounter.decrementAndGet()).isEqualTo(0);
          }
        };
    for (int i = 0; i < NUM_EXECUTOR_THREADS; i++) {
      executorService.execute(wrap(runnable));
    }
    assertThatExecutorShutsDown();
    assertThat(runnableCounter.get()).isEqualTo(NUM_EXECUTOR_THREADS);
  }

  private Runnable wrap(Runnable runnable) {
    return () -> {
      try {
        runnable.run();
      } catch (Throwable e) {
        throwableFromRunnable.compareAndSet(null, e);
      }
    };
  }

  private void assertThatExecutorShutsDown() {
    assertWithMessage("Shouldn't have been interrupted")
        .that(ExecutorUtil.interruptibleShutdown(executorService))
        .isFalse();
  }
}
