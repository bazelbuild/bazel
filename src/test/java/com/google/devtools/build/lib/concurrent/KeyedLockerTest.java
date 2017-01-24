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

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import com.google.common.base.Supplier;
import com.google.common.base.Throwables;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.devtools.build.lib.concurrent.KeyedLocker.AutoUnlocker;
import com.google.devtools.build.lib.concurrent.KeyedLocker.AutoUnlocker.IllegalUnlockException;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.util.Preconditions;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

/** Base class for tests for {@link KeyedLocker} implementations. */
public abstract class KeyedLockerTest {
  private static final int NUM_EXECUTOR_THREADS = 1000;
  private KeyedLocker<String> locker;
  protected ExecutorService executorService;
  protected ThrowableRecordingRunnableWrapper wrapper;

  protected abstract KeyedLocker<String> makeFreshLocker();

  @Before
  public final void setUp_KeyedLockerTest() {
    locker = makeFreshLocker();
    executorService = Executors.newFixedThreadPool(NUM_EXECUTOR_THREADS);
    wrapper = new ThrowableRecordingRunnableWrapper("KeyedLockerTest");
  }

  @After
  public final void shutdownExecutor() throws Exception  {
    locker = null;
    MoreExecutors.shutdownAndAwaitTermination(executorService, TestUtils.WAIT_TIMEOUT_SECONDS,
        TimeUnit.SECONDS);
  }

  private Supplier<AutoUnlocker> makeLockInvoker(final String key) {
    return new Supplier<KeyedLocker.AutoUnlocker>() {
      @Override
      public AutoUnlocker get() {
        return locker.writeLock(key);
      }
    };
  }

  private Supplier<AutoUnlocker> makeLockFn1() {
    return makeLockInvoker("1");
  }

  private Supplier<AutoUnlocker> makeLockFn2() {
    return makeLockInvoker("2");
  }

  protected void runSimpleSingleThreaded_NoUnlocks(Supplier<AutoUnlocker> lockFn1,
      Supplier<AutoUnlocker> lockFn2) {
    lockFn1.get();
    lockFn2.get();
    lockFn1.get();
    lockFn2.get();
  }

  @Test
  public void simpleSingleThreaded_NoUnlocks() {
    runSimpleSingleThreaded_NoUnlocks(makeLockFn1(), makeLockFn2());
  }

  protected void runSimpleSingleThreaded_WithUnlocks(final Supplier<AutoUnlocker> lockFn1,
      final Supplier<AutoUnlocker> lockFn2) {
    try (AutoUnlocker unlockerCat1 = lockFn1.get()) {
      try (AutoUnlocker unlockerDog1 = lockFn2.get()) {
        try (AutoUnlocker unlockerCat2 = lockFn1.get()) {
          try (AutoUnlocker unlockerDog2 = lockFn2.get()) {
          }
        }
      }
    }
  }

  @Test
  public void simpleSingleThreaded_WithUnlocks() {
    runSimpleSingleThreaded_WithUnlocks(makeLockFn1(), makeLockFn2());
  }

  protected void runDoubleUnlockOnSameAutoUnlockerNotAllowed(final Supplier<AutoUnlocker> lockFn) {
    AutoUnlocker unlocker = lockFn.get();
    unlocker.close();
    try {
      unlocker.close();
      fail();
    } catch (IllegalUnlockException expected) {
    }
  }

  @Test
  public void doubleUnlockOnSameAutoUnlockerNotAllowed() {
    runDoubleUnlockOnSameAutoUnlockerNotAllowed(makeLockFn1());
  }

  protected void runUnlockOnDifferentAutoUnlockersAllowed(final Supplier<AutoUnlocker> lockFn) {
    AutoUnlocker unlocker1 = lockFn.get();
    AutoUnlocker unlocker2 = lockFn.get();
    unlocker1.close();
    unlocker2.close();
  }

  @Test
  public void unlockOnDifferentAutoUnlockersAllowed() {
    runUnlockOnDifferentAutoUnlockersAllowed(makeLockFn1());
  }

  public void runThreadLocksMultipleTimesBeforeUnlocking(final Supplier<AutoUnlocker> lockFn)
      throws Exception {
    final AtomicReference<Long> currentThreadIdRef = new AtomicReference<>(new Long(-1L));
    final AtomicInteger count = new AtomicInteger(0);
    Runnable runnable = new Runnable() {
      @Override
      public void run() {
        Long currentThreadId = Thread.currentThread().getId();
        try (AutoUnlocker unlocker1 = lockFn.get()) {
          currentThreadIdRef.set(currentThreadId);
          try (AutoUnlocker unlocker2 = lockFn.get()) {
            assertEquals(currentThreadId, currentThreadIdRef.get());
            try (AutoUnlocker unlocker3 = lockFn.get()) {
              assertEquals(currentThreadId, currentThreadIdRef.get());
              try (AutoUnlocker unlocker4 = lockFn.get()) {
                assertEquals(currentThreadId, currentThreadIdRef.get());
                try (AutoUnlocker unlocker5 = lockFn.get()) {
                  assertEquals(currentThreadId, currentThreadIdRef.get());
                  count.incrementAndGet();
                }
              }
            }
          }
        }
      }
    };
    for (int i = 0; i < NUM_EXECUTOR_THREADS; i++) {
      @SuppressWarnings("unused") 
      Future<?> possiblyIgnoredError = executorService.submit(wrapper.wrap(runnable));
    }
    boolean interrupted = ExecutorUtil.interruptibleShutdown(executorService);
    Throwables.propagateIfPossible(wrapper.getFirstThrownError());
    if (interrupted) {
      Thread.currentThread().interrupt();
      throw new InterruptedException();
    }
    assertEquals(NUM_EXECUTOR_THREADS, count.get());
  }

  @Test
  public void threadLocksMultipleTimesBeforeUnlocking() throws Exception {
    runThreadLocksMultipleTimesBeforeUnlocking(makeLockFn1());
  }

  protected void runUnlockOnOtherThreadNotAllowed(final Supplier<AutoUnlocker> lockFn)
      throws Exception {
    final AtomicReference<AutoUnlocker> unlockerRef = new AtomicReference<>();
    final CountDownLatch unlockerRefSetLatch = new CountDownLatch(1);
    final AtomicBoolean runnableInterrupted = new AtomicBoolean(false);
    final AtomicBoolean runnable2Executed = new AtomicBoolean(false);
    Runnable runnable1 = new Runnable() {
      @Override
      public void run() {
        unlockerRef.set(lockFn.get());
        unlockerRefSetLatch.countDown();
      }
    };
    Runnable runnable2 = new Runnable() {
      @Override
      public void run() {
        try {
          unlockerRefSetLatch.await(TestUtils.WAIT_TIMEOUT_SECONDS, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
          runnableInterrupted.set(true);
        }
        try {
          Preconditions.checkNotNull(unlockerRef.get()).close();
          fail();
        } catch (IllegalMonitorStateException expected) {
          runnable2Executed.set(true);
        }
      }
    };
    @SuppressWarnings("unused")
    Future<?> possiblyIgnoredError = executorService.submit(wrapper.wrap(runnable1));
    @SuppressWarnings("unused")
    Future<?> possiblyIgnoredError1 = executorService.submit(wrapper.wrap(runnable2));
    boolean interrupted = ExecutorUtil.interruptibleShutdown(executorService);
    Throwables.propagateIfPossible(wrapper.getFirstThrownError());
    if (interrupted || runnableInterrupted.get()) {
      Thread.currentThread().interrupt();
      throw new InterruptedException();
    }
    assertTrue(runnable2Executed.get());
  }

  @Test
  public void unlockOnOtherThreadNotAllowed() throws Exception {
    runUnlockOnOtherThreadNotAllowed(makeLockFn1());
  }

  protected void runRefCountingSanity(final Supplier<AutoUnlocker> lockFn) {
    Set<AutoUnlocker> unlockers = new HashSet<>();
    for (int i = 0; i < 1000; i++) {
      try (AutoUnlocker unlocker = lockFn.get()) {
        assertTrue(unlockers.add(unlocker));
      }
    }
  }

  @Test
  public void refCountingSanity() {
    runRefCountingSanity(makeLockFn1());
  }

  protected void runSimpleMultiThreaded_MutualExclusion(final Supplier<AutoUnlocker> lockFn)
      throws InterruptedException {
    final CountDownLatch runnableLatch = new CountDownLatch(NUM_EXECUTOR_THREADS);
    final AtomicInteger mutexCounter = new AtomicInteger(0);
    final AtomicInteger runnableCounter = new AtomicInteger(0);
    final AtomicBoolean runnableInterrupted = new AtomicBoolean(false);
    Runnable runnable = new Runnable() {
      @Override
      public void run() {
        runnableLatch.countDown();
        try {
          // Wait until all the Runnables are ready to try to acquire the lock all at once.
          runnableLatch.await(TestUtils.WAIT_TIMEOUT_SECONDS, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
          runnableInterrupted.set(true);
        }
        try (AutoUnlocker unlocker = lockFn.get()) {
          runnableCounter.incrementAndGet();
          assertEquals(1, mutexCounter.incrementAndGet());
          assertEquals(0, mutexCounter.decrementAndGet());
        }
      }
    };
    for (int i = 0; i < NUM_EXECUTOR_THREADS; i++) {
      @SuppressWarnings("unused")
      Future<?> possiblyIgnoredError = executorService.submit(wrapper.wrap(runnable));
    }
    boolean interrupted = ExecutorUtil.interruptibleShutdown(executorService);
    Throwables.propagateIfPossible(wrapper.getFirstThrownError());
    if (interrupted || runnableInterrupted.get()) {
      Thread.currentThread().interrupt();
      throw new InterruptedException();
    }
    assertEquals(NUM_EXECUTOR_THREADS, runnableCounter.get());
  }

  @Test
  public void simpleMultiThreaded_MutualExclusion() throws Exception {
    runSimpleMultiThreaded_MutualExclusion(makeLockFn1());
  }
}
