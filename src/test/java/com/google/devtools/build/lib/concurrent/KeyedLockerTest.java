// Copyright 2015 Google Inc. All rights reserved.
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

import com.google.common.base.Preconditions;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.devtools.build.lib.concurrent.KeyedLocker.AutoUnlocker;
import com.google.devtools.build.lib.concurrent.KeyedLocker.AutoUnlocker.IllegalUnlockException;
import com.google.devtools.build.lib.testutil.TestUtils;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

/** Base class for tests for {@link KeyedLocker} implementations. */
public abstract class KeyedLockerTest {
  private static final int NUM_EXECUTOR_THREADS = 1000;
  private KeyedLocker<String> locker;
  private ExecutorService executorService;

  protected abstract KeyedLocker<String> makeFreshLocker();

  @Before
  public void setUp() {
    locker = makeFreshLocker();
    executorService = Executors.newFixedThreadPool(NUM_EXECUTOR_THREADS);
  }

  @After
  public void tearDown() {
    locker = null;
    MoreExecutors.shutdownAndAwaitTermination(executorService, TestUtils.WAIT_TIMEOUT_SECONDS,
        TimeUnit.SECONDS);
  }

  @Test
  public void simpleSingleThreaded_NoUnlocks() {
    locker.lock("cat");
    locker.lock("dog");
    locker.lock("cat");
    locker.lock("dog");
  }

  @Test
  public void simpleSingleThreaded_WithUnlocks() {
    try (AutoUnlocker unlockerCat1 = locker.lock("cat")) {
      try (AutoUnlocker unlockerDog1 = locker.lock("dog")) {
        try (AutoUnlocker unlockerCat2 = locker.lock("cat")) {
          try (AutoUnlocker unlockerDog2 = locker.lock("dog")) {
          }
        }
      }
    }
  }

  @Test
  public void doubleUnlockOnSameAutoUnlockerNotAllowed() {
    AutoUnlocker unlocker = locker.lock("cat");
    unlocker.close();
    try {
      unlocker.close();
      fail();
    } catch (IllegalUnlockException expected) {
    }
  }

  @Test
  public void unlockOnDifferentAutoUnlockersAllowed() {
    AutoUnlocker unlocker1 = locker.lock("cat");
    AutoUnlocker unlocker2 = locker.lock("cat");
    unlocker1.close();
    unlocker2.close();
  }

  @Test
  public void threadLocksMultipleTimesBeforeUnlocking() throws Exception {
    final AtomicReference<Long> currentThreadIdRef = new AtomicReference<>(new Long(-1L));
    final AtomicInteger count = new AtomicInteger(0);
    Runnable runnable = new Runnable() {
      @Override
      public void run() {
        try (AutoUnlocker unlocker1 = locker.lock("cat")) {
          Long currentThreadId = Thread.currentThread().getId();
          currentThreadIdRef.set(currentThreadId);
          try (AutoUnlocker unlocker2 = locker.lock("cat")) {
            assertEquals(currentThreadId, currentThreadIdRef.get());
            try (AutoUnlocker unlocker3 = locker.lock("cat")) {
              assertEquals(currentThreadId, currentThreadIdRef.get());
              try (AutoUnlocker unlocker4 = locker.lock("cat")) {
                assertEquals(currentThreadId, currentThreadIdRef.get());
                try (AutoUnlocker unlocker5 = locker.lock("cat")) {
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
      executorService.submit(runnable);
    }
    boolean interrupted = ExecutorShutdownUtil.interruptibleShutdown(executorService);
    if (interrupted) {
      Thread.currentThread().interrupt();
      throw new InterruptedException();
    }
    assertEquals(NUM_EXECUTOR_THREADS, count.get());
  }

  @Test
  public void unlockOnOtherThreadNotAllowed() throws Exception {
    final AtomicReference<AutoUnlocker> unlockerRef = new AtomicReference<>();
    final CountDownLatch unlockerRefSetLatch = new CountDownLatch(1);
    final AtomicBoolean runnableInterrupted = new AtomicBoolean(false);
    final AtomicBoolean runnable2Executed = new AtomicBoolean(false);
    Runnable runnable1 = new Runnable() {
      @Override
      public void run() {
        unlockerRef.set(locker.lock("cat"));
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
        } catch (IllegalUnlockException expected) {
          runnable2Executed.set(true);
        }
      }
    };
    executorService.submit(runnable1);
    executorService.submit(runnable2);
    boolean interrupted = ExecutorShutdownUtil.interruptibleShutdown(executorService);
    if (interrupted || runnableInterrupted.get()) {
      Thread.currentThread().interrupt();
      throw new InterruptedException();
    }
    assertTrue(runnable2Executed.get());
  }

  @Test
  public void refCountingSanity() {
    Set<AutoUnlocker> unlockers = new HashSet<>();
    for (int i = 0; i < 1000; i++) {
      try (AutoUnlocker unlocker = locker.lock("cat")) {
        assertTrue(unlockers.add(unlocker));
      }
    }
  }

  @Test
  public void simpleMultiThreaded_MutualExclusion() throws InterruptedException {
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
        try (AutoUnlocker unlocker = locker.lock("cat")) {
          runnableCounter.incrementAndGet();
          assertEquals(1, mutexCounter.incrementAndGet());
          assertEquals(0, mutexCounter.decrementAndGet());
        }
      }
    };
    for (int i = 0; i < NUM_EXECUTOR_THREADS; i++) {
      executorService.submit(runnable);
    }
    boolean interrupted = ExecutorShutdownUtil.interruptibleShutdown(executorService);
    if (interrupted || runnableInterrupted.get()) {
      Thread.currentThread().interrupt();
      throw new InterruptedException();
    }
    assertEquals(NUM_EXECUTOR_THREADS, runnableCounter.get());
  }
}
