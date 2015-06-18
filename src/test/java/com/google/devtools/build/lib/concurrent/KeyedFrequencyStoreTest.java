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
import static org.junit.Assert.assertSame;

import com.google.common.base.Throwables;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.devtools.build.lib.testutil.TestUtils;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/** Base class for tests for {@link KeyedFrequencyStore} implementations. */
public abstract class KeyedFrequencyStoreTest {
  private static final int NUM_EXECUTOR_THREADS = 1000;
  private ExecutorService executorService;
  private ThrowableRecordingRunnableWrapper wrapper;
  private KeyedFrequencyStore<String, Object> store;

  protected abstract KeyedFrequencyStore<String, Object> makeFreshStore();

  @Before
  public void setUp_KeyedFrequencyStoreTest() {
    store = makeFreshStore();
    executorService = Executors.newFixedThreadPool(NUM_EXECUTOR_THREADS);
    wrapper = new ThrowableRecordingRunnableWrapper("KeyedFrequencyStoreTest");
  }

  @After
  public void tearDown_KeyedFrequencyStoreTest() {
    store = null;
    MoreExecutors.shutdownAndAwaitTermination(executorService, TestUtils.WAIT_TIMEOUT_SECONDS,
        TimeUnit.SECONDS);
  }

  @Test
  public void zeroFrequency() {
    store.put("a", new Object(), 0);
    assertSame(null, store.consume("a"));
  }

  @Test
  public void simpleSingleThreaded() {
    int frequency = 100;
    Object objA = new Object();
    store.put("a", objA, frequency);
    for (int i = 0; i < frequency; i++) {
      assertSame(objA, store.consume("a"));
    }
    assertSame(null, store.consume("a"));
    assertSame(null, store.consume("a"));
  }

  @Test
  public void simpleMultiThreaded() throws Exception {
    int extra = 100;
    int frequency = NUM_EXECUTOR_THREADS - extra;
    final Object objA = new Object();
    store.put("a", objA, frequency);
    final AtomicInteger nullCount = new AtomicInteger(0);
    final AtomicInteger nonNullCount = new AtomicInteger(0);
    Runnable runnable = new Runnable() {
      @Override
      public void run() {
        Object obj = store.consume("a");
        if (obj == null) {
          nullCount.incrementAndGet();
        } else {
          assertSame(objA, obj);
          nonNullCount.incrementAndGet();
        }
      }
    };
    for (int i = 0; i < NUM_EXECUTOR_THREADS; i++) {
      executorService.submit(wrapper.wrap(runnable));
    }
    boolean interrupted = ExecutorUtil.interruptibleShutdown(executorService);
    Throwables.propagateIfPossible(wrapper.getFirstThrownError());
    if (interrupted) {
      Thread.currentThread().interrupt();
      throw new InterruptedException();
    }
    assertEquals(frequency, nonNullCount.get());
    assertEquals(extra, nullCount.get());
  }
}