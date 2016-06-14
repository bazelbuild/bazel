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
package com.google.devtools.build.lib.actions;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import com.google.common.testing.GcFinalization;
import com.google.devtools.build.lib.concurrent.AbstractQueueVisitor;
import com.google.devtools.build.lib.testutil.TestThread;
import com.google.devtools.build.lib.testutil.TestUtils;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.lang.ref.WeakReference;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Tests for ConcurrentMultimapWithHeadElement.
 */
@RunWith(JUnit4.class)
public class ConcurrentMultimapWithHeadElementTest {
  @Test
  public void testSmoke() throws Exception {
    ConcurrentMultimapWithHeadElement<String, String> multimap =
        new ConcurrentMultimapWithHeadElement<>();
    assertEquals("val", multimap.putAndGet("key", "val"));
    assertEquals("val", multimap.get("key"));
    assertEquals("val", multimap.putAndGet("key", "val2"));
    multimap.remove("key", "val2");
    assertEquals("val", multimap.get("key"));
    assertEquals("val", multimap.putAndGet("key", "val2"));
    multimap.remove("key", "val");
    assertEquals("val2", multimap.get("key"));
  }

  @Test
  public void testDuplicate() throws Exception {
    ConcurrentMultimapWithHeadElement<String, String> multimap =
        new ConcurrentMultimapWithHeadElement<>();
    assertEquals("val", multimap.putAndGet("key", "val"));
    assertEquals("val", multimap.get("key"));
    assertEquals("val", multimap.putAndGet("key", "val"));
    multimap.remove("key", "val");
    assertNull(multimap.get("key"));
  }

  @Test
  public void testDuplicateWithEqualsObject() throws Exception {
    ConcurrentMultimapWithHeadElement<String, String> multimap =
        new ConcurrentMultimapWithHeadElement<>();
    assertEquals("val", multimap.putAndGet("key", "val"));
    assertEquals("val", multimap.get("key"));
    assertEquals("val", multimap.putAndGet("key", "val"));
    multimap.remove("key", "val");
    assertNull(multimap.get("key"));
  }

  @Test
  public void testFailedRemoval() throws Exception {
    ConcurrentMultimapWithHeadElement<String, String> multimap =
        new ConcurrentMultimapWithHeadElement<>();
    assertEquals("val", multimap.putAndGet("key", "val"));
    multimap.remove("key", "val2");
    assertEquals("val", multimap.get("key"));
  }

  @Test
  public void testNotEmpty() throws Exception {
    ConcurrentMultimapWithHeadElement<String, String> multimap =
        new ConcurrentMultimapWithHeadElement<>();
    assertEquals("val", multimap.putAndGet("key", "val"));
    multimap.remove("key", "val2");
    assertEquals("val", multimap.get("key"));
  }

  @Test
  public void testKeyRemoved() throws Exception {
    String key = new String("key");
    ConcurrentMultimapWithHeadElement<String, String> multimap =
        new ConcurrentMultimapWithHeadElement<>();
    assertEquals("val", multimap.putAndGet(key, "val"));
    WeakReference<String> weakKey = new WeakReference<>(key);
    multimap.remove(key, "val");
    key = null;
    GcFinalization.awaitClear(weakKey);
  }

  @Test
  public void testKeyRemovedAndAddedConcurrently() throws Exception {
    final ConcurrentMultimapWithHeadElement<String, String> multimap =
        new ConcurrentMultimapWithHeadElement<>();
    // Because we have two threads racing, run the test many times. Before fixed, there was a 90%
    // chance of failure in 10,000 runs.
    for (int i = 0; i < 10000; i++) {
      assertEquals("val", multimap.putAndGet("key", "val"));
      final CountDownLatch threadStart = new CountDownLatch(1);
      TestThread testThread = new TestThread() {
        @Override
        public void runTest() throws Exception {
          threadStart.countDown();
          multimap.remove("key", "val");
        }
      };
      testThread.start();
      assertTrue(threadStart.await(TestUtils.WAIT_TIMEOUT_SECONDS, TimeUnit.SECONDS));
      assertNotNull(multimap.putAndGet("key", "val2")); // Removal may not have happened yet.
      assertNotNull(multimap.get("key")); // If put failed, this will be null.
      testThread.joinAndAssertState(2000);
      multimap.clear();
    }
  }

  private class StressTester extends AbstractQueueVisitor {
    private final ConcurrentMultimapWithHeadElement<Boolean, Integer> multimap =
        new ConcurrentMultimapWithHeadElement<>();
    private final AtomicInteger actionCount = new AtomicInteger(0);

    private StressTester() {
      super(
          /*concurrent=*/ true,
          200,
          1,
          TimeUnit.SECONDS,
          /*failFastOnException=*/ true,
          "action-graph-test");
    }

    private void addAndRemove(final Boolean key, final Integer add, final Integer remove) {
      execute(
          new Runnable() {
            @Override
            public void run() {
              assertNotNull(multimap.putAndGet(key, add));
              multimap.remove(key, remove);
              doRandom();
            }
          });
    }

    private Integer getRandomInt() {
      return (int) Math.round(Math.random() * 3.0);
    }

    private void doRandom() {
      if (actionCount.incrementAndGet() > 100000) {
        return;
      }
      Boolean key = Math.random() < 0.5;
      addAndRemove(key, getRandomInt(), getRandomInt());
    }

    private void work() throws InterruptedException {
      awaitQuiescence(/*interruptWorkers=*/ true);
    }
  }

  @Test
  public void testStressTest() throws Exception {
    StressTester stressTester = new StressTester();
    stressTester.doRandom();
    stressTester.work();
  }

}
