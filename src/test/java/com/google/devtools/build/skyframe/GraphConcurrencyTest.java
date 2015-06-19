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
package com.google.devtools.build.skyframe;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import com.google.common.base.Throwables;
import com.google.common.collect.Iterables;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.concurrent.ExecutorUtil;
import com.google.devtools.build.lib.concurrent.KeyedLocker;
import com.google.devtools.build.lib.concurrent.RefCountedMultisetKeyedLocker;
import com.google.devtools.build.lib.concurrent.ThrowableRecordingRunnableWrapper;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.util.GroupedList.GroupedListHelper;
import com.google.devtools.build.skyframe.GraphTester.StringValue;
import com.google.devtools.build.skyframe.NodeEntry.DependencyState;

import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/** Base class for concurrency sanity tests on {@link EvaluableGraph} implementations. */
public abstract class GraphConcurrencyTest {

  private static final SkyFunctionName SKY_FUNCTION_NAME =
      new SkyFunctionName("GraphConcurrencyTestKey", /*isComputed=*/false);
  private ProcessableGraph graph;
  private ThrowableRecordingRunnableWrapper wrapper;
  protected abstract ProcessableGraph getGraph();

  @Before
  public void init() {
    this.graph = getGraph();
    this.wrapper = new ThrowableRecordingRunnableWrapper("GraphConcurrencyTest");
  }

  private SkyKey key(String name) {
    return new SkyKey(SKY_FUNCTION_NAME, name);
  }

  @Test
  public void createIfAbsentSanity() {
    graph.createIfAbsent(key("cat"));
  }

  // Tests adding and removing Rdeps of a {@link NodeEntry} while a node transitions from
  // not done to done.
  @Test
  public void testAddRemoveRdeps() throws Exception {
    SkyKey key = key("foo");
    final NodeEntry entry = graph.createIfAbsent(key);
    // These numbers are arbitrary.
    int numThreads = 50;
    int numKeys = 100;
    // One chunk will be used to add and remove rdeps before setting the node value.  The second
    // chunk of work will have the node value set and the last chunk will be to add and remove
    // rdeps after the value has been set.
    final int chunkSize = 40;
    final int numIterations = chunkSize * 3;
    // This latch is used to signal that the runnables have been submitted to the executor.
    final CountDownLatch countDownLatch1 = new CountDownLatch(1);
    // This latch is used to signal to the main thread that we have begun the second chunk
    // for sufficiently many keys.  The minimum of numThreads and numKeys is used to prevent
    // thread starvation from causing a delay here.
    final CountDownLatch countDownLatch2 = new CountDownLatch(Math.min(numThreads, numKeys));
    // This latch is used to guarantee that we set the node's value before we enter the third
    // chunk for any key.
    final CountDownLatch countDownLatch3 = new CountDownLatch(1);
    ExecutorService pool = Executors.newFixedThreadPool(numThreads);
    // Add single rdep before transition to done.
    assertEquals(DependencyState.NEEDS_SCHEDULING, entry.addReverseDepAndCheckIfDone(key("rdep")));
    for (int i = 0; i < numKeys; i++) {
      final int j = i;
      Runnable r = new Runnable() {
        @Override
        public void run() {
          try {
            countDownLatch1.await();
            // Add and remove the rdep a bunch of times to test interleaving.
            for (int k = 1; k <= numIterations; k++) {
              if (k == chunkSize) {
                countDownLatch2.countDown();
              }
              entry.addReverseDepAndCheckIfDone(key("rdep" + j));
              entry.removeReverseDep(key("rdep" + j));
              if (k == chunkSize * 2) {
                countDownLatch3.await();
              }
            }
            entry.addReverseDepAndCheckIfDone(key("rdep" + j));
          } catch (InterruptedException e) {
            fail("Test failed: " + e.toString());
          }
        }
      };
      pool.execute(wrapper.wrap(r));
    }
    countDownLatch1.countDown();
    try {
      countDownLatch2.await();
    } catch (InterruptedException e) {
      fail("Test failed: " + e.toString());
    }
    entry.setValue(new StringValue("foo1"), new IntVersion(1));
    countDownLatch3.countDown();
    entry.removeReverseDep(key("rdep"));
    boolean interrupted = ExecutorUtil.interruptibleShutdown(pool);
    Throwables.propagateIfPossible(wrapper.getFirstThrownError());
    if (interrupted) {
      Thread.currentThread().interrupt();
      throw new InterruptedException();
    }
    assertEquals(new StringValue("foo1"), graph.get(key).getValue());
    assertEquals(numKeys, Iterables.size(graph.get(key).getReverseDeps()));
    // Mark the node as dirty again and check that the reverse deps have been preserved.
    entry.markDirty(true);
    startEvaluation(entry);
    entry.setValue(new StringValue("foo2"), new IntVersion(2));
    assertEquals(new StringValue("foo2"), graph.get(key).getValue());
    assertEquals(numKeys, Iterables.size(graph.get(key).getReverseDeps()));
  }

  // Tests adding inflight nodes with a given key while an existing node with the same key
  // undergoes a transition from not done to done.
  @Test
  public void testAddingInflightNodes() throws Exception {
    int numThreads = 50;
    final KeyedLocker<SkyKey> locker =
        new RefCountedMultisetKeyedLocker.Factory<SkyKey>().create();
    ExecutorService pool = Executors.newFixedThreadPool(numThreads);
    final int numKeys = 500;
    // Add each key 10 times.
    final Set<SkyKey> nodeCreated = Sets.newConcurrentHashSet();
    final Set<SkyKey> valuesSet = Sets.newConcurrentHashSet();
    for (int i = 0; i < 10; i++) {
      for (int j = 0; j < numKeys; j++) {
        final int keyNum = j;
        final SkyKey key = key("foo" + keyNum);
        Runnable r = new Runnable() {
          public void run() {
            NodeEntry entry;
            try (KeyedLocker.AutoUnlocker unlocker = locker.lock(key)) {
              entry = graph.get(key);
              if (entry == null) {
                assertTrue(nodeCreated.add(key));
              }
              entry = graph.createIfAbsent(key);
            }
            // {@code entry.addReverseDepAndCheckIfDone(null)} should return NEEDS_SCHEDULING at
            // most once.
            if (startEvaluation(entry).equals(DependencyState.NEEDS_SCHEDULING)) {
              assertTrue(valuesSet.add(key));
              // Set to done.
              entry.setValue(new StringValue("bar" + keyNum), new IntVersion(keyNum));
              assertThat(entry.isDone()).isTrue();
            }
            // This shouldn't cause any problems from the other threads.
            graph.createIfAbsent(key);
          }
        };
        pool.execute(wrapper.wrap(r));
      }
    }
    boolean interrupted = ExecutorUtil.interruptibleShutdown(pool);
    Throwables.propagateIfPossible(wrapper.getFirstThrownError());
    if (interrupted) {
      Thread.currentThread().interrupt();
      throw new InterruptedException();
    }
    // Check that all the values are as expected.
    for (int i = 0; i < numKeys; i++) {
      SkyKey key = key("foo" + i);
      assertTrue(nodeCreated.contains(key));
      assertTrue(valuesSet.contains(key));
      assertThat(graph.get(key).getValue()).isEqualTo(new StringValue("bar" + i));
      assertThat(graph.get(key).getVersion()).isEqualTo(new IntVersion(i));
    }
  }

  /**
   * Initially calling {@link NodeEntry#setValue} and then making sure concurrent calls to
   * {@link QueryableGraph#get} and {@link QueryableGraph#getBatch} do not interfere with the node.
   */
  @Test
  public void testDoneToDirty() throws Exception {
    final int numKeys = 1000;
    int numThreads = 50;
    final int numBatchRequests = 100;
    // Create a bunch of done nodes.
    for (int i = 0; i < numKeys; i++) {
      NodeEntry entry = graph.createIfAbsent(key("foo" + i));
      startEvaluation(entry);
      entry.setValue(new StringValue("bar"), new IntVersion(0));
    }

    ExecutorService pool1 = Executors.newFixedThreadPool(numThreads);
    ExecutorService pool2 = Executors.newFixedThreadPool(numThreads);
    ExecutorService pool3 = Executors.newFixedThreadPool(numThreads);

    // Only start all the threads once the batch requests are ready.
    final CountDownLatch makeBatchCountDownLatch = new CountDownLatch(numBatchRequests);
    // Do at least 5 single requests and batch requests before transitioning node.
    final CountDownLatch getBatchCountDownLatch = new CountDownLatch(5);
    final CountDownLatch getCountDownLatch = new CountDownLatch(5);

    for (int i = 0; i < numKeys; i++) {
      final int keyNum = i;
      // Transition the nodes from done to dirty and then back to done.
      Runnable r1 = new Runnable() {
        @Override
        public void run() {
          try {
            makeBatchCountDownLatch.await();
            getBatchCountDownLatch.await();
            getCountDownLatch.await();
          } catch (InterruptedException e) {
            fail("Test failed: " + e.toString());
          }
          NodeEntry entry = graph.get(key("foo" + keyNum));
          entry.markDirty(true);
          // Make some changes, like adding a dep and rdep.
          entry.addReverseDepAndCheckIfDone(key("rdep"));
          addTemporaryDirectDep(entry, key("dep"));
          entry.signalDep();
          // Move node from dirty back to done.
          entry.setValue(new StringValue("bar" + keyNum), new IntVersion(1));
        }
      };

      // Start a bunch of get() calls while the node transitions from dirty to done and back.
      Runnable r2 = new Runnable() {
        @Override
        public void run() {
          try {
            makeBatchCountDownLatch.await();
          } catch (InterruptedException e) {
            fail("Test failed: " + e.toString());
          }
          NodeEntry entry = graph.get(key("foo" + keyNum));
          assertNotEquals(null, entry);
          // Requests for the value are made at the same time that the version changes from 0 to 1.
          // Check that there is no problem in requesting the version and that the number is sane.
          assertThat(entry.getVersion()).isAnyOf(new IntVersion(0), new IntVersion(1));
          getCountDownLatch.countDown();
        }
      };
      pool1.execute(wrapper.wrap(r1));
      pool2.execute(wrapper.wrap(r2));
    }
    Random r = new Random(TestUtils.getRandomSeed());
    // Start a bunch of getBatch() calls while the node transitions from dirty to done and back.
    for (int i = 0; i < numBatchRequests; i++) {
      final List<SkyKey> batch = new ArrayList<>(numKeys);
      // Pseudorandomly uniformly sample the powerset of the keys.
      for (int j = 0; j < numKeys; j++) {
        if (r.nextBoolean()) {
          batch.add(key("foo" + j));
        }
      }
      makeBatchCountDownLatch.countDown();
      Runnable r3 = new Runnable() {
        @Override
        public void run() {
          try {
            makeBatchCountDownLatch.await();
          } catch (InterruptedException e) {
            fail("Test failed: " + e.toString());
          }
          Map<SkyKey, NodeEntry> batchMap = graph.getBatch(batch);
          getBatchCountDownLatch.countDown();
          assertEquals(batch.size(), batchMap.size());
          for (NodeEntry entry : batchMap.values()) {
            // Batch requests are made at the same time that the version changes from 0 to 1.
            // Check that there is no problem in requesting the version and that the number is sane.
            assertThat(entry.getVersion()).isAnyOf(new IntVersion(0), new IntVersion(1));
          }
        }
      };
      pool3.execute(wrapper.wrap(r3));
    }
    boolean interrupted = ExecutorUtil.interruptibleShutdown(pool1);
    interrupted |= ExecutorUtil.interruptibleShutdown(pool2);
    interrupted |= ExecutorUtil.interruptibleShutdown(pool3);
    Throwables.propagateIfPossible(wrapper.getFirstThrownError());
    if (interrupted) {
      Thread.currentThread().interrupt();
      throw new InterruptedException();
    }
    for (int i = 0; i < numKeys; i++) {
      NodeEntry entry = graph.get(key("foo" + i));
      assertThat(entry.getValue()).isEqualTo(new StringValue("bar" + i));
      assertThat(entry.getVersion()).isEqualTo(new IntVersion(1));
      for (SkyKey key : entry.getReverseDeps()) {
        assertEquals(key("rdep"), key);
      }
      for (SkyKey key : entry.getDirectDeps()) {
        assertEquals(key("dep"), key);
      }
    }
  }

  private DependencyState startEvaluation(NodeEntry entry) {
    return entry.addReverseDepAndCheckIfDone(null);
  }

  private static void addTemporaryDirectDep(NodeEntry entry, SkyKey key) {
    GroupedListHelper<SkyKey> helper = new GroupedListHelper<>();
    helper.add(key);
    entry.addTemporaryDirectDeps(helper);
  }
}
