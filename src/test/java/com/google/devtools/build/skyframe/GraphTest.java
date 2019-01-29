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
package com.google.devtools.build.skyframe;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.fail;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.concurrent.ExecutorUtil;
import com.google.devtools.build.lib.testutil.TestRunnableWrapper;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.util.GroupedList.GroupedListHelper;
import com.google.devtools.build.skyframe.GraphTester.StringValue;
import com.google.devtools.build.skyframe.NodeEntry.DependencyState;
import com.google.devtools.build.skyframe.QueryableGraph.Reason;
import com.google.devtools.build.skyframe.ThinNodeEntry.DirtyType;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import org.junit.Before;
import org.junit.Test;

/** Base class for sanity tests on {@link EvaluableGraph} implementations. */
public abstract class GraphTest {
  protected ProcessableGraph graph;
  protected TestRunnableWrapper wrapper;
  private final Version startingVersion = getStartingVersion();

  // This code should really be in a @Before method, but @Before methods are executed from the
  // top down, and this class's @Before method calls #getGraph, so makeGraph must have already
  // been called.
  protected abstract void makeGraph() throws Exception;

  protected abstract ProcessableGraph getGraph(Version version) throws Exception;

  protected abstract Version getStartingVersion();

  protected abstract Version getNextVersion(Version version);

  protected boolean checkRdeps() {
    return true;
  }

  @Before
  public void init() throws Exception {
    makeGraph();
    Version startingVersion = getStartingVersion();
    this.graph = getGraph(startingVersion);
    this.wrapper = new TestRunnableWrapper("GraphConcurrencyTest");
  }

  protected SkyKey key(String name) {
    return GraphTester.toSkyKey(name);
  }

  @Test
  public void createIfAbsentBatchSanity() throws InterruptedException {
    graph.createIfAbsentBatch(null, Reason.OTHER, ImmutableList.of(key("cat"), key("dog")));
  }

  @Test
  public void createIfAbsentConcurrentWithGet() throws InterruptedException {
    int numIters = 50;
    final SkyKey key = key("key");
    for (int i = 0; i < numIters; i++) {
      Thread t =
          new Thread(
              wrapper.wrap(
                  new Runnable() {
                    @Override
                    public void run() {
                      try {
                        graph.get(null, Reason.OTHER, key);
                      } catch (InterruptedException e) {
                        throw new IllegalStateException(e);
                      }
                    }
                  }));
      t.start();
      assertThat(graph.createIfAbsentBatch(null, Reason.OTHER, ImmutableList.of(key))).isNotEmpty();
      graph.remove(key);
    }
  }

  @Test
  public void testCreateIfAbsentWithConcurrentGet() throws Exception {
    final SkyKey key = key("foo");
    int numThreads = 50;
    final CountDownLatch startThreads = new CountDownLatch(1);
    Runnable createRunnable =
        new Runnable() {
          @Override
          public void run() {
            TrackingAwaiter.INSTANCE.awaitLatchAndTrackExceptions(
                startThreads, "threads not started");
            try {
              graph.createIfAbsentBatch(null, Reason.OTHER, ImmutableList.of(key));
            } catch (InterruptedException e) {
              throw new IllegalStateException(e);
            }
          }
        };
    Runnable noCreateRunnable =
        new Runnable() {
          @Override
          public void run() {
            TrackingAwaiter.INSTANCE.awaitLatchAndTrackExceptions(
                startThreads, "threads not started");
            try {
              graph.get(null, Reason.OTHER, key);
            } catch (InterruptedException e) {
              throw new IllegalStateException(e);
            }
          }
        };
    List<Thread> threads = new ArrayList<>(2 * numThreads);
    for (int i = 0; i < numThreads; i++) {
      Thread createThread = new Thread(createRunnable);
      createThread.start();
      threads.add(createThread);
      Thread noCreateThread = new Thread(noCreateRunnable);
      noCreateThread.start();
      threads.add(noCreateThread);
    }
    startThreads.countDown();
    for (Thread thread : threads) {
      thread.join();
    }
  }

  // Tests adding and removing Rdeps of a {@link NodeEntry} while a node transitions from
  // not done to done.
  @Test
  public void testAddRemoveRdeps() throws Exception {
    SkyKey key = key("foo");
    final NodeEntry entry =
        Iterables.getOnlyElement(
            graph.createIfAbsentBatch(null, Reason.OTHER, ImmutableList.of(key)).values());
    // These numbers are arbitrary.
    int numThreads = 50;
    int numKeys = numThreads;
    // One chunk will be used to add and remove rdeps before setting the node value.  The second
    // chunk of work will have the node value set and the last chunk will be to add and remove
    // rdeps after the value has been set.
    final int chunkSize = 40;
    final int numIterations = chunkSize * 2;
    // This latch is used to signal that the runnables have been submitted to the executor.
    final CountDownLatch waitForStart = new CountDownLatch(1);
    // This latch is used to signal to the main thread that we have begun the second chunk
    // for sufficiently many keys.  The minimum of numThreads and numKeys is used to prevent
    // thread starvation from causing a delay here.
    final CountDownLatch waitForAddedRdep = new CountDownLatch(numThreads);
    // This latch is used to guarantee that we set the node's value before we enter the third
    // chunk for any key.
    final CountDownLatch waitForSetValue = new CountDownLatch(1);
    ExecutorService pool = Executors.newFixedThreadPool(numThreads);
    // Add single rdep before transition to done.
    assertThat(entry.addReverseDepAndCheckIfDone(key("rdep")))
        .isEqualTo(DependencyState.NEEDS_SCHEDULING);
    List<SkyKey> rdepKeys = new ArrayList<>();
    for (int i = 0; i < numKeys; i++) {
      rdepKeys.add(key("rdep" + i));
    }
    graph.createIfAbsentBatch(null, Reason.OTHER, rdepKeys);
    for (int i = 0; i < numKeys; i++) {
      final int j = i;
      Runnable r =
          new Runnable() {
            @Override
            public void run() {
              try {
                waitForStart.await(TestUtils.WAIT_TIMEOUT_SECONDS, TimeUnit.SECONDS);
                assertThat(entry.addReverseDepAndCheckIfDone(key("rdep" + j)))
                    .isNotEqualTo(DependencyState.DONE);
                waitForAddedRdep.countDown();
                waitForSetValue.await(TestUtils.WAIT_TIMEOUT_SECONDS, TimeUnit.SECONDS);
                for (int k = chunkSize; k <= numIterations; k++) {
                  entry.removeReverseDep(key("rdep" + j));
                  entry.addReverseDepAndCheckIfDone(key("rdep" + j));
                  if (checkRdeps()) {
                    entry.getReverseDepsForDoneEntry();
                  }
                }
              } catch (InterruptedException e) {
                fail("Test failed: " + e.toString());
              }
            }
          };
      pool.execute(wrapper.wrap(r));
    }
    waitForStart.countDown();
    waitForAddedRdep.await(TestUtils.WAIT_TIMEOUT_SECONDS, TimeUnit.SECONDS);
    entry.setValue(new StringValue("foo1"), startingVersion, null);
    waitForSetValue.countDown();
    wrapper.waitForTasksAndMaybeThrow();
    assertThat(ExecutorUtil.interruptibleShutdown(pool)).isFalse();
    assertThat(graph.get(null, Reason.OTHER, key).getValue()).isEqualTo(new StringValue("foo1"));
    if (checkRdeps()) {
      assertThat(graph.get(null, Reason.OTHER, key).getReverseDepsForDoneEntry())
          .hasSize(numKeys + 1);
    }

    graph = getGraph(getNextVersion(startingVersion));
    NodeEntry sameEntry = Preconditions.checkNotNull(graph.get(null, Reason.OTHER, key));
    // Mark the node as dirty again and check that the reverse deps have been preserved.
    sameEntry.markDirty(DirtyType.CHANGE);
    startEvaluation(sameEntry);
    sameEntry.markRebuilding();
    sameEntry.setValue(new StringValue("foo2"), getNextVersion(startingVersion), null);
    assertThat(graph.get(null, Reason.OTHER, key).getValue()).isEqualTo(new StringValue("foo2"));
    if (checkRdeps()) {
      assertThat(graph.get(null, Reason.OTHER, key).getReverseDepsForDoneEntry())
          .hasSize(numKeys + 1);
    }
  }

  // Tests adding inflight nodes with a given key while an existing node with the same key
  // undergoes a transition from not done to done.
  @Test
  public void testAddingInflightNodes() throws Exception {
    int numThreads = 50;
    ExecutorService pool = Executors.newFixedThreadPool(numThreads);
    final int numKeys = 500;
    // Add each pair of keys 10 times.
    final Set<SkyKey> nodeCreated = Sets.newConcurrentHashSet();
    final Set<SkyKey> valuesSet = Sets.newConcurrentHashSet();
    for (int i = 0; i < 10; i++) {
      for (int j = 0; j < numKeys; j++) {
        for (int k = j + 1; k < numKeys; k++) {
          final int keyNum1 = j;
          final int keyNum2 = k;
          final SkyKey key1 = key("foo" + keyNum1);
          final SkyKey key2 = key("foo" + keyNum2);
          final Iterable<SkyKey> keys = ImmutableList.of(key1, key2);
          Runnable r =
              new Runnable() {
                @Override
                public void run() {
                  for (SkyKey key : keys) {
                    NodeEntry entry = null;
                    try {
                      entry = graph.get(null, Reason.OTHER, key);
                    } catch (InterruptedException e) {
                      throw new IllegalStateException(e);
                    }
                    if (entry == null) {
                      nodeCreated.add(key);
                    }
                  }
                  Map<SkyKey, ? extends NodeEntry> entries;
                  try {
                    entries = graph.createIfAbsentBatch(null, Reason.OTHER, keys);
                  } catch (InterruptedException e) {
                    throw new IllegalStateException(e);
                  }
                  for (Integer keyNum : ImmutableList.of(keyNum1, keyNum2)) {
                    SkyKey key = key("foo" + keyNum);
                    NodeEntry entry = entries.get(key);
                    // {@code entry.addReverseDepAndCheckIfDone(null)} should return
                    // NEEDS_SCHEDULING at most once.
                    try {
                      if (startEvaluation(entry).equals(DependencyState.NEEDS_SCHEDULING)) {
                        assertThat(valuesSet.add(key)).isTrue();
                        // Set to done.
                        entry.setValue(new StringValue("bar" + keyNum), startingVersion, null);
                        assertThat(entry.isDone()).isTrue();
                      }
                    } catch (InterruptedException e) {
                      throw new IllegalStateException(key + ", " + entry, e);
                    }
                  }
                  // This shouldn't cause any problems from the other threads.
                  try {
                    graph.createIfAbsentBatch(null, Reason.OTHER, keys);
                  } catch (InterruptedException e) {
                    throw new IllegalStateException(e);
                  }
                }
              };
          pool.execute(wrapper.wrap(r));
        }
      }
    }
    wrapper.waitForTasksAndMaybeThrow();
    assertThat(ExecutorUtil.interruptibleShutdown(pool)).isFalse();
    // Check that all the values are as expected.
    for (int i = 0; i < numKeys; i++) {
      SkyKey key = key("foo" + i);
      assertThat(nodeCreated).contains(key);
      assertThat(valuesSet).contains(key);
      assertThat(graph.get(null, Reason.OTHER, key).getValue())
          .isEqualTo(new StringValue("bar" + i));
      assertThat(graph.get(null, Reason.OTHER, key).getVersion()).isEqualTo(startingVersion);
    }
  }

  /**
   * Initially calling {@link NodeEntry#setValue} and then making sure concurrent calls to {@link
   * QueryableGraph#get} and {@link QueryableGraph#getBatch} do not interfere with the node.
   */
  @Test
  public void testDoneToDirty() throws Exception {
    final int numKeys = 1000;
    int numThreads = 50;
    final int numBatchRequests = 100;
    // Create a bunch of done nodes.
    ArrayList<SkyKey> keys = new ArrayList<>();
    for (int i = 0; i < numKeys; i++) {
      keys.add(key("foo" + i));
    }
    Map<SkyKey, ? extends NodeEntry> entries = graph.createIfAbsentBatch(null, Reason.OTHER, keys);
    for (int i = 0; i < numKeys; i++) {
      NodeEntry entry = entries.get(key("foo" + i));
      startEvaluation(entry);
      entry.setValue(new StringValue("bar"), startingVersion, null);
    }

    assertThat(graph.get(null, Reason.OTHER, key("foo" + 0))).isNotNull();
    graph = getGraph(getNextVersion(startingVersion));
    assertThat(graph.get(null, Reason.OTHER, key("foo" + 0))).isNotNull();
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
      Runnable r1 =
          new Runnable() {
            @Override
            public void run() {
              try {
                makeBatchCountDownLatch.await();
                getBatchCountDownLatch.await();
                getCountDownLatch.await();
              } catch (InterruptedException e) {
                throw new AssertionError(e);
              }
              NodeEntry entry = null;
              try {
                entry = graph.get(null, Reason.OTHER, key("foo" + keyNum));
              } catch (InterruptedException e) {
                throw new IllegalStateException(e);
              }
              try {
                entry.markDirty(DirtyType.CHANGE);

                // Make some changes, like adding a dep and rdep.
                entry.addReverseDepAndCheckIfDone(key("rdep"));
                entry.markRebuilding();
                addTemporaryDirectDep(entry, key("dep"));
                entry.signalDep();

                entry.setValue(
                    new StringValue("bar" + keyNum), getNextVersion(startingVersion), null);
              } catch (InterruptedException e) {
                throw new IllegalStateException(keyNum + ", " + entry, e);
              }
            }
          };

      // Start a bunch of get() calls while the node transitions from dirty to done and back.
      Runnable r2 =
          new Runnable() {
            @Override
            public void run() {
              try {
                makeBatchCountDownLatch.await();
              } catch (InterruptedException e) {
                throw new AssertionError(e);
              }
              NodeEntry entry = null;
              try {
                entry = graph.get(null, Reason.OTHER, key("foo" + keyNum));
              } catch (InterruptedException e) {
                throw new IllegalStateException(e);
              }
              assertThat(entry).isNotNull();
              // Requests for the value are made at the same time that the version increments from
              // the base. Check that there is no problem in requesting the version and that the
              // number is sane.
              assertThat(entry.getVersion())
                  .isAnyOf(startingVersion, getNextVersion(startingVersion));
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
      Runnable r3 =
          new Runnable() {
            @Override
            public void run() {
              try {
                makeBatchCountDownLatch.await();
              } catch (InterruptedException e) {
                throw new AssertionError(e);
              }
              Map<SkyKey, ? extends NodeEntry> batchMap = null;
              try {
                batchMap = graph.getBatch(null, Reason.OTHER, batch);
              } catch (InterruptedException e) {
                throw new IllegalStateException(e);
              }
              getBatchCountDownLatch.countDown();
              assertThat(batchMap).hasSize(batch.size());
              for (NodeEntry entry : batchMap.values()) {
                // Batch requests are made at the same time that the version increments from the
                // base. Check that there is no problem in requesting the version and that the
                // number is sane.
                assertThat(entry.getVersion())
                    .isAnyOf(startingVersion, getNextVersion(startingVersion));
              }
            }
          };
      pool3.execute(wrapper.wrap(r3));
    }
    wrapper.waitForTasksAndMaybeThrow();
    assertThat(ExecutorUtil.interruptibleShutdown(pool1)).isFalse();
    assertThat(ExecutorUtil.interruptibleShutdown(pool2)).isFalse();
    assertThat(ExecutorUtil.interruptibleShutdown(pool3)).isFalse();
    for (int i = 0; i < numKeys; i++) {
      NodeEntry entry = graph.get(null, Reason.OTHER, key("foo" + i));
      assertThat(entry.getValue()).isEqualTo(new StringValue("bar" + i));
      assertThat(entry.getVersion()).isEqualTo(getNextVersion(startingVersion));
      if (checkRdeps()) {
        for (SkyKey key : entry.getReverseDepsForDoneEntry()) {
          assertThat(key).isEqualTo(key("rdep"));
        }
      }
      for (SkyKey key : entry.getDirectDeps()) {
        assertThat(key).isEqualTo(key("dep"));
      }
    }
  }

  private static DependencyState startEvaluation(NodeEntry entry) throws InterruptedException {
    return entry.addReverseDepAndCheckIfDone(null);
  }

  private static void addTemporaryDirectDep(NodeEntry entry, SkyKey key) {
    GroupedListHelper<SkyKey> helper = new GroupedListHelper<>();
    helper.add(key);
    entry.addTemporaryDirectDeps(helper);
  }
}
