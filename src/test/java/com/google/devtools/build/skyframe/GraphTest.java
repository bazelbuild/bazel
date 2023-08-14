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
import static java.util.concurrent.TimeUnit.SECONDS;
import static org.junit.Assert.fail;
import static org.junit.Assume.assumeTrue;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.concurrent.ExecutorUtil;
import com.google.devtools.build.lib.testutil.TestRunnableWrapper;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.skyframe.GraphTester.StringValue;
import com.google.devtools.build.skyframe.NodeEntry.DependencyState;
import com.google.devtools.build.skyframe.NodeEntry.DirtyType;
import com.google.devtools.build.skyframe.QueryableGraph.Reason;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import javax.annotation.Nullable;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Base class for tests on {@link ProcessableGraph} implementations. */
@RunWith(TestParameterInjector.class)
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

  /**
   * Can be overridden to return {@code false} if the graph under test does not support
   * incrementality.
   */
  protected boolean shouldTestIncrementality() {
    return true;
  }

  protected enum BatchMethod {
    NODE_BATCH {
      @Override
      public NodeBatch get(
          ProcessableGraph graph,
          @Nullable SkyKey requestor,
          Reason reason,
          Iterable<? extends SkyKey> keys)
          throws InterruptedException {
        return graph.getBatch(requestor, reason, keys);
      }
    },
    MAP {
      @Override
      public NodeBatch get(
          ProcessableGraph graph,
          @Nullable SkyKey requestor,
          Reason reason,
          Iterable<? extends SkyKey> keys)
          throws InterruptedException {
        return graph.getBatchMap(requestor, reason, keys)::get;
      }
    };

    public abstract NodeBatch get(
        ProcessableGraph graph,
        @Nullable SkyKey requestor,
        Reason reason,
        Iterable<? extends SkyKey> keys)
        throws InterruptedException;
  }

  @Before
  public void init() throws Exception {
    makeGraph();
    Version startingVersion = getStartingVersion();
    this.graph = getGraph(startingVersion);
    this.wrapper = new TestRunnableWrapper("GraphConcurrencyTest");
  }

  protected static SkyKey key(String name) {
    return GraphTester.toSkyKey(name);
  }

  @Test
  public void createIfAbsentBatch() throws Exception {
    SkyKey cat = key("cat");
    SkyKey dog = key("dog");

    NodeBatch batch = graph.createIfAbsentBatch(null, Reason.OTHER, ImmutableList.of(cat, dog));

    NodeEntry catNode = graph.get(null, Reason.OTHER, cat);
    NodeEntry dogNode = graph.get(null, Reason.OTHER, dog);
    assertThat(catNode).isNotNull();
    assertThat(dogNode).isNotNull();
    assertThat(batch.get(cat)).isSameInstanceAs(catNode);
    assertThat(batch.get(dog)).isSameInstanceAs(dogNode);
  }

  @Test
  public void createIfAbsentBatch_interveningCallToRemove() throws Exception {
    SkyKey key = key("key");
    NodeBatch batch = graph.createIfAbsentBatch(null, Reason.OTHER, ImmutableList.of(key));
    graph.remove(key);
    assertThat(batch.get(key)).isNotNull();
  }

  @Test
  public void getBatchAndGetBatchMapConsistency() throws Exception {
    SkyKey cat = key("cat");
    SkyKey dog = key("dog");
    ImmutableList<SkyKey> keys = ImmutableList.of(cat, dog);
    graph.createIfAbsentBatch(null, Reason.OTHER, keys);

    NodeBatch batch = graph.getBatch(null, Reason.OTHER, keys);
    Map<SkyKey, ? extends NodeEntry> batchMap = graph.getBatchMap(null, Reason.OTHER, keys);

    NodeEntry catEntry = batch.get(cat);
    NodeEntry dogEntry = batch.get(dog);
    assertThat(catEntry).isNotNull();
    assertThat(dogEntry).isNotNull();
    assertThat(batchMap.get(cat)).isSameInstanceAs(catEntry);
    assertThat(batchMap.get(dog)).isSameInstanceAs(dogEntry);
    assertThat(batchMap).hasSize(2);
  }

  @Test
  public void createIfAbsentBatchConcurrentWithGet() throws Exception {
    int numIters = 50;
    SkyKey key = key("key");
    for (int i = 0; i < numIters; i++) {
      Thread t =
          new Thread(
              wrapper.wrap(
                  () -> {
                    try {
                      graph.get(null, Reason.OTHER, key);
                    } catch (InterruptedException e) {
                      throw new IllegalStateException(e);
                    }
                  }));
      t.start();
      NodeBatch batch = graph.createIfAbsentBatch(null, Reason.OTHER, ImmutableList.of(key));
      assertThat(batch.get(key)).isNotNull();
      graph.remove(key);
    }
  }

  @Test
  public void testCreateIfAbsentBatchWithConcurrentGet() throws Exception {
    SkyKey key = key("foo");
    int numThreads = 50;
    CountDownLatch startThreads = new CountDownLatch(1);
    Runnable createRunnable =
        () -> {
          TrackingAwaiter.INSTANCE.awaitLatchAndTrackExceptions(
              startThreads, "threads not started");
          try {
            graph.createIfAbsentBatch(null, Reason.OTHER, ImmutableList.of(key));
          } catch (InterruptedException e) {
            throw new IllegalStateException(e);
          }
        };
    Runnable noCreateRunnable =
        () -> {
          TrackingAwaiter.INSTANCE.awaitLatchAndTrackExceptions(
              startThreads, "threads not started");
          try {
            graph.get(null, Reason.OTHER, key);
          } catch (InterruptedException e) {
            throw new IllegalStateException(e);
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
    NodeEntry entry = graph.createIfAbsentBatch(null, Reason.OTHER, ImmutableList.of(key)).get(key);
    // These numbers are arbitrary.
    int numThreads = 50;
    int numKeys = numThreads;
    // One chunk will be used to add and remove rdeps before setting the node value.  The second
    // chunk of work will have the node value set and the last chunk will be to add and remove
    // rdeps after the value has been set.
    int chunkSize = 40;
    int numIterations = chunkSize * 2;
    // This latch is used to signal that the runnables have been submitted to the executor.
    CountDownLatch waitForStart = new CountDownLatch(1);
    // This latch is used to signal to the main thread that we have begun the second chunk
    // for sufficiently many keys.  The minimum of numThreads and numKeys is used to prevent
    // thread starvation from causing a delay here.
    CountDownLatch waitForAddedRdep = new CountDownLatch(numThreads);
    // This latch is used to guarantee that we set the node's value before we enter the third
    // chunk for any key.
    CountDownLatch waitForSetValue = new CountDownLatch(1);
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
      int j = i;
      Runnable r =
          () -> {
            try {
              waitForStart.await(TestUtils.WAIT_TIMEOUT_SECONDS, SECONDS);
              assertThat(entry.addReverseDepAndCheckIfDone(key("rdep" + j)))
                  .isNotEqualTo(DependencyState.DONE);
              waitForAddedRdep.countDown();
              waitForSetValue.await(TestUtils.WAIT_TIMEOUT_SECONDS, SECONDS);
              for (int k = chunkSize; k <= numIterations; k++) {
                if (shouldTestIncrementality()) {
                  entry.removeReverseDep(key("rdep" + j));
                }
                entry.addReverseDepAndCheckIfDone(key("rdep" + j));
                if (shouldTestIncrementality()) {
                  entry.getReverseDepsForDoneEntry();
                }
              }
            } catch (InterruptedException e) {
              fail("Test failed: " + e);
            }
          };
      pool.execute(wrapper.wrap(r));
    }
    waitForStart.countDown();
    waitForAddedRdep.await(TestUtils.WAIT_TIMEOUT_SECONDS, SECONDS);
    entry.markRebuilding();
    entry.setValue(new StringValue("foo1"), startingVersion, null);
    waitForSetValue.countDown();
    wrapper.waitForTasksAndMaybeThrow();
    assertThat(ExecutorUtil.interruptibleShutdown(pool)).isFalse();
    assertThat(graph.get(null, Reason.OTHER, key).getValue()).isEqualTo(new StringValue("foo1"));

    if (!shouldTestIncrementality()) {
      return;
    }

    assertThat(graph.get(null, Reason.OTHER, key).getReverseDepsForDoneEntry())
        .hasSize(numKeys + 1);

    graph = getGraph(getNextVersion(startingVersion));
    NodeEntry sameEntry = Preconditions.checkNotNull(graph.get(null, Reason.OTHER, key));
    // Mark the node as dirty again and check that the reverse deps have been preserved.
    sameEntry.markDirty(DirtyType.CHANGE);
    startEvaluation(sameEntry);
    sameEntry.markRebuilding();
    sameEntry.setValue(new StringValue("foo2"), getNextVersion(startingVersion), null);
    assertThat(graph.get(null, Reason.OTHER, key).getValue()).isEqualTo(new StringValue("foo2"));
    assertThat(graph.get(null, Reason.OTHER, key).getReverseDepsForDoneEntry())
        .hasSize(numKeys + 1);
  }

  // Tests adding inflight nodes with a given key while an existing node with the same key
  // undergoes a transition from not done to done.
  @Test
  public void testAddingInflightNodes() throws Exception {
    int numThreads = 50;
    ExecutorService pool = Executors.newFixedThreadPool(numThreads);
    int numKeys = 500;
    // Add each pair of keys 10 times.
    Set<SkyKey> nodeCreated = Sets.newConcurrentHashSet();
    Set<SkyKey> valuesSet = Sets.newConcurrentHashSet();
    for (int i = 0; i < 10; i++) {
      for (int j = 0; j < numKeys; j++) {
        for (int k = j + 1; k < numKeys; k++) {
          int keyNum1 = j;
          int keyNum2 = k;
          SkyKey key1 = key("foo" + keyNum1);
          SkyKey key2 = key("foo" + keyNum2);
          ImmutableList<SkyKey> keys = ImmutableList.of(key1, key2);
          Runnable r =
              () -> {
                for (SkyKey key : keys) {
                  NodeEntry entry;
                  try {
                    entry = graph.get(null, Reason.OTHER, key);
                  } catch (InterruptedException e) {
                    throw new IllegalStateException(e);
                  }
                  if (entry == null) {
                    nodeCreated.add(key);
                  }
                }
                NodeBatch entries;
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
                      entry.markRebuilding();
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
   * QueryableGraph#get} and {@link QueryableGraph#getBatchMap} do not interfere with the node.
   */
  @Test
  public void testDoneToDirty(@TestParameter BatchMethod batchMethod) throws Exception {
    assumeTrue(shouldTestIncrementality());
    int numKeys = 1000;
    int numThreads = 50;
    int numBatchRequests = 100;
    // Create a bunch of done nodes.
    ArrayList<SkyKey> keys = new ArrayList<>();
    for (int i = 0; i < numKeys; i++) {
      keys.add(key("foo" + i));
    }
    NodeBatch entries = graph.createIfAbsentBatch(null, Reason.OTHER, keys);
    for (int i = 0; i < numKeys; i++) {
      NodeEntry entry = entries.get(key("foo" + i));
      startEvaluation(entry);
      entry.markRebuilding();
      entry.setValue(new StringValue("bar"), startingVersion, null);
    }

    assertThat(graph.get(null, Reason.OTHER, key("foo" + 0))).isNotNull();
    graph = getGraph(getNextVersion(startingVersion));
    assertThat(graph.get(null, Reason.OTHER, key("foo" + 0))).isNotNull();
    ExecutorService pool1 = Executors.newFixedThreadPool(numThreads);
    ExecutorService pool2 = Executors.newFixedThreadPool(numThreads);
    ExecutorService pool3 = Executors.newFixedThreadPool(numThreads);

    // Only start all the threads once the batch requests are ready.
    CountDownLatch makeBatchCountDownLatch = new CountDownLatch(numBatchRequests);
    // Do at least 5 single requests and batch requests before transitioning node.
    CountDownLatch getBatchCountDownLatch = new CountDownLatch(5);
    CountDownLatch getCountDownLatch = new CountDownLatch(5);

    SkyKey dep = key("dep");
    for (int i = 0; i < numKeys; i++) {
      int keyNum = i;
      // Transition the nodes from done to dirty and then back to done.
      Runnable r1 =
          () -> {
            try {
              makeBatchCountDownLatch.await();
              getBatchCountDownLatch.await();
              getCountDownLatch.await();
            } catch (InterruptedException e) {
              throw new AssertionError(e);
            }
            NodeEntry entry;
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
              entry.addSingletonTemporaryDirectDep(dep);
              Version nextVersion = getNextVersion(startingVersion);
              entry.signalDep(nextVersion, dep);

              entry.setValue(new StringValue("bar" + keyNum), nextVersion, null);
            } catch (InterruptedException e) {
              throw new IllegalStateException(keyNum + ", " + entry, e);
            }
          };

      // Start a bunch of get() calls while the node transitions from dirty to done and back.
      Runnable r2 =
          () -> {
            try {
              makeBatchCountDownLatch.await();
            } catch (InterruptedException e) {
              throw new AssertionError(e);
            }
            NodeEntry entry;
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
          };
      pool1.execute(wrapper.wrap(r1));
      pool2.execute(wrapper.wrap(r2));
    }
    Random r = new Random(TestUtils.getRandomSeed());
    // Start a bunch of getBatch() calls while the node transitions from dirty to done and back.
    for (int i = 0; i < numBatchRequests; i++) {
      List<SkyKey> batch = new ArrayList<>(numKeys);
      // Pseudorandomly uniformly sample the powerset of the keys.
      for (int j = 0; j < numKeys; j++) {
        if (r.nextBoolean()) {
          batch.add(key("foo" + j));
        }
      }
      makeBatchCountDownLatch.countDown();
      Runnable r3 =
          () -> {
            try {
              makeBatchCountDownLatch.await();
            } catch (InterruptedException e) {
              throw new AssertionError(e);
            }
            NodeBatch result;
            try {
              result = batchMethod.get(graph, null, Reason.OTHER, batch);
            } catch (InterruptedException e) {
              throw new IllegalStateException(e);
            }
            getBatchCountDownLatch.countDown();
            for (SkyKey key : batch) {
              // Batch requests are made at the same time that the version increments from the
              // base. Check that there is no problem in requesting the version and that the
              // number is sane.
              NodeEntry nodeEntry = result.get(key);
              assertThat(nodeEntry).isNotNull();
              assertThat(nodeEntry.getVersion())
                  .isAnyOf(startingVersion, getNextVersion(startingVersion));
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
      assertThat(entry.getReverseDepsForDoneEntry()).containsExactly(key("rdep"));
      assertThat(entry.getDirectDeps()).containsExactly(dep);
    }
  }

  protected static DependencyState startEvaluation(NodeEntry entry) throws InterruptedException {
    return entry.addReverseDepAndCheckIfDone(null);
  }
}
