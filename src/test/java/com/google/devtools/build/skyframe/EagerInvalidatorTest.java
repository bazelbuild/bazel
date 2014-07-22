// Copyright 2014 Google Inc. All rights reserved.
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
import static com.google.devtools.build.skyframe.GraphTester.CONCATENATE;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Sets;
import com.google.common.testing.GcFinalization;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.testutil.MoreAsserts;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.skyframe.GraphTester.StringNode;
import com.google.devtools.build.skyframe.InvalidatingNodeVisitor.DirtyingInvalidationState;
import com.google.devtools.build.skyframe.InvalidatingNodeVisitor.InvalidationState;
import com.google.devtools.build.skyframe.InvalidatingNodeVisitor.InvalidationType;

import org.junit.Test;
import org.junit.experimental.runners.Enclosed;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.lang.ref.WeakReference;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

import javax.annotation.Nullable;

/**
 * Tests for {@link InvalidatingNodeVisitor}.
 */
@RunWith(Enclosed.class)
public class EagerInvalidatorTest {
  protected InMemoryGraph graph;
  protected GraphTester tester = new GraphTester();
  protected InvalidationState state = newInvalidationState();
  protected AtomicReference<InvalidatingNodeVisitor> visitor = new AtomicReference<>();

  private int graphVersion = 0;

  // The following three methods should be abstract, but junit4 does not allow us to run inner
  // classes in an abstract outer class. Thus, we provide implementations. These methods will never
  // be run because only the inner classes, annotated with @RunWith, will actually be executed.
  NodeProgressReceiver.InvalidationState expectedState() {
    throw new UnsupportedOperationException();
  }

  @SuppressWarnings("unused") // Overridden by subclasses.
  void invalidate(DirtiableGraph graph, NodeProgressReceiver invalidationReceiver,
      NodeKey... keys) throws InterruptedException { throw new UnsupportedOperationException(); }

  boolean gcExpected() { throw new UnsupportedOperationException(); }

  private boolean isInvalidated(NodeKey key) {
    NodeEntry entry = graph.get(key);
    if (gcExpected()) {
      return entry == null;
    } else {
      return entry == null || entry.isDirty();
    }
  }

  private void assertChanged(NodeKey key) {
    NodeEntry entry = graph.get(key);
    if (gcExpected()) {
      assertNull(entry);
    } else {
      assertTrue(entry.isChanged());
    }
  }

  private void assertDirtyAndNotChanged(NodeKey key) {
    NodeEntry entry = graph.get(key);
    if (gcExpected()) {
      assertNull(entry);
    } else {
      assertTrue(entry.isDirty());
      assertFalse(entry.isChanged());
    }

  }

  protected InvalidationState newInvalidationState() {
    throw new UnsupportedOperationException("Sublcasses must override");
  }

  protected InvalidationType defaultInvalidationType() {
    throw new UnsupportedOperationException("Sublcasses must override");
  }

  // Convenience method for eval-ing a single node.
  protected Node eval(boolean keepGoing, NodeKey key) throws InterruptedException {
    NodeKey[] keys = { key };
    return eval(keepGoing, keys).get(key);
  }

  protected <T extends Node> UpdateResult<T> eval(boolean keepGoing, NodeKey... keys)
    throws InterruptedException {
    Reporter reporter = new Reporter();
    ParallelEvaluator evaluator = new ParallelEvaluator(graph, graphVersion++,
        ImmutableMap.of(GraphTester.NODE_TYPE, tester.createDelegatingNodeBuilder()),
        reporter, new AutoUpdatingGraph.EmittedEventState(), keepGoing, 200, null);
    return evaluator.eval(ImmutableList.copyOf(keys));
  }

  protected void invalidateWithoutError(@Nullable NodeProgressReceiver invalidationReceiver,
      NodeKey... keys) throws InterruptedException {
    invalidate(graph, invalidationReceiver, keys);
    assertTrue(state.isEmpty());
  }

  protected void set(String name, String value) {
    tester.set(name, new StringNode(value));
  }

  protected NodeKey nodeKey(String name) {
    return GraphTester.toNodeKeys(name)[0];
  }

  protected void assertNodeValue(String name, String expectedValue) throws InterruptedException {
    StringNode node = (StringNode) eval(false, nodeKey(name));
    assertEquals(expectedValue, node.getValue());
  }

  @Test
  public void receiverWorks() throws Exception {
    final Set<String> invalidated = Sets.newConcurrentHashSet();
    NodeProgressReceiver receiver = new NodeProgressReceiver() {
      @Override
      public void invalidated(Node node, InvalidationState state) {
        Preconditions.checkState(state == expectedState());
        invalidated.add(((StringNode) node).getValue());
      }
      
      @Override
      public void enqueueing(NodeKey nodeKey) {
        throw new UnsupportedOperationException();        
      }

      @Override
      public void evaluated(NodeKey nodeKey, Node node, EvaluationState state) {
        throw new UnsupportedOperationException();
      }
    };
    graph = new InMemoryGraph();
    set("a", "a");
    set("b", "b");
    tester.getOrCreate("ab").addDependency("a").addDependency("b")
        .setComputedValue(CONCATENATE);
    assertNodeValue("ab", "ab");

    set("a", "c");
    invalidateWithoutError(receiver, nodeKey("a"));
    MoreAsserts.assertContentsAnyOrder(invalidated, "a", "ab");
    assertNodeValue("ab", "cb");
    set("b", "d");
    invalidateWithoutError(receiver, nodeKey("b"));
    MoreAsserts.assertContentsAnyOrder(invalidated, "a", "ab", "b", "cb");
  }

  @Test
  public void receiverIsNotNotifiedAboutNodesInError() throws Exception {
    final Set<String> invalidated = Sets.newConcurrentHashSet();
    NodeProgressReceiver receiver = new NodeProgressReceiver() {
      @Override
      public void invalidated(Node node, InvalidationState state) {
        Preconditions.checkState(state == expectedState());
        invalidated.add(((StringNode) node).getValue());
      }
      
      @Override
      public void enqueueing(NodeKey nodeKey) {
        throw new UnsupportedOperationException();        
      }

      @Override
      public void evaluated(NodeKey nodeKey, Node node, EvaluationState state) {
        throw new UnsupportedOperationException();
      }
    };

    graph = new InMemoryGraph();
    set("a", "a");
    tester.getOrCreate("ab").addDependency("a").setHasError(true);
    eval(false, nodeKey("ab"));

    invalidateWithoutError(receiver, nodeKey("a"));
    MoreAsserts.assertContentsInOrder(invalidated, "a");  // "ab" is not present
  }

  @Test
  public void invalidateNodesNotInGraph() throws Exception {
    final Set<String> invalidated = Sets.newConcurrentHashSet();
    NodeProgressReceiver receiver = new NodeProgressReceiver() {
      @Override
      public void invalidated(Node node, InvalidationState state) {
        Preconditions.checkState(state == InvalidationState.DIRTY);
        invalidated.add(((StringNode) node).getValue());
      }
      
      @Override
      public void enqueueing(NodeKey nodeKey) {
        throw new UnsupportedOperationException();        
      }

      @Override
      public void evaluated(NodeKey nodeKey, Node node, EvaluationState state) {
        throw new UnsupportedOperationException();
      }
    };
    graph = new InMemoryGraph();
    invalidateWithoutError(receiver, nodeKey("a"));
    assertThat(invalidated).isEmpty();
    set("a", "a");
    assertNodeValue("a", "a");
    invalidateWithoutError(receiver, nodeKey("b"));
    assertThat(invalidated).isEmpty();
  }

  @Test
  public void invalidatedNodesAreGCedAsExpected() throws Exception {
    NodeKey key = GraphTester.nodeKey("a");
    HeavyNode heavyNode = new HeavyNode();
    WeakReference<HeavyNode> weakRef = new WeakReference<>(heavyNode);
    tester.set("a", heavyNode);

    graph = new InMemoryGraph();
    eval(false, key);
    invalidate(graph, null, key);

    tester = null;
    heavyNode = null;
    if (gcExpected()) {
      GcFinalization.awaitClear(weakRef);
    } else {
      // Not a reliable check, but better than nothing.
      System.gc();
      Thread.sleep(300);
      assertNotNull(weakRef.get());
    }
  }

  @Test
  public void reverseDepsConsistent() throws Exception {
    graph = new InMemoryGraph();
    set("a", "a");
    set("b", "b");
    set("c", "c");
    tester.getOrCreate("ab").addDependency("a").addDependency("b").setComputedValue(CONCATENATE);
    tester.getOrCreate("bc").addDependency("b").addDependency("c").setComputedValue(CONCATENATE);
    tester.getOrCreate("ab_c").addDependency("ab").addDependency("c")
        .setComputedValue(CONCATENATE);
    eval(false, nodeKey("ab_c"), nodeKey("bc"));

    MoreAsserts.assertContentsAnyOrder(graph.get(nodeKey("a")).getReverseDeps(),
        nodeKey("ab"));
    MoreAsserts.assertContentsAnyOrder(graph.get(nodeKey("b")).getReverseDeps(),
        nodeKey("ab"), nodeKey("bc"));
    MoreAsserts.assertContentsAnyOrder(graph.get(nodeKey("c")).getReverseDeps(),
        nodeKey("ab_c"), nodeKey("bc"));

    invalidateWithoutError(null, nodeKey("ab"));
    eval(false);

    // The graph nodes should be gone.
    assertTrue(isInvalidated(nodeKey("ab")));
    assertTrue(isInvalidated(nodeKey("abc")));

    // The reverse deps to ab and ab_c should have been removed.
    assertThat(graph.get(nodeKey("a")).getReverseDeps()).isEmpty();
    MoreAsserts.assertContentsAnyOrder(graph.get(nodeKey("b")).getReverseDeps(),
        nodeKey("bc"));
    MoreAsserts.assertContentsAnyOrder(graph.get(nodeKey("c")).getReverseDeps(),
        nodeKey("bc"));
  }

  @Test
  public void interruptChild() throws Exception {
    graph = new InMemoryGraph();
    int numNodes = 50; // More nodes than the invalidator has threads.
    final NodeKey[] family = new NodeKey[numNodes];
    final NodeKey child = GraphTester.nodeKey("child");
    final StringNode childNode = new StringNode("child");
    tester.set(child, childNode);
    family[0] = child;
    for (int i = 1; i < numNodes; i++) {
      NodeKey member = nodeKey(Integer.toString(i));
      tester.getOrCreate(member).addDependency(family[i - 1]).setComputedValue(CONCATENATE);
      family[i] = member;
    }
    NodeKey parent = GraphTester.nodeKey("parent");
    tester.getOrCreate(parent).addDependency(family[numNodes - 1]).setComputedValue(CONCATENATE);
    eval(/*keepGoing=*/false, parent);
    final Thread mainThread = Thread.currentThread();
    final AtomicReference<Node> badNode = new AtomicReference<>();
    NodeProgressReceiver receiver = new NodeProgressReceiver() {
      @Override
      public void invalidated(Node node, InvalidationState state) {
        if (node == childNode) {
          // Interrupt on the very first invalidate
          mainThread.interrupt();
        } else if (!childNode.equals(node)) {
          // All other invalidations should be of the same node.
          // Exceptions thrown here may be silently dropped, so keep track of errors ourselves.
          badNode.set(node);
        }
        try {
          assertTrue(visitor.get().awaitInterruptionForTestingOnly(2, TimeUnit.HOURS));
        } catch (InterruptedException e) {
          // We may well have thrown here because by the time we try to await, the main thread is
          // already interrupted.
        }
      }
      
      @Override
      public void enqueueing(NodeKey nodeKey) {
        throw new UnsupportedOperationException();        
      }

      @Override
      public void evaluated(NodeKey nodeKey, Node node, EvaluationState state) {
        throw new UnsupportedOperationException();
      }
    };
    try {
      invalidateWithoutError(receiver, child);
      fail();
    } catch (InterruptedException e) {
      // Expected.
    }
    assertNull(badNode.get());
    assertFalse(state.isEmpty());
    final Set<Node> invalidated = new HashSet<>();
    assertFalse(isInvalidated(parent));
    Node parentNode = graph.getNode(parent);
    assertNotNull(parentNode);
    receiver = new NodeProgressReceiver() {
      @Override
      public void invalidated(Node node, InvalidationState state) {
        invalidated.add(node);
      }
      
      @Override
      public void enqueueing(NodeKey nodeKey) {
        throw new UnsupportedOperationException();        
      }

      @Override
      public void evaluated(NodeKey nodeKey, Node node, EvaluationState state) {
        throw new UnsupportedOperationException();
      }
    };
    invalidateWithoutError(receiver);
    assertThat(invalidated).has().item(parentNode);
    assertThat(state.getInvalidationsForTesting()).isEmpty();

    // Regression test coverage:
    // "all pending nodes are marked changed on interrupt".
    assertTrue(isInvalidated(child));
    assertChanged(child);
    for (int i = 1; i < numNodes; i++) {
      assertDirtyAndNotChanged(family[i]);
    }
    assertDirtyAndNotChanged(parent);
  }

  private NodeKey[] constructLargeGraph(int size) {
    Random random = new Random(TestUtils.getRandomSeed());
    NodeKey[] nodes = new NodeKey[size];
    for (int i = 0; i < size; i++) {
      String iString = Integer.toString(i);
      NodeKey iKey = GraphTester.toNodeKey(iString);
      set(iString, iString);
      for (int j = 0; j < i; j++) {
        if (random.nextInt(3) == 0) {
          tester.getOrCreate(iKey).addDependency(Integer.toString(j));
        }
      }
      nodes[i] = iKey;
    }
    return nodes;
  }

  /** Return a subset of {@code nodes} that are still valid and so can be invalidated. */
  private Set<Pair<NodeKey, InvalidationType>> getNodesToInvalidate(NodeKey[] nodes) {
    Set<Pair<NodeKey, InvalidationType>> result = new HashSet<>();
    Random random = new Random(TestUtils.getRandomSeed());
    for (NodeKey node : nodes) {
      if (!isInvalidated(node)) {
        if (result.isEmpty() || random.nextInt(3) == 0) {
          // Add at least one node, if we can.
          result.add(Pair.of(node, defaultInvalidationType()));
        }
      }
    }
    return result;
  }

  @Test
  public void interruptThreadInReceiver() throws Exception {
    Random random = new Random(TestUtils.getRandomSeed());
    int graphSize = 1000;
    int tries = 5;
    graph = new InMemoryGraph();
    NodeKey[] nodes = constructLargeGraph(graphSize);
    eval(/*keepGoing=*/false, nodes);
    final Thread mainThread = Thread.currentThread();
    for (int run = 0; run < tries; run++) {
      Set<Pair<NodeKey, InvalidationType>> nodesToInvalidate = getNodesToInvalidate(nodes);
      // Find how many invalidations will actually be enqueued for invalidation in the first round,
      // so that we can interrupt before all of them are done.
      int validNodesToDo =
          Sets.difference(nodesToInvalidate, state.getInvalidationsForTesting()).size();
      for (Pair<NodeKey, InvalidationType> pair : state.getInvalidationsForTesting()) {
        if (!isInvalidated(pair.first)) {
          validNodesToDo++;
        }
      }
      int countDownStart = validNodesToDo > 0 ? random.nextInt(validNodesToDo) : 0;
      final CountDownLatch countDownToInterrupt = new CountDownLatch(countDownStart);
      final NodeProgressReceiver receiver = new NodeProgressReceiver() {
        @Override
        public void invalidated(Node node, InvalidationState state) {
          countDownToInterrupt.countDown();
          if (countDownToInterrupt.getCount() == 0) {
            mainThread.interrupt();
            try {
              // Wait for the main thread to be interrupted uninterruptibly, because the main thread
              // is going to interrupt us, and we don't want to get into an interrupt fight. Only
              // if we get interrupted without the main thread also being interrupted will this
              // throw an InterruptedException.
              TrackingAwaiter.waitAndMaybeThrowInterrupt(
                  visitor.get().getInterruptionLatchForTestingOnly(),
                  "Main thread was not interrupted");
            } catch (InterruptedException e) {
              throw new IllegalStateException(e);
            }
          }
        }

        @Override
        public void enqueueing(NodeKey nodeKey) {
          throw new UnsupportedOperationException();
        }

        @Override
        public void evaluated(NodeKey nodeKey, Node node, EvaluationState state) {
          throw new UnsupportedOperationException();
        }
      };
      try {
        invalidate(graph, receiver,
            Sets.newHashSet(
                Iterables.transform(nodesToInvalidate,
                    Pair.<NodeKey, InvalidationType>firstFunction())).toArray(new NodeKey[0]));
        assertThat(state.getInvalidationsForTesting()).isEmpty();
      } catch (InterruptedException e) {
        // Expected.
      }
      if (state.isEmpty()) {
        // Ran out of nodes to invalidate.
        break;
      }
    }

    eval(/*keepGoing=*/false, nodes);
  }

  private static class HeavyNode implements Node {
  }

  /**
   * Test suite for the deleting invalidator.
   */
  @RunWith(JUnit4.class)
  public static class DeletingInvalidatorTest extends EagerInvalidatorTest {
    @Override
    protected void invalidate(DirtiableGraph graph, NodeProgressReceiver invalidationReceiver,
        NodeKey... keys) throws InterruptedException {
      InvalidatingNodeVisitor invalidatingVisitor =
          EagerInvalidator.createVisitor(/*delete=*/true, graph, ImmutableList.copyOf(keys),
              invalidationReceiver, state);
      if (invalidatingVisitor != null) {
        visitor.set(invalidatingVisitor);
        invalidatingVisitor.run();
      }
    }

    @Override
    NodeProgressReceiver.InvalidationState expectedState() {
      return NodeProgressReceiver.InvalidationState.DELETED;
    }

    @Override
    boolean gcExpected() {
      return true;
    }

    @Override
    protected InvalidationState newInvalidationState() {
      return new InvalidatingNodeVisitor.DeletingInvalidationState();
    }

    @Override
    protected InvalidationType defaultInvalidationType() {
      return InvalidationType.DELETED;
    }
  }

  /**
   * Test suite for the dirtying invalidator.
   */
  @RunWith(JUnit4.class)
  public static class DirtyingInvalidatorTest extends EagerInvalidatorTest {
    @Override
    protected void invalidate(DirtiableGraph graph, NodeProgressReceiver invalidationReceiver,
        NodeKey... keys) throws InterruptedException {
      InvalidatingNodeVisitor invalidatingVisitor =
          EagerInvalidator.createVisitor(/*delete=*/false, graph, ImmutableList.copyOf(keys),
              invalidationReceiver, state);
      if (invalidatingVisitor != null) {
        visitor.set(invalidatingVisitor);
        invalidatingVisitor.run();
      }
    }

    @Override
    NodeProgressReceiver.InvalidationState expectedState() {
      return NodeProgressReceiver.InvalidationState.DIRTY;
    }

    @Override
    boolean gcExpected() {
      return false;
    }

    @Override
    protected InvalidationState newInvalidationState() {
      return new DirtyingInvalidationState();
    }

    @Override
    protected InvalidationType defaultInvalidationType() {
      return InvalidationType.CHANGED;
    }
  }
}
