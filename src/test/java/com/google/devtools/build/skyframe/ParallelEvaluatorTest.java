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

import static com.google.devtools.build.skyframe.GraphTester.CONCATENATE;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;
import static org.truth0.Truth.ASSERT;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.events.DelegatingErrorEventListener;
import com.google.devtools.build.lib.events.ErrorEventListener;
import com.google.devtools.build.lib.events.EventCollector;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.events.OutputFilter.RegexOutputFilter;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.testutil.JunitTestUtils;
import com.google.devtools.build.lib.testutil.MoreAsserts;
import com.google.devtools.build.lib.testutil.TestThread;
import com.google.devtools.build.skyframe.GraphTester.SomeErrorException;
import com.google.devtools.build.skyframe.GraphTester.StringNode;
import com.google.devtools.build.skyframe.NotifyingInMemoryGraph.EventType;
import com.google.devtools.build.skyframe.NotifyingInMemoryGraph.Listener;
import com.google.devtools.build.skyframe.NotifyingInMemoryGraph.Order;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

import javax.annotation.Nullable;

/**
 * Tests for {@link ParallelEvaluator}.
 */
@RunWith(JUnit4.class)
public class ParallelEvaluatorTest {
  protected ProcessableGraph graph;
  protected GraphTester tester = new GraphTester();

  private EventCollector eventCollector;
  private ErrorEventListener reporter;

  private NodeProgressReceiver revalidationReceiver;

  @Before
  public void initializeReporter() {
    eventCollector = new EventCollector(EventKind.ALL_EVENTS);
    reporter = new Reporter(eventCollector);
  }

  private ParallelEvaluator makeEvaluator(ProcessableGraph graph,
      ImmutableMap<NodeType, ? extends NodeBuilder> builders, boolean keepGoing) {
    return new ParallelEvaluator(graph, /*graphVersion=*/0L,
        builders, reporter,  new AutoUpdatingGraph.EmittedEventState(), keepGoing,
        AutoUpdatingGraph.DEFAULT_THREAD_COUNT, revalidationReceiver);
  }

  /** Convenience method for eval-ing a single node. */
  protected Node eval(boolean keepGoing, NodeKey key) throws InterruptedException {
    return eval(keepGoing, ImmutableList.of(key)).get(key);
  }

  protected ErrorInfo evalNodeInError(NodeKey key) throws InterruptedException {
    return eval(true, ImmutableList.of(key)).getError(key);
  }

  protected <T extends Node> UpdateResult<T> eval(boolean keepGoing, NodeKey... keys)
      throws InterruptedException {
    return eval(keepGoing, ImmutableList.copyOf(keys));
  }

  protected <T extends Node> UpdateResult<T> eval(boolean keepGoing, Iterable<NodeKey> keys)
      throws InterruptedException {
    ParallelEvaluator evaluator = makeEvaluator(graph,
        ImmutableMap.of(GraphTester.NODE_TYPE, tester.createDelegatingNodeBuilder()),
        keepGoing);
    return evaluator.eval(keys);
  }

  protected GraphTester.TestNodeBuilder set(String name, String value) {
    return tester.set(name, new StringNode(value));
  }

  @Test
  public void smoke() throws Exception {
    graph = new InMemoryGraph();
    set("a", "a");
    set("b", "b");
    tester.getOrCreate("ab").addDependency("a").addDependency("b").setComputedValue(CONCATENATE);
    StringNode node = (StringNode) eval(false, GraphTester.toNodeKey("ab"));
    assertEquals("ab", node.getValue());
    JunitTestUtils.assertNoEvents(eventCollector);
  }

  /**
   * Test interruption handling when a long-running NodeBuilder gets interrupted.
   */
  @Test
  public void interruptedNodeBuilder() throws Exception {
    runInterruptionTest(new NodeBuilderFactory() {
      @Override
      public NodeBuilder create(final Semaphore threadStarted, final String[] errorMessage) {
        return new NodeBuilder() {
          @Override
          public Node build(NodeKey key, Environment env) throws InterruptedException {
            // Signal the waiting test thread that the evaluator thread has really started.
            threadStarted.release();

            // Simulate a NodeBuilder that runs for 10 seconds (this number was chosen arbitrarily).
            // The main thread should interrupt it shortly after it got started.
            Thread.sleep(10 * 1000);

            // Set an error message to indicate that the expected interruption didn't happen.
            // We can't use Assert.fail(String) on an async thread.
            errorMessage[0] = "NodeBuilder should have been interrupted";
            return null;
          }

          @Nullable
          @Override
          public String extractTag(NodeKey nodeKey) {
            return null;
          }
        };
      }
    });
  }

  /**
   * Test interruption handling when the Evaluator is in-between running NodeBuilders.
   *
   * <p>This is the point in time after a NodeBuilder requested a dependency which is not yet built
   * so the builder returned null to the Evaluator, and the latter is about to schedule evaluation
   * of the missing dependency but gets interrupted before the dependency's NodeBuilder could start.
   */
  @Test
  public void interruptedEvaluatorThread() throws Exception {
    runInterruptionTest(new NodeBuilderFactory() {
      @Override
      public NodeBuilder create(final Semaphore threadStarted, final String[] errorMessage) {
        return new NodeBuilder() {
          // No need to synchronize access to this field; we always request just one more
          // dependency, so it's only one NodeBuilder running at any time.
          private int nodeIdCounter = 0;

          @Override
          public Node build(NodeKey key, Environment env) {
            // Signal the waiting test thread that the Evaluator thread has really started.
            threadStarted.release();

            // Keep the evaluator busy until the test's thread gets scheduled and can
            // interrupt the Evaluator's thread.
            env.getDep(GraphTester.toNodeKey("a" + nodeIdCounter++));

            // This method never throws InterruptedException, therefore it's the responsibility
            // of the Evaluator to detect the interrupt and avoid calling subsequent NodeBuilders.
            return null;
          }

          @Nullable
          @Override
          public String extractTag(NodeKey nodeKey) {
            return null;
          }
        };
      }
    });
  }

  private void runPartialResultOnInterruption(boolean buildFastFirst) throws Exception {
    graph = new InMemoryGraph();
    // Two runs for fastKey's builder and one for the start of waitKey's builder.
    final CountDownLatch allNodesReady = new CountDownLatch(3); 
    final NodeKey waitKey = GraphTester.toNodeKey("wait");
    final NodeKey fastKey = GraphTester.toNodeKey("fast");
    NodeKey leafKey = GraphTester.toNodeKey("leaf");
    tester.getOrCreate(waitKey).setBuilder(new NodeBuilder() {
          @Override
          public Node build(NodeKey nodeKey, Environment env) throws InterruptedException {
            allNodesReady.countDown();
            Thread.sleep(10000);
            throw new AssertionError("Should have been interrupted");
          }

          @Override
          public String extractTag(NodeKey nodeKey) {
            return null;
          }
        });
    tester.getOrCreate(fastKey).setBuilder(new ChainedNodeBuilder(null, null, allNodesReady, false,
        new StringNode("fast"), ImmutableList.of(leafKey)));
    tester.set(leafKey, new StringNode("leaf"));
    if (buildFastFirst) {
      eval(/*keepGoing=*/false, fastKey);
    }
    final Set<NodeKey> receivedNodes = Sets.newConcurrentHashSet();
    revalidationReceiver = new NodeProgressReceiver() {
      @Override
      public void invalidated(Node node, InvalidationState state) {}

      @Override
      public void enqueueing(NodeKey key) {}

      @Override
      public void evaluated(NodeKey nodeKey, Node node, EvaluationState state) {
        receivedNodes.add(nodeKey);
      }
    };
    TestThread evalThread = new TestThread() {
      @Override
      public void runTest() throws Exception {
        try {
          eval(/*keepGoing=*/true, waitKey, fastKey);
          fail();
        } catch (InterruptedException e) {
          // Expected.
        }
      }
    };
    evalThread.start();
    assertTrue(allNodesReady.await(2, TimeUnit.SECONDS));
    evalThread.interrupt();
    evalThread.join(5000);
    assertFalse(evalThread.isAlive());
    if (buildFastFirst) {
      // If leafKey was already built, it is not reported to the receiver.
      MoreAsserts.assertContentsAnyOrder(receivedNodes, fastKey);
    } else {
      // On first time being built, leafKey is registered too.
      MoreAsserts.assertContentsAnyOrder(receivedNodes, fastKey, leafKey);
    }
  }

  @Test
  public void partialResultOnInterruption() throws Exception {
    runPartialResultOnInterruption(/*buildFastFirst=*/false);
  }

  @Test
  public void partialCachedResultOnInterruption() throws Exception {
    runPartialResultOnInterruption(/*buildFastFirst=*/true);
  }

  /**
   * Factory for NodeBuilders for interruption testing (see {@link #runInterruptionTest}).
   */
  private interface NodeBuilderFactory {
    /**
     * Creates a NodeBuilder suitable for a specific test scenario.
     *
     * @param threadStarted a latch which the returned NodeBuilder must
     *     {@link Semaphore#release() release} once it started (otherwise the test won't work)
     * @param errorMessage a single-element array; the NodeBuilder can put a error message in it
     *     to indicate that an assertion failed (calling {@code fail} from async thread doesn't
     *     work)
     */
    NodeBuilder create(final Semaphore threadStarted, final String[] errorMessage);
  }

  /**
   * Test that we can handle the Evaluator getting interrupted at various points.
   *
   * <p>This method creates an Evaluator with the specified NodeBuilder for GraphTested.NODE_TYPE,
   * then starts a thread, requests evaluation and asserts that evaluation started. It then
   * interrupts the Evaluator thread and asserts that it acknowledged the interruption.
   *
   * @param nodeBuilderFactory creates a NodeBuilder which may or may not handle interruptions
   *     (depending on the test)
   */
  private void runInterruptionTest(NodeBuilderFactory nodeBuilderFactory) throws Exception {
    final Semaphore threadStarted = new Semaphore(0);
    final Semaphore threadInterrupted = new Semaphore(0);
    final String[] wasError = new String[] { null };
    final ParallelEvaluator evaluator = makeEvaluator(new InMemoryGraph(),
        ImmutableMap.of(GraphTester.NODE_TYPE, nodeBuilderFactory.create(threadStarted, wasError)),
        false);

    Thread t = new Thread(new Runnable() {
        @Override
        public void run() {
          try {
            evaluator.eval(ImmutableList.of(GraphTester.toNodeKey("a")));

            // There's no real need to set an error here. If the thread is not interrupted then
            // threadInterrupted is not released and the test thread will fail to acquire it.
            wasError[0] = "evaluation should have been interrupted";
          } catch (InterruptedException e) {
            // This is the interrupt we are waiting for. It should come straight from the
            // evaluator (more precisely, the AbstractQueueVisitor).
            // Signal the waiting test thread that the interrupt was acknowledged.
            threadInterrupted.release();
          }
        }
    });

    // Start the thread and wait for a semaphore. This ensures that the thread was really started.
    // The timeout guarantees the test won't hang in case the thread can't start for some reason.
    // (The timeout was chosen arbitrarily; 500 ms looks like it should be enough.)
    t.start();
    assertTrue(threadStarted.tryAcquire(500, TimeUnit.MILLISECONDS));

    // Interrupt the thread and wait for a semaphore. This ensures that the thread was really
    // interrupted and this fact was acknowledged.
    // The timeout guarantees the test won't hang in case the thread can't acknowledge the
    // interruption for some reason.
    // (The timeout was chosen arbitrarily; 500 ms looks like it should be enough.)
    t.interrupt();
    assertTrue(threadInterrupted.tryAcquire(500, TimeUnit.MILLISECONDS));

    // The NodeBuilder may have reported an error.
    if (wasError[0] != null) {
      fail(wasError[0]);
    }

    // Wait for the thread to finish.
    // (The timeout was chosen arbitrarily; 500 ms looks like it should be enough.)
    t.join(500);
  }

  @Test
  public void unrecoverableError() throws Exception {
    class CustomRuntimeException extends RuntimeException {}
    final CustomRuntimeException expected = new CustomRuntimeException();

    final NodeBuilder builder = new NodeBuilder() {
      @Override
      @Nullable
      public Node build(NodeKey nodeKey, Environment env)
          throws NodeBuilderException, InterruptedException {
        throw expected;
      }

      @Override
      @Nullable
      public String extractTag(NodeKey nodeKey) {
        return null;
      }
    };

    final ParallelEvaluator evaluator = makeEvaluator(new InMemoryGraph(),
        ImmutableMap.of(GraphTester.NODE_TYPE, builder),
        false);

    NodeKey nodeToEval = GraphTester.toNodeKey("a");
    try {
      evaluator.eval(ImmutableList.of(nodeToEval));
    } catch (RuntimeException re) {
      assertTrue(re.getMessage()
          .contains("Unrecoverable error while evaluating node '" + nodeToEval.toString() + "'"));
      assertTrue(re.getCause() instanceof CustomRuntimeException);
    }
  }

  @Test
  public void simpleWarning() throws Exception {
    graph = new InMemoryGraph();
    set("a", "a").setWarning("warning on 'a'");
    StringNode node = (StringNode) eval(false, GraphTester.toNodeKey("a"));
    assertEquals("a", node.getValue());
    JunitTestUtils.assertContainsEvent(eventCollector, "warning on 'a'");
    JunitTestUtils.assertEventCount(1, eventCollector);
  }

  @Test
  public void warningMatchesRegex() throws Exception {
    graph = new InMemoryGraph();
    ((Reporter) reporter).setOutputFilter(RegexOutputFilter.forRegex("a"));
    set("example", "a value").setWarning("warning message");
    NodeKey a = GraphTester.toNodeKey("example");
    tester.getOrCreate(a).setTag("a");
    StringNode node = (StringNode) eval(false, a);
    assertEquals("a value", node.getValue());
    JunitTestUtils.assertContainsEvent(eventCollector, "warning message");
    JunitTestUtils.assertEventCount(1, eventCollector);
  }

  @Test
  public void warningMatchesRegexOnlyTag() throws Exception {
    graph = new InMemoryGraph();
    ((Reporter) reporter).setOutputFilter(RegexOutputFilter.forRegex("a"));
    set("a", "a value").setWarning("warning on 'a'");
    NodeKey a = GraphTester.toNodeKey("a");
    tester.getOrCreate(a).setTag("b");
    StringNode node = (StringNode) eval(false, a);
    assertEquals("a value", node.getValue());
    JunitTestUtils.assertEventCount(0, eventCollector);  }

  @Test
  public void warningDoesNotMatchRegex() throws Exception {
    graph = new InMemoryGraph();
    ((Reporter) reporter).setOutputFilter(RegexOutputFilter.forRegex("b"));
    set("a", "a").setWarning("warning on 'a'");
    NodeKey a = GraphTester.toNodeKey("a");
    tester.getOrCreate(a).setTag("a");
    StringNode node = (StringNode) eval(false, a);
    assertEquals("a", node.getValue());
    JunitTestUtils.assertEventCount(0, eventCollector);
  }

  /** Regression test: events from already-done node not replayed. */
  @Test
  public void eventFromDoneChildRecorded() throws Exception {
    graph = new InMemoryGraph();
    set("a", "a").setWarning("warning on 'a'");
    NodeKey a = GraphTester.toNodeKey("a");
    NodeKey top = GraphTester.toNodeKey("top");
    tester.getOrCreate(top).addDependency(a).setComputedValue(CONCATENATE);
    // Build a so that it is already in the graph.
    eval(false, a);
    JunitTestUtils.assertEventCount(1, eventCollector);
    eventCollector.clear();
    // Build top. The warning from a should be reprinted.
    eval(false, top);
    JunitTestUtils.assertEventCount(1, eventCollector);
    eventCollector.clear();
    // Build top again. The warning should have been stored in the node.
    eval(false, top);
    JunitTestUtils.assertEventCount(1, eventCollector);
  }

  @Test
  public void shouldCreateErrorNodeWithRootCause() throws Exception {
    graph = new InMemoryGraph();
    set("a", "a");
    NodeKey parentErrorKey = GraphTester.toNodeKey("parent");
    NodeKey errorKey = GraphTester.toNodeKey("error");
    tester.getOrCreate(parentErrorKey).addDependency("a").addDependency(errorKey)
    .setComputedValue(CONCATENATE);
    tester.getOrCreate(errorKey).setHasError(true);
    ErrorInfo error = evalNodeInError(parentErrorKey);
    MoreAsserts.assertContentsAnyOrder(error.getRootCauses(), errorKey);
  }

  @Test
  public void shouldBuildOneTarget() throws Exception {
    graph = new InMemoryGraph();
    set("a", "a");
    set("b", "b");
    NodeKey parentErrorKey = GraphTester.toNodeKey("parent");
    NodeKey errorFreeKey = GraphTester.toNodeKey("ab");
    NodeKey errorKey = GraphTester.toNodeKey("error");
    tester.getOrCreate(parentErrorKey).addDependency(errorKey).addDependency("a")
    .setComputedValue(CONCATENATE);
    tester.getOrCreate(errorKey).setHasError(true);
    tester.getOrCreate(errorFreeKey).addDependency("a").addDependency("b")
    .setComputedValue(CONCATENATE);
    UpdateResult<StringNode> result = eval(true, parentErrorKey, errorFreeKey);
    ErrorInfo error = result.getError(parentErrorKey);
    MoreAsserts.assertContentsAnyOrder(error.getRootCauses(), errorKey);
    StringNode abNode = result.get(errorFreeKey);
    assertEquals("ab", abNode.getValue());
  }

  @Test
  public void parentFailureDoesntAffectChild() throws Exception {
    graph = new InMemoryGraph();
    NodeKey parentKey = GraphTester.toNodeKey("parent");
    tester.getOrCreate(parentKey).setHasError(true);
    NodeKey childKey = GraphTester.toNodeKey("child");
    set("child", "onions");
    tester.getOrCreate(parentKey).addDependency(childKey).setComputedValue(CONCATENATE);
    UpdateResult<StringNode> result = eval(/*keepGoing=*/true, parentKey, childKey);
    // Child is guaranteed to complete successfully before parent can run (and fail),
    // since parent depends on it.
    StringNode childNode = result.get(childKey);
    Assert.assertNotNull(childNode);
    assertEquals("onions", childNode.getValue());
    ErrorInfo error = result.getError(parentKey);
    Assert.assertNotNull(error);
    MoreAsserts.assertContentsAnyOrder(error.getRootCauses(), parentKey);
  }

  @Test
  public void newParentOfErrorShouldHaveError() throws Exception {
    graph = new InMemoryGraph();
    NodeKey errorKey = GraphTester.toNodeKey("error");
    tester.getOrCreate(errorKey).setHasError(true);
    ErrorInfo error = evalNodeInError(errorKey);
    MoreAsserts.assertContentsAnyOrder(error.getRootCauses(), errorKey);
    NodeKey parentKey = GraphTester.toNodeKey("parent");
    tester.getOrCreate(parentKey).addDependency("error").setComputedValue(CONCATENATE);
    error = evalNodeInError(parentKey);
    MoreAsserts.assertContentsAnyOrder(error.getRootCauses(), errorKey);
  }

  @Test
  public void errorTwoLevelsDeep() throws Exception {
    graph = new InMemoryGraph();
    NodeKey parentKey = GraphTester.toNodeKey("parent");
    NodeKey errorKey = GraphTester.toNodeKey("error");
    tester.getOrCreate(errorKey).setHasError(true);
    tester.getOrCreate("mid").addDependency(errorKey).setComputedValue(CONCATENATE);
    tester.getOrCreate(parentKey).addDependency("mid").setComputedValue(CONCATENATE);
    ErrorInfo error = evalNodeInError(parentKey);
    MoreAsserts.assertContentsAnyOrder(error.getRootCauses(), errorKey);
  }

  /**
   * A recreation of BuildViewTest#testHasErrorRaceCondition.  Also similar to errorTwoLevelsDeep,
   * except here we request multiple toplevel nodes.
   */
  @Test
  public void errorPropagationToTopLevelNodes() throws Exception {
    graph = new InMemoryGraph();
    NodeKey topKey = GraphTester.toNodeKey("top");
    NodeKey midKey = GraphTester.toNodeKey("mid");
    NodeKey badKey = GraphTester.toNodeKey("bad");
    tester.getOrCreate(topKey).addDependency(midKey).setComputedValue(CONCATENATE);
    tester.getOrCreate(midKey).addDependency(badKey).setComputedValue(CONCATENATE);
    tester.getOrCreate(badKey).setHasError(true);
    UpdateResult<Node> result = eval(/*keepGoing=*/false, topKey, midKey);
    MoreAsserts.assertContentsAnyOrder(result.getError(midKey).getRootCauses(), badKey);
    // Do it again with keepGoing.  We should also see an error for the top key this time.
    result = eval(/*keepGoing=*/true, topKey, midKey);
    MoreAsserts.assertContentsAnyOrder(result.getError(midKey).getRootCauses(), badKey);
    MoreAsserts.assertContentsAnyOrder(result.getError(topKey).getRootCauses(), badKey);
  }

  @Test
  public void valueNotUsedInFailFastErrorRecovery() throws Exception {
    graph = new InMemoryGraph();
    NodeKey topKey = GraphTester.toNodeKey("top");
    NodeKey recoveryKey = GraphTester.toNodeKey("midRecovery");
    NodeKey badKey = GraphTester.toNodeKey("bad");

    tester.getOrCreate(topKey).addDependency(recoveryKey).setComputedValue(CONCATENATE);
    tester.getOrCreate(recoveryKey).addErrorDependency(badKey, new StringNode("i recovered"))
        .setComputedValue(CONCATENATE);
    tester.getOrCreate(badKey).setHasError(true);

    UpdateResult<Node> result = eval(/*keepGoing=*/true, ImmutableList.of(recoveryKey));
    ASSERT.that(result.errorMap()).isEmpty();
    assertTrue(result.hasError());
    assertEquals(new StringNode("i recovered"), result.get(recoveryKey));

    result = eval(/*keepGoing=*/false, ImmutableList.of(topKey));
    assertTrue(result.hasError());
    ASSERT.that(result.keyNames()).isEmpty();
    assertEquals(1, result.errorMap().size());
    assertNotNull(result.getError(topKey).getException());
  }

  /**
   * Regression test: "clearing incomplete nodes on --keep_going build is racy".
   * Tests that if a node is requested on the first (non-keep-going) build and its child throws
   * an error, when the second (keep-going) build runs, there is not a race that keeps it as a
   * reverse dep of its children.
   */
  @Test
  public void raceClearingIncompleteNodes() throws Exception {
    NodeKey topKey = GraphTester.toNodeKey("top");
    final NodeKey midKey = GraphTester.toNodeKey("mid");
    NodeKey badKey = GraphTester.toNodeKey("bad");
    final AtomicBoolean waitForSecondCall = new AtomicBoolean(false);
    final TrackingAwaiter trackingAwaiter = new TrackingAwaiter();
    final CountDownLatch otherThreadWinning = new CountDownLatch(1);
    final AtomicReference<Thread> firstThread = new AtomicReference<>();
    graph = new NotifyingInMemoryGraph(new Listener() {
      @Override
      public void accept(NodeKey key, EventType type, Order order, Object context) {
        if (!waitForSecondCall.get()) {
          return;
        }
        if (key.equals(midKey)) {
          if (type == EventType.CREATE_IF_ABSENT) {
            // The first thread to create midKey will not be the first thread to add a reverse dep
            // to it.
            firstThread.compareAndSet(null, Thread.currentThread());
            return;
          }
          if (type == EventType.ADD_REVERSE_DEP) {
            if (order == Order.BEFORE && Thread.currentThread().equals(firstThread.get())) {
              // If this thread created midKey, block until the other thread adds a dep on it.
              trackingAwaiter.awaitLatchAndTrackExceptions(otherThreadWinning,
                  "other thread didn't pass this one");
            } else if (order == Order.AFTER && !Thread.currentThread().equals(firstThread.get())) {
              // This thread has added a dep. Allow the other thread to proceed.
              otherThreadWinning.countDown();
            }
          }
        }
      }
    });
    tester.getOrCreate(topKey).addDependency(midKey).setComputedValue(CONCATENATE);
    tester.getOrCreate(midKey).addDependency(badKey).setComputedValue(CONCATENATE);
    tester.getOrCreate(badKey).setHasError(true);
    UpdateResult<Node> result = eval(/*keepGoing=*/false, topKey, midKey);
    MoreAsserts.assertContentsAnyOrder(result.getError(midKey).getRootCauses(), badKey);
    waitForSecondCall.set(true);
    result = eval(/*keepGoing=*/true, topKey, midKey);
    trackingAwaiter.assertNoErrors();
    assertNotNull(firstThread.get());
    assertEquals(0, otherThreadWinning.getCount());
    MoreAsserts.assertContentsAnyOrder(result.getError(midKey).getRootCauses(), badKey);
    MoreAsserts.assertContentsAnyOrder(result.getError(topKey).getRootCauses(), badKey);
  }

  /**
   * Regression test: IllegalStateException on hasInflightParent check.
   * A parent node may be signaled by its child, realize it is done, restart itself, and build, all
   * before the child actually throws an exception and stops the threadpool. To test this, we have
   * the following sequence, with the node top depending on error and on the many nodes [deps]:
   * <pre>
   * 0.  [deps] are all built (before the main evaluation).
   * 1.  top requests its children.
   * 2.  top's builder exits, and ParallelEvaluator registers all of top's new deps.
   * 2'. (Concurrent with 2.) Since it was requested by top, error builds and throws an exception,
   *     as well as outputting a warning.
   * 3'. error is written to the graph, signaling top that it is done. If top is still engaged in
   *     registering its deps, this signal will make top ready when it finishes registering them,
   *     causing top to restart its builder.
   * 4'. reporter is called to output error's warning. It blocks on top's builder's second run. If
   *     error signaled top after top had finished, this second run will not happen, and so the
   *     reporter will give up. In that case, this run did not exercise the desired codepath, and so
   *     it will be run again. In testing, this happened approximately 1% of the time.
   * 5.  top's builder restarts, and blocks on the threadpool catching an exception.
   * 6.  reporter finishes. ParallelEvaluator throws a SchedulerException.
   * 7.  top finishes. Its value is written to the graph, but the SchedulerException thrown by
   *     ParallelEvaluator on its behalf is ignored, since the threadpool is shutting down.
   * 8.  The exception thrown by error is bubbled up the graph.
   * </pre>
   *
   * A time diagram (time flows down, and 2', 3', 4' take place concurrently with 2):
   * <pre>
   *                         0
   *                         |
   *                         1
   *                         |
   *                         2- -2'
   *                         |   |
   *                         |   3'
   *                         |   |
   *                         5   4'
   *                         \   /
   *                           6
   *                           |
   *                           7
   *                           |
   *                           8
   * </pre>
   */
  @Test
  public void slowChildCleanup() throws Exception {
    // Node to be built. It will be signaled to restart its builder before it has finished
    // registering its deps.
    final NodeKey top = GraphTester.toNodeKey("top");
    // Dep that blocks before it acknowledges being added as a dep by top, so the errorKey node has
    // time to signal top.
    final NodeKey slowAddingDep = GraphTester.toNodeKey("slowAddingDep");
    final StringNode depNode = new StringNode("dep");
    tester.set(slowAddingDep, depNode);
    final CountDownLatch topRestartedBuild = new CountDownLatch(1);
    final CountDownLatch topSignaled = new CountDownLatch(1);
    final TrackingAwaiter trackingAwaiter = new TrackingAwaiter();
    final AtomicBoolean topBuilderEntered = new AtomicBoolean(false);
    graph = new NotifyingInMemoryGraph(new Listener() {
      @Override
      public void accept(NodeKey key, EventType type, Order order, Object context) {
        if (key.equals(top) && type == EventType.SIGNAL && order == Order.AFTER) {
          // top is signaled by errorKey (since slowAddingDep is blocking), so slowAddingDep is now
          // free to acknowledge top as a parent.
          topSignaled.countDown();
          return;
        }
        if (key.equals(slowAddingDep) && type == EventType.ADD_REVERSE_DEP
            && top.equals(context) && order == Order.BEFORE) {
          // If top is trying to declare a dep on slowAddingDep, wait until errorKey has signaled
          // top. Then this add dep will return DONE and top will be signaled, making it ready, so
          // it will be enqueued.
          trackingAwaiter.awaitLatchAndTrackExceptions(topSignaled,
              "error key didn't signal top in time");
        }
      }
    });
    // Node that will throw an error when it is built, but will wait to actually throw the error
    // until top's builder restarts. We enforce the wait by having the listener to which it reports
    // its warning block until top's builder restarts.
    final NodeKey errorKey = GraphTester.nodeKey("error");
    String warningText = "warning text";
    tester.getOrCreate(errorKey).setHasError(true).setWarning(warningText);
    // On its first run, top's builder just requests both its deps. On its second run, the builder
    // waits for an exception to have been thrown (by errorKey) and then returns.
    NodeBuilder topBuilder = new NodeBuilder() {
      @Override
      public Node build(NodeKey key, NodeBuilder.Environment env) {
        // The reporter will be given errorKey's warning to emit when it is requested as a dep
        // below, if errorKey is already built, so we release the reporter's latch beforehand.
        boolean firstTime = topBuilderEntered.compareAndSet(false, true);
        if (!firstTime) {
          topRestartedBuild.countDown();
        }
        assertNull(env.getDep(errorKey));
        assertEquals(depNode, env.getDep(slowAddingDep));
        if (firstTime) {
          return null;
        }
        trackingAwaiter.awaitLatchAndTrackExceptions(env.getExceptionLatchForTesting(),
            "top did not get exception in time");
        return null;
      }

      @Override
      public String extractTag(NodeKey nodeKey) {
        return nodeKey.toString();
      }
    };
    reporter = new DelegatingErrorEventListener(reporter) {
      @Override
      public void warn(Location location, String message) {
        super.warn(location, message);
        trackingAwaiter.awaitLatchAndTrackExceptions(topRestartedBuild,
            "top's builder did not restart in time");
      }
    };
    tester.getOrCreate(top).setBuilder(topBuilder);
    // Make sure slowAddingDep is already in the graph, so it will be DONE.
    eval(/*keepGoing=*/false, slowAddingDep);
    UpdateResult<StringNode> result = eval(/*keepGoing=*/false, ImmutableList.of(top));
    ASSERT.that(result.keyNames()).isEmpty(); // No successfully evaluated nodes.
    ErrorInfo errorInfo = result.getError(top);
    MoreAsserts.assertContentsAnyOrder(errorInfo.getRootCauses(), errorKey);
    JunitTestUtils.assertContainsEvent(eventCollector, warningText);
    assertTrue(topBuilderEntered.get());
    trackingAwaiter.assertNoErrors();
    assertEquals(0, topRestartedBuild.getCount());
    assertEquals(0, topSignaled.getCount());
  }

  @Test
  public void multipleRootCauses() throws Exception {
    graph = new InMemoryGraph();
    NodeKey parentKey = GraphTester.toNodeKey("parent");
    NodeKey errorKey = GraphTester.toNodeKey("error");
    NodeKey errorKey2 = GraphTester.toNodeKey("error2");
    NodeKey errorKey3 = GraphTester.toNodeKey("error3");
    tester.getOrCreate(errorKey).setHasError(true);
    tester.getOrCreate(errorKey2).setHasError(true);
    tester.getOrCreate(errorKey3).setHasError(true);
    tester.getOrCreate("mid").addDependency(errorKey).addDependency(errorKey2)
      .setComputedValue(CONCATENATE);
    tester.getOrCreate(parentKey)
      .addDependency("mid").addDependency(errorKey2).addDependency(errorKey3)
      .setComputedValue(CONCATENATE);
    ErrorInfo error = evalNodeInError(parentKey);
    MoreAsserts.assertContentsAnyOrder(error.getRootCauses(),
        errorKey, errorKey2, errorKey3);
  }

  @Test
  public void rootCauseWithNoKeepGoing() throws Exception {
    graph = new InMemoryGraph();
    NodeKey parentKey = GraphTester.toNodeKey("parent");
    NodeKey errorKey = GraphTester.toNodeKey("error");
    tester.getOrCreate(errorKey).setHasError(true);
    tester.getOrCreate("mid").addDependency(errorKey).setComputedValue(CONCATENATE);
    tester.getOrCreate(parentKey).addDependency("mid").setComputedValue(CONCATENATE);
    UpdateResult<StringNode> result = eval(false, ImmutableList.of(parentKey));
    Map.Entry<NodeKey, ErrorInfo> error = Iterables.getOnlyElement(result.errorMap().entrySet());
    assertEquals(parentKey, error.getKey());
    MoreAsserts.assertContentsAnyOrder(error.getValue().getRootCauses(), errorKey);
  }

  @Test
  public void noKeepGoingAfterKeepGoingFails() throws Exception {
    graph = new InMemoryGraph();
    NodeKey errorKey = GraphTester.toNodeKey("my_error_node");
    tester.getOrCreate(errorKey).setHasError(true);
    NodeKey parentKey = GraphTester.toNodeKey("parent");
    tester.getOrCreate(parentKey).addDependency(errorKey);
    ErrorInfo error = evalNodeInError(parentKey);
    MoreAsserts.assertContentsAnyOrder(error.getRootCauses(), errorKey);
    NodeKey[] list = { parentKey };
    UpdateResult<StringNode> result = eval(false, list);
    ErrorInfo errorInfo = result.getError();
    assertEquals(errorKey, Iterables.getOnlyElement(errorInfo.getRootCauses()));
    assertEquals(errorKey.toString(), errorInfo.getException().getMessage());
  }

  @Test
  public void twoErrors() throws Exception {
    graph = new InMemoryGraph();
    NodeKey firstError = GraphTester.toNodeKey("error1");
    NodeKey secondError = GraphTester.toNodeKey("error2");
    CountDownLatch firstStart = new CountDownLatch(1);
    CountDownLatch secondStart = new CountDownLatch(1);
    tester.getOrCreate(firstError).setBuilder(new ChainedNodeBuilder(firstStart, secondStart,
        /*notifyFinish=*/null, /*waitForException=*/false, /*value=*/null,
        ImmutableList.<NodeKey>of()));
    tester.getOrCreate(secondError).setBuilder(new ChainedNodeBuilder(secondStart, firstStart,
        /*notifyFinish=*/null, /*waitForException=*/false, /*value=*/null,
        ImmutableList.<NodeKey>of()));
    UpdateResult<StringNode> result = eval(/*keepGoing=*/false, firstError, secondError);
    assertTrue(result.toString(), result.hasError());
    assertNotNull(result.toString(), result.getError(firstError));
    assertNotNull(result.toString(), result.getError(secondError));
    assertNotNull(result.toString(), result.getError());
  }

  @Test
  public void simpleCycle() throws Exception {
    graph = new InMemoryGraph();
    NodeKey aKey = GraphTester.toNodeKey("a");
    NodeKey bKey = GraphTester.toNodeKey("b");
    tester.getOrCreate(aKey).addDependency(bKey);
    tester.getOrCreate(bKey).addDependency(aKey);
    ErrorInfo errorInfo = eval(false, ImmutableList.of(aKey)).getError();
    assertEquals(null, errorInfo.getException());
    CycleInfo cycleInfo = Iterables.getOnlyElement(errorInfo.getCycleInfo());
    MoreAsserts.assertContentsInOrder(cycleInfo.getCycle(), aKey, bKey);
    assertTrue(cycleInfo.getPathToCycle().isEmpty());
  }

  @Test
  public void cycleWithHead() throws Exception {
    graph = new InMemoryGraph();
    NodeKey aKey = GraphTester.toNodeKey("a");
    NodeKey bKey = GraphTester.toNodeKey("b");
    NodeKey topKey = GraphTester.toNodeKey("top");
    NodeKey midKey = GraphTester.toNodeKey("mid");
    tester.getOrCreate(topKey).addDependency(midKey);
    tester.getOrCreate(midKey).addDependency(aKey);
    tester.getOrCreate(aKey).addDependency(bKey);
    tester.getOrCreate(bKey).addDependency(aKey);
    ErrorInfo errorInfo = eval(false, ImmutableList.of(topKey)).getError();
    assertEquals(null, errorInfo.getException());
    CycleInfo cycleInfo = Iterables.getOnlyElement(errorInfo.getCycleInfo());
    MoreAsserts.assertContentsInOrder(cycleInfo.getCycle(), aKey, bKey);
    MoreAsserts.assertContentsInOrder(cycleInfo.getPathToCycle(), topKey, midKey);
  }

  @Test
  public void selfEdgeWithHead() throws Exception {
    graph = new InMemoryGraph();
    NodeKey aKey = GraphTester.toNodeKey("a");
    NodeKey topKey = GraphTester.toNodeKey("top");
    NodeKey midKey = GraphTester.toNodeKey("mid");
    tester.getOrCreate(topKey).addDependency(midKey);
    tester.getOrCreate(midKey).addDependency(aKey);
    tester.getOrCreate(aKey).addDependency(aKey);
    ErrorInfo errorInfo = eval(false, ImmutableList.of(topKey)).getError();
    assertEquals(null, errorInfo.getException());
    CycleInfo cycleInfo = Iterables.getOnlyElement(errorInfo.getCycleInfo());
    MoreAsserts.assertContentsInOrder(cycleInfo.getCycle(), aKey);
    MoreAsserts.assertContentsInOrder(cycleInfo.getPathToCycle(), topKey, midKey);
  }

  @Test
  public void cycleWithKeepGoing() throws Exception {
    graph = new InMemoryGraph();
    NodeKey aKey = GraphTester.toNodeKey("a");
    NodeKey bKey = GraphTester.toNodeKey("b");
    NodeKey topKey = GraphTester.toNodeKey("top");
    NodeKey midKey = GraphTester.toNodeKey("mid");
    NodeKey goodKey = GraphTester.toNodeKey("good");
    StringNode goodValue = new StringNode("good");
    tester.set(goodKey, goodValue);
    tester.getOrCreate(topKey).addDependency(midKey);
    tester.getOrCreate(midKey).addDependency(aKey);
    tester.getOrCreate(aKey).addDependency(bKey);
    tester.getOrCreate(bKey).addDependency(aKey);
    UpdateResult<StringNode> result = eval(true, topKey, goodKey);
    assertEquals(goodValue, result.get(goodKey));
    assertEquals(null, result.get(topKey));
    ErrorInfo errorInfo = result.getError(topKey);
    CycleInfo cycleInfo = Iterables.getOnlyElement(errorInfo.getCycleInfo());
    MoreAsserts.assertContentsInOrder(cycleInfo.getCycle(), aKey, bKey);
    MoreAsserts.assertContentsInOrder(cycleInfo.getPathToCycle(), topKey, midKey);
  }

  @Test
  public void twoCycles() throws Exception {
    graph = new InMemoryGraph();
    NodeKey aKey = GraphTester.toNodeKey("a");
    NodeKey bKey = GraphTester.toNodeKey("b");
    NodeKey cKey = GraphTester.toNodeKey("c");
    NodeKey dKey = GraphTester.toNodeKey("d");
    NodeKey topKey = GraphTester.toNodeKey("top");
    tester.getOrCreate(topKey).addDependency(aKey).addDependency(cKey);
    tester.getOrCreate(aKey).addDependency(bKey);
    tester.getOrCreate(bKey).addDependency(aKey);
    tester.getOrCreate(cKey).addDependency(dKey);
    tester.getOrCreate(dKey).addDependency(cKey);
    UpdateResult<StringNode> result = eval(false, ImmutableList.of(topKey));
    assertEquals(null, result.get(topKey));
    ErrorInfo errorInfo = result.getError(topKey);
    Iterable<CycleInfo> cycles = CycleInfo.prepareCycles(topKey,
        ImmutableList.of(new CycleInfo(ImmutableList.of(aKey, bKey)),
        new CycleInfo(ImmutableList.of(cKey, dKey))));
    MoreAsserts.assertContains(cycles, Iterables.getOnlyElement(errorInfo.getCycleInfo()));
  }


  @Test
  public void twoCyclesKeepGoing() throws Exception {
    graph = new InMemoryGraph();
    NodeKey aKey = GraphTester.toNodeKey("a");
    NodeKey bKey = GraphTester.toNodeKey("b");
    NodeKey cKey = GraphTester.toNodeKey("c");
    NodeKey dKey = GraphTester.toNodeKey("d");
    NodeKey topKey = GraphTester.toNodeKey("top");
    tester.getOrCreate(topKey).addDependency(aKey).addDependency(cKey);
    tester.getOrCreate(aKey).addDependency(bKey);
    tester.getOrCreate(bKey).addDependency(aKey);
    tester.getOrCreate(cKey).addDependency(dKey);
    tester.getOrCreate(dKey).addDependency(cKey);
    UpdateResult<StringNode> result = eval(true, ImmutableList.of(topKey));
    assertEquals(null, result.get(topKey));
    ErrorInfo errorInfo = result.getError(topKey);
    CycleInfo aCycle = new CycleInfo(ImmutableList.of(topKey), ImmutableList.of(aKey, bKey));
    CycleInfo cCycle = new CycleInfo(ImmutableList.of(topKey), ImmutableList.of(cKey, dKey));
    MoreAsserts.assertContentsAnyOrder(errorInfo.getCycleInfo(), aCycle, cCycle);
  }

  @Test
  public void triangleBelowHeadCycle() throws Exception {
    graph = new InMemoryGraph();
    NodeKey aKey = GraphTester.toNodeKey("a");
    NodeKey bKey = GraphTester.toNodeKey("b");
    NodeKey cKey = GraphTester.toNodeKey("c");
    NodeKey topKey = GraphTester.toNodeKey("top");
    tester.getOrCreate(topKey).addDependency(aKey);
    tester.getOrCreate(aKey).addDependency(bKey).addDependency(cKey);
    tester.getOrCreate(bKey).addDependency(cKey);
    tester.getOrCreate(cKey).addDependency(topKey);
    UpdateResult<StringNode> result = eval(true, ImmutableList.of(topKey));
    assertEquals(null, result.get(topKey));
    ErrorInfo errorInfo = result.getError(topKey);
    CycleInfo topCycle = new CycleInfo(ImmutableList.of(topKey, aKey, cKey));
    MoreAsserts.assertContentsAnyOrder(errorInfo.getCycleInfo(), topCycle);
  }

  @Test
  public void longCycle() throws Exception {
    graph = new InMemoryGraph();
    NodeKey aKey = GraphTester.toNodeKey("a");
    NodeKey bKey = GraphTester.toNodeKey("b");
    NodeKey cKey = GraphTester.toNodeKey("c");
    NodeKey topKey = GraphTester.toNodeKey("top");
    tester.getOrCreate(topKey).addDependency(aKey);
    tester.getOrCreate(aKey).addDependency(bKey);
    tester.getOrCreate(bKey).addDependency(cKey);
    tester.getOrCreate(cKey).addDependency(topKey);
    UpdateResult<StringNode> result = eval(true, ImmutableList.of(topKey));
    assertEquals(null, result.get(topKey));
    ErrorInfo errorInfo = result.getError(topKey);
    CycleInfo topCycle = new CycleInfo(ImmutableList.of(topKey, aKey, bKey, cKey));
    MoreAsserts.assertContentsAnyOrder(errorInfo.getCycleInfo(), topCycle);
  }

  @Test
  public void cycleWithTail() throws Exception {
    graph = new InMemoryGraph();
    NodeKey aKey = GraphTester.toNodeKey("a");
    NodeKey bKey = GraphTester.toNodeKey("b");
    NodeKey cKey = GraphTester.toNodeKey("c");
    NodeKey topKey = GraphTester.toNodeKey("top");
    tester.getOrCreate(topKey).addDependency(aKey).addDependency(cKey);
    tester.getOrCreate(aKey).addDependency(bKey);
    tester.getOrCreate(bKey).addDependency(aKey).addDependency(cKey);
    tester.getOrCreate(cKey);
    tester.set(cKey, new StringNode("cNode"));
    UpdateResult<StringNode> result = eval(false, ImmutableList.of(topKey));
    assertEquals(null, result.get(topKey));
    ErrorInfo errorInfo = result.getError(topKey);
    CycleInfo cycleInfo = Iterables.getOnlyElement(errorInfo.getCycleInfo());
    MoreAsserts.assertContentsInOrder(cycleInfo.getCycle(), aKey, bKey);
    MoreAsserts.assertContentsInOrder(cycleInfo.getPathToCycle(), topKey);
  }

  /** Regression test: "node cannot be ready in a cycle". */
  @Test
  public void selfEdgeWithExtraChildrenUnderCycle() throws Exception {
    graph = new InMemoryGraph();
    NodeKey aKey = GraphTester.toNodeKey("a");
    NodeKey bKey = GraphTester.toNodeKey("b");
    NodeKey cKey = GraphTester.toNodeKey("c");
    tester.getOrCreate(aKey).addDependency(bKey);
    tester.getOrCreate(bKey).addDependency(cKey).addDependency(bKey);
    tester.getOrCreate(cKey).addDependency(aKey);
    UpdateResult<StringNode> result = eval(/*keepGoing=*/true, ImmutableList.of(aKey));
    assertEquals(null, result.get(aKey));
    ErrorInfo errorInfo = result.getError(aKey);
    CycleInfo cycleInfo = Iterables.getOnlyElement(errorInfo.getCycleInfo());
    MoreAsserts.assertContentsInOrder(cycleInfo.getCycle(), bKey);
    MoreAsserts.assertContentsInOrder(cycleInfo.getPathToCycle(), aKey);
  }

  /** Regression test: "node cannot be ready in a cycle". */
  @Test
  public void cycleWithExtraChildrenUnderCycle() throws Exception {
    graph = new InMemoryGraph();
    NodeKey aKey = GraphTester.toNodeKey("a");
    NodeKey bKey = GraphTester.toNodeKey("b");
    NodeKey cKey = GraphTester.toNodeKey("c");
    NodeKey dKey = GraphTester.toNodeKey("d");
    tester.getOrCreate(aKey).addDependency(bKey);
    tester.getOrCreate(bKey).addDependency(cKey).addDependency(dKey);
    tester.getOrCreate(cKey).addDependency(aKey);
    tester.getOrCreate(dKey).addDependency(bKey);
    UpdateResult<StringNode> result = eval(/*keepGoing=*/true, ImmutableList.of(aKey));
    assertEquals(null, result.get(aKey));
    ErrorInfo errorInfo = result.getError(aKey);
    CycleInfo cycleInfo = Iterables.getOnlyElement(errorInfo.getCycleInfo());
    MoreAsserts.assertContentsInOrder(cycleInfo.getCycle(), bKey, dKey);
    MoreAsserts.assertContentsInOrder(cycleInfo.getPathToCycle(), aKey);
  }

  /** Regression test: "node cannot be ready in a cycle". */
  @Test
  public void cycleAboveIndependentCycle() throws Exception {
    graph = new InMemoryGraph();
    NodeKey aKey = GraphTester.toNodeKey("a");
    NodeKey bKey = GraphTester.toNodeKey("b");
    NodeKey cKey = GraphTester.toNodeKey("c");
    tester.getOrCreate(aKey).addDependency(bKey);
    tester.getOrCreate(bKey).addDependency(cKey);
    tester.getOrCreate(cKey).addDependency(aKey).addDependency(bKey);
    UpdateResult<StringNode> result = eval(/*keepGoing=*/true, ImmutableList.of(aKey));
    assertEquals(null, result.get(aKey));
    MoreAsserts.assertContentsAnyOrder(result.getError(aKey).getCycleInfo(),
        new CycleInfo(ImmutableList.of(aKey, bKey, cKey)),
        new CycleInfo(ImmutableList.of(aKey), ImmutableList.of(bKey, cKey)));
 }

  public void nodeAboveCycleAndExceptionReportsException() throws Exception {
    graph = new InMemoryGraph();
    NodeKey aKey = GraphTester.toNodeKey("a");
    NodeKey errorKey = GraphTester.toNodeKey("error");
    NodeKey bKey = GraphTester.toNodeKey("b");
    tester.getOrCreate(aKey).addDependency(bKey).addDependency(errorKey);
    tester.getOrCreate(bKey).addDependency(bKey);
    tester.getOrCreate(errorKey).setHasError(true);
    UpdateResult<StringNode> result = eval(/*keepGoing=*/true, ImmutableList.of(aKey));
    assertEquals(null, result.get(aKey));
    assertNotNull(result.getError(aKey).getException());
    CycleInfo cycleInfo = Iterables.getOnlyElement(result.getError(aKey).getCycleInfo());
    MoreAsserts.assertContentsInOrder(cycleInfo.getCycle(), bKey);
    MoreAsserts.assertContentsInOrder(cycleInfo.getPathToCycle(), aKey);
  }

  @Test
  public void errorNodeStored() throws Exception {
    graph = new InMemoryGraph();
    NodeKey errorKey = GraphTester.toNodeKey("my_error_node");
    tester.getOrCreate(errorKey).setHasError(true);
    UpdateResult<StringNode> result = eval(false, ImmutableList.of(errorKey));
    ASSERT.that(result.keyNames()).isEmpty();
    MoreAsserts.assertContentsAnyOrder(result.errorMap().keySet(), errorKey);
    ErrorInfo errorInfo = result.getError();
    MoreAsserts.assertContentsAnyOrder(errorInfo.getRootCauses(), errorKey);
    // Update node. But builder won't rebuild it.
    tester.getOrCreate(errorKey).setHasError(false);
    tester.set(errorKey, new StringNode("no error?"));
    result = eval(false, ImmutableList.of(errorKey));
    ASSERT.that(result.keyNames()).isEmpty();
    MoreAsserts.assertContentsAnyOrder(result.errorMap().keySet(), errorKey);
    errorInfo = result.getError();
    MoreAsserts.assertContentsAnyOrder(errorInfo.getRootCauses(), errorKey);
  }

  /**
   * Regression test: "OOM in Skyframe cycle detection".
   * We only store the first 20 cycles found below any given root node.
   */
  @Test
  public void manyCycles() throws Exception {
    graph = new InMemoryGraph();
    NodeKey topKey = GraphTester.toNodeKey("top");
    for (int i = 0; i < 100; i++) {
      NodeKey dep = GraphTester.toNodeKey(Integer.toString(i));
      tester.getOrCreate(topKey).addDependency(dep);
      tester.getOrCreate(dep).addDependency(dep);
    }
    UpdateResult<StringNode> result = eval(/*keepGoing=*/true, ImmutableList.of(topKey));
    assertEquals(null, result.get(topKey));
    assertManyCycles(result.getError(topKey), topKey, /*selfEdge=*/false);
  }

  /**
   * Regression test: "OOM in Skyframe cycle detection".
   * We filter out multiple paths to a cycle that go through the same child node.
   */
  @Test
  public void manyPathsToCycle() throws Exception {
    graph = new InMemoryGraph();
    NodeKey topKey = GraphTester.toNodeKey("top");
    NodeKey midKey = GraphTester.toNodeKey("mid");
    NodeKey cycleKey = GraphTester.toNodeKey("cycle");
    tester.getOrCreate(topKey).addDependency(midKey);
    tester.getOrCreate(cycleKey).addDependency(cycleKey);
    for (int i = 0; i < 100; i++) {
      NodeKey dep = GraphTester.toNodeKey(Integer.toString(i));
      tester.getOrCreate(midKey).addDependency(dep);
      tester.getOrCreate(dep).addDependency(cycleKey);
    }
    UpdateResult<StringNode> result = eval(/*keepGoing=*/true, ImmutableList.of(topKey));
    assertEquals(null, result.get(topKey));
    CycleInfo cycleInfo = Iterables.getOnlyElement(result.getError(topKey).getCycleInfo());
    assertEquals(1, cycleInfo.getCycle().size());
    assertEquals(3, cycleInfo.getPathToCycle().size());
    MoreAsserts.assertContentsInOrder(cycleInfo.getPathToCycle().subList(0, 2),
        topKey, midKey);
  }

  /**
   * Checks that errorInfo has many self-edge cycles, and that one of them is a self-edge of
   * topKey, if {@code selfEdge} is true.
   */
  private static void assertManyCycles(ErrorInfo errorInfo, NodeKey topKey, boolean selfEdge) {
    MoreAsserts.assertGreaterThan(1, Iterables.size(errorInfo.getCycleInfo()));
    MoreAsserts.assertLessThan(50, Iterables.size(errorInfo.getCycleInfo()));
    boolean foundSelfEdge = false;
    for (CycleInfo cycle : errorInfo.getCycleInfo()) {
      assertEquals(1, cycle.getCycle().size()); // Self-edge.
      if (!Iterables.isEmpty(cycle.getPathToCycle())) {
        MoreAsserts.assertContentsInOrder(cycle.getPathToCycle(), topKey);
      } else {
        MoreAsserts.assertContentsInOrder(cycle.getCycle(), topKey);
        foundSelfEdge = true;
      }
    }
    assertEquals(errorInfo + ", " + topKey, selfEdge, foundSelfEdge);
  }

  @Test
  public void manyUnprocessedNodesInCycle() throws Exception {
    graph = new InMemoryGraph();
    NodeKey lastSelfKey = GraphTester.toNodeKey("lastSelf");
    NodeKey firstSelfKey = GraphTester.toNodeKey("firstSelf");
    NodeKey midSelfKey = GraphTester.toNodeKey("midSelf");
    // We add firstSelf first so that it is processed last in cycle detection (LIFO), meaning that
    // none of the dep nodes have to be cleared from firstSelf.
    tester.getOrCreate(firstSelfKey).addDependency(firstSelfKey);
    for (int i = 0; i < 100; i++) {
      NodeKey firstDep = GraphTester.toNodeKey("first" + i);
      NodeKey midDep = GraphTester.toNodeKey("mid" + i);
      NodeKey lastDep = GraphTester.toNodeKey("last" + i);
      tester.getOrCreate(firstSelfKey).addDependency(firstDep);
      tester.getOrCreate(midSelfKey).addDependency(midDep);
      tester.getOrCreate(lastSelfKey).addDependency(lastDep);
      if (i == 90) {
        // Most of the deps will be cleared from midSelf.
        tester.getOrCreate(midSelfKey).addDependency(midSelfKey);
      }
      tester.getOrCreate(firstDep).addDependency(firstDep);
      tester.getOrCreate(midDep).addDependency(midDep);
      tester.getOrCreate(lastDep).addDependency(lastDep);
    }
    // All the deps will be cleared from lastSelf.
    tester.getOrCreate(lastSelfKey).addDependency(lastSelfKey);
    UpdateResult<StringNode> result = eval(/*keepGoing=*/true,
        ImmutableList.of(lastSelfKey, firstSelfKey, midSelfKey));
    ASSERT.withFailureMessage(result.toString()).that(result.keyNames()).isEmpty();
    MoreAsserts.assertContentsAnyOrder(result.errorMap().keySet(),
        lastSelfKey, firstSelfKey, midSelfKey);

    // Check lastSelfKey.
    ErrorInfo errorInfo = result.getError(lastSelfKey);
    assertEquals(errorInfo.toString(), 1, Iterables.size(errorInfo.getCycleInfo()));
    CycleInfo cycleInfo = Iterables.getOnlyElement(errorInfo.getCycleInfo());
    MoreAsserts.assertContentsAnyOrder(cycleInfo.getCycle(), lastSelfKey);
    ASSERT.that(cycleInfo.getPathToCycle()).isEmpty();

    // Check firstSelfKey. It should not have discovered its own self-edge, because there were too
    // many other nodes before it in the queue.
    assertManyCycles(result.getError(firstSelfKey), firstSelfKey, /*selfEdge=*/false);

    // Check midSelfKey. It should have discovered its own self-edge.
    assertManyCycles(result.getError(midSelfKey), midSelfKey, /*selfEdge=*/true);
  }

  @Test
  public void errorNodeStoredWithKeepGoing() throws Exception {
    graph = new InMemoryGraph();
    NodeKey errorKey = GraphTester.toNodeKey("my_error_node");
    tester.getOrCreate(errorKey).setHasError(true);
    UpdateResult<StringNode> result = eval(true, ImmutableList.of(errorKey));
    ASSERT.that(result.keyNames()).isEmpty();
    MoreAsserts.assertContentsAnyOrder(result.errorMap().keySet(), errorKey);
    ErrorInfo errorInfo = result.getError();
    MoreAsserts.assertContentsAnyOrder(errorInfo.getRootCauses(), errorKey);
    // Update node. But builder won't rebuild it.
    tester.getOrCreate(errorKey).setHasError(false);
    tester.set(errorKey, new StringNode("no error?"));
    result = eval(true, ImmutableList.of(errorKey));
    ASSERT.that(result.keyNames()).isEmpty();
    MoreAsserts.assertContentsAnyOrder(result.errorMap().keySet(), errorKey);
    errorInfo = result.getError();
    MoreAsserts.assertContentsAnyOrder(errorInfo.getRootCauses(), errorKey);
  }

  @Test
  public void continueWithErrorDep() throws Exception {
    graph = new InMemoryGraph();
    NodeKey errorKey = GraphTester.toNodeKey("my_error_node");
    tester.getOrCreate(errorKey).setHasError(true);
    tester.set("after", new StringNode("after"));
    NodeKey parentKey = GraphTester.toNodeKey("parent");
    tester.getOrCreate(parentKey).addErrorDependency(errorKey, new StringNode("recovered"))
        .setComputedValue(CONCATENATE).addDependency("after");
    UpdateResult<StringNode> result = eval(/*keepGoing=*/true, ImmutableList.of(parentKey));
    ASSERT.that(result.errorMap()).isEmpty();
    assertEquals("recoveredafter", result.get(parentKey).getValue());
    result = eval(/*keepGoing=*/false, ImmutableList.of(parentKey));
    ASSERT.that(result.keyNames()).isEmpty();
    Map.Entry<NodeKey, ErrorInfo> error = Iterables.getOnlyElement(result.errorMap().entrySet());
    assertEquals(parentKey, error.getKey());
    MoreAsserts.assertContentsAnyOrder(error.getValue().getRootCauses(), errorKey);
  }

  @Test
  public void breakWithErrorDep() throws Exception {
    graph = new InMemoryGraph();
    NodeKey errorKey = GraphTester.toNodeKey("my_error_node");
    tester.getOrCreate(errorKey).setHasError(true);
    tester.set("after", new StringNode("after"));
    NodeKey parentKey = GraphTester.toNodeKey("parent");
    tester.getOrCreate(parentKey).addErrorDependency(errorKey, new StringNode("recovered"))
        .setComputedValue(CONCATENATE).addDependency("after");
    UpdateResult<StringNode> result = eval(/*keepGoing=*/false, ImmutableList.of(parentKey));
    ASSERT.that(result.keyNames()).isEmpty();
    Map.Entry<NodeKey, ErrorInfo> error = Iterables.getOnlyElement(result.errorMap().entrySet());
    assertEquals(parentKey, error.getKey());
    MoreAsserts.assertContentsAnyOrder(error.getValue().getRootCauses(), errorKey);
    result = eval(/*keepGoing=*/true, ImmutableList.of(parentKey));
    ASSERT.that(result.errorMap()).isEmpty();
    assertEquals("recoveredafter", result.get(parentKey).getValue());
  }

  @Test
  public void breakWithInterruptibleErrorDep() throws Exception {
    graph = new InMemoryGraph();
    NodeKey errorKey = GraphTester.toNodeKey("my_error_node");
    tester.getOrCreate(errorKey).setHasError(true);
    NodeKey parentKey = GraphTester.toNodeKey("parent");
    tester.getOrCreate(parentKey).addErrorDependency(errorKey, new StringNode("recovered"))
        .setComputedValue(CONCATENATE);
    // When the error node throws, the propagation will cause an interrupted exception in parent.
    UpdateResult<StringNode> result = eval(/*keepGoing=*/false, ImmutableList.of(parentKey));
    ASSERT.that(result.keyNames()).isEmpty();
    Map.Entry<NodeKey, ErrorInfo> error = Iterables.getOnlyElement(result.errorMap().entrySet());
    assertEquals(parentKey, error.getKey());
    MoreAsserts.assertContentsAnyOrder(error.getValue().getRootCauses(), errorKey);
    assertFalse(Thread.interrupted());
    result = eval(/*keepGoing=*/true, ImmutableList.of(parentKey));
    ASSERT.that(result.errorMap()).isEmpty();
    assertEquals("recovered", result.get(parentKey).getValue());
  }

  @Test
  public void transformErrorDep() throws Exception {
    graph = new InMemoryGraph();
    NodeKey errorKey = GraphTester.toNodeKey("my_error_node");
    tester.getOrCreate(errorKey).setHasError(true);
    NodeKey parentErrorKey = GraphTester.toNodeKey("parent");
    tester.getOrCreate(parentErrorKey).addErrorDependency(errorKey, new StringNode("recovered"))
        .setHasError(true);
    UpdateResult<StringNode> result = eval(/*keepGoing=*/false, ImmutableList.of(parentErrorKey));
    ASSERT.that(result.keyNames()).isEmpty();
    Map.Entry<NodeKey, ErrorInfo> error = Iterables.getOnlyElement(result.errorMap().entrySet());
    assertEquals(parentErrorKey, error.getKey());
    MoreAsserts.assertContentsAnyOrder(error.getValue().getRootCauses(), parentErrorKey);
  }

  @Test
  public void transformErrorDepKeepGoing() throws Exception {
    graph = new InMemoryGraph();
    NodeKey errorKey = GraphTester.toNodeKey("my_error_node");
    tester.getOrCreate(errorKey).setHasError(true);
    NodeKey parentErrorKey = GraphTester.toNodeKey("parent");
    tester.getOrCreate(parentErrorKey).addErrorDependency(errorKey, new StringNode("recovered"))
        .setHasError(true);
    UpdateResult<StringNode> result = eval(/*keepGoing=*/true, ImmutableList.of(parentErrorKey));
    ASSERT.that(result.keyNames()).isEmpty();
    Map.Entry<NodeKey, ErrorInfo> error = Iterables.getOnlyElement(result.errorMap().entrySet());
    assertEquals(parentErrorKey, error.getKey());
    MoreAsserts.assertContentsAnyOrder(error.getValue().getRootCauses(), parentErrorKey);
  }

  @Test
  public void transformErrorDepOneLevelDownKeepGoing() throws Exception {
    graph = new InMemoryGraph();
    NodeKey errorKey = GraphTester.toNodeKey("my_error_node");
    tester.getOrCreate(errorKey).setHasError(true);
    tester.set("after", new StringNode("after"));
    NodeKey parentErrorKey = GraphTester.toNodeKey("parent");
    tester.getOrCreate(parentErrorKey).addErrorDependency(errorKey, new StringNode("recovered"));
    tester.set(parentErrorKey, new StringNode("parent value"));
    NodeKey topKey = GraphTester.toNodeKey("top");
    tester.getOrCreate(topKey).addDependency(parentErrorKey).addDependency("after")
        .setComputedValue(CONCATENATE);
    UpdateResult<StringNode> result = eval(/*keepGoing=*/true, ImmutableList.of(topKey));
    MoreAsserts.assertContentsAnyOrder(
        ImmutableList.<String>copyOf(result.<String>keyNames()), "top");
    assertEquals("parent valueafter", result.get(topKey).getValue());
    ASSERT.that(result.errorMap()).isEmpty();
  }

  @Test
  public void transformErrorDepOneLevelDownNoKeepGoing() throws Exception {
    graph = new InMemoryGraph();
    NodeKey errorKey = GraphTester.toNodeKey("my_error_node");
    tester.getOrCreate(errorKey).setHasError(true);
    tester.set("after", new StringNode("after"));
    NodeKey parentErrorKey = GraphTester.toNodeKey("parent");
    tester.getOrCreate(parentErrorKey).addErrorDependency(errorKey, new StringNode("recovered"));
    tester.set(parentErrorKey, new StringNode("parent value"));
    NodeKey topKey = GraphTester.toNodeKey("top");
    tester.getOrCreate(topKey).addDependency(parentErrorKey).addDependency("after")
        .setComputedValue(CONCATENATE);
    UpdateResult<StringNode> result = eval(/*keepGoing=*/false, ImmutableList.of(topKey));
    ASSERT.that(result.keyNames()).isEmpty();
    Map.Entry<NodeKey, ErrorInfo> error = Iterables.getOnlyElement(result.errorMap().entrySet());
    assertEquals(topKey, error.getKey());
    MoreAsserts.assertContentsAnyOrder(error.getValue().getRootCauses(), errorKey);
  }

  /**
   * Make sure that multiple unfinished children can be cleared from a cycle node.
   */
  @Test
  public void cycleWithMultipleUnfinishedChildren() throws Exception {
    graph = new InMemoryGraph();
    tester = new GraphTester();
    NodeKey cycleKey = GraphTester.toNodeKey("cycle");
    NodeKey midKey = GraphTester.toNodeKey("mid");
    NodeKey topKey = GraphTester.toNodeKey("top");
    NodeKey selfEdge1 = GraphTester.toNodeKey("selfEdge1");
    NodeKey selfEdge2 = GraphTester.toNodeKey("selfEdge2");
    tester.getOrCreate(topKey).addDependency(midKey).setComputedValue(CONCATENATE);
    // selfEdge* come before cycleKey, so cycleKey's path will be checked first (LIFO), and the
    // cycle with mid will be detected before the selfEdge* cycles are.
    tester.getOrCreate(midKey).addDependency(selfEdge1).addDependency(selfEdge2)
        .addDependency(cycleKey)
    .setComputedValue(CONCATENATE);
    tester.getOrCreate(cycleKey).addDependency(midKey);
    tester.getOrCreate(selfEdge1).addDependency(selfEdge1);
    tester.getOrCreate(selfEdge2).addDependency(selfEdge2);
    UpdateResult<StringNode> result = eval(/*keepGoing=*/true, ImmutableSet.of(topKey));
    MoreAsserts.assertContentsAnyOrder(result.errorMap().keySet(), topKey);
    Iterable<CycleInfo> cycleInfos = result.getError(topKey).getCycleInfo();
    CycleInfo cycleInfo = Iterables.getOnlyElement(cycleInfos);
    MoreAsserts.assertContentsAnyOrder(cycleInfo.getPathToCycle(), topKey);
    MoreAsserts.assertContentsAnyOrder(cycleInfo.getCycle(), midKey, cycleKey);
  }

  /**
   * Regression test: "node in cycle depends on error".
   * The mid node will have two parents -- top and cycle. Error bubbles up from mid to cycle, and we
   * should detect cycle.
   */
  private void cycleAndErrorInBubbleUp(boolean keepGoing) throws Exception {
    graph = new DeterministicInMemoryGraph();
    tester = new GraphTester();
    NodeKey errorKey = GraphTester.toNodeKey("error");
    NodeKey cycleKey = GraphTester.toNodeKey("cycle");
    NodeKey midKey = GraphTester.toNodeKey("mid");
    NodeKey topKey = GraphTester.toNodeKey("top");
    tester.getOrCreate(topKey).addDependency(midKey).setComputedValue(CONCATENATE);
    tester.getOrCreate(midKey).addDependency(errorKey).addDependency(cycleKey)
        .setComputedValue(CONCATENATE);

    // We need to ensure that cycle node has finished his work, and we have recorded dependencies
    CountDownLatch cycleFinish = new CountDownLatch(1);
    tester.getOrCreate(cycleKey).setBuilder(new ChainedNodeBuilder(null,
        null, cycleFinish, false, new StringNode(""), ImmutableSet.<NodeKey>of(midKey)));
    tester.getOrCreate(errorKey).setBuilder(new ChainedNodeBuilder(null, cycleFinish,
        null, /*waitForException=*/false, null, ImmutableSet.<NodeKey>of()));

    UpdateResult<StringNode> result = eval(keepGoing, ImmutableSet.of(topKey));
    MoreAsserts.assertContentsAnyOrder(result.errorMap().keySet(), topKey);
    Iterable<CycleInfo> cycleInfos = result.getError(topKey).getCycleInfo();
    if (keepGoing) {
      // The error thrown will only be recorded in keep_going mode.
      MoreAsserts.assertContentsAnyOrder(result.getError().getRootCauses(), errorKey);
    }
    ASSERT.that(cycleInfos).isNotEmpty();
    CycleInfo cycleInfo = Iterables.getOnlyElement(cycleInfos);
    MoreAsserts.assertContentsAnyOrder(cycleInfo.getPathToCycle(), topKey);
    MoreAsserts.assertContentsAnyOrder(cycleInfo.getCycle(), midKey, cycleKey);
  }

  @Test
  public void cycleAndErrorInBubbleUpNoKeepGoing() throws Exception {
    cycleAndErrorInBubbleUp(false);
  }

  @Test
  public void cycleAndErrorInBubbleUpKeepGoing() throws Exception {
    cycleAndErrorInBubbleUp(true);
  }

  /**
   * Regression test: "node in cycle depends on error".
   * We add another node that won't finish building before the threadpool shuts down, to check that
   * the cycle detection can handle unfinished nodes.
   */
  @Test
  public void cycleAndErrorAndOtherInBubbleUp() throws Exception {
    graph = new DeterministicInMemoryGraph();
    tester = new GraphTester();
    NodeKey errorKey = GraphTester.toNodeKey("error");
    NodeKey cycleKey = GraphTester.toNodeKey("cycle");
    NodeKey midKey = GraphTester.toNodeKey("mid");
    NodeKey topKey = GraphTester.toNodeKey("top");
    tester.getOrCreate(topKey).addDependency(midKey).setComputedValue(CONCATENATE);
    // We should add cycleKey first and errorKey afterwards. Otherwise there is a chance that
    // during error propagation cycleKey will not be processed, and we will not detect the cycle.
    tester.getOrCreate(midKey).addDependency(errorKey).addDependency(cycleKey)
        .setComputedValue(CONCATENATE);
    NodeKey otherTop = GraphTester.toNodeKey("otherTop");
    CountDownLatch topStartAndCycleFinish = new CountDownLatch(2);
    // In nokeep_going mode, otherTop will wait until the threadpool has received an exception,
    // then request its own dep. This guarantees that there is a node that is not finished when
    // cycle detection happens.
    tester.getOrCreate(otherTop).setBuilder(new ChainedNodeBuilder(topStartAndCycleFinish,
        new CountDownLatch(0), null, /*waitForException=*/true, new StringNode("never returned"),
        ImmutableSet.<NodeKey>of(GraphTester.toNodeKey("dep that never builds"))));

    tester.getOrCreate(cycleKey).setBuilder(new ChainedNodeBuilder(null, null, 
        topStartAndCycleFinish, /*waitForException=*/false, new StringNode(""),
        ImmutableSet.<NodeKey>of(midKey)));
    // error waits until otherTop starts and cycle finishes, to make sure otherTop will request 
    // its dep before the threadpool shuts down.
    tester.getOrCreate(errorKey).setBuilder(new ChainedNodeBuilder(null, topStartAndCycleFinish,
        null, /*waitForException=*/false, null,
        ImmutableSet.<NodeKey>of()));
    UpdateResult<StringNode> result =
        eval(/*keepGoing=*/false, ImmutableSet.of(topKey, otherTop));
    MoreAsserts.assertContentsAnyOrder(result.errorMap().keySet(), topKey);
    Iterable<CycleInfo> cycleInfos = result.getError(topKey).getCycleInfo();
    ASSERT.that(cycleInfos).isNotEmpty();
    CycleInfo cycleInfo = Iterables.getOnlyElement(cycleInfos);
    MoreAsserts.assertContentsAnyOrder(cycleInfo.getPathToCycle(), topKey);
    MoreAsserts.assertContentsAnyOrder(cycleInfo.getCycle(), midKey, cycleKey);
  }

  /**
   * Regression test: "node in cycle depends on error".
   * Here, we add an additional top-level key in error, just to mix it up.
   */
  private void cycleAndErrorAndError(boolean keepGoing) throws Exception {
    graph = new DeterministicInMemoryGraph();
    tester = new GraphTester();
    NodeKey errorKey = GraphTester.toNodeKey("error");
    NodeKey cycleKey = GraphTester.toNodeKey("cycle");
    NodeKey midKey = GraphTester.toNodeKey("mid");
    NodeKey topKey = GraphTester.toNodeKey("top");
    tester.getOrCreate(topKey).addDependency(midKey).setComputedValue(CONCATENATE);
    tester.getOrCreate(midKey).addDependency(errorKey).addDependency(cycleKey)
        .setComputedValue(CONCATENATE);
    NodeKey otherTop = GraphTester.toNodeKey("otherTop");
    CountDownLatch topStartAndCycleFinish = new CountDownLatch(2);
    // In nokeep_going mode, otherTop will wait until the threadpool has received an exception,
    // then throw its own exception. This guarantees that its exception will not be the one
    // bubbling up, but that there is a top-level node with an exception by the time the bubbling
    // up starts.
    tester.getOrCreate(otherTop).setBuilder(new ChainedNodeBuilder(topStartAndCycleFinish,
        new CountDownLatch(0), null, /*waitForException=*/!keepGoing, null,
        ImmutableSet.<NodeKey>of()));
    // error waits until otherTop starts and cycle finishes, to make sure otherTop will request 
    // its dep before the threadpool shuts down.
    tester.getOrCreate(errorKey).setBuilder(new ChainedNodeBuilder(null, topStartAndCycleFinish,
        null, /*waitForException=*/false, null,
        ImmutableSet.<NodeKey>of()));
    tester.getOrCreate(cycleKey).setBuilder(new ChainedNodeBuilder(null, null, 
        topStartAndCycleFinish, /*waitForException=*/false, new StringNode(""),
        ImmutableSet.<NodeKey>of(midKey)));
    UpdateResult<StringNode> result =
        eval(keepGoing, ImmutableSet.of(topKey, otherTop));
    MoreAsserts.assertContentsAnyOrder(result.errorMap().keySet(), otherTop, topKey);
    MoreAsserts.assertContentsAnyOrder(result.getError(otherTop).getRootCauses(), otherTop);
    Iterable<CycleInfo> cycleInfos = result.getError(topKey).getCycleInfo();
    if (keepGoing) {
      // The error thrown will only be recorded in keep_going mode.
      MoreAsserts.assertContentsAnyOrder(result.getError(topKey).getRootCauses(), errorKey);
    }
    ASSERT.that(cycleInfos).isNotEmpty();
    CycleInfo cycleInfo = Iterables.getOnlyElement(cycleInfos);
    MoreAsserts.assertContentsAnyOrder(cycleInfo.getPathToCycle(), topKey);
    MoreAsserts.assertContentsAnyOrder(cycleInfo.getCycle(), midKey, cycleKey);
  }

  @Test
  public void cycleAndErrorAndErrorNoKeepGoing() throws Exception {
    cycleAndErrorAndError(false);
  }

  @Test
  public void cycleAndErrorAndErrorKeepGoing() throws Exception {
    cycleAndErrorAndError(true);
  }

  @Test
  public void testNodeBuilderCrashTrace() throws Exception {
    final NodeType childType = new NodeType("child", false);
    final NodeType parentType = new NodeType("parent", false);

    class ChildNodeBuilder implements NodeBuilder {
      @Override
      public Node build(NodeKey nodeKey, Environment env) {
        throw new IllegalStateException("I WANT A PONY!!!");
      }

      @Override public String extractTag(NodeKey nodeKey) { return null; }
    }

    class ParentNodeBuilder implements NodeBuilder {
      @Override
      public Node build(NodeKey nodeKey, Environment env) {
        Node dep = env.getDep(new NodeKey(childType, "billy the kid"));
        if (dep == null) {
          return null;
        }
        throw new IllegalStateException();  // Should never get here.
      }

      @Override public String extractTag(NodeKey nodeKey) { return null; }
    }

    ImmutableMap<NodeType, NodeBuilder> nodeBuilders = ImmutableMap.of(
        childType, new ChildNodeBuilder(),
        parentType, new ParentNodeBuilder());
    ParallelEvaluator evaluator = makeEvaluator(new InMemoryGraph(),
        nodeBuilders, false);

    try {
      evaluator.eval(ImmutableList.of(new NodeKey(parentType, "octodad")));
      fail();
    } catch (RuntimeException e) {
      assertEquals("I WANT A PONY!!!", e.getCause().getMessage());
      assertEquals("Unrecoverable error while evaluating node 'child:billy the kid' "
          + "(requested by nodes 'parent:octodad')", e.getMessage());
    }
  }

  private void unexpectedErrorDep(boolean keepGoing) throws Exception {
    graph = new InMemoryGraph();
    NodeKey errorKey = GraphTester.toNodeKey("my_error_node");
    final Exception exception = new Exception("error exception");
    tester.getOrCreate(errorKey).setBuilder(new NodeBuilder() {
      @Override
      public Node build(NodeKey nodeKey, Environment env) throws GenericNodeBuilderException {
        throw new GenericNodeBuilderException(nodeKey, exception);
      }

      @Override
      public String extractTag(NodeKey nodeKey) {
        throw new UnsupportedOperationException();
      }
    });
    NodeKey topKey = GraphTester.toNodeKey("top");
    tester.getOrCreate(topKey).addErrorDependency(errorKey, new StringNode("recovered"))
        .setComputedValue(CONCATENATE);
    UpdateResult<StringNode> result = eval(keepGoing, ImmutableList.of(topKey));
    ASSERT.that(result.keyNames()).isEmpty();
    assertSame(exception, result.getError(topKey).getException());
    MoreAsserts.assertContentsAnyOrder(result.getError(topKey).getRootCauses(), errorKey);
  }

  /**
   * This and the following three tests are in response a bug: "Skyframe error propagation model is
   * problematic". They ensure that exceptions a child throws that a node does not specify it can
   * handle in getDepOrThrow do not cause a crash.
   */
  @Test
  public void unexpectedErrorDepKeepGoing() throws Exception {
    unexpectedErrorDep(true);
  }

  @Test
  public void unexpectedErrorDepNoKeepGoing() throws Exception {
    unexpectedErrorDep(false);
  }

  private void unexpectedErrorDepOneLevelDown(final boolean keepGoing) throws Exception {
    graph = new InMemoryGraph();
    NodeKey errorKey = GraphTester.toNodeKey("my_error_node");
    final Exception exception = new Exception("error exception");
    final Exception topException = new Exception("top exception");
    final StringNode topNode = new StringNode("top");
    tester.getOrCreate(errorKey).setBuilder(new NodeBuilder() {
      @Override
      public Node build(NodeKey nodeKey, Environment env) throws GenericNodeBuilderException {
        throw new GenericNodeBuilderException(nodeKey, exception);
      }

      @Override
      public String extractTag(NodeKey nodeKey) {
        throw new UnsupportedOperationException();
      }
    });
    NodeKey topKey = GraphTester.toNodeKey("top");
    final NodeKey parentKey = GraphTester.toNodeKey("parent");
    tester.getOrCreate(parentKey).addDependency(errorKey).setComputedValue(CONCATENATE);
    tester.getOrCreate(topKey).setBuilder(new NodeBuilder() {
      @Override
      public Node build(NodeKey nodeKey, Environment env) throws GenericNodeBuilderException {
        try {
          if (env.getDepOrThrow(parentKey, Exception.class) == null) {
            return null;
          }
        } catch (Exception e) {
          assertEquals(e.toString(), exception, e);
        }
        if (keepGoing) {
          return topNode;
        } else {
          throw new GenericNodeBuilderException(nodeKey, topException);
        }
      }
      @Override
      public String extractTag(NodeKey nodeKey) {
        throw new UnsupportedOperationException();
      }
    });
    tester.getOrCreate(topKey).addErrorDependency(errorKey, new StringNode("recovered"))
        .setComputedValue(CONCATENATE);
    UpdateResult<StringNode> result = eval(keepGoing, ImmutableList.of(topKey));
    if (!keepGoing) {
      ASSERT.that(result.keyNames()).isEmpty();
      assertEquals(topException, result.getError(topKey).getException());
      MoreAsserts.assertContentsAnyOrder(result.getError(topKey).getRootCauses(), topKey);
      assertTrue(result.hasError());
    } else {
      // result.hasError() is set to true even if the top-level node returned has recovered from
      // an error.
      assertTrue(result.hasError());
      assertSame(topNode, result.get(topKey));
    }
  }

  @Test
  public void unexpectedErrorDepOneLevelDownKeepGoing() throws Exception {
    unexpectedErrorDepOneLevelDown(true);
  }

  @Test
  public void unexpectedErrorDepOneLevelDownNoKeepGoing() throws Exception {
    unexpectedErrorDepOneLevelDown(false);
  }

  /**
   * Exercises various situations involving groups of deps that overlap -- request one group, then
   * request another group that has a dep in common with the first group.
   *
   * @param sameFirst whether the dep in common in the two groups should be the first dep.
   * @param twoCalls whether the two groups should be requested in two different builder calls.
   * @param depsOrThrow whether the deps should be requested using getDepsOrThrow.
   */
  private void sameDepInTwoGroups(final boolean sameFirst, final boolean twoCalls,
      final boolean depsOrThrow) throws Exception {
    graph = new InMemoryGraph();
    NodeKey topKey = GraphTester.toNodeKey("top");
    final List<NodeKey> leaves = new ArrayList<>();
    for (int i = 1; i <= 3; i++) {
      NodeKey leaf = GraphTester.toNodeKey("leaf" + i);
      leaves.add(leaf);
      tester.set(leaf, new StringNode("leaf" + i));
    }
    final NodeKey leaf4 = GraphTester.toNodeKey("leaf4");
    tester.set(leaf4, new StringNode("leaf" + 4));
    tester.getOrCreate(topKey).setBuilder(new NodeBuilder() {
      @Override
      public Node build(NodeKey nodeKey, Environment env) throws NodeBuilderException,
          InterruptedException {
        if (depsOrThrow) {
          env.getDepsOrThrow(leaves, Exception.class);
        } else {
          env.getDeps(leaves);
        }
        if (twoCalls && env.depsMissing()) {
          return null;
        }
        NodeKey first = sameFirst ? leaves.get(0) : leaf4;
        NodeKey second = sameFirst ? leaf4 : leaves.get(2);
        List<NodeKey> secondRequest = ImmutableList.of(first, second);
        if (depsOrThrow) {
          env.getDepsOrThrow(secondRequest, Exception.class);
        } else {
          env.getDeps(secondRequest);
        }
        if (env.depsMissing()) {
          return null;
        }
        return new StringNode("top");
      }

      @Override
      public String extractTag(NodeKey nodeKey) {
        return null;
      }
    });
    eval(/*keepGoing=*/false, topKey);
    assertEquals(new StringNode("top"), eval(/*keepGoing=*/false, topKey));
  }

  @Test
  public void sameDepInTwoGroups_Same_Two_Throw() throws Exception {
    sameDepInTwoGroups(/*sameFirst=*/true, /*twoCalls=*/true, /*depsOrThrow=*/true);
  }

  @Test
  public void sameDepInTwoGroups_Same_Two_Deps() throws Exception {
    sameDepInTwoGroups(/*sameFirst=*/true, /*twoCalls=*/true, /*depsOrThrow=*/false);
  }

  @Test
  public void sameDepInTwoGroups_Same_One_Throw() throws Exception {
    sameDepInTwoGroups(/*sameFirst=*/true, /*twoCalls=*/false, /*depsOrThrow=*/true);
  }

  @Test
  public void sameDepInTwoGroups_Same_One_Deps() throws Exception {
    sameDepInTwoGroups(/*sameFirst=*/true, /*twoCalls=*/false, /*depsOrThrow=*/false);
  }

  @Test
  public void sameDepInTwoGroups_Different_Two_Throw() throws Exception {
    sameDepInTwoGroups(/*sameFirst=*/false, /*twoCalls=*/true, /*depsOrThrow=*/true);
  }

  @Test
  public void sameDepInTwoGroups_Different_Two_Deps() throws Exception {
    sameDepInTwoGroups(/*sameFirst=*/false, /*twoCalls=*/true, /*depsOrThrow=*/false);
  }

  @Test
  public void sameDepInTwoGroups_Different_One_Throw() throws Exception {
    sameDepInTwoGroups(/*sameFirst=*/false, /*twoCalls=*/false, /*depsOrThrow=*/true);
  }

  @Test
  public void sameDepInTwoGroups_Different_One_Deps() throws Exception {
    sameDepInTwoGroups(/*sameFirst=*/false, /*twoCalls=*/false, /*depsOrThrow=*/false);
  }

  private void getDepsOrThrowWithErrors(boolean keepGoing) throws Exception {
    graph = new InMemoryGraph();
    NodeKey parentKey = GraphTester.toNodeKey("parent");
    final NodeKey errorDep = GraphTester.toNodeKey("errorChild");
    final SomeErrorException childExn = new SomeErrorException("child error");
    tester.getOrCreate(errorDep).setBuilder(new NodeBuilder() {
      @Override
      public Node build(NodeKey nodeKey, Environment env) throws NodeBuilderException {
        throw new GenericNodeBuilderException(nodeKey, childExn);
      }

      @Override
      public String extractTag(NodeKey nodeKey) {
        return null;
      }
    });
    final List<NodeKey> deps = new ArrayList<>();
    for (int i = 1; i <= 3; i++) {
      NodeKey dep = GraphTester.toNodeKey("child" + i);
      deps.add(dep);
      tester.set(dep, new StringNode("child" + i));
    }
    final SomeErrorException parentExn = new SomeErrorException("parent error");
    tester.getOrCreate(parentKey).setBuilder(new NodeBuilder() {
      @Override
      public Node build(NodeKey nodeKey, Environment env) throws NodeBuilderException {
        try {
          Node node = env.getDepOrThrow(errorDep, SomeErrorException.class);
          if (node == null) {
            return null;
          }
        } catch (SomeErrorException e) {
          // Recover from the child error.
        }
        env.getDeps(deps);
        if (env.depsMissing()) {
          return null;
        }
        throw new GenericNodeBuilderException(nodeKey, parentExn);
      }

      @Override
      public String extractTag(NodeKey nodeKey) {
        return null;
      }
    });
    UpdateResult<StringNode> updateResult = eval(keepGoing, ImmutableList.of(parentKey));
    assertTrue(updateResult.hasError());
    assertEquals(keepGoing ? parentExn : childExn, updateResult.getError().getException());
  }

  @Test
  public void getDepsOrThrowWithErrors_NoKeepGoing() throws Exception {
    getDepsOrThrowWithErrors(/*keepGoing=*/false);
  }

  @Test
  public void getDepsOrThrowWithErrors_KeepGoing() throws Exception {
    getDepsOrThrowWithErrors(/*keepGoing=*/true);
  }

  @Test
  public void duplicateCycles() throws Exception {
    graph = new InMemoryGraph();
    NodeKey grandparentKey = GraphTester.toNodeKey("grandparent");
    NodeKey parentKey1 = GraphTester.toNodeKey("parent1");
    NodeKey parentKey2 = GraphTester.toNodeKey("parent2");
    NodeKey loopKey1 = GraphTester.toNodeKey("loop1");
    NodeKey loopKey2 = GraphTester.toNodeKey("loop2");
    tester.getOrCreate(loopKey1).addDependency(loopKey2);
    tester.getOrCreate(loopKey2).addDependency(loopKey1);
    tester.getOrCreate(parentKey1).addDependency(loopKey1);
    tester.getOrCreate(parentKey2).addDependency(loopKey2);
    tester.getOrCreate(grandparentKey).addDependency(parentKey1);
    tester.getOrCreate(grandparentKey).addDependency(parentKey2);

    ErrorInfo errorInfo = evalNodeInError(grandparentKey);
    List<ImmutableList<NodeKey>> cycles = Lists.newArrayList();
    for (CycleInfo cycleInfo : errorInfo.getCycleInfo()) {
      cycles.add(cycleInfo.getCycle());
    }
    // Skyframe doesn't automatically dedupe cycles that are the same except for entry point.
    assertEquals(2, cycles.size());
    int numUniqueCycles = 0;
    CycleDeduper<NodeKey> cycleDeduper = new CycleDeduper<NodeKey>();
    for (ImmutableList<NodeKey> cycle : cycles) {
      if (cycleDeduper.seen(cycle)) {
        numUniqueCycles++;
      }
    }
    assertEquals(1, numUniqueCycles);
  }

  @Test
  public void signalNodeEnqueued() throws Exception {
    final Set<NodeKey> enqueuedNodes = new HashSet<>();
    NodeProgressReceiver progressReceiver = new NodeProgressReceiver() {
      @Override
      public void invalidated(Node node, InvalidationState state) {
        throw new IllegalStateException();
      }

      @Override
      public void enqueueing(NodeKey nodeKey) {
        enqueuedNodes.add(nodeKey);
      }

      @Override
      public void evaluated(NodeKey nodeKey, Node node, EvaluationState state) {}
    };

    ErrorEventListener reporter = new ErrorEventListener() {
      @Override
      public void warn(Location location, String message) {
        throw new IllegalStateException();
      }

      @Override
      public boolean showOutput(String tag) {
        throw new IllegalStateException();
      }

      @Override
      public void report(EventKind kind, Location location, String message) {
        throw new IllegalStateException();
      }

      @Override
      public void progress(Location location, String message) {
        throw new IllegalStateException();
      }

      @Override
      public void info(Location location, String message) {
        throw new IllegalStateException();
      }

      @Override
      public void error(Location location, String message) {
        throw new IllegalStateException();
      }
    };

    AutoUpdatingGraph aug = new InMemoryAutoUpdatingGraph(
        ImmutableMap.of(GraphTester.NODE_TYPE, tester.getNodeBuilder()), progressReceiver);

    tester.getOrCreate("top1").setComputedValue(CONCATENATE)
        .addDependency("d1").addDependency("d2");
    tester.getOrCreate("top2").setComputedValue(CONCATENATE).addDependency("d3");
    tester.getOrCreate("top3");
    ASSERT.that(enqueuedNodes).isEmpty();

    tester.set("d1", new StringNode("1"));
    tester.set("d2", new StringNode("2"));
    tester.set("d3", new StringNode("3"));

    aug.update(ImmutableList.of(GraphTester.toNodeKey("top1")), false,
        AutoUpdatingGraph.DEFAULT_THREAD_COUNT, reporter);
    MoreAsserts.assertContentsAnyOrder(enqueuedNodes, GraphTester.toNodeKeys("top1", "d1", "d2"));
    enqueuedNodes.clear();

    aug.update(ImmutableList.of(GraphTester.toNodeKey("top2")), false,
        AutoUpdatingGraph.DEFAULT_THREAD_COUNT, reporter);
    MoreAsserts.assertContentsAnyOrder(enqueuedNodes, GraphTester.toNodeKeys("top2", "d3"));
    enqueuedNodes.clear();

    aug.update(ImmutableList.of(GraphTester.toNodeKey("top1")), false,
        AutoUpdatingGraph.DEFAULT_THREAD_COUNT, reporter);
    ASSERT.that(enqueuedNodes).isEmpty();
  }
}
