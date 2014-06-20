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
import static com.google.devtools.build.skyframe.GraphTester.COPY;
import static com.google.devtools.build.skyframe.GraphTester.NODE_TYPE;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;
import static org.truth0.Truth.ASSERT;

import com.google.common.base.Preconditions;
import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Sets;
import com.google.common.testing.GcFinalization;
import com.google.common.util.concurrent.Uninterruptibles;
import com.google.devtools.build.lib.events.DelegatingErrorEventListener;
import com.google.devtools.build.lib.events.ErrorEventListener;
import com.google.devtools.build.lib.events.EventCollector;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.testutil.JunitTestUtils;
import com.google.devtools.build.lib.testutil.MoreAsserts;
import com.google.devtools.build.lib.testutil.TestThread;
import com.google.devtools.build.skyframe.AutoUpdatingGraph.NodeProgressReceiver;
import com.google.devtools.build.skyframe.GraphTester.NodeComputer;
import com.google.devtools.build.skyframe.GraphTester.SomeErrorException;
import com.google.devtools.build.skyframe.GraphTester.StringNode;
import com.google.devtools.build.skyframe.GraphTester.TestNodeBuilder;
import com.google.devtools.build.skyframe.NotifyingInMemoryGraph.EventType;
import com.google.devtools.build.skyframe.NotifyingInMemoryGraph.Listener;
import com.google.devtools.build.skyframe.NotifyingInMemoryGraph.Order;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.lang.ref.WeakReference;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

import javax.annotation.Nullable;

/**
 * Tests for {@link AutoUpdatingGraph}.
 */
@RunWith(JUnit4.class)
public class AutoUpdatingGraphTest {

  private AutoUpdatingGraphTester tester;
  private EventCollector eventCollector;
  private ErrorEventListener reporter;
  private AutoUpdatingGraph.EmittedEventState emittedEventState;

  // Knobs that control the size / duration of larger tests.
  private static final int TEST_NODE_COUNT = 100;
  private static final int TESTED_NODES = 10;
  private static final int RUNS = 10;

  @Before
  public void initializeTester() {
    initializeTester(null);
  }

  public void initializeTester(@Nullable TrackingInvalidationReceiver customInvalidationReceiver) {
    emittedEventState = new AutoUpdatingGraph.EmittedEventState();
    tester = new AutoUpdatingGraphTester();
    if (customInvalidationReceiver != null) {
      tester.setInvalidationReceiver(customInvalidationReceiver);
    }
    tester.initialize();
  }

  @Before
  public void initializeReporter() {
    eventCollector = new EventCollector(EventKind.ALL_EVENTS);
    reporter = new Reporter(eventCollector);
    tester.resetPlayedEvents();
  }

  protected static NodeKey toNodeKey(String name) {
    return new NodeKey(NODE_TYPE, name);
  }

  @Test
  public void smoke() throws Exception {
    tester.set("x", new StringNode("y"));
    StringNode node = (StringNode) tester.evalAndGet("x");
    assertEquals("y", node.getValue());
  }

  @Test
  public void invalidationWithNothingChanged() throws Exception {
    tester.set("x", new StringNode("y")).setWarning("fizzlepop");
    StringNode node = (StringNode) tester.evalAndGet("x");
    assertEquals("y", node.getValue());
    JunitTestUtils.assertContainsEvent(eventCollector, "fizzlepop");
    JunitTestUtils.assertEventCount(1, eventCollector);

    initializeReporter();
    tester.invalidate();
    node = (StringNode) tester.evalAndGet("x");
    assertEquals("y", node.getValue());
    JunitTestUtils.assertContainsEvent(eventCollector, "fizzlepop");
    JunitTestUtils.assertEventCount(1, eventCollector);
  }

  private abstract static class NoExtractorNodeBuilder implements NodeBuilder {
    @Override
    public final String extractTag(NodeKey nodeKey) {
      return null;
    }
  }

  @Test
  // Regression test for bug: "[skyframe-m1]: registerIfDone() crash".
  public void bubbleRace() throws Exception {
    // The top-level node declares dependencies on a "badNode" in error, and a "sleepyNode"
    // which is very slow. After "badNode" fails, the builder interrupts the "sleepyNode" and
    // attempts to re-run "top" for error bubbling. Make sure this doesn't cause a precondition
    // failure because "top" still has an outstanding dep ("sleepyNode").
    tester.getOrCreate("top").setBuilder(new NoExtractorNodeBuilder() {
      @Override
      public Node build(NodeKey nodeKey, Environment env) throws InterruptedException {
        env.getDep(toNodeKey("sleepyNode"));
        try {
          env.getDepOrThrow(toNodeKey("badNode"), SomeErrorException.class);
        } catch (SomeErrorException e) {
          // In order to trigger this bug, we need to request a dep on an already computed node.
          env.getDep(toNodeKey("otherNode1"));
        }
        if (!env.depsMissing()) {
          throw new AssertionError("SleepyNode should always be unavailable");
        }
        return null;
      }
    });
    tester.getOrCreate("sleepyNode").setBuilder(new NoExtractorNodeBuilder() {
      @Override
      public Node build(NodeKey nodeKey, Environment env) throws InterruptedException {
        Thread.sleep(99999);
        throw new AssertionError("I should have been interrupted");
      }
    });
    tester.getOrCreate("badNode").addDependency("otherNode1").setHasError(true);
    tester.getOrCreate("otherNode1").setConstantValue(new StringNode("otherVal1"));

    UpdateResult<Node> result = tester.eval(false, "top");
    assertTrue(result.hasError());
    assertEquals(toNodeKey("badNode"), Iterables.getOnlyElement(result.getError().getRootCauses()));
    ASSERT.that(result.keyNames()).isEmpty();
  }

  @Test
  public void deleteNodes() throws Exception {
    tester.getOrCreate("top").setComputedValue(CONCATENATE)
        .addDependency("d1").addDependency("d2").addDependency("d3");
    tester.set("d1", new StringNode("1"));
    StringNode d2 = new StringNode("2");
    tester.set("d2", d2);
    StringNode d3 = new StringNode("3");
    tester.set("d3", d3);
    tester.eval(true, "top");

    tester.delete("d1");
    tester.eval(true, "d3");

    assertEquals(ImmutableSet.of(), tester.getDirtyNodes());
    assertEquals(ImmutableSet.of(new StringNode("1"), new StringNode("123")),
        tester.getDeletedNodes());
    assertEquals(null, tester.getExistingNode("top"));
    assertEquals(null, tester.getExistingNode("d1"));
    assertEquals(d2, tester.getExistingNode("d2"));
    assertEquals(d3, tester.getExistingNode("d3"));
  }

  @Test
  public void deleteNonexistentNodes() throws Exception {
    tester.getOrCreate("d1").setConstantValue(new StringNode("1"));
    tester.delete("d1");
    tester.delete("d2");
    tester.eval(true, "d1");
  }

  @Test
  public void signalNodeEnqueued() throws Exception {
    tester.getOrCreate("top1").setComputedValue(CONCATENATE)
        .addDependency("d1").addDependency("d2");
    tester.getOrCreate("top2").setComputedValue(CONCATENATE).addDependency("d3");
    tester.getOrCreate("top3");
    ASSERT.that(tester.getEnqueuedNodes()).isEmpty();

    tester.set("d1", new StringNode("1"));
    tester.set("d2", new StringNode("2"));
    tester.set("d3", new StringNode("3"));
    tester.eval(true, "top1");
    MoreAsserts.assertContentsAnyOrder(tester.getEnqueuedNodes(),
        AutoUpdatingGraphTester.toNodeKeys("top1", "d1", "d2"));

    tester.eval(true, "top2");
    MoreAsserts.assertContentsAnyOrder(tester.getEnqueuedNodes(),
        AutoUpdatingGraphTester.toNodeKeys("top1", "d1", "d2", "top2", "d3"));
  }

  // NOTE: Some of these tests exercising errors/warnings run through a size-2 for loop in order
  // to ensure that we are properly recording and replyaing these messages on null builds.
  @Test
  public void warningViaMultiplePaths() throws Exception {
    tester.set("d1", new StringNode("d1")).setWarning("warn-d1");
    tester.set("d2", new StringNode("d2")).setWarning("warn-d2");
    tester.getOrCreate("top").setComputedValue(CONCATENATE).addDependency("d1").addDependency("d2");
    for (int i = 0; i < 2; i++) {
      initializeReporter();
      tester.evalAndGet("top");
      JunitTestUtils.assertContainsEvent(eventCollector, "warn-d1");
      JunitTestUtils.assertContainsEvent(eventCollector, "warn-d2");
      JunitTestUtils.assertEventCount(2, eventCollector);
    }
  }

  @Test
  public void warningBeforeErrorOnFailFastBuild() throws Exception {
    tester.set("dep", new StringNode("dep")).setWarning("warn-dep");
    NodeKey topKey = GraphTester.toNodeKey("top");
    tester.getOrCreate(topKey).setHasError(true).addDependency("dep");
    for (int i = 0; i < 2; i++) {
      initializeReporter();
      UpdateResult<StringNode> result = tester.eval(false, "top");
      assertTrue(result.hasError());
      MoreAsserts.assertContentsAnyOrder(result.getError(topKey).getRootCauses(), topKey);
      assertEquals(topKey.toString(), result.getError(topKey).getException().getMessage());
      assertTrue(result.getError(topKey).getException() instanceof
                    GraphTester.SomeErrorException);
      JunitTestUtils.assertContainsEvent(eventCollector, "warn-dep");
      JunitTestUtils.assertEventCount(1, eventCollector);
    }
  }

  @Test
  public void warningAndErrorOnFailFastBuild() throws Exception {
    NodeKey topKey = GraphTester.toNodeKey("top");
    tester.set(topKey, new StringNode("top")).setWarning("warning msg").setHasError(true);
    for (int i = 0; i < 2; i++) {
      initializeReporter();
      UpdateResult<StringNode> result = tester.eval(false, "top");
      assertTrue(result.hasError());
      MoreAsserts.assertContentsAnyOrder(result.getError(topKey).getRootCauses(), topKey);
      assertEquals(topKey.toString(), result.getError(topKey).getException().getMessage());
      assertTrue(result.getError(topKey).getException() instanceof GraphTester.SomeErrorException);
      JunitTestUtils.assertContainsEvent(eventCollector, "warning msg");
      JunitTestUtils.assertEventCount(1, eventCollector);
    }
  }

  @Test
  public void warningAndErrorOnFailFastBuildAfterKeepGoingBuild() throws Exception {
    NodeKey topKey = GraphTester.toNodeKey("top");
    tester.set(topKey, new StringNode("top")).setWarning("warning msg").setHasError(true);
    for (int i = 0; i < 2; i++) {
      initializeReporter();
      UpdateResult<StringNode> result = tester.eval(i == 0, "top");
      assertTrue(result.hasError());
      MoreAsserts.assertContentsAnyOrder(result.getError(topKey).getRootCauses(), topKey);
      assertEquals(topKey.toString(), result.getError(topKey).getException().getMessage());
      assertTrue(result.getError(topKey).getException() instanceof GraphTester.SomeErrorException);
      JunitTestUtils.assertContainsEvent(eventCollector, "warning msg");
      JunitTestUtils.assertEventCount(1, eventCollector);
    }
  }

  @Test
  public void twoTLTsOnOneWarningNode() throws Exception {
    tester.set("t1", new StringNode("t1")).addDependency("dep");
    tester.set("t2", new StringNode("t2")).addDependency("dep");
    tester.set("dep", new StringNode("dep")).setWarning("look both ways before crossing");
    for (int i = 0; i < 2; i++) {
      // Make sure we see the warning exactly once.
      initializeReporter();
      tester.eval(/*keepGoing=*/false, "t1", "t2");
      JunitTestUtils.assertContainsEvent(eventCollector, "look both ways before crossing");
      JunitTestUtils.assertEventCount(1, eventCollector);
    }
  }

  @Test
  public void errorNodeDepOnWarningNode() throws Exception {
    tester.getOrCreate("error-node").setHasError(true).addDependency("warning-node");
    tester.set("warning-node", new StringNode("warning-node"))
        .setWarning("don't chew with your mouth open");

    for (int i = 0; i < 2; i++) {
      initializeReporter();
      tester.evalAndGetError("error-node");
      JunitTestUtils.assertContainsEvent(eventCollector, "don't chew with your mouth open");
      JunitTestUtils.assertEventCount(1, eventCollector);
    }

    initializeReporter();
    tester.evalAndGet("warning-node");
    JunitTestUtils.assertContainsEvent(eventCollector, "don't chew with your mouth open");
    JunitTestUtils.assertEventCount(1, eventCollector);
  }

  @Test
  public void progressMessageOnlyPrintedTheFirstTime() throws Exception {
    // The framework keeps track of warning and error messages, but not progress messages.
    // So here we see both the progress and warning on the first build, but only the warning
    // on the subsequent null build.
    tester.set("x", new StringNode("y")).setWarning("fizzlepop")
        .setProgress("just letting you know");

    StringNode node = (StringNode) tester.evalAndGet("x");
    assertEquals("y", node.getValue());
    JunitTestUtils.assertContainsEvent(eventCollector, "fizzlepop");
    JunitTestUtils.assertContainsEvent(eventCollector, "just letting you know");
    JunitTestUtils.assertEventCount(2, eventCollector);

    // On the rebuild, we only replay warning messages.
    initializeReporter();
    node = (StringNode) tester.evalAndGet("x");
    assertEquals("y", node.getValue());
    JunitTestUtils.assertContainsEvent(eventCollector, "fizzlepop");
    JunitTestUtils.assertEventCount(1, eventCollector);
  }

  @Test
  public void invalidationWithChangeAndThenNothingChanged() throws Exception {
    tester.getOrCreate("a")
        .addDependency("b")
        .setComputedValue(COPY);
    tester.set("b", new StringNode("y"));
    StringNode original = (StringNode) tester.evalAndGet("a");
    assertEquals("y", original.getValue());
    tester.set("b", new StringNode("z"));
    tester.invalidate();
    StringNode old = (StringNode) tester.evalAndGet("a");
    assertEquals("z", old.getValue());
    tester.invalidate();
    StringNode current = (StringNode) tester.evalAndGet("a");
    assertSame(old, current);
  }

  @Test
  public void errorNodeInvalidation() throws Exception {
    // Verify that invalidating errors causes all error nodes to be rerun.
    tester.getOrCreate("error-node").setHasError(true).setProgress("just letting you know");

    tester.evalAndGetError("error-node");
    JunitTestUtils.assertContainsEvent(eventCollector, "just letting you know");
    JunitTestUtils.assertEventCount(1, eventCollector);

    // Change the progress message.
    tester.getOrCreate("error-node").setHasError(true).setProgress("letting you know more");

    // Without invalidating errors, we shouldn't show the new progress message.
    for (int i = 0; i < 2; i++) {
      initializeReporter();
      tester.evalAndGetError("error-node");
      JunitTestUtils.assertNoEvents(eventCollector);
    }

    // When invalidating errors, we should show the new progress message.
    initializeReporter();
    tester.invalidateErrors();
    tester.evalAndGetError("error-node");
    JunitTestUtils.assertContainsEvent(eventCollector, "letting you know more");
    JunitTestUtils.assertEventCount(1, eventCollector);
  }

  @Test
  public void simpleDependency() throws Exception {
    tester.getOrCreate("ab")
        .addDependency("a")
        .setComputedValue(COPY);
    tester.set("a", new StringNode("me"));
    StringNode node = (StringNode) tester.evalAndGet("ab");
    assertEquals("me", node.getValue());
  }

  @Test
  public void incrementalSimpleDependency() throws Exception {
    tester.getOrCreate("ab")
        .addDependency("a")
        .setComputedValue(COPY);
    tester.set("a", new StringNode("me"));
    tester.evalAndGet("ab");

    tester.set("a", new StringNode("other"));
    tester.invalidate();
    StringNode node = (StringNode) tester.evalAndGet("ab");
    assertEquals("other", node.getValue());
  }

  @Test
  public void diamondDependency() throws Exception {
    setupDiamondDependency();
    tester.set("d", new StringNode("me"));
    StringNode node = (StringNode) tester.evalAndGet("a");
    assertEquals("meme", node.getValue());
  }

  @Test
  public void incrementalDiamondDependency() throws Exception {
    setupDiamondDependency();
    tester.set("d", new StringNode("me"));
    tester.evalAndGet("a");

    tester.set("d", new StringNode("other"));
    tester.invalidate();
    StringNode node = (StringNode) tester.evalAndGet("a");
    assertEquals("otherother", node.getValue());
  }

  private void setupDiamondDependency() {
    tester.getOrCreate("a")
        .addDependency("b")
        .addDependency("c")
        .setComputedValue(CONCATENATE);
    tester.getOrCreate("b")
        .addDependency("d")
        .setComputedValue(COPY);
    tester.getOrCreate("c")
        .addDependency("d")
        .setComputedValue(COPY);
  }

  // Regression test: ParallelEvaluator notifies NodeProgressReceiver of already-built top-level
  // nodes in error: we built "top" and "mid" as top-level targets; "mid" contains an error. We make
  // sure "mid" is built as a dependency of "top" before enqueuing mid as a top-level target (by
  // using a latch), so that the top-level enqueuing finds that mid has already been built. The
  // progress receiver should not be notified of any node having been evaluated.
  @Test
  public void testAlreadyAnalyzedBadTarget() throws Exception {
    final NodeKey mid = GraphTester.toNodeKey("mid");
    final CountDownLatch valueSet = new CountDownLatch(1);
    final TrackingAwaiter trackingAwaiter = new TrackingAwaiter();
    tester.graph.setGraphForTesting(new NotifyingInMemoryGraph(new Listener() {
      @Override
      public void accept(NodeKey key, EventType type, Order order, Object context) {
        if (!key.equals(mid)) {
          return;
        }
        switch (type) {
          case ADD_REVERSE_DEP:
            if (context == null) {
              // Context is null when we are enqueuing this node as a top-level job.
              trackingAwaiter.awaitLatchAndTrackExceptions(valueSet, "value not set");
            }
            break;
          case SET_VALUE:
            valueSet.countDown();
            break;
          default:
            break;
        }
      }
    }));
    NodeKey top = GraphTester.nodeKey("top");
    tester.getOrCreate(top).addDependency(mid).setComputedValue(CONCATENATE);
    tester.getOrCreate(mid).setHasError(true);
    tester.eval(/*keepGoing=*/false, top, mid);
    assertEquals(0L, valueSet.getCount());
    trackingAwaiter.assertNoErrors();
    ASSERT.that(tester.invalidationReceiver.evaluated).isEmpty();
  }

  @Test
  public void incrementalAddedDependency() throws Exception {
    tester.getOrCreate("a")
        .addDependency("b")
        .setComputedValue(CONCATENATE);
    tester.set("b", new StringNode("first"));
    tester.set("c", new StringNode("second"));
    tester.evalAndGet("a");

    tester.getOrCreate("a").addDependency("c");
    tester.set("b", new StringNode("now"));
    tester.invalidate();
    StringNode node = (StringNode) tester.evalAndGet("a");
    assertEquals("nowsecond", node.getValue());
  }

  @Test
  public void manyNodesDependOnSingleNode() throws Exception {
    initializeTester();
    String[] nodes = new String[TEST_NODE_COUNT];
    for (int i = 0; i < nodes.length; i++) {
      nodes[i] = Integer.toString(i);
      tester.getOrCreate(nodes[i])
          .addDependency("leaf")
          .setComputedValue(COPY);
    }
    tester.set("leaf", new StringNode("leaf"));

    UpdateResult<StringNode> result = tester.eval(/*keep_going=*/false, nodes);
    for (int i = 0; i < nodes.length; i++) {
      Node actual = result.get(new NodeKey(GraphTester.NODE_TYPE, nodes[i]));
      assertEquals(new StringNode("leaf"), actual);
    }

    for (int j = 0; j < TESTED_NODES; j++) {
      tester.set("leaf", new StringNode("other" + j));
      tester.invalidate();
      result = tester.eval(/*keep_going=*/false, nodes);
      for (int i = 0; i < nodes.length; i++) {
        Node actual = result.get(new NodeKey(GraphTester.NODE_TYPE, nodes[i]));
        assertEquals("Run " + j + ", node " + i, new StringNode("other" + j), actual);
      }
    }
  }

  @Test
  public void singleNodeDependsOnManyNodes() throws Exception {
    initializeTester();
    String[] nodes = new String[TEST_NODE_COUNT];
    StringBuilder expected = new StringBuilder();
    for (int i = 0; i < nodes.length; i++) {
      nodes[i] = Integer.toString(i);
      tester.set(nodes[i], new StringNode(nodes[i]));
      expected.append(nodes[i]);
    }
    NodeKey rootKey = new NodeKey(GraphTester.NODE_TYPE, "root");
    TestNodeBuilder node = tester.getOrCreate(rootKey)
        .setComputedValue(CONCATENATE);
    for (int i = 0; i < nodes.length; i++) {
      node.addDependency(nodes[i]);
    }

    UpdateResult<StringNode> result = tester.eval(/*keep_going=*/false, rootKey);
    assertEquals(new StringNode(expected.toString()), result.get(rootKey));

    for (int j = 0; j < 10; j++) {
      expected.setLength(0);
      for (int i = 0; i < nodes.length; i++) {
        String value = "other" + i + " " + j;
        tester.set(nodes[i], new StringNode(value));
        expected.append(value);
      }
      tester.invalidate();

      result = tester.eval(/*keep_going=*/false, rootKey);
      assertEquals(new StringNode(expected.toString()), result.get(rootKey));
    }
  }

  @Test
  public void twoRailLeftRightDependencies() throws Exception {
    initializeTester();
    String[] leftNodes = new String[TEST_NODE_COUNT];
    String[] rightNodes = new String[TEST_NODE_COUNT];
    for (int i = 0; i < leftNodes.length; i++) {
      leftNodes[i] = "left-" + i;
      rightNodes[i] = "right-" + i;
      if (i == 0) {
        tester.getOrCreate(leftNodes[i])
              .addDependency("leaf")
              .setComputedValue(COPY);
        tester.getOrCreate(rightNodes[i])
              .addDependency("leaf")
              .setComputedValue(COPY);
      } else {
        tester.getOrCreate(leftNodes[i])
              .addDependency(leftNodes[i - 1])
              .addDependency(rightNodes[i - 1])
              .setComputedValue(new PassThroughSelected(toNodeKey(leftNodes[i - 1])));
        tester.getOrCreate(rightNodes[i])
              .addDependency(leftNodes[i - 1])
              .addDependency(rightNodes[i - 1])
              .setComputedValue(new PassThroughSelected(toNodeKey(rightNodes[i - 1])));
      }
    }
    tester.set("leaf", new StringNode("leaf"));

    String lastLeft = "left-" + (TEST_NODE_COUNT - 1);
    String lastRight = "right-" + (TEST_NODE_COUNT - 1);

    UpdateResult<StringNode> result = tester.eval(/*keep_going=*/false, lastLeft, lastRight);
    assertEquals(new StringNode("leaf"), result.get(toNodeKey(lastLeft)));
    assertEquals(new StringNode("leaf"), result.get(toNodeKey(lastRight)));

    for (int j = 0; j < TESTED_NODES; j++) {
      String value = "other" + j;
      tester.set("leaf", new StringNode(value));
      tester.invalidate();
      result = tester.eval(/*keep_going=*/false, lastLeft, lastRight);
      assertEquals(new StringNode(value), result.get(toNodeKey(lastLeft)));
      assertEquals(new StringNode(value), result.get(toNodeKey(lastRight)));
    }
  }

  @Test
  public void noKeepGoingAfterKeepGoingCycle() throws Exception {
    initializeTester();
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
    UpdateResult<StringNode> result = tester.eval(/*keepGoing=*/true, topKey, goodKey);
    assertEquals(goodValue, result.get(goodKey));
    assertEquals(null, result.get(topKey));
    ErrorInfo errorInfo = result.getError(topKey);
    CycleInfo cycleInfo = Iterables.getOnlyElement(errorInfo.getCycleInfo());
    MoreAsserts.assertContentsInOrder(cycleInfo.getCycle(), aKey, bKey);
    MoreAsserts.assertContentsInOrder(cycleInfo.getPathToCycle(), topKey, midKey);

    tester.invalidate();
    result = tester.eval(/*keepGoing=*/false, topKey, goodKey);
    assertEquals(null, result.get(topKey));
    errorInfo = result.getError(topKey);
    cycleInfo = Iterables.getOnlyElement(errorInfo.getCycleInfo());
    MoreAsserts.assertContentsInOrder(cycleInfo.getCycle(), aKey, bKey);
    MoreAsserts.assertContentsInOrder(cycleInfo.getPathToCycle(), topKey, midKey);
  }

  @Test
  public void changeCycle() throws Exception {
    initializeTester();
    NodeKey aKey = GraphTester.toNodeKey("a");
    NodeKey bKey = GraphTester.toNodeKey("b");
    NodeKey topKey = GraphTester.toNodeKey("top");
    NodeKey midKey = GraphTester.toNodeKey("mid");
    tester.getOrCreate(topKey).addDependency(midKey).setComputedValue(COPY);
    tester.getOrCreate(midKey).addDependency(aKey).setComputedValue(COPY);
    tester.getOrCreate(aKey).addDependency(bKey).setComputedValue(COPY);
    tester.getOrCreate(bKey).addDependency(aKey);
    UpdateResult<StringNode> result = tester.eval(/*keepGoing=*/false, topKey);
    assertEquals(null, result.get(topKey));
    ErrorInfo errorInfo = result.getError(topKey);
    CycleInfo cycleInfo = Iterables.getOnlyElement(errorInfo.getCycleInfo());
    MoreAsserts.assertContentsInOrder(cycleInfo.getCycle(), aKey, bKey);
    MoreAsserts.assertContentsInOrder(cycleInfo.getPathToCycle(), topKey, midKey);

    tester.getOrCreate(bKey).removeDependency(aKey);
    tester.set(bKey, new StringNode("bNode"));
    tester.invalidate();
    result = tester.eval(/*keepGoing=*/false, topKey);
    assertEquals(new StringNode("bNode"), result.get(topKey));
    assertEquals(null, result.getError(topKey));
  }

  /** Regression test: "crash in cycle checker with dirty nodes". */
  @Test
  public void cycleAndSelfEdgeWithDirtyNode() throws Exception {
    initializeTester();
    NodeKey cycleKey1 = GraphTester.toNodeKey("cycleKey1");
    NodeKey cycleKey2 = GraphTester.toNodeKey("cycleKey2");
    tester.getOrCreate(cycleKey1).addDependency(cycleKey2).addDependency(cycleKey1)
    .setComputedValue(CONCATENATE);
    tester.getOrCreate(cycleKey2).addDependency(cycleKey1).setComputedValue(COPY);
    UpdateResult<StringNode> result = tester.eval(/*keepGoing=*/true, cycleKey1);
    assertEquals(null, result.get(cycleKey1));
    ErrorInfo errorInfo = result.getError(cycleKey1);
    CycleInfo cycleInfo = Iterables.getOnlyElement(errorInfo.getCycleInfo());
    MoreAsserts.assertContentsInOrder(cycleInfo.getCycle(), cycleKey1);
    ASSERT.that(cycleInfo.getPathToCycle()).isEmpty();
    tester.getOrCreate(cycleKey1, /*markAsModified=*/true);
    tester.invalidate();
    result = tester.eval(/*keepGoing=*/true, cycleKey1, cycleKey2);
    assertEquals(null, result.get(cycleKey1));
    errorInfo = result.getError(cycleKey1);
    cycleInfo = Iterables.getOnlyElement(errorInfo.getCycleInfo());
    MoreAsserts.assertContentsInOrder(cycleInfo.getCycle(), cycleKey1);
    ASSERT.that(cycleInfo.getPathToCycle()).isEmpty();
    cycleInfo =
        Iterables.getOnlyElement(tester.graph.getExistingErrorForTesting(cycleKey2).getCycleInfo());
    MoreAsserts.assertContentsInOrder(cycleInfo.getCycle(), cycleKey1);
    MoreAsserts.assertContentsInOrder(cycleInfo.getPathToCycle(), cycleKey2);
  }

  /** Regression test: "crash in cycle checker with dirty nodes". */
  @Test
  public void cycleWithDirtyNode() throws Exception {
    initializeTester();
    NodeKey cycleKey1 = GraphTester.toNodeKey("cycleKey1");
    NodeKey cycleKey2 = GraphTester.toNodeKey("cycleKey2");
    tester.getOrCreate(cycleKey1).addDependency(cycleKey2).setComputedValue(COPY);
    tester.getOrCreate(cycleKey2).addDependency(cycleKey1).setComputedValue(COPY);
    UpdateResult<StringNode> result = tester.eval(/*keepGoing=*/true, cycleKey1);
    assertEquals(null, result.get(cycleKey1));
    ErrorInfo errorInfo = result.getError(cycleKey1);
    CycleInfo cycleInfo = Iterables.getOnlyElement(errorInfo.getCycleInfo());
    MoreAsserts.assertContentsInOrder(cycleInfo.getCycle(), cycleKey1, cycleKey2);
    ASSERT.that(cycleInfo.getPathToCycle()).isEmpty();
    tester.getOrCreate(cycleKey1, /*markAsModified=*/true);
    tester.invalidate();
    result = tester.eval(/*keepGoing=*/true, cycleKey1);
    assertEquals(null, result.get(cycleKey1));
    errorInfo = result.getError(cycleKey1);
    cycleInfo = Iterables.getOnlyElement(errorInfo.getCycleInfo());
    MoreAsserts.assertContentsInOrder(cycleInfo.getCycle(), cycleKey1, cycleKey2);
    ASSERT.that(cycleInfo.getPathToCycle()).isEmpty();
  }

  /**
   * Regression test: IllegalStateException in BuildingState.isReady(). The ParallelEvaluator used
   * to assume during cycle-checking that all nodes had been built as fully as possible -- that
   * evaluation had not been interrupted. However, we also do cycle-checking in nokeep-going mode
   * when a node throws an error (possibly prematurely shutting down evaluation) but that error then
   * bubbles up into a cycle.
   *
   * <p>We want to achieve the following state: we are checking for a cycle; the node we examine has
   * not yet finished checking its children to see if they are dirty; but all children checked so
   * far have been unchanged. This node is "otherTop". We first build otherTop, then mark its first
   * child changed (without actually changing it), and then do a second build. On the second build,
   * we also build "top", which requests a cycle that depends on an error. We wait to signal
   * otherTop that its first child is done until the error throws and shuts down evaluation. The
   * error then bubbles up to the cycle, and so the bubbling is aborted. Finally, cycle checking
   * happens, and otherTop is examined, as desired.
   */
  @Test
  public void cycleAndErrorAndReady() throws Exception {
    // This node will not have finished building on the second build when the error is thrown.
    final NodeKey otherTop = GraphTester.toNodeKey("otherTop");
    final NodeKey errorKey = GraphTester.toNodeKey("error");
    // Is the graph state all set up and ready for the error to be thrown?
    final CountDownLatch nodesReady = new CountDownLatch(3);
    // Is evaluation being shut down? This is counted down by the exceptionMarker's builder, after
    // it has waited for the threadpool's exception latch to be released.
    final CountDownLatch errorThrown = new CountDownLatch(1);
    // We don't do anything on the first build.
    final AtomicBoolean secondBuild = new AtomicBoolean(false);
    final TrackingAwaiter trackingAwaiter = new TrackingAwaiter();
    tester.graph.setGraphForTesting(new DeterministicInMemoryGraph(new Listener() {
      @Override
      public void accept(NodeKey key, EventType type, Order order, Object context) {
        if (!secondBuild.get()) {
          return;
        }
        if (key.equals(errorKey) && type == EventType.SET_VALUE) {
          // If the error is about to be thrown, make sure all listeners are ready.
          trackingAwaiter.awaitLatchAndTrackExceptions(nodesReady, "waiting nodes not ready");
          return;
        }
        if (key.equals(otherTop) && type == EventType.SIGNAL) {
          // otherTop is being signaled that dep1 is done. Tell the error node that it is ready,
          // then wait until the error is thrown, so that otherTop's builder is not re-entered.
          nodesReady.countDown();
          trackingAwaiter.awaitLatchAndTrackExceptions(errorThrown, "error not thrown");
          return;
        }
      }
    }));
    final NodeKey dep1 = GraphTester.toNodeKey("dep1");
    tester.set(dep1, new StringNode("dep1"));
    final NodeKey dep2 = GraphTester.toNodeKey("dep2");
    tester.set(dep2, new StringNode("dep2"));
    // otherTop should request the deps one at a time, so that it can be in the CHECK_DEPENDENCIES
    // state even after one dep is re-evaluated.
    tester.getOrCreate(otherTop).setBuilder(new NoExtractorNodeBuilder() {
      @Override
      public Node build(NodeKey nodeKey, Environment env) {
        env.getDep(dep1);
        if (env.depsMissing()) {
          return null;
        }
        env.getDep(dep2);
        return env.depsMissing() ? null : new StringNode("otherTop");
      }
    });
    // Prime the graph with otherTop, so we can dirty it next build.
    assertEquals(new StringNode("otherTop"), tester.evalAndGet(/*keepGoing=*/false, otherTop));
    // Mark dep1 changed, so otherTop will be dirty and request re-evaluation of dep1.
    tester.getOrCreate(dep1, /*markAsModified=*/true);
    NodeKey topKey = GraphTester.toNodeKey("top");
    // Note that since DeterministicInMemoryGraph alphabetizes reverse deps, it is important that
    // "cycle2" comes before "top".
    final NodeKey cycle1Key = GraphTester.toNodeKey("cycle1");
    final NodeKey cycle2Key = GraphTester.toNodeKey("cycle2");
    tester.getOrCreate(topKey).addDependency(cycle1Key).setComputedValue(CONCATENATE);
    tester.getOrCreate(cycle1Key).addDependency(errorKey).addDependency(cycle2Key)
        .setComputedValue(CONCATENATE);
    tester.getOrCreate(errorKey).setHasError(true);
    // Make sure cycle2Key has declared its dependence on cycle1Key before error throws.
    tester.getOrCreate(cycle2Key).setBuilder(new ChainedNodeBuilder(/*notifyStart=*/nodesReady,
        null, null, false, new StringNode("never returned"), ImmutableList.<NodeKey>of(cycle1Key)));
    // Node that waits until an exception is thrown to finish building. We use it just to be
    // informed when the threadpool is shutting down.
    final NodeKey exceptionMarker = GraphTester.toNodeKey("exceptionMarker");
    tester.getOrCreate(exceptionMarker).setBuilder(new ChainedNodeBuilder(
        /*notifyStart=*/nodesReady, /*waitToFinish=*/new CountDownLatch(0),
        /*notifyFinish=*/errorThrown,
        /*waitForException=*/true, new StringNode("exception marker"),
        ImmutableList.<NodeKey>of()));
    tester.invalidate();
    secondBuild.set(true);
    // otherTop must be first, since we check top-level nodes for cycles in the order in which
    // they appear here.
    UpdateResult<StringNode> result =
        tester.eval(/*keepGoing=*/false, otherTop, topKey, exceptionMarker);
    trackingAwaiter.assertNoErrors();
    MoreAsserts.assertContentsAnyOrder(result.errorMap().keySet(), topKey);
    Iterable<CycleInfo> cycleInfos = result.getError(topKey).getCycleInfo();
    ASSERT.withFailureMessage(result.toString()).that(cycleInfos).isNotEmpty();
    CycleInfo cycleInfo = Iterables.getOnlyElement(cycleInfos);
    MoreAsserts.assertContentsAnyOrder(cycleInfo.getPathToCycle(), topKey);
    MoreAsserts.assertContentsAnyOrder(cycleInfo.getCycle(), cycle1Key, cycle2Key);
  }

  @Test
  public void limitEvaluatorThreads() throws Exception {
    initializeTester();

    int numKeys = 10;
    final Object lock = new Object();
    final AtomicInteger inProgressCount = new AtomicInteger();
    final int[] maxValue = {0};

    NodeKey topLevel = GraphTester.toNodeKey("toplevel");
    TestNodeBuilder topLevelBuilder = tester.getOrCreate(topLevel);
    for (int i = 0; i < numKeys; i++) {
      topLevelBuilder.addDependency("subKey" + i);
      tester.getOrCreate("subKey" + i).setComputedValue(new NodeComputer() {
        @Override
        public Node compute(Map<NodeKey, Node> deps, NodeBuilder.Environment env) {
          int val = inProgressCount.incrementAndGet();
          synchronized (lock) {
            if (val > maxValue[0]) {
              maxValue[0] = val;
            }
          }
          Uninterruptibles.sleepUninterruptibly(5, TimeUnit.SECONDS);

          inProgressCount.decrementAndGet();
          return new StringNode("abc");
        }
      });
    }
    topLevelBuilder.setConstantValue(new StringNode("xyz"));

    UpdateResult<StringNode> result = tester.eval(/*keepGoing=*/true, /*numThreads=*/5, topLevel);
    assertFalse(result.hasError());
    assertEquals(5, maxValue[0]);
  }

  /**
   * Regression test: error on clearMaybeDirtyNode. We do an update of topKey, which registers
   * dependencies on midKey and errorKey. midKey enqueues slowKey, and waits. errorKey throws an
   * error, which bubbles up to topKey. If topKey does not unregister its dependence on midKey, it
   * will have a dangling reference to midKey after unfinished nodes are cleaned from the graph.
   * Note that slowKey will wait until errorKey has thrown and the threadpool has caught the
   * exception before returning, so the Evaluator will already have stopped enqueuing new jobs, so
   * midKey is not evaluated.
   */
  @Test
  public void incompleteDirectDepsAreClearedBeforeInvalidation() throws Exception {
    initializeTester();
    CountDownLatch slowStart = new CountDownLatch(1);
    CountDownLatch errorFinish = new CountDownLatch(1);
    NodeKey errorKey = GraphTester.toNodeKey("error");
    tester.getOrCreate(errorKey).setBuilder(
        new ChainedNodeBuilder(/*notifyStart=*/null, /*waitToFinish=*/slowStart,
            /*notifyFinish=*/errorFinish, /*waitForException=*/false, /*value=*/null,
            /*deps=*/ImmutableList.<NodeKey>of()));
    NodeKey slowKey = GraphTester.toNodeKey("slow");
    tester.getOrCreate(slowKey).setBuilder(
        new ChainedNodeBuilder(/*notifyStart=*/slowStart, /*waitToFinish=*/errorFinish,
            /*notifyFinish=*/null, /*waitForException=*/true, new StringNode("slow"),
            /*deps=*/ImmutableList.<NodeKey>of()));
    NodeKey midKey = GraphTester.toNodeKey("mid");
    tester.getOrCreate(midKey).addDependency(slowKey).setComputedValue(COPY);
    NodeKey topKey = GraphTester.toNodeKey("top");
    tester.getOrCreate(topKey).addDependency(midKey).addDependency(errorKey)
        .setComputedValue(CONCATENATE);
    // slowKey starts -> errorKey finishes, written to graph -> slowKey finishes & (Visitor aborts)
    // -> topKey builds.
    UpdateResult<StringNode> result = tester.eval(/*keepGoing=*/false, topKey);
    MoreAsserts.assertContentsAnyOrder(result.getError().getRootCauses(), errorKey);
    // Make sure midKey didn't finish building.
    assertEquals(null, tester.graph.getExistingNodeForTesting(midKey));
    // Give slowKey a nice ordinary builder.
    tester.getOrCreate(slowKey, /*markAsModified=*/false).setBuilder(null)
        .setConstantValue(new StringNode("slow"));
    // Put midKey into the graph. It won't have a reverse dependence on topKey.
    tester.evalAndGet(/*keepGoing=*/false, midKey);
    tester.invalidateErrors();
    // topKey should not access midKey as if it were already registered as a dependency.
    tester.eval(/*keepGoing=*/false, topKey);
  }

  /**
   * Regression test: error on clearMaybeDirtyNode. Same as the previous test, but the second update
   * is keepGoing, which should cause an access of the children of topKey.
   */
  @Test
  public void incompleteDirectDepsAreClearedBeforeKeepGoing() throws Exception {
    initializeTester();
    CountDownLatch slowStart = new CountDownLatch(1);
    CountDownLatch errorFinish = new CountDownLatch(1);
    NodeKey errorKey = GraphTester.toNodeKey("error");
    tester.getOrCreate(errorKey).setBuilder(
        new ChainedNodeBuilder(/*notifyStart=*/null, /*waitToFinish=*/slowStart,
            /*notifyFinish=*/errorFinish, /*waitForException=*/false, /*value=*/null,
            /*deps=*/ImmutableList.<NodeKey>of()));
    NodeKey slowKey = GraphTester.toNodeKey("slow");
    tester.getOrCreate(slowKey).setBuilder(
        new ChainedNodeBuilder(/*notifyStart=*/slowStart, /*waitToFinish=*/errorFinish,
            /*notifyFinish=*/null, /*waitForException=*/true, new StringNode("slow"),
            /*deps=*/ImmutableList.<NodeKey>of()));
    NodeKey midKey = GraphTester.toNodeKey("mid");
    tester.getOrCreate(midKey).addDependency(slowKey).setComputedValue(COPY);
    NodeKey topKey = GraphTester.toNodeKey("top");
    tester.getOrCreate(topKey).addDependency(midKey).addDependency(errorKey)
        .setComputedValue(CONCATENATE);
    // slowKey starts -> errorKey finishes, written to graph -> slowKey finishes & (Visitor aborts)
    // -> topKey builds.
    UpdateResult<StringNode> result = tester.eval(/*keepGoing=*/false, topKey);
    MoreAsserts.assertContentsAnyOrder(result.getError().getRootCauses(), errorKey);
    // Make sure midKey didn't finish building.
    assertEquals(null, tester.graph.getExistingNodeForTesting(midKey));
    // Give slowKey a nice ordinary builder.
    tester.getOrCreate(slowKey, /*markAsModified=*/false).setBuilder(null)
        .setConstantValue(new StringNode("slow"));
    // Put midKey into the graph. It won't have a reverse dependence on topKey.
    tester.evalAndGet(/*keepGoing=*/false, midKey);
    // topKey should not access midKey as if it were already registered as a dependency.
    // We don't invalidate errors, but because topKey wasn't actually written to the graph last
    // build, it should be rebuilt here.
    tester.eval(/*keepGoing=*/true, topKey);
  }

  /**
   * Regression test: tests that pass before other build actions fail yield crash in non -k builds.
   */
  @Test
  public void passThenFailToBuild() throws Exception {
    CountDownLatch blocker = new CountDownLatch(1);
    NodeKey successKey = GraphTester.toNodeKey("success");
    tester.getOrCreate(successKey).setBuilder(
        new ChainedNodeBuilder(/*notifyStart=*/null, /*waitToFinish=*/null,
            /*notifyFinish=*/blocker, /*waitForException=*/false, new StringNode("yippee"),
            /*deps=*/ImmutableList.<NodeKey>of()));
    NodeKey slowFailKey = GraphTester.toNodeKey("slow_then_fail");
    tester.getOrCreate(slowFailKey).setBuilder(
        new ChainedNodeBuilder(/*notifyStart=*/null, /*waitToFinish=*/blocker,
            /*notifyFinish=*/null, /*waitForException=*/false, /*value=*/null,
            /*deps=*/ImmutableList.<NodeKey>of()));

    UpdateResult<StringNode> result = tester.eval(/*keepGoing=*/false, successKey, slowFailKey);
    MoreAsserts.assertContentsAnyOrder(result.getError().getRootCauses(), slowFailKey);
    MoreAsserts.assertContentsAnyOrder(result.values(), new StringNode("yippee"));
  }

  @Test
  public void passThenFailToBuildAlternateOrder() throws Exception {
    CountDownLatch blocker = new CountDownLatch(1);
    NodeKey successKey = GraphTester.toNodeKey("success");
    tester.getOrCreate(successKey).setBuilder(
        new ChainedNodeBuilder(/*notifyStart=*/null, /*waitToFinish=*/null,
            /*notifyFinish=*/blocker, /*waitForException=*/false, new StringNode("yippee"),
            /*deps=*/ImmutableList.<NodeKey>of()));
    NodeKey slowFailKey = GraphTester.toNodeKey("slow_then_fail");
    tester.getOrCreate(slowFailKey).setBuilder(
        new ChainedNodeBuilder(/*notifyStart=*/null, /*waitToFinish=*/blocker,
            /*notifyFinish=*/null, /*waitForException=*/false, /*value=*/null,
            /*deps=*/ImmutableList.<NodeKey>of()));

    UpdateResult<StringNode> result = tester.eval(/*keepGoing=*/false, slowFailKey, successKey);
    MoreAsserts.assertContentsAnyOrder(result.getError().getRootCauses(), slowFailKey);
    MoreAsserts.assertContentsAnyOrder(result.values(), new StringNode("yippee"));
  }

  @Test
  public void incompleteDirectDepsForDirtyNode() throws Exception {
    initializeTester();
    NodeKey topKey = GraphTester.toNodeKey("top");
    tester.set(topKey, new StringNode("initial"));
    // Put topKey into graph so it will be dirtied on next run.
    assertEquals(new StringNode("initial"), tester.evalAndGet(/*keepGoing=*/false, topKey));
    CountDownLatch slowStart = new CountDownLatch(1);
    CountDownLatch errorFinish = new CountDownLatch(1);
    NodeKey errorKey = GraphTester.toNodeKey("error");
    tester.getOrCreate(errorKey).setBuilder(
        new ChainedNodeBuilder(/*notifyStart=*/null, /*waitToFinish=*/slowStart,
            /*notifyFinish=*/errorFinish,
            /*waitForException=*/false, /*value=*/null, /*deps=*/ImmutableList.<NodeKey>of()));
    NodeKey slowKey = GraphTester.toNodeKey("slow");
    tester.getOrCreate(slowKey).setBuilder(
        new ChainedNodeBuilder(/*notifyStart=*/slowStart, /*waitToFinish=*/errorFinish,
            /*notifyFinish=*/null, /*waitForException=*/true,
            new StringNode("slow"), /*deps=*/ImmutableList.<NodeKey>of()));
    NodeKey midKey = GraphTester.toNodeKey("mid");
    tester.getOrCreate(midKey).addDependency(slowKey).setComputedValue(COPY);
    tester.set(topKey, null);
    tester.getOrCreate(topKey).addDependency(midKey).addDependency(errorKey)
        .setComputedValue(CONCATENATE);
    tester.invalidate();
    // slowKey starts -> errorKey finishes, written to graph -> slowKey finishes & (Visitor aborts)
    // -> topKey builds.
    UpdateResult<StringNode> result = tester.eval(/*keepGoing=*/false, topKey);
    MoreAsserts.assertContentsAnyOrder(result.getError().getRootCauses(), errorKey);
    // Make sure midKey didn't finish building.
    assertEquals(null, tester.graph.getExistingNodeForTesting(midKey));
    // Give slowKey a nice ordinary builder.
    tester.getOrCreate(slowKey, /*markAsModified=*/false).setBuilder(null)
        .setConstantValue(new StringNode("slow"));
    // Put midKey into the graph. It won't have a reverse dependence on topKey.
    tester.evalAndGet(/*keepGoing=*/false, midKey);
    // topKey should not access midKey as if it were already registered as a dependency.
    // We don't invalidate errors, but since topKey wasn't actually written to the graph before, it
    // will be rebuilt.
    tester.eval(/*keepGoing=*/true, topKey);
  }

  @Test
  public void continueWithErrorDep() throws Exception {
    initializeTester();
    NodeKey errorKey = GraphTester.toNodeKey("my_error_node");
    tester.getOrCreate(errorKey).setHasError(true);
    tester.set("after", new StringNode("after"));
    NodeKey parentKey = GraphTester.toNodeKey("parent");
    tester.getOrCreate(parentKey).addErrorDependency(errorKey, new StringNode("recovered"))
        .setComputedValue(CONCATENATE).addDependency("after");
    UpdateResult<StringNode> result = tester.eval(/*keepGoing=*/true, parentKey);
    ASSERT.that(result.errorMap()).isEmpty();
    assertEquals("recoveredafter", result.get(parentKey).getValue());
    tester.set("after", new StringNode("before"));
    tester.invalidate();
    result = tester.eval(/*keepGoing=*/true, parentKey);
    ASSERT.that(result.errorMap()).isEmpty();
    assertEquals("recoveredbefore", result.get(parentKey).getValue());
  }

  @Test
  public void continueWithErrorDepTurnedGood() throws Exception {
    initializeTester();
    NodeKey errorKey = GraphTester.toNodeKey("my_error_node");
    tester.getOrCreate(errorKey).setHasError(true);
    tester.set("after", new StringNode("after"));
    NodeKey parentKey = GraphTester.toNodeKey("parent");
    tester.getOrCreate(parentKey).addErrorDependency(errorKey, new StringNode("recovered"))
        .setComputedValue(CONCATENATE).addDependency("after");
    UpdateResult<StringNode> result = tester.eval(/*keepGoing=*/true, parentKey);
    ASSERT.that(result.errorMap()).isEmpty();
    assertEquals("recoveredafter", result.get(parentKey).getValue());
    tester.set(errorKey, new StringNode("reformed")).setHasError(false);
    tester.invalidate();
    result = tester.eval(/*keepGoing=*/true, parentKey);
    ASSERT.that(result.errorMap()).isEmpty();
    assertEquals("reformedafter", result.get(parentKey).getValue());
  }

  @Test
  public void errorDepAlreadyThereThenTurnedGood() throws Exception {
    initializeTester();
    NodeKey errorKey = GraphTester.toNodeKey("my_error_node");
    tester.getOrCreate(errorKey).setHasError(true);
    NodeKey parentKey = GraphTester.toNodeKey("parent");
    tester.getOrCreate(parentKey).addErrorDependency(errorKey, new StringNode("recovered"))
        .setHasError(true);
    // Prime the graph by putting the error node in it beforehand.
    MoreAsserts.assertContentsAnyOrder(tester.evalAndGetError(errorKey).getRootCauses(), errorKey);
    UpdateResult<StringNode> result = tester.eval(/*keepGoing=*/false, parentKey);
    // Request the parent.
    ASSERT.that(result.getError(parentKey).getRootCauses()).iteratesAs(parentKey);
    // Change the error node to no longer throw.
    tester.set(errorKey, new StringNode("reformed")).setHasError(false);
    tester.getOrCreate(parentKey, /*markAsModified=*/false).setHasError(false)
        .setComputedValue(COPY);
    tester.invalidate();
    // Request the parent again. This time it should succeed.
    result = tester.eval(/*keepGoing=*/false, parentKey);
    ASSERT.that(result.errorMap()).isEmpty();
    assertEquals("reformed", result.get(parentKey).getValue());
    // Confirm that the parent no longer depends on the error transience node -- make it unbuildable
    // again, but without invalidating it, and invalidate errors. The parent should not be rebuilt.
    tester.getOrCreate(parentKey, /*markAsModified=*/false).setHasError(true);
    tester.invalidateErrors();
    result = tester.eval(/*keepGoing=*/false, parentKey);
    ASSERT.that(result.errorMap()).isEmpty();
    assertEquals("reformed", result.get(parentKey).getValue());
  }

  /**
   * Regression test for 2014 bug: error transience node is registered before newly requested deps.
   * A node requests a child, gets it back immediately, and then throws, causing the error
   * transience node to be registered as a dep. The following build, the error is invalidated via
   * that child.
   */
  @Test
  public void doubleDepOnErrorTransienceNode() throws Exception {
    initializeTester();
    NodeKey leafKey = GraphTester.toNodeKey("leaf");
    tester.set(leafKey, new StringNode("leaf"));
    // Prime the graph by putting leaf in beforehand.
    assertEquals(new StringNode("leaf"), tester.evalAndGet(/*keepGoing=*/false, leafKey));
    NodeKey topKey = GraphTester.toNodeKey("top");
    tester.getOrCreate(topKey).addDependency(leafKey).setHasError(true);
    // Build top -- it has an error.
    ASSERT.that(tester.evalAndGetError(topKey).getRootCauses()).iteratesAs(topKey);
    // Invalidate top via leaf, and rebuild.
    tester.set(leafKey, new StringNode("leaf2"));
    tester.invalidate();
    ASSERT.that(tester.evalAndGetError(topKey).getRootCauses()).iteratesAs(topKey);
  }

  @Test
  public void incompleteNodeAlreadyThereNotUsed() throws Exception {
    initializeTester();
    NodeKey errorKey = GraphTester.toNodeKey("my_error_node");
    tester.getOrCreate(errorKey).setHasError(true);
    NodeKey midKey = GraphTester.toNodeKey("mid");
    tester.getOrCreate(midKey).addErrorDependency(errorKey, new StringNode("recovered"))
        .setComputedValue(COPY);
    NodeKey parentKey = GraphTester.toNodeKey("parent");
    tester.getOrCreate(parentKey).addErrorDependency(midKey, new StringNode("don't use this"))
        .setComputedValue(COPY);
    // Prime the graph by evaluating the mid-level node. It shouldn't be stored in the graph because
    // it was only called during the bubbling-up phase.
    UpdateResult<StringNode> result = tester.eval(/*keepGoing=*/false, midKey);
    assertEquals(null, result.get(midKey));
    MoreAsserts.assertContentsAnyOrder(result.getError().getRootCauses(), errorKey);
    // In a keepGoing build, midKey should be re-evaluated.
    assertEquals("recovered",
        ((StringNode) tester.evalAndGet(/*keepGoing=*/true, parentKey)).getValue());
  }

  /**
   * "top" requests a dependency group in which the first node, called "error", throws an exception,
   * so "mid" and "mid2", which depend on "slow", never get built.
   */
  @Test
  public void errorInDependencyGroup() throws Exception {
    initializeTester();
    NodeKey topKey = GraphTester.toNodeKey("top");
    CountDownLatch slowStart = new CountDownLatch(1);
    CountDownLatch errorFinish = new CountDownLatch(1);
    final NodeKey errorKey = GraphTester.toNodeKey("error");
    tester.getOrCreate(errorKey).setBuilder(
        new ChainedNodeBuilder(/*notifyStart=*/null, /*waitToFinish=*/slowStart,
            /*notifyFinish=*/errorFinish, /*waitForException=*/false,
            // ChainedNodeBuilder throws when value is null.
            /*value=*/null, /*deps=*/ImmutableList.<NodeKey>of()));
    NodeKey slowKey = GraphTester.toNodeKey("slow");
    tester.getOrCreate(slowKey).setBuilder(
        new ChainedNodeBuilder(/*notifyStart=*/slowStart, /*waitToFinish=*/errorFinish,
            /*notifyFinish=*/null, /*waitForException=*/true,
            new StringNode("slow"), /*deps=*/ImmutableList.<NodeKey>of()));
    final NodeKey midKey = GraphTester.toNodeKey("mid");
    tester.getOrCreate(midKey).addDependency(slowKey).setComputedValue(COPY);
    final NodeKey mid2Key = GraphTester.toNodeKey("mid2");
    tester.getOrCreate(mid2Key).addDependency(slowKey).setComputedValue(COPY);
    tester.set(topKey, null);
    tester.getOrCreate(topKey).setBuilder(new NodeBuilder() {
      @Override
      public Node build(NodeKey nodeKey, Environment env) throws NodeBuilderException,
          InterruptedException {
        env.getDeps(ImmutableList.of(errorKey, midKey, mid2Key));
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

    // Assert that build fails and "error" really is in error.
    UpdateResult<StringNode> result = tester.eval(/*keepGoing=*/false, topKey);
    assertTrue(result.hasError());
    MoreAsserts.assertContentsAnyOrder(result.getError(topKey).getRootCauses(), errorKey);

    // Ensure that evaluation succeeds if errorKey does not throw an error.
    tester.getOrCreate(errorKey).setBuilder(null);
    tester.set(errorKey, new StringNode("ok"));
    tester.invalidate();
    assertEquals(new StringNode("top"), tester.evalAndGet("top"));
  }

  @Test
  public void errorOnlyEmittedOnce() throws Exception {
    initializeTester();
    tester.set("x", new StringNode("y")).setWarning("fizzlepop");
    StringNode node = (StringNode) tester.evalAndGet("x");
    assertEquals("y", node.getValue());
    JunitTestUtils.assertContainsEvent(eventCollector, "fizzlepop");
    JunitTestUtils.assertEventCount(1, eventCollector);

    tester.invalidate();
    node = (StringNode) tester.evalAndGet("x");
    assertEquals("y", node.getValue());
    // No new events emitted.
    JunitTestUtils.assertEventCount(1, eventCollector);
  }

  /**
   * We are checking here that we are resilient to a race condition in which a node that is checking
   * its children for dirtiness is signaled by all of its children, putting it in a ready state,
   * before the thread has terminated. Optionally, one of its children may throw an error, shutting
   * down the threadpool. This is similar to {@link ParallelEvaluatorTest#slowChildCleanup}: a child
   * about to throw signals its parent and the parent's builder restarts itself before the exception
   * is thrown. Here, the signaling happens while dirty dependencies are being checked, as opposed
   * to during actual evaluation, but the principle is the same. We control the timing by blocking
   * "top"'s registering itself on its deps 
   */
  private void dirtyChildEnqueuesParentDuringCheckDependencies(boolean throwError)
      throws Exception {
    // Node to be built. It will be signaled to rebuild before it has finished checking its deps.
    final NodeKey top = GraphTester.toNodeKey("top");
    // Dep that blocks before it acknowledges being added as a dep by top, so the firstKey node has
    // time to signal top.
    final NodeKey slowAddingDep = GraphTester.toNodeKey("dep");
    // Don't perform any blocking on the first build.
    final AtomicBoolean delayTopSignaling = new AtomicBoolean(false);
    final CountDownLatch topSignaled = new CountDownLatch(1);
    final CountDownLatch topRestartedBuild = new CountDownLatch(1);
    final TrackingAwaiter trackingAwaiter = new TrackingAwaiter();
    tester.graph.setGraphForTesting(new NotifyingInMemoryGraph(new Listener() {
      @Override
      public void accept(NodeKey key, EventType type, Order order, Object context) {
        if (!delayTopSignaling.get()) {
          return;
        }
        if (key.equals(top) && type == EventType.SIGNAL && order == Order.AFTER) {
          // top is signaled by firstKey (since slowAddingDep is blocking), so slowAddingDep is now
          // free to acknowledge top as a parent.
          topSignaled.countDown();
          return;
        }
        if (key.equals(slowAddingDep) && type == EventType.ADD_REVERSE_DEP
            && context.equals(top) && order == Order.BEFORE) {
          // If top is trying to declare a dep on slowAddingDep, wait until firstKey has signaled
          // top. Then this add dep will return DONE and top will be signaled, making it ready, so
          // it will be enqueued.
          trackingAwaiter.awaitLatchAndTrackExceptions(topSignaled,
              "first key didn't signal top in time");
        }
      }
    }));
    // Node that is modified on the second build. Its thread won't finish until it signals top,
    // which will wait for the signal before it enqueues its next dep. We prevent the thread from
    // finishing by having the listener to which it reports its warning block until top's builder
    // starts.
    final NodeKey firstKey = GraphTester.nodeKey("first");
    tester.set(firstKey, new StringNode("biding"));
    tester.set(slowAddingDep, new StringNode("dep"));
    // top's builder just requests both deps in a group.
    tester.getOrCreate(top).setBuilder(new NoExtractorNodeBuilder() {
      @Override
      public Node build(NodeKey key, NodeBuilder.Environment env) {
        if (delayTopSignaling.get()) {
          // The reporter will be given firstKey's warning to emit when it is requested as a dep
          // below, if firstKey is already built, so we release the reporter's latch beforehand.
          topRestartedBuild.countDown();
        }
        env.getDepsOrThrow(ImmutableList.of(firstKey, slowAddingDep), Exception.class);
        return env.depsMissing() ? null : new StringNode("top");
      }
    });
    reporter = new DelegatingErrorEventListener(reporter) {
      @Override
      public void warn(Location location, String message) {
        super.warn(location, message);
        trackingAwaiter.awaitLatchAndTrackExceptions(topRestartedBuild,
            "top's builder did not start in time");
      }
    };
    // First build : just prime the graph.
    UpdateResult<StringNode> result = tester.eval(/*keepGoing=*/false, top);
    assertFalse(result.hasError());
    assertEquals(new StringNode("top"), result.get(top));
    // Now dirty the graph, and maybe have firstKey throw an error.
    String warningText = "warning text";
    tester.getOrCreate(firstKey, /*markAsModified=*/true).setHasError(throwError)
        .setWarning(warningText);
    tester.invalidate();
    delayTopSignaling.set(true);
    result = tester.eval(/*keepGoing=*/false, top);
    trackingAwaiter.assertNoErrors();
    if (throwError) {
      assertTrue(result.hasError());
      ASSERT.that(result.keyNames()).isEmpty(); // No successfully evaluated nodes.
      ErrorInfo errorInfo = result.getError(top);
      MoreAsserts.assertContentsAnyOrder(errorInfo.getRootCauses(), firstKey);
    } else {
      assertEquals(new StringNode("top"), result.get(top));
      assertFalse(result.hasError());
    }
    JunitTestUtils.assertContainsEvent(eventCollector, warningText);
    assertEquals(0, topSignaled.getCount());
    assertEquals(0, topRestartedBuild.getCount());
  }

  @Test
  public void dirtyChildEnqueuesParentDuringCheckDependencies_Throw() throws Exception {
    dirtyChildEnqueuesParentDuringCheckDependencies(/*throwError=*/true);
  }

  @Test
  public void dirtyChildEnqueuesParentDuringCheckDependencies_NoThrow() throws Exception {
    dirtyChildEnqueuesParentDuringCheckDependencies(/*throwError=*/false);
  }

  /**
   * The same dep is requested in two groups, but its value determines what the other dep in the
   * second group is. When it changes, the other dep in the second group should not be requested.
   */
  @Test
  public void sameDepInTwoGroups() throws Exception {
    initializeTester();

    // leaf4 should not built in the second build.
    final NodeKey leaf4 = GraphTester.toNodeKey("leaf4");
    final AtomicBoolean shouldNotBuildLeaf4 = new AtomicBoolean(false);
    tester.graph.setGraphForTesting(new NotifyingInMemoryGraph(new Listener() {
      @Override
      public void accept(NodeKey key, EventType type, Order order, Object context) {
        if (shouldNotBuildLeaf4.get() && key.equals(leaf4)) {
          throw new IllegalStateException("leaf4 should not have been considered this build: "
              + type + ", " + order + ", " + context);
        }
      }
    }));
    tester.set(leaf4, new StringNode("leaf4"));

    // Create leaf0, leaf1 and leaf2 nodes with values "leaf2", "leaf3", "leaf4" respectively.
    // These will be requested as one dependency group. In the second build, leaf2 will have the
    // value "leaf5".
    final List<NodeKey> leaves = new ArrayList<>();
    for (int i = 0; i <= 2; i++) {
      NodeKey leaf = GraphTester.toNodeKey("leaf" + i);
      leaves.add(leaf);
      tester.set(leaf, new StringNode("leaf" + (i + 2)));
    }

    // Create "top" node. It depends on all leaf nodes in two overlapping dependency groups.
    NodeKey topKey = GraphTester.toNodeKey("top");
    final Node topValue = new StringNode("top");
    tester.getOrCreate(topKey).setBuilder(new NoExtractorNodeBuilder() {
      @Override
      public Node build(NodeKey nodeKey, Environment env) throws NodeBuilderException,
          InterruptedException {
        // Request the first group, [leaf0, leaf1, leaf2].
        // In the first build, it has values ["leaf2", "leaf3", "leaf4"].
        // In the second build it has values ["leaf2", "leaf3", "leaf5"]
        Map<NodeKey, Node> nodes = env.getDeps(leaves);
        if (env.depsMissing()) {
          return null;
        }

        // Request the second group. In the first build it's [leaf2, leaf4].
        // In the second build it's [leaf2, leaf5]
        env.getDeps(ImmutableList.of(leaves.get(2),
            GraphTester.toNodeKey(((StringNode) nodes.get(leaves.get(2))).getValue())));
        if (env.depsMissing()) {
          return null;
        }

        return topValue;
      }
    });

    // First build: assert we can evaluate "top".
    assertEquals(topValue, tester.evalAndGet(/*keepGoing=*/false, topKey));

    // Second build: replace "leaf4" by "leaf5" in leaf2's value. Assert leaf4 is not requested.
    final NodeKey leaf5 = GraphTester.toNodeKey("leaf5");
    tester.set(leaf5, new StringNode("leaf5"));
    tester.set(leaves.get(2), new StringNode("leaf5"));
    tester.invalidate();
    shouldNotBuildLeaf4.set(true);
    assertEquals(topValue, tester.evalAndGet(/*keepGoing=*/false, topKey));
  }

  @Test
  public void dirtyAndChanged() throws Exception {
    initializeTester();
    NodeKey leaf = GraphTester.toNodeKey("leaf");
    NodeKey mid = GraphTester.toNodeKey("mid");
    NodeKey top = GraphTester.toNodeKey("top");
    tester.getOrCreate(top).addDependency(mid).setComputedValue(COPY);
    tester.getOrCreate(mid).addDependency(leaf).setComputedValue(COPY);
    tester.set(leaf, new StringNode("leafy"));
    // For invalidation.
    tester.set("dummy", new StringNode("dummy"));
    StringNode topNode = (StringNode) tester.evalAndGet("top");
    assertEquals("leafy", topNode.getValue());
    tester.set(leaf, new StringNode("crunchy"));
    tester.invalidate();
    // For invalidation.
    tester.evalAndGet("dummy");
    tester.getOrCreate(mid, /*markAsModified=*/true);
    tester.invalidate();
    topNode = (StringNode) tester.evalAndGet("top");
    assertEquals("crunchy", topNode.getValue());
  }

  /**
   * Test whether a node that was already marked changed will be incorrectly marked dirty, not
   * changed, if another thread tries to mark it just dirty. To exercise this, we need to have a
   * race condition where both threads see that the node is not dirty yet, then the "changed" thread
   * marks the node changed before the "dirty" thread marks the node dirty. To accomplish this, we
   * use a countdown latch to make the "dirty" thread wait until the "changed" thread is done, and
   * another countdown latch to make both of them wait until they have both checked if the node
   * is currently clean.
   */
  @Test
  public void dirtyAndChangedNodeIsChanged() throws Exception {
    final NodeKey parent = GraphTester.toNodeKey("parent");
    final AtomicBoolean blockingEnabled = new AtomicBoolean(false);
    final CountDownLatch waitForChanged = new CountDownLatch(1);
    // changed thread checks node entry once (to see if it is changed). dirty thread checks twice,
    // to see if it is changed, and if it is dirty.
    final CountDownLatch threadsStarted = new CountDownLatch(3);
    final TrackingAwaiter trackingAwaiter = new TrackingAwaiter();
    tester.graph.setGraphForTesting(new NotifyingInMemoryGraph(new Listener() {
      @Override
      public void accept(NodeKey key, EventType type, Order order, Object context) {
        if (!blockingEnabled.get()) {
          return;
        }
        if (!key.equals(parent)) {
          return;
        }
        if (type == EventType.IS_CHANGED && order == Order.BEFORE) {
          threadsStarted.countDown();
        }
        // Dirtiness only checked by dirty thread.
        if (type == EventType.IS_DIRTY && order == Order.BEFORE) {
          threadsStarted.countDown();
        }
        if (type == EventType.MARK_DIRTY) {
          trackingAwaiter.awaitLatchAndTrackExceptions(threadsStarted,
              "Both threads did not query if node isChanged in time");
          boolean isChanged = (Boolean) context;
          if (order == Order.BEFORE && !isChanged) {
            trackingAwaiter.awaitLatchAndTrackExceptions(waitForChanged,
                "'changed' thread did not mark node changed in time");
            return;
          }
          if (order == Order.AFTER && isChanged) {
            waitForChanged.countDown();
          }
        }
      }
    }));
    NodeKey leaf = GraphTester.toNodeKey("leaf");
    tester.set(leaf, new StringNode("leaf"));
    tester.getOrCreate(parent).addDependency(leaf).setComputedValue(CONCATENATE);
    UpdateResult<StringNode> result;
    result = tester.eval(/*keepGoing=*/false, parent);
    assertEquals("leaf", result.get(parent).getValue());
    // Invalidate leaf, but don't actually change it. It will transitively dirty parent
    // concurrently with parent directly dirtying itself.
    tester.getOrCreate(leaf, /*markAsModified=*/true);
    NodeKey other2 = GraphTester.toNodeKey("other2");
    tester.set(other2, new StringNode("other2"));
    // Invalidate parent, actually changing it.
    tester.getOrCreate(parent, /*markAsModified=*/true).addDependency(other2);
    tester.invalidate();
    blockingEnabled.set(true);
    result = tester.eval(/*keepGoing=*/false, parent);
    assertEquals("leafother2", result.get(parent).getValue());
    trackingAwaiter.assertNoErrors();
    assertEquals(0, waitForChanged.getCount());
    assertEquals(0, threadsStarted.getCount());
  }

  @Test
  public void singleNodeDependsOnManyDirtyNodes() throws Exception {
    initializeTester();
    NodeKey[] nodes = new NodeKey[TEST_NODE_COUNT];
    StringBuilder expected = new StringBuilder();
    for (int i = 0; i < nodes.length; i++) {
      String nodeName = Integer.toString(i);
      nodes[i] = GraphTester.toNodeKey(nodeName);
      tester.set(nodes[i], new StringNode(nodeName));
      expected.append(nodeName);
    }
    NodeKey topKey = new NodeKey(GraphTester.NODE_TYPE, "top");
    TestNodeBuilder node = tester.getOrCreate(topKey)
        .setComputedValue(CONCATENATE);
    for (int i = 0; i < nodes.length; i++) {
      node.addDependency(nodes[i]);
    }

    UpdateResult<StringNode> result = tester.eval(/*keepGoing=*/false, topKey);
    assertEquals(new StringNode(expected.toString()), result.get(topKey));

    for (int j = 0; j < RUNS; j++) {
      for (int i = 0; i < nodes.length; i++) {
        tester.getOrCreate(nodes[i], /*markAsModified=*/true);
      }
      // This node has an error, but we should never discover it because it is not marked changed
      // and all of its dependencies re-evaluate to the same thing.
      tester.getOrCreate(topKey, /*markAsModified=*/false).setHasError(true);
      tester.invalidate();

      result = tester.eval(/*keep_going=*/false, topKey);
      assertEquals(new StringNode(expected.toString()), result.get(topKey));
    }
  }

  /**
   * Tests scenario where we have dirty nodes in the graph, and then one of them is deleted since
   * its evaluation did not complete before an error was thrown. Can either test the graph via an
   * update of that deleted node, or an invalidation of a child, and can either remove the thrown
   * error or throw it again on that update.
   */
  private void dirtyNodeChildrenProperlyRemovedOnEarlyBuildAbort(
      boolean reevaluateMissingNode, boolean removeError) throws Exception {
    initializeTester();
    NodeKey errorKey = GraphTester.toNodeKey("error");
    tester.set(errorKey, new StringNode("biding time"));
    NodeKey slowKey = GraphTester.toNodeKey("slow");
    tester.set(slowKey, new StringNode("slow"));
    NodeKey midKey = GraphTester.toNodeKey("mid");
    tester.getOrCreate(midKey).addDependency(slowKey).setComputedValue(COPY);
    NodeKey lastKey = GraphTester.toNodeKey("last");
    tester.set(lastKey, new StringNode("last"));
    NodeKey motherKey = GraphTester.toNodeKey("mother");
    tester.getOrCreate(motherKey).addDependency(errorKey)
        .addDependency(midKey).addDependency(lastKey).setComputedValue(CONCATENATE);
    NodeKey fatherKey = GraphTester.toNodeKey("father");
    tester.getOrCreate(fatherKey).addDependency(errorKey)
        .addDependency(midKey).addDependency(lastKey).setComputedValue(CONCATENATE);
    UpdateResult<StringNode> result = tester.eval(/*keepGoing=*/false, motherKey, fatherKey);
    assertEquals("biding timeslowlast", result.get(motherKey).getValue());
    assertEquals("biding timeslowlast", result.get(fatherKey).getValue());
    tester.set(slowKey, null);
    // Each parent depends on errorKey, midKey, lastKey. We keep slowKey waiting until errorKey is
    // finished. So there is no way lastKey can be enqueued by either parent. Thus, the parent that
    // is cleaned has not interacted with lastKey this build. Still, lastKey's reverse dep on that
    // parent should be removed.
    CountDownLatch errorFinish = new CountDownLatch(1);
    tester.set(errorKey, null);
    tester.getOrCreate(errorKey).setBuilder(
        new ChainedNodeBuilder(/*notifyStart=*/null, /*waitToFinish=*/null,
            /*notifyFinish=*/errorFinish, /*waitForException=*/false, /*value=*/null,
            /*deps=*/ImmutableList.<NodeKey>of()));
    tester.getOrCreate(slowKey).setBuilder(
        new ChainedNodeBuilder(/*notifyStart=*/null, /*waitToFinish=*/errorFinish,
            /*notifyFinish=*/null, /*waitForException=*/true, new StringNode("leaf2"),
            /*deps=*/ImmutableList.<NodeKey>of()));
    tester.invalidate();
    // errorKey finishes, written to graph -> leafKey maybe starts+finishes & (Visitor aborts)
    // -> one of mother or father builds. The other one should be cleaned, and no references to it
    // left in the graph.
    result = tester.eval(/*keepGoing=*/false, motherKey, fatherKey);
    assertTrue(result.hasError());
    // Only one of mother or father should be in the graph.
    assertTrue(result.getError(motherKey) + ", " + result.getError(fatherKey),
        (result.getError(motherKey) == null) != (result.getError(fatherKey) == null));
    NodeKey parentKey = (reevaluateMissingNode == (result.getError(motherKey) == null)) ?
        motherKey : fatherKey;
    // Give slowKey a nice ordinary builder.
    tester.getOrCreate(slowKey, /*markAsModified=*/false).setBuilder(null)
        .setConstantValue(new StringNode("leaf2"));
    if (removeError) {
      tester.getOrCreate(errorKey, /*markAsModified=*/true).setBuilder(null)
          .setConstantValue(new StringNode("reformed"));
    }
    String lastString = "last";
    if (!reevaluateMissingNode) {
      // Mark the last key modified if we're not trying the absent node again. This invalidation
      // will test if lastKey still has a reference to the absent node.
      lastString = "last2";
      tester.set(lastKey, new StringNode(lastString));
    }
    tester.invalidate();
    result = tester.eval(/*keepGoing=*/false, parentKey);
    if (removeError) {
      assertEquals("reformedleaf2" + lastString, result.get(parentKey).getValue());
    } else {
      assertNotNull(result.getError(parentKey));
    }
  }

  /**
   * The following four tests (dirtyChildrenProperlyRemovedWith*) test the consistency of the graph
   * after a failed build in which a dirty node should have been deleted from the graph. The
   * consistency is tested via either evaluating the missing node, or the re-evaluating the present
   * node, and either clearing the error or keeping it. To evaluate the present node, we invalidate
   * the error node to force re-evaluation. Related to bug "skyframe m1: graph may not be properly
   * cleaned on interrupt or failure".
   */
  @Test
  public void dirtyChildrenProperlyRemovedWithInvalidateRemoveError() throws Exception {
    dirtyNodeChildrenProperlyRemovedOnEarlyBuildAbort(/*reevaluateMissingNode=*/false,
        /*removeError=*/true);
  }

  @Test
  public void dirtyChildrenProperlyRemovedWithInvalidateKeepError() throws Exception {
    dirtyNodeChildrenProperlyRemovedOnEarlyBuildAbort(/*reevaluateMissingNode=*/false,
        /*removeError=*/false);
  }

  @Test
  public void dirtyChildrenProperlyRemovedWithReevaluateRemoveError() throws Exception {
    dirtyNodeChildrenProperlyRemovedOnEarlyBuildAbort(/*reevaluateMissingNode=*/true,
        /*removeError=*/true);
  }

  @Test
  public void dirtyChildrenProperlyRemovedWithReevaluateKeepError() throws Exception {
    dirtyNodeChildrenProperlyRemovedOnEarlyBuildAbort(/*reevaluateMissingNode=*/true,
        /*removeError=*/false);
  }

  /**
   * Regression test: enqueue so many nodes that some of them won't have started processing, and
   * then either interrupt processing or have a child throw an error. In the latter case, this also
   * tests that a node that hasn't started processing can still have a child error bubble up to it.
   * In both cases, it tests that the graph is properly cleaned of the dirty nodes and references to
   * them.
   */
  private void manyDirtyNodesClearChildrenOnFail(boolean interrupt) throws Exception {
    NodeKey leafKey = GraphTester.toNodeKey("leaf");
    tester.set(leafKey, new StringNode("leafy"));
    NodeKey lastKey = GraphTester.toNodeKey("last");
    tester.set(lastKey, new StringNode("last"));
    final List<NodeKey> tops = new ArrayList<>();
    // Request far more top-level nodes than there are threads, so some of them will block until the
    // leaf child is enqueued for processing.
    for (int i = 0; i < 10000; i++) {
      NodeKey topKey = GraphTester.toNodeKey("top" + i);
      tester.getOrCreate(topKey).addDependency(leafKey).addDependency(lastKey)
          .setComputedValue(CONCATENATE);
      tops.add(topKey);
    }
    tester.eval(/*keepGoing=*/false, tops.toArray(new NodeKey[0]));
    final CountDownLatch notifyStart = new CountDownLatch(1);
    tester.set(leafKey, null);
    if (interrupt) {
      // leaf will wait for an interrupt if desired. We cannot use the usual ChainedNodeBuilder
      // because we need to actually throw the interrupt.
      final AtomicBoolean shouldSleep = new AtomicBoolean(true);
      tester.getOrCreate(leafKey, /*markAsModified=*/true).setBuilder(
          new NoExtractorNodeBuilder() {
            @Override
            public Node build(NodeKey nodeKey, Environment env) throws InterruptedException {
              notifyStart.countDown();
              if (shouldSleep.get()) {
                // Should be interrupted within 5 seconds.
                Thread.sleep(5000);
                throw new AssertionError("leaf was not interrupted");
              }
              return new StringNode("crunchy");
            }
          });
      tester.invalidate();
      TestThread evalThread = new TestThread() {
        @Override
        public void runTest() {
          try {
            tester.eval(/*keepGoing=*/false, tops.toArray(new NodeKey[0]));
            Assert.fail();
          } catch (InterruptedException e) {
            // Expected.
          }
        }
      };
      evalThread.start();
      assertTrue(notifyStart.await(2, TimeUnit.SECONDS));
      evalThread.interrupt();
      // 5000 ms arbitrarily chosen because 1000 caused occasional flakiness.
      evalThread.joinAndAssertState(5000);
      // Free leafKey to compute next time.
      shouldSleep.set(false);
    } else {
      // Non-interrupt case. Just throw an error in the child.
      tester.getOrCreate(leafKey, /*markAsModified=*/true).setHasError(true);
      tester.invalidate();
      // The error thrown may non-deterministically bubble up to a parent that has not yet started
      // processing, but has been enqueued for processing.
      tester.eval(/*keepGoing=*/false, tops.toArray(new NodeKey[0]));
      tester.getOrCreate(leafKey, /*markAsModified=*/true).setHasError(false);
      tester.set(leafKey, new StringNode("crunchy"));
    }
    // lastKey was not touched during the previous build, but its reverse deps on its parents should
    // still be accurate.
    tester.set(lastKey, new StringNode("new last"));
    tester.invalidate();
    UpdateResult<StringNode> result =
        tester.eval(/*keepGoing=*/false, tops.toArray(new NodeKey[0]));
    for (NodeKey topKey : tops) {
      assertEquals(topKey.toString(), "crunchynew last", result.get(topKey).getValue());
    }
  }

  /**
   * Regression test: make sure that if an evaluation fails before a dirty node starts evaluation
   * (in particular, before it is reset), the graph remains consistent.
   */
  @Test
  public void manyDirtyNodesClearChildrenOnError() throws Exception {
    manyDirtyNodesClearChildrenOnFail(/*interrupt=*/false);
  }

  /**
   * Regression test: Make sure that if an evaluation is interrupted before a dirty node starts
   * evaluation (in particular, before it is reset), the graph remains consistent.
   */
  @Test
  public void manyDirtyNodesClearChildrenOnInterrupt() throws Exception {
    manyDirtyNodesClearChildrenOnFail(/*interrupt=*/true);
  }

  @Test
  public void changePruning() throws Exception {
    initializeTester();
    NodeKey leaf = GraphTester.toNodeKey("leaf");
    NodeKey mid = GraphTester.toNodeKey("mid");
    NodeKey top = GraphTester.toNodeKey("top");
    tester.getOrCreate(top).addDependency(mid).setComputedValue(COPY);
    tester.getOrCreate(mid).addDependency(leaf).setComputedValue(COPY);
    tester.set(leaf, new StringNode("leafy"));
    StringNode topNode = (StringNode) tester.evalAndGet("top");
    assertEquals("leafy", topNode.getValue());
    // Mark leaf changed, but don't actually change it.
    tester.getOrCreate(leaf, /*markAsModified=*/true);
    // mid will give an error if re-evaluated, but it shouldn't be because it is not marked changed,
    // and its dirty child will evaluate to the same element.
    tester.getOrCreate(mid, /*markAsModified=*/false).setHasError(true);
    tester.invalidate();
    UpdateResult<StringNode> result = tester.eval(/*keepGoing=*/false, top);
    assertFalse(result.hasError());
    topNode = result.get(top);
    assertEquals("leafy", topNode.getValue());
    ASSERT.that(tester.getDirtyNodes()).isEmpty();
    ASSERT.that(tester.getDeletedNodes()).isEmpty();
  }

  @Test
  public void changePruningWithDoneNode() throws Exception {
    initializeTester();
    NodeKey leaf = GraphTester.toNodeKey("leaf");
    NodeKey mid = GraphTester.toNodeKey("mid");
    NodeKey top = GraphTester.toNodeKey("top");
    NodeKey suffix = GraphTester.toNodeKey("suffix");
    StringNode suffixNode = new StringNode("suffix");
    tester.set(suffix, suffixNode);
    tester.getOrCreate(top).addDependency(mid).addDependency(suffix).setComputedValue(CONCATENATE);
    tester.getOrCreate(mid).addDependency(leaf).addDependency(suffix).setComputedValue(CONCATENATE);
    Node leafyNode = new StringNode("leafy");
    tester.set(leaf, leafyNode);
    StringNode node = (StringNode) tester.evalAndGet("top");
    assertEquals("leafysuffixsuffix", node.getValue());
    // Mark leaf changed, but don't actually change it.
    tester.getOrCreate(leaf, /*markAsModified=*/true);
    // mid will give an error if re-evaluated, but it shouldn't be because it is not marked changed,
    // and its dirty child will evaluate to the same element.
    tester.getOrCreate(mid, /*markAsModified=*/false).setHasError(true);
    tester.invalidate();
    node = (StringNode) tester.evalAndGet("leaf");
    assertEquals("leafy", node.getValue());
    MoreAsserts.assertContentsAnyOrder(tester.getDirtyNodes(), new StringNode("leafysuffix"),
        new StringNode("leafysuffixsuffix"));
    ASSERT.that(tester.getDeletedNodes()).isEmpty();
    UpdateResult<StringNode> result = tester.eval(/*keepGoing=*/false, top);
    assertFalse(result.hasError());
    node = result.get(top);
    assertEquals("leafysuffixsuffix", node.getValue());
    ASSERT.that(tester.getDirtyNodes()).isEmpty();
    ASSERT.that(tester.getDeletedNodes()).isEmpty();
  }

  @Test
  public void changedChildChangesDepOfParent() throws Exception {
    initializeTester();
    final NodeKey buildFile = GraphTester.toNodeKey("buildFile");
    NodeComputer authorDrink = new NodeComputer() {
      @Override
      public Node compute(Map<NodeKey, Node> deps, NodeBuilder.Environment env) {
        String author = ((StringNode) deps.get(buildFile)).getValue();
        StringNode beverage;
        switch (author) {
          case "hemingway":
            beverage = (StringNode) env.getDep(GraphTester.toNodeKey("absinthe"));
            break;
          case "joyce":
            beverage = (StringNode) env.getDep(GraphTester.toNodeKey("whiskey"));
            break;
          default:
              throw new IllegalStateException(author);
        }
        if (beverage == null) {
          return null;
        }
        return new StringNode(author + " drank " + beverage.getValue());
      }
    };

    tester.set(buildFile, new StringNode("hemingway"));
    NodeKey absinthe = GraphTester.toNodeKey("absinthe");
    tester.set(absinthe, new StringNode("absinthe"));
    NodeKey whiskey = GraphTester.toNodeKey("whiskey");
    tester.set(whiskey, new StringNode("whiskey"));
    NodeKey top = GraphTester.toNodeKey("top");
    tester.getOrCreate(top).addDependency(buildFile).setComputedValue(authorDrink);
    StringNode topNode = (StringNode) tester.evalAndGet("top");
    assertEquals("hemingway drank absinthe", topNode.getValue());
    tester.set(buildFile, new StringNode("joyce"));
    // Don't evaluate absinthe successfully anymore.
    tester.getOrCreate(absinthe).setHasError(true);
    tester.invalidate();
    topNode = (StringNode) tester.evalAndGet("top");
    assertEquals("joyce drank whiskey", topNode.getValue());
    MoreAsserts.assertContentsAnyOrder(tester.getDirtyNodes(), new StringNode("hemingway"),
        new StringNode("hemingway drank absinthe"));
    ASSERT.that(tester.getDeletedNodes()).isEmpty();
  }

  @Test
  public void dirtyDepIgnoresChildren() throws Exception {
    initializeTester();
    NodeKey leaf = GraphTester.toNodeKey("leaf");
    NodeKey mid = GraphTester.toNodeKey("mid");
    NodeKey top = GraphTester.toNodeKey("top");
    tester.set(mid, new StringNode("ignore"));
    tester.getOrCreate(top).addDependency(mid).setComputedValue(COPY);
    tester.getOrCreate(mid).addDependency(leaf);
    tester.set(leaf, new StringNode("leafy"));
    StringNode topNode = (StringNode) tester.evalAndGet("top");
    assertEquals("ignore", topNode.getValue());
    ASSERT.that(tester.getDirtyNodes()).isEmpty();
    ASSERT.that(tester.getDeletedNodes()).isEmpty();
    // Change leaf.
    tester.set(leaf, new StringNode("crunchy"));
    tester.invalidate();
    topNode = (StringNode) tester.evalAndGet("top");
    assertEquals("ignore", topNode.getValue());
    MoreAsserts.assertContentsAnyOrder(tester.getDirtyNodes(), new StringNode("leafy"));
    ASSERT.that(tester.getDeletedNodes()).isEmpty();
    tester.set(leaf, new StringNode("smushy"));
    tester.invalidate();
    topNode = (StringNode) tester.evalAndGet("top");
    assertEquals("ignore", topNode.getValue());
    MoreAsserts.assertContentsAnyOrder(tester.getDirtyNodes(), new StringNode("crunchy"));
    ASSERT.that(tester.getDeletedNodes()).isEmpty();
  }

  private static final NodeBuilder INTERRUPT_BUILDER = new NodeBuilder() {

    @Override
    public Node build(NodeKey nodeKey, Environment env) throws NodeBuilderException,
        InterruptedException {
      throw new InterruptedException();
    }

    @Override
    public String extractTag(NodeKey nodeKey) {
      throw new UnsupportedOperationException();
    }
  };

  /**
   * Utility function to induce a graph clean of whatever node is requested, by trying to build this
   * node and interrupting the build as soon as this node's builder starts.
   */
  private void failBuildAndRemoveNode(final NodeKey node) {
    tester.set(node, null);
    // Evaluator will think leaf was interrupted because it threw, so it will be cleaned from graph.
    tester.getOrCreate(node, /*markAsModified=*/true).setBuilder(INTERRUPT_BUILDER);
    tester.invalidate();
    try {
      tester.eval(/*keepGoing=*/false, node);
      Assert.fail();
    } catch (InterruptedException e) {
      // Expected.
    }
    tester.getOrCreate(node, /*markAsModified=*/false).setBuilder(null);
  }

  /**
   * Make sure that when a dirty node is building, the fact that a child may no longer exist in the
   * graph doesn't cause problems.
   */
  @Test
  public void dirtyBuildAfterFailedBuild() throws Exception {
    initializeTester();
    final NodeKey leaf = GraphTester.toNodeKey("leaf");
    NodeKey top = GraphTester.toNodeKey("top");
    tester.getOrCreate(top).addDependency(leaf).setComputedValue(COPY);
    tester.set(leaf, new StringNode("leafy"));
    StringNode topNode = (StringNode) tester.evalAndGet("top");
    assertEquals("leafy", topNode.getValue());
    ASSERT.that(tester.getDirtyNodes()).isEmpty();
    ASSERT.that(tester.getDeletedNodes()).isEmpty();
    failBuildAndRemoveNode(leaf);
    // Leaf should no longer exist in the graph. Check that this doesn't cause problems.
    tester.set(leaf, null);
    tester.set(leaf, new StringNode("crunchy"));
    tester.invalidate();
    topNode = (StringNode) tester.evalAndGet("top");
    assertEquals("crunchy", topNode.getValue());
  }

  /**
   * Regression test: error when clearing reverse deps on dirty node about to be rebuilt, because
   * child nodes were deleted and recreated in interim, forgetting they had reverse dep on dirty
   * node in the first place.
   */
  @Test
  public void changedBuildAfterFailedThenSuccessfulBuild() throws Exception {
    initializeTester();
    final NodeKey leaf = GraphTester.toNodeKey("leaf");
    NodeKey top = GraphTester.toNodeKey("top");
    tester.getOrCreate(top).addDependency(leaf).setComputedValue(COPY);
    tester.set(leaf, new StringNode("leafy"));
    StringNode topNode = (StringNode) tester.evalAndGet("top");
    assertEquals("leafy", topNode.getValue());
    ASSERT.that(tester.getDirtyNodes()).isEmpty();
    ASSERT.that(tester.getDeletedNodes()).isEmpty();
    failBuildAndRemoveNode(leaf);
    tester.set(leaf, new StringNode("crunchy"));
    tester.invalidate();
    tester.eval(/*keepGoing=*/false, leaf);
    // Leaf no longer has reverse dep on top. Check that this doesn't cause problems, even if the
    // top node is evaluated unconditionally.
    tester.getOrCreate(top, /*markAsModified=*/true);
    tester.invalidate();
    topNode = (StringNode) tester.evalAndGet("top");
    assertEquals("crunchy", topNode.getValue());
  }

  /**
   * Regression test: child node that has been deleted since it and its parent were marked dirty no
   * longer knows it has a reverse dep on its parent.
   *
   * <p>Start with:
   * <pre>
   *              top0  ... top1000
   *                  \  | /
   *                   leaf
   * </pre>
   * Then fail to build leaf. Now the entry for leaf should have no "memory" that it was ever
   * depended on by tops. Now build tops, but fail again.
   */
  @Test
  public void manyDirtyNodesClearChildrenOnSecondFail() throws Exception {
    final NodeKey leafKey = GraphTester.toNodeKey("leaf");
    tester.set(leafKey, new StringNode("leafy"));
    NodeKey lastKey = GraphTester.toNodeKey("last");
    tester.set(lastKey, new StringNode("last"));
    final List<NodeKey> tops = new ArrayList<>();
    // Request far more top-level nodes than there are threads, so some of them will block until the
    // leaf child is enqueued for processing.
    for (int i = 0; i < 10000; i++) {
      NodeKey topKey = GraphTester.toNodeKey("top" + i);
      tester.getOrCreate(topKey).addDependency(leafKey).addDependency(lastKey)
          .setComputedValue(CONCATENATE);
      tops.add(topKey);
    }
    tester.eval(/*keepGoing=*/false, tops.toArray(new NodeKey[0]));
    failBuildAndRemoveNode(leafKey);
    // Request the tops. Since leaf was deleted from the graph last build, it no longer knows that
    // its parents depend on it. When leaf throws, at least one of its parents (hopefully) will not
    // have re-informed leaf that the parent depends on it, exposing the bug, since the parent
    // should then not try to clean the reverse dep from leaf.
    tester.set(leafKey, null);
    // Evaluator will think leaf was interrupted because it threw, so it will be cleaned from graph.
    tester.getOrCreate(leafKey, /*markAsModified=*/true).setBuilder(INTERRUPT_BUILDER);
    tester.invalidate();
    try {
      tester.eval(/*keepGoing=*/false, tops.toArray(new NodeKey[0]));
      Assert.fail();
    } catch (InterruptedException e) {
      // Expected.
    }
  }

  @Test
  public void failedDirtyBuild() throws Exception {
    initializeTester();
    NodeKey leaf = GraphTester.toNodeKey("leaf");
    NodeKey top = GraphTester.toNodeKey("top");
    tester.getOrCreate(top).addErrorDependency(leaf, new StringNode("recover"))
        .setComputedValue(COPY);
    tester.set(leaf, new StringNode("leafy"));
    StringNode topNode = (StringNode) tester.evalAndGet("top");
    assertEquals("leafy", topNode.getValue());
    ASSERT.that(tester.getDirtyNodes()).isEmpty();
    ASSERT.that(tester.getDeletedNodes()).isEmpty();
    // Change leaf.
    tester.getOrCreate(leaf, /*markAsModified=*/true).setHasError(true);
    tester.getOrCreate(top, /*markAsModified=*/false).setHasError(true);
    tester.invalidate();
    UpdateResult<StringNode> result = tester.eval(/*keepGoing=*/false, top);
    assertNull("node should not have completed evaluation", result.get(top));
    MoreAsserts.assertContentsAnyOrder(
        "The error thrown by leaf should have been swallowed by the error thrown by top",
        result.getError().getRootCauses(), top);
  }

  @Test
  public void failedDirtyBuildInBuilder() throws Exception {
    initializeTester();
    NodeKey leaf = GraphTester.toNodeKey("leaf");
    NodeKey secondError = GraphTester.toNodeKey("secondError");
    NodeKey top = GraphTester.toNodeKey("top");
    tester.getOrCreate(top).addDependency(leaf)
        .addErrorDependency(secondError, new StringNode("recover")).setComputedValue(CONCATENATE);
    tester.set(secondError, new StringNode("secondError")).addDependency(leaf);
    tester.set(leaf, new StringNode("leafy"));
    StringNode topNode = (StringNode) tester.evalAndGet("top");
    assertEquals("leafysecondError", topNode.getValue());
    ASSERT.that(tester.getDirtyNodes()).isEmpty();
    ASSERT.that(tester.getDeletedNodes()).isEmpty();
    // Invalidate leaf.
    tester.getOrCreate(leaf, /*markAsModified=*/true);
    tester.set(leaf, new StringNode("crunchy"));
    tester.getOrCreate(secondError, /*markAsModified=*/true).setHasError(true);
    tester.getOrCreate(top, /*markAsModified=*/false).setHasError(true);
    tester.invalidate();
    UpdateResult<StringNode> result = tester.eval(/*keepGoing=*/false, top);
    assertNull("node should not have completed evaluation", result.get(top));
    MoreAsserts.assertContentsAnyOrder(
        "The error thrown by leaf should have been swallowed by the error thrown by top",
        result.getError().getRootCauses(), top);
  }

  @Test
  public void dirtyErrorTransienceNode() throws Exception {
    initializeTester();
    NodeKey error = GraphTester.toNodeKey("error");
    tester.getOrCreate(error).setHasError(true);
    assertNotNull(tester.evalAndGetError(error));
    tester.invalidateErrors();
    NodeKey secondError = GraphTester.toNodeKey("secondError");
    tester.getOrCreate(secondError).setHasError(true);
    // secondError declares a new dependence on ErrorTransienceNode, but not until it has already
    // thrown an error.
    assertNotNull(tester.evalAndGetError(secondError));
  }

  @Test
  public void dirtyDependsOnErrorTurningGood() throws Exception {
    initializeTester();
    NodeKey error = GraphTester.toNodeKey("error");
    tester.getOrCreate(error).setHasError(true);
    NodeKey topKey = GraphTester.toNodeKey("top");
    tester.getOrCreate(topKey).addDependency(error).setComputedValue(COPY);
    UpdateResult<StringNode> result = tester.eval(/*keepGoing=*/false, topKey);
    MoreAsserts.assertContentsAnyOrder(result.getError(topKey).getRootCauses(), error);
    tester.invalidateErrors();
    result = tester.eval(/*keepGoing=*/false, topKey);
    MoreAsserts.assertContentsAnyOrder(result.getError(topKey).getRootCauses(), error);
    tester.getOrCreate(error).setHasError(false);
    StringNode val = new StringNode("reformed");
    tester.set(error, val);
    tester.invalidate();
    result = tester.eval(/*keepGoing=*/false, topKey);
    assertEquals(val, result.get(topKey));
    assertFalse(result.hasError());
  }

  @Test
  public void dirtyWithOwnErrorDependsOnErrorTurningGood() throws Exception {
    initializeTester();
    final NodeKey error = GraphTester.toNodeKey("error");
    tester.getOrCreate(error).setHasError(true);
    NodeKey topKey = GraphTester.toNodeKey("top");
    final AtomicBoolean firstTime = new AtomicBoolean(true);
    NodeBuilder errorNodeBuilder = new NodeBuilder() {
      @Override
      public Node build(NodeKey nodeKey, Environment env) throws GenericNodeBuilderException,
          InterruptedException {
        env.getDep(error);
        if (firstTime.getAndSet(false)) {
          return null;
        }
        throw new GenericNodeBuilderException(nodeKey, new Exception());
      }

      @Override
      public String extractTag(NodeKey nodeKey) {
        throw new UnsupportedOperationException();
      }
    };
    tester.getOrCreate(topKey).setBuilder(errorNodeBuilder);
    UpdateResult<StringNode> result = tester.eval(/*keepGoing=*/false, topKey);
    MoreAsserts.assertContentsAnyOrder(result.getError(topKey).getRootCauses(), topKey);
    tester.invalidateErrors();
    firstTime.set(true);
    result = tester.eval(/*keepGoing=*/false, topKey);
    MoreAsserts.assertContentsAnyOrder(result.getError(topKey).getRootCauses(), topKey);
    tester.getOrCreate(error).setHasError(false);
    StringNode reformed = new StringNode("reformed");
    tester.set(error, reformed);
    tester.getOrCreate(topKey).setBuilder(null).addDependency(error).setComputedValue(COPY);
    tester.invalidate();
    result = tester.eval(/*keepGoing=*/false, topKey);
    assertEquals(reformed, result.get(topKey));
    assertFalse(result.hasError());
  }

  /**
   * Make sure that when an error is thrown, it is given for handling only to parents that have
   * already registered a dependence on the node that threw the error.
   *
   * <pre>
   *  topBubbleKey  topErrorFirstKey
   *    |       \    /
   *  midKey  errorKey
   *    |
   * slowKey
   * </pre>
   *
   * On the second build, errorKey throws, and the threadpool aborts before midKey finishes.
   * topBubbleKey therefore has not yet requested errorKey this build. If errorKey bubbles up to it,
   * topBubbleKey must be able to handle that. (The evaluator can deal with this either by not
   * allowing errorKey to bubble up to topBubbleKey, or by dealing with that case.)
   */
  @Test
  public void errorOnlyBubblesToRequestingParents() throws Exception {
    // We need control over the order of reverse deps, so use a deterministic graph.
    tester.graph.setGraphForTesting(new DeterministicInMemoryGraph());
    NodeKey errorKey = GraphTester.toNodeKey("error");
    tester.set(errorKey, new StringNode("biding time"));
    NodeKey slowKey = GraphTester.toNodeKey("slow");
    tester.set(slowKey, new StringNode("slow"));
    NodeKey midKey = GraphTester.toNodeKey("mid");
    tester.getOrCreate(midKey).addDependency(slowKey).setComputedValue(COPY);
    NodeKey topErrorFirstKey = GraphTester.toNodeKey("2nd top alphabetically");
    tester.getOrCreate(topErrorFirstKey).addDependency(errorKey).setComputedValue(CONCATENATE);
    NodeKey topBubbleKey = GraphTester.toNodeKey("1st top alphabetically");
    tester.getOrCreate(topBubbleKey).addDependency(midKey).addDependency(errorKey)
        .setComputedValue(CONCATENATE);
    // First error-free evaluation, to put all nodes in graph.
    UpdateResult<StringNode> result = tester.eval(/*keepGoing=*/false,
        topErrorFirstKey, topBubbleKey);
    assertEquals("biding time", result.get(topErrorFirstKey).getValue());
    assertEquals("slowbiding time", result.get(topBubbleKey).getValue());
    // Set up timing of child nodes: slowKey waits to finish until errorKey has thrown an
    // exception that has been caught by the threadpool.
    tester.set(slowKey, null);
    CountDownLatch errorFinish = new CountDownLatch(1);
    tester.set(errorKey, null);
    tester.getOrCreate(errorKey).setBuilder(
        new ChainedNodeBuilder(/*notifyStart=*/null, /*waitToFinish=*/null,
            /*notifyFinish=*/errorFinish, /*waitForException=*/false, /*value=*/null,
            /*deps=*/ImmutableList.<NodeKey>of()));
    tester.getOrCreate(slowKey).setBuilder(
        new ChainedNodeBuilder(/*notifyStart=*/null, /*waitToFinish=*/errorFinish,
            /*notifyFinish=*/null, /*waitForException=*/true, new StringNode("leaf2"),
            /*deps=*/ImmutableList.<NodeKey>of()));
    tester.invalidate();
    // errorKey finishes, written to graph -> slowKey maybe starts+finishes & (Visitor aborts)
    // -> some top key builds.
    result = tester.eval(/*keepGoing=*/false, topErrorFirstKey, topBubbleKey);
    assertTrue(result.hasError());
    assertNotNull(result.getError(topErrorFirstKey));
  }

  @Test
  public void dirtyWithRecoveryErrorDependsOnErrorTurningGood() throws Exception {
    initializeTester();
    final NodeKey error = GraphTester.toNodeKey("error");
    tester.getOrCreate(error).setHasError(true);
    NodeKey topKey = GraphTester.toNodeKey("top");
    NodeBuilder recoveryErrorNodeBuilder = new NodeBuilder() {
      @Override
      public Node build(NodeKey nodeKey, Environment env) throws NodeBuilderException,
          InterruptedException {
        try {
          env.getDepOrThrow(error, SomeErrorException.class);
        } catch (SomeErrorException e) {
          throw new GenericNodeBuilderException(nodeKey, e);
        }
        return null;
      }

      @Override
      public String extractTag(NodeKey nodeKey) {
        throw new UnsupportedOperationException();
      }
    };
    tester.getOrCreate(topKey).setBuilder(recoveryErrorNodeBuilder);
    UpdateResult<StringNode> result = tester.eval(/*keepGoing=*/false, topKey);
    MoreAsserts.assertContentsAnyOrder(result.getError(topKey).getRootCauses(), topKey);
    tester.invalidateErrors();
    result = tester.eval(/*keepGoing=*/false, topKey);
    MoreAsserts.assertContentsAnyOrder(result.getError(topKey).getRootCauses(), topKey);
    tester.getOrCreate(error).setHasError(false);
    StringNode reformed = new StringNode("reformed");
    tester.set(error, reformed);
    tester.getOrCreate(topKey).setBuilder(null).addDependency(error).setComputedValue(COPY);
    tester.invalidate();
    result = tester.eval(/*keepGoing=*/false, topKey);
    assertEquals(reformed, result.get(topKey));
    assertFalse(result.hasError());
  }


  @Test
  public void absentParent() throws Exception {
    initializeTester();
    NodeKey errorKey = GraphTester.toNodeKey("my_error_node");
    tester.set(errorKey, new StringNode("biding time"));
    NodeKey absentParentKey = GraphTester.toNodeKey("absentParent");
    tester.getOrCreate(absentParentKey).addDependency(errorKey).setComputedValue(CONCATENATE);
    assertEquals(new StringNode("biding time"),
        tester.evalAndGet(/*keepGoing=*/false, absentParentKey));
    tester.getOrCreate(errorKey, /*markAsModified=*/true).setHasError(true);
    NodeKey newParent = GraphTester.toNodeKey("newParent");
    tester.getOrCreate(newParent).addDependency(errorKey).setComputedValue(CONCATENATE);
    tester.invalidate();
    UpdateResult<StringNode> result = tester.eval(/*keepGoing=*/false, newParent);
    ErrorInfo error = result.getError(newParent);
    MoreAsserts.assertContentsAnyOrder(error.getRootCauses(), errorKey);
  }

  // Tests that we have a sane implementation of error transience.
  @Test
  public void errorTransienceBug() throws Exception {
    tester.getOrCreate("key").setHasError(true);
    assertNotNull(tester.evalAndGetError("key").getException());
    StringNode node = new StringNode("hi");
    tester.getOrCreate("key").setHasError(false).setConstantValue(node);
    tester.invalidateErrors();
    assertEquals(node, tester.evalAndGet("key"));
    // This works because the version of the NodeEntry for the ErrorTransience node is always
    // increased on each InMemoryAutoUpdatingGraph#update call. But that's not the only way to
    // implement error transience; another valid implementation would be to unconditionally mark
    // nodes depending on the ErrorTransience node as being changed (rather than merely dirtied)
    // during invalidation.
  }

  @Test
  public void transientErrorTurningGoodHasNoError() throws Exception {
    initializeTester();
    NodeKey errorKey = GraphTester.toNodeKey("my_error_node");
    tester.getOrCreate(errorKey).setHasError(true);
    ErrorInfo errorInfo = tester.evalAndGetError(errorKey);
    assertNotNull(errorInfo);
    MoreAsserts.assertContentsAnyOrder(errorInfo.getRootCauses(), errorKey);
    // Re-evaluates to same thing when errors are invalidated
    tester.invalidateErrors();
    errorInfo = tester.evalAndGetError(errorKey);
    assertNotNull(errorInfo);
    StringNode value = new StringNode("reformed");
    MoreAsserts.assertContentsAnyOrder(errorInfo.getRootCauses(), errorKey);
    tester.getOrCreate(errorKey, /*markAsModified=*/false).setHasError(false)
        .setConstantValue(value);
    tester.invalidateErrors();
    StringNode node = (StringNode) tester.evalAndGet(/*keepGoing=*/true, errorKey);
    assertSame(node, value);
    // Node builder will now throw, but we should never get to it because it isn't dirty.
    tester.getOrCreate(errorKey, /*markAsModified=*/false).setHasError(true);
    tester.invalidateErrors();
    node = (StringNode) tester.evalAndGet(/*keepGoing=*/true, errorKey);
    assertSame(node, value);
  }

  @Test
  public void deleteInvalidatedNode() throws Exception {
    initializeTester();
    NodeKey top = GraphTester.toNodeKey("top");
    NodeKey toDelete = GraphTester.toNodeKey("toDelete");
    // Must be a concatenation -- COPY doesn't actually copy.
    tester.getOrCreate(top).addDependency(toDelete).setComputedValue(CONCATENATE);
    tester.set(toDelete, new StringNode("toDelete"));
    Node node = tester.evalAndGet("top");
    NodeKey forceInvalidation = GraphTester.toNodeKey("forceInvalidation");
    tester.set(forceInvalidation, new StringNode("forceInvalidation"));
    tester.getOrCreate(toDelete, /*markAsModified=*/true);
    tester.invalidate();
    tester.eval(/*keepGoing=*/false, forceInvalidation);
    tester.delete("toDelete");
    WeakReference<Node> ref = new WeakReference<>(node);
    node = null;
    tester.eval(/*keepGoing=*/false, forceInvalidation);
    tester.invalidate(); // So that invalidation receiver doesn't hang on to reference.
    GcFinalization.awaitClear(ref);
  }

  /**
   * General stress/fuzz test of the evaluator with failure. Construct a large graph, and then throw
   * exceptions during building at various points.
   */
  @Test
  public void twoRailLeftRightDependenciesWithFailure() throws Exception {
    initializeTester();
    NodeKey[] leftNodes = new NodeKey[TEST_NODE_COUNT];
    NodeKey[] rightNodes = new NodeKey[TEST_NODE_COUNT];
    for (int i = 0; i < TEST_NODE_COUNT; i++) {
      leftNodes[i] = GraphTester.toNodeKey("left-" + i);
      rightNodes[i] = GraphTester.toNodeKey("right-" + i);
      if (i == 0) {
        tester.getOrCreate(leftNodes[i])
              .addDependency("leaf")
              .setComputedValue(COPY);
        tester.getOrCreate(rightNodes[i])
              .addDependency("leaf")
              .setComputedValue(COPY);
      } else {
        tester.getOrCreate(leftNodes[i])
              .addDependency(leftNodes[i - 1])
              .addDependency(rightNodes[i - 1])
              .setComputedValue(new PassThroughSelected(leftNodes[i - 1]));
        tester.getOrCreate(rightNodes[i])
              .addDependency(leftNodes[i - 1])
              .addDependency(rightNodes[i - 1])
              .setComputedValue(new PassThroughSelected(rightNodes[i - 1]));
      }
    }
    tester.set("leaf", new StringNode("leaf"));

    String lastLeft = "left-" + (TEST_NODE_COUNT - 1);
    String lastRight = "right-" + (TEST_NODE_COUNT - 1);

    for (int i = 0; i < TESTED_NODES; i++) {
      try {
        tester.getOrCreate(leftNodes[i], /*markAsModified=*/true).setHasError(true);
        tester.invalidate();
        UpdateResult<StringNode> result = tester.eval(/*keep_going=*/false, lastLeft, lastRight);
        assertTrue(result.hasError());
        tester.invalidateErrors();
        result = tester.eval(/*keep_going=*/false, lastLeft, lastRight);
        assertTrue(result.hasError());
        tester.getOrCreate(leftNodes[i], /*markAsModified=*/true).setHasError(false);
        tester.invalidate();
        result = tester.eval(/*keep_going=*/false, lastLeft, lastRight);
        assertEquals(new StringNode("leaf"), result.get(toNodeKey(lastLeft)));
        assertEquals(new StringNode("leaf"), result.get(toNodeKey(lastRight)));
      } catch (Exception e) {
        System.err.println("twoRailLeftRightDependenciesWithFailure exception on run " + i);
        throw e;
      }
    }
  }

  @Test
  public void nodeInjection() throws Exception {
    NodeKey key = GraphTester.toNodeKey("new_node");
    Node val = new StringNode("val");

    tester.graph.inject(ImmutableMap.of(key, val));
    assertEquals(val, tester.evalAndGet("new_node"));
  }

  @Test
  public void nodeInjectionOverExistingEntry() throws Exception {
    NodeKey key = GraphTester.toNodeKey("node");
    Node val = new StringNode("val");

    tester.getOrCreate(key).setConstantValue(new StringNode("old_val"));
    tester.graph.inject(ImmutableMap.of(key, val));
    assertEquals(val, tester.evalAndGet("node"));
  }

  @Test
  public void nodeInjectionOverExistingDirtyEntry() throws Exception {
    NodeKey key = GraphTester.toNodeKey("node");
    Node val = new StringNode("val");

    tester.getOrCreate(key).setConstantValue(new StringNode("old_val"));
    tester.graph.inject(ImmutableMap.of(key, val));
    tester.eval(/*keepGoing=*/false, new NodeKey[0]); // Create the node.

    tester.graph.invalidate(ImmutableList.of(key));
    tester.eval(/*keepGoing=*/false, new NodeKey[0]); // Mark node as dirty.

    tester.graph.inject(ImmutableMap.of(key, val));
    tester.eval(/*keepGoing=*/false, new NodeKey[0]); // Inject again.
    assertEquals(val, tester.evalAndGet("node"));
  }

  @Test
  public void nodeInjectionOverExistingEntryMarkedForInvalidation() throws Exception {
    NodeKey key = GraphTester.toNodeKey("node");
    Node val = new StringNode("val");

    tester.getOrCreate(key).setConstantValue(new StringNode("old_val"));
    tester.graph.invalidate(ImmutableList.of(key));
    tester.graph.inject(ImmutableMap.of(key, val));
    assertEquals(val, tester.evalAndGet("node"));
  }

  @Test
  public void nodeInjectionOverExistingEntryMarkedForDeletion() throws Exception {
    NodeKey key = GraphTester.toNodeKey("node");
    Node val = new StringNode("val");

    tester.getOrCreate(key).setConstantValue(new StringNode("old_val"));
    tester.graph.delete(Predicates.<NodeKey>alwaysTrue());
    tester.graph.inject(ImmutableMap.of(key, val));
    assertEquals(val, tester.evalAndGet("node"));
  }

  @Test
  public void nodeInjectionOverExistingEqualEntryMarkedForInvalidation() throws Exception {
    NodeKey key = GraphTester.toNodeKey("node");
    Node val = new StringNode("val");

    tester.graph.inject(ImmutableMap.of(key, val));
    assertEquals(val, tester.evalAndGet("node"));

    tester.graph.invalidate(ImmutableList.of(key));
    tester.graph.inject(ImmutableMap.of(key, val));
    assertEquals(val, tester.evalAndGet("node"));
  }

  @Test
  public void nodeInjectionOverExistingEqualEntryMarkedForDeletion() throws Exception {
    NodeKey key = GraphTester.toNodeKey("node");
    Node val = new StringNode("val");

    tester.graph.inject(ImmutableMap.of(key, val));
    assertEquals(val, tester.evalAndGet("node"));

    tester.graph.delete(Predicates.<NodeKey>alwaysTrue());
    tester.graph.inject(ImmutableMap.of(key, val));
    assertEquals(val, tester.evalAndGet("node"));
  }

  @Test
  public void nodeInjectionOverNodeWithDeps() throws Exception {
    NodeKey key = GraphTester.toNodeKey("node");
    Node val = new StringNode("val");
    StringNode prevVal = new StringNode("foo");

    tester.getOrCreate("other").setConstantValue(prevVal);
    tester.getOrCreate(key).addDependency("other").setComputedValue(COPY);
    assertEquals(prevVal, tester.evalAndGet("node"));
    tester.graph.inject(ImmutableMap.of(key, val));
    try {
      tester.evalAndGet("node");
      Assert.fail("injection over node with deps should have failed");
    } catch (IllegalStateException e) {
      assertEquals("existing entry for Type:node has deps: [Type:other]", e.getMessage());
    }
  }

  @Test
  public void nodeInjectionOverEqualNodeWithDeps() throws Exception {
    NodeKey key = GraphTester.toNodeKey("node");
    Node val = new StringNode("val");

    tester.getOrCreate("other").setConstantValue(val);
    tester.getOrCreate(key).addDependency("other").setComputedValue(COPY);
    assertEquals(val, tester.evalAndGet("node"));
    tester.graph.inject(ImmutableMap.of(key, val));
    try {
      tester.evalAndGet("node");
      Assert.fail("injection over node with deps should have failed");
    } catch (IllegalStateException e) {
      assertEquals("existing entry for Type:node has deps: [Type:other]", e.getMessage());
    }
  }

  @Test
  public void nodeInjectionOverNodeWithErrors() throws Exception {
    NodeKey key = GraphTester.toNodeKey("node");
    Node val = new StringNode("val");

    tester.getOrCreate(key).setHasNonTransientError(true);
    tester.evalAndGetError(key);

    tester.graph.inject(ImmutableMap.of(key, val));
    assertEquals(val, tester.evalAndGet(false, key));
  }

  @Test
  public void nodeInjectionInvalidatesReverseDeps() throws Exception {
    NodeKey childKey = GraphTester.toNodeKey("child");
    NodeKey parentKey = GraphTester.toNodeKey("parent");
    StringNode oldVal = new StringNode("old_val");

    tester.getOrCreate(childKey).setConstantValue(oldVal);
    tester.getOrCreate(parentKey).addDependency("child").setComputedValue(COPY);

    UpdateResult<Node> result = tester.eval(false, parentKey);
    assertFalse(result.hasError());
    assertEquals(oldVal, result.get(parentKey));

    Node val = new StringNode("val");
    tester.graph.inject(ImmutableMap.of(childKey, val));
    assertEquals(val, tester.evalAndGet("child"));
    // Injecting a new child should have invalidated the parent.
    Assert.assertNull(tester.getExistingNode("parent"));

    tester.eval(false, childKey);
    assertEquals(val, tester.getExistingNode("child"));
    Assert.assertNull(tester.getExistingNode("parent"));
    assertEquals(val, tester.evalAndGet("parent"));
  }

  @Test
  public void nodeInjectionOverExistingEqualEntryDoesNotInvalidate() throws Exception {
    NodeKey childKey = GraphTester.toNodeKey("child");
    NodeKey parentKey = GraphTester.toNodeKey("parent");
    Node val = new StringNode("same_val");

    tester.getOrCreate(parentKey).addDependency("child").setComputedValue(COPY);
    tester.getOrCreate(childKey).setConstantValue(new StringNode("same_val"));
    assertEquals(val, tester.evalAndGet("parent"));

    tester.graph.inject(ImmutableMap.of(childKey, val));
    assertEquals(val, tester.getExistingNode("child"));
    // Since we are injecting an equal value, the parent should not have been invalidated.
    assertEquals(val, tester.getExistingNode("parent"));
  }

  private static final class PassThroughSelected implements NodeComputer {
    private final NodeKey key;

    public PassThroughSelected(NodeKey key) {
      this.key = key;
    }

    @Override
    public Node compute(Map<NodeKey, Node> deps, NodeBuilder.Environment env) {
      return Preconditions.checkNotNull(deps.get(key));
    }
  }

  private static class TrackingInvalidationReceiver implements NodeProgressReceiver {
    public final Set<Node> dirty = Sets.newConcurrentHashSet();
    public final Set<Node> deleted = Sets.newConcurrentHashSet();
    public final Set<NodeKey> enqueued = Sets.newConcurrentHashSet();
    public final Set<NodeKey> evaluated = Sets.newConcurrentHashSet();

    @Override
    public void invalidated(Node node, InvalidationState state) {
      switch (state) {
        case DELETED:
          dirty.remove(node);
          deleted.add(node);
          break;
        case DIRTY:
          dirty.add(node);
          Preconditions.checkState(!deleted.contains(node));
          break;
      }
    }

    @Override
    public void enqueueing(NodeKey nodeKey) {
      enqueued.add(nodeKey);
    }

    @Override
    public void evaluated(NodeKey nodeKey, Node node, EvaluationState state) {
      evaluated.add(nodeKey);
      switch (state) {
        default:
          dirty.remove(node);
          deleted.remove(node);
          break;
      }
    }

    public void clear() {
      dirty.clear();
      deleted.clear();
    }
  }

  /**
   * A graph tester that is specific to the auto-updating graph, with some convenience methods.
   */
  private class AutoUpdatingGraphTester extends GraphTester {
    private AutoUpdatingGraph graph;
    private TrackingInvalidationReceiver invalidationReceiver = new TrackingInvalidationReceiver();

    public void initialize() {
      this.graph = new InMemoryAutoUpdatingGraph(
          ImmutableMap.of(NODE_TYPE, createDelegatingNodeBuilder()), invalidationReceiver,
          emittedEventState);
    }

    public void setInvalidationReceiver(TrackingInvalidationReceiver customInvalidationReceiver) {
      Preconditions.checkState(graph == null, "graph already initialized");
      invalidationReceiver = customInvalidationReceiver;
    }

    public void invalidate() {
      graph.invalidate(getModifiedNodes());
      getModifiedNodes().clear();
      invalidationReceiver.clear();
    }

    public void invalidateErrors() {
      graph.invalidateErrors();
    }

    public void delete(String key) {
      graph.delete(Predicates.equalTo(GraphTester.nodeKey(key)));
    }

    public void resetPlayedEvents() {
      emittedEventState.clear();
    }

    public Set<Node> getDirtyNodes() {
      return invalidationReceiver.dirty;
    }

    public Set<Node> getDeletedNodes() {
      return invalidationReceiver.deleted;
    }

    public Set<NodeKey> getEnqueuedNodes() {
      return invalidationReceiver.enqueued;
    }

    public <T extends Node> UpdateResult<T> eval(boolean keepGoing, int numThreads, NodeKey... keys)
        throws InterruptedException {
      Preconditions.checkState(getModifiedNodes().isEmpty());
      return graph.update(ImmutableList.copyOf(keys), keepGoing, numThreads, reporter);
    }

    public <T extends Node> UpdateResult<T> eval(boolean keepGoing, NodeKey... keys)
        throws InterruptedException {
      return eval(keepGoing, AutoUpdatingGraph.DEFAULT_THREAD_COUNT, keys);
    }

    public <T extends Node> UpdateResult<T> eval(boolean keepGoing, String... keys)
        throws InterruptedException {
      return eval(keepGoing, toNodeKeys(keys));
    }

    public Node evalAndGet(boolean keepGoing, String key)
        throws InterruptedException {
      return evalAndGet(keepGoing, new NodeKey(NODE_TYPE, key));
    }

    public Node evalAndGet(String key) throws InterruptedException {
      return evalAndGet(/*keepGoing=*/false, key);
    }

    public Node evalAndGet(boolean keepGoing, NodeKey key)
        throws InterruptedException {
      UpdateResult<StringNode> updateResult = eval(keepGoing, key);
      Node result = updateResult.get(key);
      assertNotNull(updateResult.toString(), result);
      return result;
    }

    public ErrorInfo evalAndGetError(NodeKey key) throws InterruptedException {
      UpdateResult<StringNode> updateResult = eval(/*keepGoing=*/true, key);
      ErrorInfo result = updateResult.getError(key);
      assertNotNull(updateResult.toString(), result);
      return result;
    }

    public ErrorInfo evalAndGetError(String key) throws InterruptedException {
      return evalAndGetError(new NodeKey(NODE_TYPE, key));
    }

    @Nullable
    public Node getExistingNode(String key) {
      return graph.getExistingNodeForTesting(new NodeKey(NODE_TYPE, key));
    }
  }
}
