// Copyright 2014 The Bazel Authors. All rights reserved.
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
import static com.google.common.truth.Truth.assertWithMessage;
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertContainsEvent;
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertEventCount;
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertNoEvents;
import static com.google.devtools.build.skyframe.ErrorInfoSubjectFactory.assertThatErrorInfo;
import static com.google.devtools.build.skyframe.EvaluationResultSubjectFactory.assertThatEvaluationResult;
import static com.google.devtools.build.skyframe.GraphTester.CONCATENATE;
import static com.google.devtools.build.skyframe.GraphTester.COPY;
import static com.google.devtools.build.skyframe.GraphTester.NODE_TYPE;
import static com.google.devtools.build.skyframe.GraphTester.skyKey;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.testing.GcFinalization;
import com.google.common.util.concurrent.Uninterruptibles;
import com.google.devtools.build.lib.events.DelegatingEventHandler;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventCollector;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.testutil.TestThread;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.skyframe.GraphTester.NotComparableStringValue;
import com.google.devtools.build.skyframe.GraphTester.StringValue;
import com.google.devtools.build.skyframe.GraphTester.TestFunction;
import com.google.devtools.build.skyframe.GraphTester.ValueComputer;
import com.google.devtools.build.skyframe.NotifyingInMemoryGraph.EventType;
import com.google.devtools.build.skyframe.NotifyingInMemoryGraph.Listener;
import com.google.devtools.build.skyframe.NotifyingInMemoryGraph.Order;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;

import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.lang.ref.WeakReference;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import javax.annotation.Nullable;

/**
 * Tests for {@link MemoizingEvaluator}.
 */
@RunWith(JUnit4.class)
public class MemoizingEvaluatorTest {

  protected MemoizingEvaluatorTester tester;
  private EventCollector eventCollector;
  private EventHandler reporter;
  protected MemoizingEvaluator.EmittedEventState emittedEventState;
  @Nullable NotifyingInMemoryGraph graph = null;

  // Knobs that control the size / duration of larger tests.
  private static final int TEST_NODE_COUNT = 100;
  private static final int TESTED_NODES = 10;
  private static final int RUNS = 10;

  @Before
  public void initializeTester() {
    initializeTester(null);
  }

  @After
  public void assertNoTrackedErrors() {
    TrackingAwaiter.INSTANCE.assertNoErrors();
    if (graph != null) {
      graph.assertNoExceptions();
    }
  }

  public void initializeTester(@Nullable TrackingInvalidationReceiver customInvalidationReceiver) {
    emittedEventState = new MemoizingEvaluator.EmittedEventState();
    tester = new MemoizingEvaluatorTester();
    if (customInvalidationReceiver != null) {
      tester.setInvalidationReceiver(customInvalidationReceiver);
    }
    tester.initialize();
  }

  protected MemoizingEvaluator getMemoizingEvaluator(
      Map<SkyFunctionName, ? extends SkyFunction> functions,
      Differencer differencer,
      EvaluationProgressReceiver invalidationReceiver) {
    return new InMemoryMemoizingEvaluator(
        functions, differencer, invalidationReceiver, emittedEventState, true);
  }

  protected BuildDriver getBuildDriver(MemoizingEvaluator evaluator) {
    return new SequentialBuildDriver(evaluator);
  }

  protected boolean eventsStored() {
    return true;
  }

  protected boolean rootCausesStored() {
    return true;
  }

  @Before
  public void initializeReporter() {
    eventCollector = new EventCollector();
    reporter = eventCollector;
    tester.resetPlayedEvents();
  }

  protected static SkyKey toSkyKey(String name) {
    return new SkyKey(NODE_TYPE, name);
  }

  @Test
  public void smoke() throws Exception {
    tester.set("x", new StringValue("y"));
    StringValue value = (StringValue) tester.evalAndGet("x");
    assertEquals("y", value.getValue());
  }

  @Test
  public void invalidationWithNothingChanged() throws Exception {
    tester.set("x", new StringValue("y")).setWarning("fizzlepop");
    StringValue value = (StringValue) tester.evalAndGet("x");
    assertEquals("y", value.getValue());
    assertContainsEvent(eventCollector, "fizzlepop");
    assertEventCount(1, eventCollector);

    initializeReporter();
    tester.invalidate();
    value = (StringValue) tester.evalAndGet("x");
    assertEquals("y", value.getValue());
    if (eventsStored()) {
      assertContainsEvent(eventCollector, "fizzlepop");
      assertEventCount(1, eventCollector);
    }
  }

  private abstract static class NoExtractorFunction implements SkyFunction {
    @Override
    public final String extractTag(SkyKey skyKey) {
      return null;
    }
  }

  @Test
  // Regression test for bug: "[skyframe-m1]: registerIfDone() crash".
  public void bubbleRace() throws Exception {
    // The top-level value declares dependencies on a "badValue" in error, and a "sleepyValue"
    // which is very slow. After "badValue" fails, the builder interrupts the "sleepyValue" and
    // attempts to re-run "top" for error bubbling. Make sure this doesn't cause a precondition
    // failure because "top" still has an outstanding dep ("sleepyValue").
    tester.getOrCreate("top").setBuilder(new NoExtractorFunction() {
      @Override
      public SkyValue compute(SkyKey skyKey, Environment env) throws InterruptedException {
        env.getValue(toSkyKey("sleepyValue"));
        try {
          env.getValueOrThrow(toSkyKey("badValue"), SomeErrorException.class);
        } catch (SomeErrorException e) {
          // In order to trigger this bug, we need to request a dep on an already computed value.
          env.getValue(toSkyKey("otherValue1"));
        }
        if (!env.valuesMissing()) {
          throw new AssertionError("SleepyValue should always be unavailable");
        }
        return null;
      }
    });
    tester.getOrCreate("sleepyValue").setBuilder(new NoExtractorFunction() {
      @Override
      public SkyValue compute(SkyKey skyKey, Environment env) throws InterruptedException {
        Thread.sleep(99999);
        throw new AssertionError("I should have been interrupted");
      }
    });
    tester.getOrCreate("badValue").addDependency("otherValue1").setHasError(true);
    tester.getOrCreate("otherValue1").setConstantValue(new StringValue("otherVal1"));

    EvaluationResult<SkyValue> result = tester.eval(false, "top");
    assertTrue(result.hasError());
    assertEquals(toSkyKey("badValue"), Iterables.getOnlyElement(result.getError().getRootCauses()));
    assertThat(result.keyNames()).isEmpty();
  }

  @Test
  public void crashAfterInterruptCrashes() throws Exception {
    SkyKey failKey = GraphTester.skyKey("fail");
    SkyKey badInterruptkey = GraphTester.skyKey("bad-interrupt");
    // Given a SkyFunction implementation which is improperly coded to throw a runtime exception
    // when it is interrupted,
    final CountDownLatch badInterruptStarted = new CountDownLatch(1);
    tester
        .getOrCreate(badInterruptkey)
        .setBuilder(
            new SkyFunction() {
              @Nullable
              @Override
              public SkyValue compute(SkyKey skyKey, Environment env) {
                badInterruptStarted.countDown();
                try {
                  Thread.sleep(TestUtils.WAIT_TIMEOUT_MILLISECONDS);
                  throw new AssertionError("Shouldn't have slept so long");
                } catch (InterruptedException e) {
                  throw new RuntimeException("I don't like being woken up!");
                }
              }

              @Nullable
              @Override
              public String extractTag(SkyKey skyKey) {
                return null;
              }
            });
    // And another SkyFunction that waits for the first to start, and then throws,
    tester
        .getOrCreate(failKey)
        .setBuilder(
            new ChainedFunction(
                null,
                badInterruptStarted,
                null,
                /*waitForException=*/ false,
                null,
                ImmutableList.<SkyKey>of()));

    try {
      // When it is interrupted during evaluation (here, caused by the failure of the throwing
      // SkyFunction during a no-keep-going evaluation),
      EvaluationResult<StringValue> unexpectedResult =
          tester.eval(/*keepGoing=*/ false, badInterruptkey, failKey);
      fail(unexpectedResult.toString());
    } catch (RuntimeException e) {
      // Then the Evaluator#evaluate call throws a RuntimeException e where e.getCause() is the
      // RuntimeException thrown by that SkyFunction.
      assertThat(e.getCause()).hasMessage("I don't like being woken up!");
    }
  }

  @Test
  public void interruptAfterFailFails() throws Exception {
    SkyKey failKey = GraphTester.skyKey("fail");
    SkyKey interruptedKey = GraphTester.skyKey("interrupted");
    // Given a SkyFunction implementation that is properly coded to as not to throw a
    // runtime exception when it is interrupted,
    final CountDownLatch interruptStarted = new CountDownLatch(1);
    tester
        .getOrCreate(interruptedKey)
        .setBuilder(
            new SkyFunction() {
              @Nullable
              @Override
              public SkyValue compute(SkyKey skyKey, Environment env) throws InterruptedException {
                interruptStarted.countDown();
                Thread.sleep(TestUtils.WAIT_TIMEOUT_MILLISECONDS);
                throw new AssertionError("Shouldn't have slept so long");
              }

              @Nullable
              @Override
              public String extractTag(SkyKey skyKey) {
                return null;
              }
            });
    // And another SkyFunction that waits for the first to start, and then throws,
    tester
        .getOrCreate(failKey)
        .setBuilder(
            new ChainedFunction(
                null,
                interruptStarted,
                null,
                /*waitForException=*/ false,
                null,
                ImmutableList.<SkyKey>of()));

    // When it is interrupted during evaluation (here, caused by the failure of a sibling node
    // during a no-keep-going evaluation),
    EvaluationResult<StringValue> result =
        tester.eval(/*keepGoing=*/ false, interruptedKey, failKey);
    // Then the Evaluator#evaluate call returns an EvaluationResult that has no error for the
    // interrupted SkyFunction.
    assertWithMessage(result.toString()).that(result.hasError()).isTrue();
    assertWithMessage(result.toString()).that(result.getError(failKey)).isNotNull();
    assertWithMessage(result.toString()).that(result.getError(interruptedKey)).isNull();
  }

  @Test
  public void deleteValues() throws Exception {
    tester.getOrCreate("top").setComputedValue(CONCATENATE)
        .addDependency("d1").addDependency("d2").addDependency("d3");
    tester.set("d1", new StringValue("1"));
    StringValue d2 = new StringValue("2");
    tester.set("d2", d2);
    StringValue d3 = new StringValue("3");
    tester.set("d3", d3);
    tester.eval(true, "top");

    tester.delete("d1");
    tester.eval(true, "d3");

    assertThat(tester.getDirtyKeys()).isEmpty();
    assertEquals(
        ImmutableSet.of(skyKey("d1"), skyKey("top")), tester.getDeletedKeys());
    assertEquals(null, tester.getExistingValue("top"));
    assertEquals(null, tester.getExistingValue("d1"));
    assertEquals(d2, tester.getExistingValue("d2"));
    assertEquals(d3, tester.getExistingValue("d3"));
  }

  @Test
  public void deleteOldNodesTest() throws Exception {
    tester.getOrCreate("top").setComputedValue(CONCATENATE).addDependency("d1").addDependency("d2");
    tester.set("d1", new StringValue("one"));
    tester.set("d2", new StringValue("two"));
    tester.eval(true, "top");

    tester.set("d2", new StringValue("three"));
    tester.invalidate();
    tester.eval(true, "d2");

    // The graph now contains the three above nodes (and ERROR_TRANSIENCE).
    assertThat(tester.evaluator.getValues().keySet())
        .containsExactly(skyKey("top"), skyKey("d1"), skyKey("d2"), ErrorTransienceValue.KEY);

    String[] noKeys = {};
    tester.evaluator.deleteDirty(2);
    tester.eval(true, noKeys);

    // The top node's value is dirty, but less than two generations old, so it wasn't deleted.
    assertThat(tester.evaluator.getValues().keySet())
        .containsExactly(skyKey("top"), skyKey("d1"), skyKey("d2"), ErrorTransienceValue.KEY);

    tester.evaluator.deleteDirty(2);
    tester.eval(true, noKeys);

    // The top node's value was dirty, and was two generations old, so it was deleted.
    assertThat(tester.evaluator.getValues().keySet())
        .containsExactly(skyKey("d1"), skyKey("d2"), ErrorTransienceValue.KEY);
  }

  @Test
  public void deleteDirtyCleanedValue() throws Exception {
    SkyKey leafKey = GraphTester.skyKey("leafKey");
    tester.getOrCreate(leafKey).setConstantValue(new StringValue("value"));
    SkyKey topKey = GraphTester.skyKey("topKey");
    tester.getOrCreate(topKey).addDependency(leafKey).setComputedValue(CONCATENATE);

    assertThat(tester.evalAndGet(/*keepGoing=*/false, topKey)).isEqualTo(new StringValue("value"));
    failBuildAndRemoveValue(leafKey);
    tester.evaluator.deleteDirty(0);
  }

  @Test
  public void deleteNonexistentValues() throws Exception {
    tester.getOrCreate("d1").setConstantValue(new StringValue("1"));
    tester.delete("d1");
    tester.delete("d2");
    tester.eval(true, "d1");
  }

  @Test
  public void signalValueEnqueued() throws Exception {
    tester.getOrCreate("top1").setComputedValue(CONCATENATE)
        .addDependency("d1").addDependency("d2");
    tester.getOrCreate("top2").setComputedValue(CONCATENATE).addDependency("d3");
    tester.getOrCreate("top3");
    assertThat(tester.getEnqueuedValues()).isEmpty();

    tester.set("d1", new StringValue("1"));
    tester.set("d2", new StringValue("2"));
    tester.set("d3", new StringValue("3"));
    tester.eval(true, "top1");
    assertThat(tester.getEnqueuedValues()).containsExactlyElementsIn(
        Arrays.asList(MemoizingEvaluatorTester.toSkyKeys("top1", "d1", "d2")));

    tester.eval(true, "top2");
    assertThat(tester.getEnqueuedValues()).containsExactlyElementsIn(
        Arrays.asList(MemoizingEvaluatorTester.toSkyKeys("top1", "d1", "d2", "top2", "d3")));
  }

  // NOTE: Some of these tests exercising errors/warnings run through a size-2 for loop in order
  // to ensure that we are properly recording and replyaing these messages on null builds.
  @Test
  public void warningViaMultiplePaths() throws Exception {
    tester.set("d1", new StringValue("d1")).setWarning("warn-d1");
    tester.set("d2", new StringValue("d2")).setWarning("warn-d2");
    tester.getOrCreate("top").setComputedValue(CONCATENATE).addDependency("d1").addDependency("d2");
    for (int i = 0; i < 2; i++) {
      initializeReporter();
      tester.evalAndGet("top");
      if (i == 0 || eventsStored()) {
        assertContainsEvent(eventCollector, "warn-d1");
        assertContainsEvent(eventCollector, "warn-d2");
        assertEventCount(2, eventCollector);
      }
    }
  }

  @Test
  public void warningBeforeErrorOnFailFastBuild() throws Exception {
    tester.set("dep", new StringValue("dep")).setWarning("warn-dep");
    SkyKey topKey = GraphTester.toSkyKey("top");
    tester.getOrCreate(topKey).setHasError(true).addDependency("dep");
    for (int i = 0; i < 2; i++) {
      initializeReporter();
      EvaluationResult<StringValue> result = tester.eval(false, "top");
      assertTrue(result.hasError());
      if (rootCausesStored()) {
        assertThat(result.getError(topKey).getRootCauses()).containsExactly(topKey);
      }
      assertEquals(topKey.toString(), result.getError(topKey).getException().getMessage());
      assertTrue(result.getError(topKey).getException() instanceof SomeErrorException);
      if (i == 0 || eventsStored()) {
        assertContainsEvent(eventCollector, "warn-dep");
        assertEventCount(1, eventCollector);
      }
    }
  }

  @Test
  public void warningAndErrorOnFailFastBuild() throws Exception {
    SkyKey topKey = GraphTester.toSkyKey("top");
    tester.set(topKey, new StringValue("top")).setWarning("warning msg").setHasError(true);
    for (int i = 0; i < 2; i++) {
      initializeReporter();
      EvaluationResult<StringValue> result = tester.eval(false, "top");
      assertTrue(result.hasError());
      if (rootCausesStored()) {
        assertThat(result.getError(topKey).getRootCauses()).containsExactly(topKey);
      }
      assertEquals(topKey.toString(), result.getError(topKey).getException().getMessage());
      assertTrue(result.getError(topKey).getException() instanceof SomeErrorException);
      if (i == 0 || eventsStored()) {
        assertContainsEvent(eventCollector, "warning msg");
        assertEventCount(1, eventCollector);
      }
    }
  }

  @Test
  public void warningAndErrorOnFailFastBuildAfterKeepGoingBuild() throws Exception {
    SkyKey topKey = GraphTester.toSkyKey("top");
    tester.set(topKey, new StringValue("top")).setWarning("warning msg").setHasError(true);
    for (int i = 0; i < 2; i++) {
      initializeReporter();
      EvaluationResult<StringValue> result = tester.eval(i == 0, "top");
      assertTrue(result.hasError());
      if (rootCausesStored()) {
        assertThat(result.getError(topKey).getRootCauses()).containsExactly(topKey);
      }
      assertEquals(topKey.toString(), result.getError(topKey).getException().getMessage());
      assertTrue(result.getError(topKey).getException() instanceof SomeErrorException);
      if (i == 0 || eventsStored()) {
        assertContainsEvent(eventCollector, "warning msg");
        assertEventCount(1, eventCollector);
      }
    }
  }

  @Test
  public void twoTLTsOnOneWarningValue() throws Exception {
    tester.set("t1", new StringValue("t1")).addDependency("dep");
    tester.set("t2", new StringValue("t2")).addDependency("dep");
    tester.set("dep", new StringValue("dep")).setWarning("look both ways before crossing");
    for (int i = 0; i < 2; i++) {
      // Make sure we see the warning exactly once.
      initializeReporter();
      tester.eval(/*keepGoing=*/false, "t1", "t2");
      if (i == 0 || eventsStored()) {
        assertContainsEvent(eventCollector, "look both ways before crossing");
        assertEventCount(1, eventCollector);
      }
    }
  }

  @Test
  public void errorValueDepOnWarningValue() throws Exception {
    tester.getOrCreate("error-value").setHasError(true).addDependency("warning-value");
    tester.set("warning-value", new StringValue("warning-value"))
        .setWarning("don't chew with your mouth open");

    for (int i = 0; i < 2; i++) {
      initializeReporter();
      tester.evalAndGetError("error-value");
      if (i == 0 || eventsStored()) {
        assertContainsEvent(eventCollector, "don't chew with your mouth open");
        assertEventCount(1, eventCollector);
      }
    }

    initializeReporter();
    tester.evalAndGet("warning-value");
    if (eventsStored()) {
      assertContainsEvent(eventCollector, "don't chew with your mouth open");
      assertEventCount(1, eventCollector);
    }
  }

  @Test
  public void progressMessageOnlyPrintedTheFirstTime() throws Exception {
    // The framework keeps track of warning and error messages, but not progress messages.
    // So here we see both the progress and warning on the first build, but only the warning
    // on the subsequent null build.
    tester.set("x", new StringValue("y")).setWarning("fizzlepop")
        .setProgress("just letting you know");

    StringValue value = (StringValue) tester.evalAndGet("x");
    assertEquals("y", value.getValue());
    assertContainsEvent(eventCollector, "fizzlepop");
    assertContainsEvent(eventCollector, "just letting you know");
    assertEventCount(2, eventCollector);

    if (eventsStored()) {
      // On the rebuild, we only replay warning messages.
      initializeReporter();
      value = (StringValue) tester.evalAndGet("x");
      assertEquals("y", value.getValue());
      assertContainsEvent(eventCollector, "fizzlepop");
      assertEventCount(1, eventCollector);
    }
  }

  @Test
  public void invalidationWithChangeAndThenNothingChanged() throws Exception {
    tester.getOrCreate("a")
        .addDependency("b")
        .setComputedValue(COPY);
    tester.set("b", new StringValue("y"));
    StringValue original = (StringValue) tester.evalAndGet("a");
    assertEquals("y", original.getValue());
    tester.set("b", new StringValue("z"));
    tester.invalidate();
    StringValue old = (StringValue) tester.evalAndGet("a");
    assertEquals("z", old.getValue());
    tester.invalidate();
    StringValue current = (StringValue) tester.evalAndGet("a");
    assertThat(current).isEqualTo(old);
  }

  @Test
  public void transientErrorValueInvalidation() throws Exception {
    // Verify that invalidating errors causes all transient error values to be rerun.
    tester.getOrCreate("error-value").setHasTransientError(true).setProgress(
        "just letting you know");

    tester.evalAndGetError("error-value");
    assertContainsEvent(eventCollector, "just letting you know");
    assertEventCount(1, eventCollector);

    // Change the progress message.
    tester.getOrCreate("error-value").setHasTransientError(true).setProgress(
        "letting you know more");

    // Without invalidating errors, we shouldn't show the new progress message.
    for (int i = 0; i < 2; i++) {
      initializeReporter();
      tester.evalAndGetError("error-value");
      assertNoEvents(eventCollector);
    }

    // When invalidating errors, we should show the new progress message.
    initializeReporter();
    tester.invalidateTransientErrors();
    tester.evalAndGetError("error-value");
    assertContainsEvent(eventCollector, "letting you know more");
    assertEventCount(1, eventCollector);
  }

  @Test
  public void transientPruning() throws Exception {
    SkyKey leaf = GraphTester.toSkyKey("leaf");
    tester.getOrCreate("top").setHasTransientError(true).addDependency(leaf);
    tester.set(leaf, new StringValue("leafy"));
    tester.evalAndGetError("top");
    tester.getOrCreate(leaf, /*markAsModified=*/true);
    tester.invalidate();
    tester.evalAndGetError("top");
  }

  @Test
  public void simpleDependency() throws Exception {
    tester.getOrCreate("ab")
        .addDependency("a")
        .setComputedValue(COPY);
    tester.set("a", new StringValue("me"));
    StringValue value = (StringValue) tester.evalAndGet("ab");
    assertEquals("me", value.getValue());
  }

  @Test
  public void incrementalSimpleDependency() throws Exception {
    tester.getOrCreate("ab")
        .addDependency("a")
        .setComputedValue(COPY);
    tester.set("a", new StringValue("me"));
    tester.evalAndGet("ab");

    tester.set("a", new StringValue("other"));
    tester.invalidate();
    StringValue value = (StringValue) tester.evalAndGet("ab");
    assertEquals("other", value.getValue());
  }

  @Test
  public void diamondDependency() throws Exception {
    setupDiamondDependency();
    tester.set("d", new StringValue("me"));
    StringValue value = (StringValue) tester.evalAndGet("a");
    assertEquals("meme", value.getValue());
  }

  @Test
  public void incrementalDiamondDependency() throws Exception {
    setupDiamondDependency();
    tester.set("d", new StringValue("me"));
    tester.evalAndGet("a");

    tester.set("d", new StringValue("other"));
    tester.invalidate();
    StringValue value = (StringValue) tester.evalAndGet("a");
    assertEquals("otherother", value.getValue());
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

  // ParallelEvaluator notifies ValueProgressReceiver of already-built top-level values in error: we
  // built "top" and "mid" as top-level targets; "mid" contains an error. We make sure "mid" is
  // built as a dependency of "top" before enqueuing mid as a top-level target (by using a latch),
  // so that the top-level enqueuing finds that mid has already been built. The progress receiver
  // should be notified that mid has been built.
  @Test
  public void alreadyAnalyzedBadTarget() throws Exception {
    final SkyKey mid = GraphTester.toSkyKey("mid");
    final CountDownLatch valueSet = new CountDownLatch(1);
    setGraphForTesting(
        new NotifyingInMemoryGraph(
            new Listener() {
              @Override
              public void accept(SkyKey key, EventType type, Order order, Object context) {
                if (!key.equals(mid)) {
                  return;
                }
                switch (type) {
                  case ADD_REVERSE_DEP:
                    if (context == null) {
                      // Context is null when we are enqueuing this value as a top-level job.
                      TrackingAwaiter.INSTANCE.awaitLatchAndTrackExceptions(
                          valueSet, "value not set");
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
    SkyKey top = GraphTester.skyKey("top");
    tester.getOrCreate(top).addDependency(mid).setComputedValue(CONCATENATE);
    tester.getOrCreate(mid).setHasError(true);
    tester.eval(/*keepGoing=*/false, top, mid);
    assertEquals(0L, valueSet.getCount());
    assertThat(tester.invalidationReceiver.evaluated).containsExactly(mid);
  }

  @Test
  public void receiverToldOfVerifiedValueDependingOnCycle() throws Exception {
    SkyKey leaf = GraphTester.toSkyKey("leaf");
    SkyKey cycle = GraphTester.toSkyKey("cycle");
    SkyKey top = GraphTester.toSkyKey("top");
    tester.set(leaf, new StringValue("leaf"));
    tester.getOrCreate(cycle).addDependency(cycle);
    tester.getOrCreate(top).addDependency(leaf).addDependency(cycle);
    tester.eval(/*keepGoing=*/true, top);
    assertThat(tester.invalidationReceiver.evaluated).containsExactly(leaf, top, cycle);
    tester.invalidationReceiver.clear();
    tester.getOrCreate(leaf, /*markAsModified=*/true);
    tester.invalidate();
    tester.eval(/*keepGoing=*/true, top);
    assertThat(tester.invalidationReceiver.evaluated).containsExactly(leaf, top);
  }

  @Test
  public void incrementalAddedDependency() throws Exception {
    tester.getOrCreate("a")
        .addDependency("b")
        .setComputedValue(CONCATENATE);
    tester.set("b", new StringValue("first"));
    tester.set("c", new StringValue("second"));
    tester.evalAndGet("a");

    tester.getOrCreate("a").addDependency("c");
    tester.set("b", new StringValue("now"));
    tester.invalidate();
    StringValue value = (StringValue) tester.evalAndGet("a");
    assertEquals("nowsecond", value.getValue());
  }

  @Test
  public void manyValuesDependOnSingleValue() throws Exception {
    initializeTester();
    String[] values = new String[TEST_NODE_COUNT];
    for (int i = 0; i < values.length; i++) {
      values[i] = Integer.toString(i);
      tester.getOrCreate(values[i])
          .addDependency("leaf")
          .setComputedValue(COPY);
    }
    tester.set("leaf", new StringValue("leaf"));

    EvaluationResult<StringValue> result = tester.eval(/*keep_going=*/false, values);
    for (int i = 0; i < values.length; i++) {
      SkyValue actual = result.get(toSkyKey(values[i]));
      assertEquals(new StringValue("leaf"), actual);
    }

    for (int j = 0; j < TESTED_NODES; j++) {
      tester.set("leaf", new StringValue("other" + j));
      tester.invalidate();
      result = tester.eval(/*keep_going=*/false, values);
      for (int i = 0; i < values.length; i++) {
        SkyValue actual = result.get(toSkyKey(values[i]));
        assertEquals("Run " + j + ", value " + i, new StringValue("other" + j), actual);
      }
    }
  }

  @Test
  public void singleValueDependsOnManyValues() throws Exception {
    initializeTester();
    String[] values = new String[TEST_NODE_COUNT];
    StringBuilder expected = new StringBuilder();
    for (int i = 0; i < values.length; i++) {
      values[i] = Integer.toString(i);
      tester.set(values[i], new StringValue(values[i]));
      expected.append(values[i]);
    }
    SkyKey rootKey = toSkyKey("root");
    TestFunction value = tester.getOrCreate(rootKey)
        .setComputedValue(CONCATENATE);
    for (int i = 0; i < values.length; i++) {
      value.addDependency(values[i]);
    }

    EvaluationResult<StringValue> result = tester.eval(/*keep_going=*/false, rootKey);
    assertEquals(new StringValue(expected.toString()), result.get(rootKey));

    for (int j = 0; j < 10; j++) {
      expected.setLength(0);
      for (int i = 0; i < values.length; i++) {
        String s = "other" + i + " " + j;
        tester.set(values[i], new StringValue(s));
        expected.append(s);
      }
      tester.invalidate();

      result = tester.eval(/*keep_going=*/false, rootKey);
      assertEquals(new StringValue(expected.toString()), result.get(rootKey));
    }
  }

  @Test
  public void twoRailLeftRightDependencies() throws Exception {
    initializeTester();
    String[] leftValues = new String[TEST_NODE_COUNT];
    String[] rightValues = new String[TEST_NODE_COUNT];
    for (int i = 0; i < leftValues.length; i++) {
      leftValues[i] = "left-" + i;
      rightValues[i] = "right-" + i;
      if (i == 0) {
        tester.getOrCreate(leftValues[i])
              .addDependency("leaf")
              .setComputedValue(COPY);
        tester.getOrCreate(rightValues[i])
              .addDependency("leaf")
              .setComputedValue(COPY);
      } else {
        tester.getOrCreate(leftValues[i])
              .addDependency(leftValues[i - 1])
              .addDependency(rightValues[i - 1])
              .setComputedValue(new PassThroughSelected(toSkyKey(leftValues[i - 1])));
        tester.getOrCreate(rightValues[i])
              .addDependency(leftValues[i - 1])
              .addDependency(rightValues[i - 1])
              .setComputedValue(new PassThroughSelected(toSkyKey(rightValues[i - 1])));
      }
    }
    tester.set("leaf", new StringValue("leaf"));

    String lastLeft = "left-" + (TEST_NODE_COUNT - 1);
    String lastRight = "right-" + (TEST_NODE_COUNT - 1);

    EvaluationResult<StringValue> result = tester.eval(/*keep_going=*/false, lastLeft, lastRight);
    assertEquals(new StringValue("leaf"), result.get(toSkyKey(lastLeft)));
    assertEquals(new StringValue("leaf"), result.get(toSkyKey(lastRight)));

    for (int j = 0; j < TESTED_NODES; j++) {
      String value = "other" + j;
      tester.set("leaf", new StringValue(value));
      tester.invalidate();
      result = tester.eval(/*keep_going=*/false, lastLeft, lastRight);
      assertEquals(new StringValue(value), result.get(toSkyKey(lastLeft)));
      assertEquals(new StringValue(value), result.get(toSkyKey(lastRight)));
    }
  }

  @Test
  public void noKeepGoingAfterKeepGoingCycle() throws Exception {
    initializeTester();
    SkyKey aKey = GraphTester.toSkyKey("a");
    SkyKey bKey = GraphTester.toSkyKey("b");
    SkyKey topKey = GraphTester.toSkyKey("top");
    SkyKey midKey = GraphTester.toSkyKey("mid");
    SkyKey goodKey = GraphTester.toSkyKey("good");
    StringValue goodValue = new StringValue("good");
    tester.set(goodKey, goodValue);
    tester.getOrCreate(topKey).addDependency(midKey);
    tester.getOrCreate(midKey).addDependency(aKey);
    tester.getOrCreate(aKey).addDependency(bKey);
    tester.getOrCreate(bKey).addDependency(aKey);
    EvaluationResult<StringValue> result = tester.eval(/*keepGoing=*/true, topKey, goodKey);
    assertEquals(goodValue, result.get(goodKey));
    assertEquals(null, result.get(topKey));
    ErrorInfo errorInfo = result.getError(topKey);
    CycleInfo cycleInfo = Iterables.getOnlyElement(errorInfo.getCycleInfo());
    assertThat(cycleInfo.getCycle()).containsExactly(aKey, bKey).inOrder();
    assertThat(cycleInfo.getPathToCycle()).containsExactly(topKey, midKey).inOrder();

    tester.invalidate();
    result = tester.eval(/*keepGoing=*/false, topKey, goodKey);
    assertEquals(null, result.get(topKey));
    errorInfo = result.getError(topKey);
    cycleInfo = Iterables.getOnlyElement(errorInfo.getCycleInfo());
    assertThat(cycleInfo.getCycle()).containsExactly(aKey, bKey).inOrder();
    assertThat(cycleInfo.getPathToCycle()).containsExactly(topKey, midKey).inOrder();
  }

  private void changeCycle(boolean keepGoing) throws Exception {
    initializeTester();
    SkyKey aKey = GraphTester.toSkyKey("a");
    SkyKey bKey = GraphTester.toSkyKey("b");
    SkyKey topKey = GraphTester.toSkyKey("top");
    SkyKey midKey = GraphTester.toSkyKey("mid");
    tester.getOrCreate(topKey).addDependency(midKey).setComputedValue(COPY);
    tester.getOrCreate(midKey).addDependency(aKey).setComputedValue(COPY);
    tester.getOrCreate(aKey).addDependency(bKey).setComputedValue(COPY);
    tester.getOrCreate(bKey).addDependency(aKey);
    EvaluationResult<StringValue> result = tester.eval(keepGoing, topKey);
    assertEquals(null, result.get(topKey));
    ErrorInfo errorInfo = result.getError(topKey);
    CycleInfo cycleInfo = Iterables.getOnlyElement(errorInfo.getCycleInfo());
    assertThat(cycleInfo.getCycle()).containsExactly(aKey, bKey).inOrder();
    assertThat(cycleInfo.getPathToCycle()).containsExactly(topKey, midKey).inOrder();

    tester.getOrCreate(bKey).removeDependency(aKey);
    tester.set(bKey, new StringValue("bValue"));
    tester.invalidate();
    result = tester.eval(keepGoing, topKey);
    assertEquals(new StringValue("bValue"), result.get(topKey));
    assertEquals(null, result.getError(topKey));
  }

  @Test
  public void changeCycle_NoKeepGoing() throws Exception {
    changeCycle(false);
  }

  @Test
  public void changeCycle_KeepGoing() throws Exception {
    changeCycle(true);
  }

  /** @see ParallelEvaluatorTest#cycleAboveIndependentCycle() */
  @Test
  public void cycleAboveIndependentCycle() throws Exception {
    SkyKey aKey = GraphTester.toSkyKey("a");
    final SkyKey bKey = GraphTester.toSkyKey("b");
    SkyKey cKey = GraphTester.toSkyKey("c");
    final SkyKey leafKey = GraphTester.toSkyKey("leaf");
    // When aKey depends on leafKey and bKey,
    tester
        .getOrCreate(aKey)
        .setBuilder(
            new SkyFunction() {
              @Nullable
              @Override
              public SkyValue compute(SkyKey skyKey, Environment env) {
                env.getValues(ImmutableList.of(leafKey, bKey));
                return null;
              }

              @Nullable
              @Override
              public String extractTag(SkyKey skyKey) {
                return null;
              }
            });
    // And bKey depends on cKey,
    tester.getOrCreate(bKey).addDependency(cKey);
    // And cKey depends on aKey and bKey in that order,
    tester.getOrCreate(cKey).addDependency(aKey).addDependency(bKey);
    // And leafKey is a leaf node,
    tester.set(leafKey, new StringValue("leafy"));
    // Then when we evaluate,
    EvaluationResult<StringValue> result = tester.eval(/*keepGoing=*/ true, aKey);
    // aKey has an error,
    assertEquals(null, result.get(aKey));
    // And both cycles were found underneath aKey: the (aKey->bKey->cKey) cycle, and the
    // aKey->(bKey->cKey) cycle. This is because cKey depended on aKey and then bKey, so it pushed
    // them down on the stack in that order, so bKey was processed first. It found its cycle, then
    // popped off the stack, and then aKey was processed and found its cycle.
    assertThat(result.getError(aKey).getCycleInfo())
        .containsExactly(
            new CycleInfo(ImmutableList.of(aKey, bKey, cKey)),
            new CycleInfo(ImmutableList.of(aKey), ImmutableList.of(bKey, cKey)));
    // When leafKey is changed, so that aKey will be marked as NEEDS_REBUILDING,
    tester.set(leafKey, new StringValue("crunchy"));
    // And cKey is invalidated, so that cycle checking will have to explore the full graph,
    tester.getOrCreate(cKey, /*markAsModified=*/ true);
    tester.invalidate();
    // Then when we evaluate,
    EvaluationResult<StringValue> result2 = tester.eval(/*keepGoing=*/ true, aKey);
    // Things are just as before.
    assertEquals(null, result2.get(aKey));
    assertThat(result2.getError(aKey).getCycleInfo())
        .containsExactly(
            new CycleInfo(ImmutableList.of(aKey, bKey, cKey)),
            new CycleInfo(ImmutableList.of(aKey), ImmutableList.of(bKey, cKey)));
  }

  /** Regression test: "crash in cycle checker with dirty values". */
  @Test
  public void cycleAndSelfEdgeWithDirtyValue() throws Exception {
    initializeTester();
    SkyKey cycleKey1 = GraphTester.toSkyKey("cycleKey1");
    SkyKey cycleKey2 = GraphTester.toSkyKey("cycleKey2");
    tester.getOrCreate(cycleKey1).addDependency(cycleKey2).addDependency(cycleKey1)
    .setComputedValue(CONCATENATE);
    tester.getOrCreate(cycleKey2).addDependency(cycleKey1).setComputedValue(COPY);
    EvaluationResult<StringValue> result = tester.eval(/*keepGoing=*/true, cycleKey1);
    assertEquals(null, result.get(cycleKey1));
    ErrorInfo errorInfo = result.getError(cycleKey1);
    CycleInfo cycleInfo = Iterables.getOnlyElement(errorInfo.getCycleInfo());
    assertThat(cycleInfo.getCycle()).containsExactly(cycleKey1).inOrder();
    assertThat(cycleInfo.getPathToCycle()).isEmpty();
    tester.getOrCreate(cycleKey1, /*markAsModified=*/true);
    tester.invalidate();
    result = tester.eval(/*keepGoing=*/true, cycleKey1, cycleKey2);
    assertEquals(null, result.get(cycleKey1));
    errorInfo = result.getError(cycleKey1);
    cycleInfo = Iterables.getOnlyElement(errorInfo.getCycleInfo());
    assertThat(cycleInfo.getCycle()).containsExactly(cycleKey1).inOrder();
    assertThat(cycleInfo.getPathToCycle()).isEmpty();
    cycleInfo =
        Iterables.getOnlyElement(
            tester.driver.getExistingErrorForTesting(cycleKey2).getCycleInfo());
    assertThat(cycleInfo.getCycle()).containsExactly(cycleKey1).inOrder();
    assertThat(cycleInfo.getPathToCycle()).containsExactly(cycleKey2).inOrder();
  }

  @Test
  public void cycleAndSelfEdgeWithDirtyValueInSameGroup() throws Exception {
    setGraphForTesting(new DeterministicInMemoryGraph());
    final SkyKey cycleKey1 = GraphTester.toSkyKey("zcycleKey1");
    final SkyKey cycleKey2 = GraphTester.toSkyKey("acycleKey2");
    tester.getOrCreate(cycleKey2).addDependency(cycleKey2).setComputedValue(CONCATENATE);
    tester
        .getOrCreate(cycleKey1)
        .setBuilder(
            new SkyFunction() {
              @Nullable
              @Override
              public SkyValue compute(SkyKey skyKey, Environment env)
                  throws SkyFunctionException, InterruptedException {
                Map<SkyKey, SkyValue> result =
                    env.getValues(ImmutableList.of(cycleKey1, cycleKey2));
                Preconditions.checkState(env.valuesMissing(), result);
                return null;
              }

              @Nullable
              @Override
              public String extractTag(SkyKey skyKey) {
                return null;
              }
            });
    // Evaluate twice to make sure nothing strange happens with invalidation the second time.
    for (int i = 0; i < 2; i++) {
      EvaluationResult<SkyValue> result = tester.eval(/*keepGoing=*/ true, cycleKey1);
      assertEquals(null, result.get(cycleKey1));
      ErrorInfo errorInfo = result.getError(cycleKey1);
      CycleInfo cycleInfo = Iterables.getOnlyElement(errorInfo.getCycleInfo());
      assertThat(cycleInfo.getCycle()).containsExactly(cycleKey1).inOrder();
      assertThat(cycleInfo.getPathToCycle()).isEmpty();
    }
  }

  /** Regression test: "crash in cycle checker with dirty values". */
  @Test
  public void cycleWithDirtyValue() throws Exception {
    initializeTester();
    SkyKey cycleKey1 = GraphTester.toSkyKey("cycleKey1");
    SkyKey cycleKey2 = GraphTester.toSkyKey("cycleKey2");
    tester.getOrCreate(cycleKey1).addDependency(cycleKey2).setComputedValue(COPY);
    tester.getOrCreate(cycleKey2).addDependency(cycleKey1).setComputedValue(COPY);
    EvaluationResult<StringValue> result = tester.eval(/*keepGoing=*/true, cycleKey1);
    assertEquals(null, result.get(cycleKey1));
    ErrorInfo errorInfo = result.getError(cycleKey1);
    CycleInfo cycleInfo = Iterables.getOnlyElement(errorInfo.getCycleInfo());
    assertThat(cycleInfo.getCycle()).containsExactly(cycleKey1, cycleKey2).inOrder();
    assertThat(cycleInfo.getPathToCycle()).isEmpty();
    tester.getOrCreate(cycleKey1, /*markAsModified=*/true);
    tester.invalidate();
    result = tester.eval(/*keepGoing=*/true, cycleKey1);
    assertEquals(null, result.get(cycleKey1));
    errorInfo = result.getError(cycleKey1);
    cycleInfo = Iterables.getOnlyElement(errorInfo.getCycleInfo());
    assertThat(cycleInfo.getCycle()).containsExactly(cycleKey1, cycleKey2).inOrder();
    assertThat(cycleInfo.getPathToCycle()).isEmpty();
  }

  /**
   * {@link ParallelEvaluator} can be configured to not store errors alongside recovered values.
   * In that case, transient errors that are recovered from do not make the parent transient.
   */
  protected void parentOfCycleAndTransientInternal(boolean errorsStoredAlongsideValues)
      throws Exception {
    initializeTester();
    SkyKey cycleKey1 = GraphTester.toSkyKey("cycleKey1");
    SkyKey cycleKey2 = GraphTester.toSkyKey("cycleKey2");
    SkyKey mid = GraphTester.toSkyKey("mid");
    SkyKey errorKey = GraphTester.toSkyKey("errorKey");
    tester.getOrCreate(cycleKey1).addDependency(cycleKey2).setComputedValue(COPY);
    tester.getOrCreate(cycleKey2).addDependency(cycleKey1).setComputedValue(COPY);
    tester.getOrCreate(errorKey).setHasTransientError(true);
    tester.getOrCreate(mid).addErrorDependency(errorKey, new StringValue("recovered"))
        .setComputedValue(COPY);
    SkyKey top = GraphTester.toSkyKey("top");
    CountDownLatch topEvaluated = new CountDownLatch(2);
    tester.getOrCreate(top).setBuilder(new ChainedFunction(topEvaluated, null, null, false,
        new StringValue("unused"), ImmutableList.of(mid, cycleKey1)));
    EvaluationResult<StringValue> evalResult = tester.eval(true, top);
    assertThatEvaluationResult(evalResult).hasError();
    ErrorInfo errorInfo = evalResult.getError(top);
    assertThat(topEvaluated.getCount()).isEqualTo(1);
    if (errorsStoredAlongsideValues) {
      // The parent should be transitively transient, since it transitively depends on a transient
      // error.
      assertThat(errorInfo.isTransient()).isTrue();
      assertThat(errorInfo.getException()).hasMessage(NODE_TYPE.getName() + ":errorKey");
      assertThat(errorInfo.getRootCauseOfException()).isEqualTo(errorKey);
    } else {
      assertThatErrorInfo(errorInfo).isNotTransient();
      assertThatErrorInfo(errorInfo).hasExceptionThat().isNull();
    }
    assertWithMessage(errorInfo.toString())
        .that(errorInfo.getCycleInfo())
        .containsExactly(
            new CycleInfo(ImmutableList.of(top), ImmutableList.of(cycleKey1, cycleKey2)));
    // But the parent itself shouldn't have a direct dep on the special error transience node.
    assertThatEvaluationResult(evalResult)
        .hasDirectDepsInGraphThat(top)
        .doesNotContain(ErrorTransienceValue.KEY);
  }

  @Test
  public void parentOfCycleAndTransient() throws Exception {
    parentOfCycleAndTransientInternal(/*errorsStoredAlongsideValues=*/ true);
  }

  /**
   * Regression test: IllegalStateException in BuildingState.isReady(). The ParallelEvaluator used
   * to assume during cycle-checking that all values had been built as fully as possible -- that
   * evaluation had not been interrupted. However, we also do cycle-checking in nokeep-going mode
   * when a value throws an error (possibly prematurely shutting down evaluation) but that error
   * then bubbles up into a cycle.
   *
   * <p>We want to achieve the following state: we are checking for a cycle; the value we examine
   * has not yet finished checking its children to see if they are dirty; but all children checked
   * so far have been unchanged. This value is "otherTop". We first build otherTop, then mark its
   * first child changed (without actually changing it), and then do a second build. On the second
   * build, we also build "top", which requests a cycle that depends on an error. We wait to signal
   * otherTop that its first child is done until the error throws and shuts down evaluation. The
   * error then bubbles up to the cycle, and so the bubbling is aborted. Finally, cycle checking
   * happens, and otherTop is examined, as desired.
   */
  @Test
  public void cycleAndErrorAndReady() throws Exception {
    // This value will not have finished building on the second build when the error is thrown.
    final SkyKey otherTop = GraphTester.toSkyKey("otherTop");
    final SkyKey errorKey = GraphTester.toSkyKey("error");
    // Is the graph state all set up and ready for the error to be thrown? The three values are
    // exceptionMarker, cycle2Key, and dep1 (via signaling otherTop).
    final CountDownLatch valuesReady = new CountDownLatch(3);
    // Is evaluation being shut down? This is counted down by the exceptionMarker's builder, after
    // it has waited for the threadpool's exception latch to be released.
    final CountDownLatch errorThrown = new CountDownLatch(1);
    // We don't do anything on the first build.
    final AtomicBoolean secondBuild = new AtomicBoolean(false);
    setGraphForTesting(
        new DeterministicInMemoryGraph(
            new Listener() {
              @Override
              public void accept(SkyKey key, EventType type, Order order, Object context) {
                if (!secondBuild.get()) {
                  return;
                }
                if (key.equals(otherTop) && type == EventType.SIGNAL) {
                  // otherTop is being signaled that dep1 is done. Tell the error value that it is
                  // ready, then wait until the error is thrown, so that otherTop's builder is not
                  // re-entered.
                  valuesReady.countDown();
                  TrackingAwaiter.INSTANCE.awaitLatchAndTrackExceptions(
                      errorThrown, "error not thrown");
                  return;
                }
              }
            }));
    final SkyKey dep1 = GraphTester.toSkyKey("dep1");
    tester.set(dep1, new StringValue("dep1"));
    final SkyKey dep2 = GraphTester.toSkyKey("dep2");
    tester.set(dep2, new StringValue("dep2"));
    // otherTop should request the deps one at a time, so that it can be in the CHECK_DEPENDENCIES
    // state even after one dep is re-evaluated.
    tester.getOrCreate(otherTop).setBuilder(new NoExtractorFunction() {
      @Override
      public SkyValue compute(SkyKey skyKey, Environment env) {
        env.getValue(dep1);
        if (env.valuesMissing()) {
          return null;
        }
        env.getValue(dep2);
        return env.valuesMissing() ? null : new StringValue("otherTop");
      }
    });
    // Prime the graph with otherTop, so we can dirty it next build.
    assertEquals(new StringValue("otherTop"), tester.evalAndGet(/*keepGoing=*/false, otherTop));
    // Mark dep1 changed, so otherTop will be dirty and request re-evaluation of dep1.
    tester.getOrCreate(dep1, /*markAsModified=*/true);
    SkyKey topKey = GraphTester.toSkyKey("top");
    // Note that since DeterministicInMemoryGraph alphabetizes reverse deps, it is important that
    // "cycle2" comes before "top".
    final SkyKey cycle1Key = GraphTester.toSkyKey("cycle1");
    final SkyKey cycle2Key = GraphTester.toSkyKey("cycle2");
    tester.getOrCreate(topKey).addDependency(cycle1Key).setComputedValue(CONCATENATE);
    tester.getOrCreate(cycle1Key).addDependency(errorKey).addDependency(cycle2Key)
        .setComputedValue(CONCATENATE);
    tester.getOrCreate(errorKey).setBuilder(new ChainedFunction(/*notifyStart=*/null,
    /*waitToFinish=*/valuesReady, /*notifyFinish=*/null, /*waitForException=*/false, /*value=*/null,
        ImmutableList.<SkyKey>of()));
    // Make sure cycle2Key has declared its dependence on cycle1Key before error throws.
    tester.getOrCreate(cycle2Key).setBuilder(new ChainedFunction(/*notifyStart=*/valuesReady,
        null, null, false, new StringValue("never returned"), ImmutableList.<SkyKey>of(cycle1Key)));
    // Value that waits until an exception is thrown to finish building. We use it just to be
    // informed when the threadpool is shutting down.
    final SkyKey exceptionMarker = GraphTester.toSkyKey("exceptionMarker");
    tester.getOrCreate(exceptionMarker).setBuilder(new ChainedFunction(
        /*notifyStart=*/valuesReady, /*waitToFinish=*/new CountDownLatch(0),
        /*notifyFinish=*/errorThrown,
        /*waitForException=*/true, new StringValue("exception marker"),
        ImmutableList.<SkyKey>of()));
    tester.invalidate();
    secondBuild.set(true);
    // otherTop must be first, since we check top-level values for cycles in the order in which
    // they appear here.
    EvaluationResult<StringValue> result =
        tester.eval(/*keepGoing=*/false, otherTop, topKey, exceptionMarker);
    assertThat(result.errorMap().keySet()).containsExactly(topKey);
    Iterable<CycleInfo> cycleInfos = result.getError(topKey).getCycleInfo();
    assertWithMessage(result.toString()).that(cycleInfos).isNotEmpty();
    CycleInfo cycleInfo = Iterables.getOnlyElement(cycleInfos);
    assertThat(cycleInfo.getPathToCycle()).containsExactly(topKey);
    assertThat(cycleInfo.getCycle()).containsExactly(cycle1Key, cycle2Key);
  }

  @Test
  public void limitEvaluatorThreads() throws Exception {
    initializeTester();

    int numKeys = 10;
    final Object lock = new Object();
    final AtomicInteger inProgressCount = new AtomicInteger();
    final int[] maxValue = {0};

    SkyKey topLevel = GraphTester.toSkyKey("toplevel");
    TestFunction topLevelBuilder = tester.getOrCreate(topLevel);
    for (int i = 0; i < numKeys; i++) {
      topLevelBuilder.addDependency("subKey" + i);
      tester.getOrCreate("subKey" + i).setComputedValue(new ValueComputer() {
        @Override
        public SkyValue compute(Map<SkyKey, SkyValue> deps, SkyFunction.Environment env) {
          int val = inProgressCount.incrementAndGet();
          synchronized (lock) {
            if (val > maxValue[0]) {
              maxValue[0] = val;
            }
          }
          Uninterruptibles.sleepUninterruptibly(5, TimeUnit.SECONDS);

          inProgressCount.decrementAndGet();
          return new StringValue("abc");
        }
      });
    }
    topLevelBuilder.setConstantValue(new StringValue("xyz"));

    EvaluationResult<StringValue> result = tester.eval(
        /*keepGoing=*/true, /*numThreads=*/5, topLevel);
    assertFalse(result.hasError());
    assertEquals(5, maxValue[0]);
  }

  /**
   * Regression test: error on clearMaybeDirtyValue. We do an evaluation of topKey, which registers
   * dependencies on midKey and errorKey. midKey enqueues slowKey, and waits. errorKey throws an
   * error, which bubbles up to topKey. If topKey does not unregister its dependence on midKey, it
   * will have a dangling reference to midKey after unfinished values are cleaned from the graph.
   * Note that slowKey will wait until errorKey has thrown and the threadpool has caught the
   * exception before returning, so the Evaluator will already have stopped enqueuing new jobs, so
   * midKey is not evaluated.
   */
  @Test
  public void incompleteDirectDepsAreClearedBeforeInvalidation() throws Exception {
    initializeTester();
    CountDownLatch slowStart = new CountDownLatch(1);
    CountDownLatch errorFinish = new CountDownLatch(1);
    SkyKey errorKey = GraphTester.toSkyKey("error");
    tester.getOrCreate(errorKey).setBuilder(
        new ChainedFunction(/*notifyStart=*/null, /*waitToFinish=*/slowStart,
            /*notifyFinish=*/errorFinish, /*waitForException=*/false, /*value=*/null,
            /*deps=*/ImmutableList.<SkyKey>of()));
    SkyKey slowKey = GraphTester.toSkyKey("slow");
    tester.getOrCreate(slowKey).setBuilder(
        new ChainedFunction(/*notifyStart=*/slowStart, /*waitToFinish=*/errorFinish,
            /*notifyFinish=*/null, /*waitForException=*/true, new StringValue("slow"),
            /*deps=*/ImmutableList.<SkyKey>of()));
    SkyKey midKey = GraphTester.toSkyKey("mid");
    tester.getOrCreate(midKey).addDependency(slowKey).setComputedValue(COPY);
    SkyKey topKey = GraphTester.toSkyKey("top");
    tester.getOrCreate(topKey).addDependency(midKey).addDependency(errorKey)
        .setComputedValue(CONCATENATE);
    // slowKey starts -> errorKey finishes, written to graph -> slowKey finishes & (Visitor aborts)
    // -> topKey builds.
    EvaluationResult<StringValue> result = tester.eval(/*keepGoing=*/false, topKey);
    assertThat(result.getError().getRootCauses()).containsExactly(errorKey);
    // Make sure midKey didn't finish building.
    assertEquals(null, tester.getExistingValue(midKey));
    // Give slowKey a nice ordinary builder.
    tester.getOrCreate(slowKey, /*markAsModified=*/false).setBuilder(null)
        .setConstantValue(new StringValue("slow"));
    // Put midKey into the graph. It won't have a reverse dependence on topKey.
    tester.evalAndGet(/*keepGoing=*/false, midKey);
    tester.differencer.invalidate(ImmutableList.of(errorKey));
    // topKey should not access midKey as if it were already registered as a dependency.
    tester.eval(/*keepGoing=*/false, topKey);
  }

  /**
   * Regression test: error on clearMaybeDirtyValue. Same as the previous test, but the second
   * evaluation is keepGoing, which should cause an access of the children of topKey.
   */
  @Test
  public void incompleteDirectDepsAreClearedBeforeKeepGoing() throws Exception {
    initializeTester();
    CountDownLatch slowStart = new CountDownLatch(1);
    CountDownLatch errorFinish = new CountDownLatch(1);
    SkyKey errorKey = GraphTester.toSkyKey("error");
    tester.getOrCreate(errorKey).setBuilder(
        new ChainedFunction(/*notifyStart=*/null, /*waitToFinish=*/slowStart,
            /*notifyFinish=*/errorFinish, /*waitForException=*/false, /*value=*/null,
            /*deps=*/ImmutableList.<SkyKey>of()));
    SkyKey slowKey = GraphTester.toSkyKey("slow");
    tester.getOrCreate(slowKey).setBuilder(
        new ChainedFunction(/*notifyStart=*/slowStart, /*waitToFinish=*/errorFinish,
            /*notifyFinish=*/null, /*waitForException=*/true, new StringValue("slow"),
            /*deps=*/ImmutableList.<SkyKey>of()));
    SkyKey midKey = GraphTester.toSkyKey("mid");
    tester.getOrCreate(midKey).addDependency(slowKey).setComputedValue(COPY);
    SkyKey topKey = GraphTester.toSkyKey("top");
    tester.getOrCreate(topKey).addDependency(midKey).addDependency(errorKey)
        .setComputedValue(CONCATENATE);
    // slowKey starts -> errorKey finishes, written to graph -> slowKey finishes & (Visitor aborts)
    // -> topKey builds.
    EvaluationResult<StringValue> result = tester.eval(/*keepGoing=*/false, topKey);
    assertThat(result.getError().getRootCauses()).containsExactly(errorKey);
    // Make sure midKey didn't finish building.
    assertEquals(null, tester.getExistingValue(midKey));
    // Give slowKey a nice ordinary builder.
    tester.getOrCreate(slowKey, /*markAsModified=*/false).setBuilder(null)
        .setConstantValue(new StringValue("slow"));
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
    SkyKey successKey = GraphTester.toSkyKey("success");
    tester.getOrCreate(successKey).setBuilder(
        new ChainedFunction(/*notifyStart=*/null, /*waitToFinish=*/null,
            /*notifyFinish=*/blocker, /*waitForException=*/false, new StringValue("yippee"),
            /*deps=*/ImmutableList.<SkyKey>of()));
    SkyKey slowFailKey = GraphTester.toSkyKey("slow_then_fail");
    tester.getOrCreate(slowFailKey).setBuilder(
        new ChainedFunction(/*notifyStart=*/null, /*waitToFinish=*/blocker,
            /*notifyFinish=*/null, /*waitForException=*/false, /*value=*/null,
            /*deps=*/ImmutableList.<SkyKey>of()));

    EvaluationResult<StringValue> result = tester.eval(
        /*keepGoing=*/false, successKey, slowFailKey);
    assertThat(result.getError().getRootCauses()).containsExactly(slowFailKey);
    assertThat(result.values()).containsExactly(new StringValue("yippee"));
  }

  @Test
  public void passThenFailToBuildAlternateOrder() throws Exception {
    CountDownLatch blocker = new CountDownLatch(1);
    SkyKey successKey = GraphTester.toSkyKey("success");
    tester.getOrCreate(successKey).setBuilder(
        new ChainedFunction(/*notifyStart=*/null, /*waitToFinish=*/null,
            /*notifyFinish=*/blocker, /*waitForException=*/false, new StringValue("yippee"),
            /*deps=*/ImmutableList.<SkyKey>of()));
    SkyKey slowFailKey = GraphTester.toSkyKey("slow_then_fail");
    tester.getOrCreate(slowFailKey).setBuilder(
        new ChainedFunction(/*notifyStart=*/null, /*waitToFinish=*/blocker,
            /*notifyFinish=*/null, /*waitForException=*/false, /*value=*/null,
            /*deps=*/ImmutableList.<SkyKey>of()));

    EvaluationResult<StringValue> result = tester.eval(
        /*keepGoing=*/false, slowFailKey, successKey);
    assertThat(result.getError().getRootCauses()).containsExactly(slowFailKey);
    assertThat(result.values()).containsExactly(new StringValue("yippee"));
  }

  @Test
  public void incompleteDirectDepsForDirtyValue() throws Exception {
    initializeTester();
    SkyKey topKey = GraphTester.toSkyKey("top");
    tester.set(topKey, new StringValue("initial"));
    // Put topKey into graph so it will be dirtied on next run.
    assertEquals(new StringValue("initial"), tester.evalAndGet(/*keepGoing=*/false, topKey));
    CountDownLatch slowStart = new CountDownLatch(1);
    CountDownLatch errorFinish = new CountDownLatch(1);
    SkyKey errorKey = GraphTester.toSkyKey("error");
    tester.getOrCreate(errorKey).setBuilder(
        new ChainedFunction(/*notifyStart=*/null, /*waitToFinish=*/slowStart,
            /*notifyFinish=*/errorFinish,
            /*waitForException=*/false, /*value=*/null, /*deps=*/ImmutableList.<SkyKey>of()));
    SkyKey slowKey = GraphTester.toSkyKey("slow");
    tester.getOrCreate(slowKey).setBuilder(
        new ChainedFunction(/*notifyStart=*/slowStart, /*waitToFinish=*/errorFinish,
            /*notifyFinish=*/null, /*waitForException=*/true,
            new StringValue("slow"), /*deps=*/ImmutableList.<SkyKey>of()));
    SkyKey midKey = GraphTester.toSkyKey("mid");
    tester.getOrCreate(midKey).addDependency(slowKey).setComputedValue(COPY);
    tester.set(topKey, null);
    tester.getOrCreate(topKey).addDependency(midKey).addDependency(errorKey)
        .setComputedValue(CONCATENATE);
    tester.invalidate();
    // slowKey starts -> errorKey finishes, written to graph -> slowKey finishes & (Visitor aborts)
    // -> topKey builds.
    EvaluationResult<StringValue> result = tester.eval(/*keepGoing=*/false, topKey);
    assertThat(result.getError().getRootCauses()).containsExactly(errorKey);
    // Make sure midKey didn't finish building.
    assertEquals(null, tester.getExistingValue(midKey));
    // Give slowKey a nice ordinary builder.
    tester.getOrCreate(slowKey, /*markAsModified=*/false).setBuilder(null)
        .setConstantValue(new StringValue("slow"));
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
    SkyKey errorKey = GraphTester.toSkyKey("my_error_value");
    tester.getOrCreate(errorKey).setHasError(true);
    tester.set("after", new StringValue("after"));
    SkyKey parentKey = GraphTester.toSkyKey("parent");
    tester.getOrCreate(parentKey).addErrorDependency(errorKey, new StringValue("recovered"))
        .setComputedValue(CONCATENATE).addDependency("after");
    EvaluationResult<StringValue> result = tester.eval(/*keepGoing=*/true, parentKey);
    assertThat(result.errorMap()).isEmpty();
    assertEquals("recoveredafter", result.get(parentKey).getValue());
    tester.set("after", new StringValue("before"));
    tester.invalidate();
    result = tester.eval(/*keepGoing=*/true, parentKey);
    assertThat(result.errorMap()).isEmpty();
    assertEquals("recoveredbefore", result.get(parentKey).getValue());
  }

  @Test
  public void continueWithErrorDepTurnedGood() throws Exception {
    initializeTester();
    SkyKey errorKey = GraphTester.toSkyKey("my_error_value");
    tester.getOrCreate(errorKey).setHasError(true);
    tester.set("after", new StringValue("after"));
    SkyKey parentKey = GraphTester.toSkyKey("parent");
    tester.getOrCreate(parentKey).addErrorDependency(errorKey, new StringValue("recovered"))
        .setComputedValue(CONCATENATE).addDependency("after");
    EvaluationResult<StringValue> result = tester.eval(/*keepGoing=*/true, parentKey);
    assertThat(result.errorMap()).isEmpty();
    assertEquals("recoveredafter", result.get(parentKey).getValue());
    tester.set(errorKey, new StringValue("reformed")).setHasError(false);
    tester.invalidate();
    result = tester.eval(/*keepGoing=*/true, parentKey);
    assertThat(result.errorMap()).isEmpty();
    assertEquals("reformedafter", result.get(parentKey).getValue());
  }

  @Test
  public void errorDepAlreadyThereThenTurnedGood() throws Exception {
    initializeTester();
    SkyKey errorKey = GraphTester.toSkyKey("my_error_value");
    tester.getOrCreate(errorKey).setHasError(true);
    SkyKey parentKey = GraphTester.toSkyKey("parent");
    tester.getOrCreate(parentKey).addErrorDependency(errorKey, new StringValue("recovered"))
        .setHasError(true);
    // Prime the graph by putting the error value in it beforehand.
    assertThat(tester.evalAndGetError(errorKey).getRootCauses()).containsExactly(errorKey);
    EvaluationResult<StringValue> result = tester.eval(/*keepGoing=*/false, parentKey);
    // Request the parent.
    assertThat(result.getError(parentKey).getRootCauses()).containsExactly(parentKey).inOrder();
    // Change the error value to no longer throw.
    tester.set(errorKey, new StringValue("reformed")).setHasError(false);
    tester.getOrCreate(parentKey, /*markAsModified=*/false).setHasError(false)
        .setComputedValue(COPY);
    tester.differencer.invalidate(ImmutableList.of(errorKey));
    tester.invalidate();
    // Request the parent again. This time it should succeed.
    result = tester.eval(/*keepGoing=*/false, parentKey);
    assertThat(result.errorMap()).isEmpty();
    assertEquals("reformed", result.get(parentKey).getValue());
    // Confirm that the parent no longer depends on the error transience value -- make it
    // unbuildable again, but without invalidating it, and invalidate transient errors. The parent
    // should not be rebuilt.
    tester.getOrCreate(parentKey, /*markAsModified=*/false).setHasError(true);
    tester.invalidateTransientErrors();
    result = tester.eval(/*keepGoing=*/false, parentKey);
    assertThat(result.errorMap()).isEmpty();
    assertEquals("reformed", result.get(parentKey).getValue());
  }

  /**
   * Regression test for 2014 bug: error transience value is registered before newly requested deps.
   * A value requests a child, gets it back immediately, and then throws, causing the error
   * transience value to be registered as a dep. The following build, the error is invalidated via
   * that child.
   */
  @Test
  public void doubleDepOnErrorTransienceValue() throws Exception {
    initializeTester();
    SkyKey leafKey = GraphTester.toSkyKey("leaf");
    tester.set(leafKey, new StringValue("leaf"));
    // Prime the graph by putting leaf in beforehand.
    assertEquals(new StringValue("leaf"), tester.evalAndGet(/*keepGoing=*/false, leafKey));
    SkyKey topKey = GraphTester.toSkyKey("top");
    tester.getOrCreate(topKey).addDependency(leafKey).setHasError(true);
    // Build top -- it has an error.
    assertThat(tester.evalAndGetError(topKey).getRootCauses()).containsExactly(topKey).inOrder();
    // Invalidate top via leaf, and rebuild.
    tester.set(leafKey, new StringValue("leaf2"));
    tester.invalidate();
    assertThat(tester.evalAndGetError(topKey).getRootCauses()).containsExactly(topKey).inOrder();
  }

  /** Regression test for crash bug. */
  @Test
  public void errorTransienceDepCleared() throws Exception {
    initializeTester();
    final SkyKey top = GraphTester.toSkyKey("top");
    SkyKey leaf = GraphTester.toSkyKey("leaf");
    tester.set(leaf, new StringValue("leaf"));
    tester.getOrCreate(top).addDependency(leaf).setHasTransientError(true);
    EvaluationResult<StringValue> result = tester.eval(/*keepGoing=*/false, top);
    assertTrue(result.toString(), result.hasError());
    tester.getOrCreate(leaf, /*markAsModified=*/true);
    tester.invalidate();
    SkyKey irrelevant = GraphTester.toSkyKey("irrelevant");
    tester.set(irrelevant, new StringValue("irrelevant"));
    tester.eval(/*keepGoing=*/true, irrelevant);
    tester.invalidateTransientErrors();
    result = tester.eval(/*keepGoing=*/true, top);
    assertTrue(result.toString(), result.hasError());
  }

  @Test
  public void incompleteValueAlreadyThereNotUsed() throws Exception {
    initializeTester();
    SkyKey errorKey = GraphTester.toSkyKey("my_error_value");
    tester.getOrCreate(errorKey).setHasError(true);
    SkyKey midKey = GraphTester.toSkyKey("mid");
    tester.getOrCreate(midKey).addErrorDependency(errorKey, new StringValue("recovered"))
        .setComputedValue(COPY);
    SkyKey parentKey = GraphTester.toSkyKey("parent");
    tester.getOrCreate(parentKey).addErrorDependency(midKey, new StringValue("don't use this"))
        .setComputedValue(COPY);
    // Prime the graph by evaluating the mid-level value. It shouldn't be stored in the graph
    // because
    // it was only called during the bubbling-up phase.
    EvaluationResult<StringValue> result = tester.eval(/*keepGoing=*/false, midKey);
    assertEquals(null, result.get(midKey));
    assertThat(result.getError().getRootCauses()).containsExactly(errorKey);
    // In a keepGoing build, midKey should be re-evaluated.
    assertEquals("recovered",
        ((StringValue) tester.evalAndGet(/*keepGoing=*/true, parentKey)).getValue());
  }

  /**
   * "top" requests a dependency group in which the first value, called "error", throws an
   * exception, so "mid" and "mid2", which depend on "slow", never get built.
   */
  @Test
  public void errorInDependencyGroup() throws Exception {
    initializeTester();
    SkyKey topKey = GraphTester.toSkyKey("top");
    CountDownLatch slowStart = new CountDownLatch(1);
    CountDownLatch errorFinish = new CountDownLatch(1);
    final SkyKey errorKey = GraphTester.toSkyKey("error");
    tester.getOrCreate(errorKey).setBuilder(
        new ChainedFunction(/*notifyStart=*/null, /*waitToFinish=*/slowStart,
            /*notifyFinish=*/errorFinish, /*waitForException=*/false,
            // ChainedFunction throws when value is null.
            /*value=*/null, /*deps=*/ImmutableList.<SkyKey>of()));
    SkyKey slowKey = GraphTester.toSkyKey("slow");
    tester.getOrCreate(slowKey).setBuilder(
        new ChainedFunction(/*notifyStart=*/slowStart, /*waitToFinish=*/errorFinish,
            /*notifyFinish=*/null, /*waitForException=*/true,
            new StringValue("slow"), /*deps=*/ImmutableList.<SkyKey>of()));
    final SkyKey midKey = GraphTester.toSkyKey("mid");
    tester.getOrCreate(midKey).addDependency(slowKey).setComputedValue(COPY);
    final SkyKey mid2Key = GraphTester.toSkyKey("mid2");
    tester.getOrCreate(mid2Key).addDependency(slowKey).setComputedValue(COPY);
    tester.set(topKey, null);
    tester.getOrCreate(topKey).setBuilder(new SkyFunction() {
      @Override
      public SkyValue compute(SkyKey skyKey, Environment env) throws SkyFunctionException,
          InterruptedException {
        env.getValues(ImmutableList.of(errorKey, midKey, mid2Key));
        if (env.valuesMissing()) {
          return null;
        }
        return new StringValue("top");
      }

      @Override
      public String extractTag(SkyKey skyKey) {
        return null;
      }
    });

    // Assert that build fails and "error" really is in error.
    EvaluationResult<StringValue> result = tester.eval(/*keepGoing=*/false, topKey);
    assertTrue(result.hasError());
    assertThat(result.getError(topKey).getRootCauses()).containsExactly(errorKey);

    // Ensure that evaluation succeeds if errorKey does not throw an error.
    tester.getOrCreate(errorKey).setBuilder(null);
    tester.set(errorKey, new StringValue("ok"));
    tester.invalidate();
    assertEquals(new StringValue("top"), tester.evalAndGet("top"));
  }

  /**
   * Regression test -- if value top requests {depA, depB}, depC, with depA and depC there and depB
   * absent, and then throws an exception, the stored deps should be depA, depC (in different
   * groups), not {depA, depC} (same group).
   */
  @Test
  public void valueInErrorWithGroups() throws Exception {
    initializeTester();
    SkyKey topKey = GraphTester.toSkyKey("top");
    final SkyKey groupDepA = GraphTester.toSkyKey("groupDepA");
    final SkyKey groupDepB = GraphTester.toSkyKey("groupDepB");
    SkyKey depC = GraphTester.toSkyKey("depC");
    tester.set(groupDepA, new StringValue("depC"));
    tester.set(groupDepB, new StringValue(""));
    tester.getOrCreate(depC).setHasError(true);
    tester.getOrCreate(topKey).setBuilder(new NoExtractorFunction() {
      @Override
      public SkyValue compute(SkyKey skyKey, Environment env) throws SkyFunctionException {
        StringValue val = ((StringValue) env.getValues(
            ImmutableList.of(groupDepA, groupDepB)).get(groupDepA));
        if (env.valuesMissing()) {
          return null;
        }
        String nextDep = val.getValue();
        try {
          env.getValueOrThrow(GraphTester.toSkyKey(nextDep), SomeErrorException.class);
        } catch (SomeErrorException e) {
          throw new GenericFunctionException(e, Transience.PERSISTENT);
        }
        return env.valuesMissing() ? null : new StringValue("top");
      }
    });

    EvaluationResult<StringValue> evaluationResult = tester.eval(
        /*keepGoing=*/true, groupDepA, depC);
    assertTrue(evaluationResult.hasError());
    assertEquals("depC", evaluationResult.get(groupDepA).getValue());
    assertThat(evaluationResult.getError(depC).getRootCauses()).containsExactly(depC).inOrder();
    evaluationResult = tester.eval(/*keepGoing=*/false, topKey);
    assertTrue(evaluationResult.hasError());
    assertThat(evaluationResult.getError(topKey).getRootCauses()).containsExactly(topKey).inOrder();

    tester.set(groupDepA, new StringValue("groupDepB"));
    tester.getOrCreate(depC, /*markAsModified=*/true);
    tester.invalidate();
    evaluationResult = tester.eval(/*keepGoing=*/false, topKey);
    assertFalse(evaluationResult.toString(), evaluationResult.hasError());
    assertEquals("top", evaluationResult.get(topKey).getValue());
  }

  @Test
  public void errorOnlyEmittedOnce() throws Exception {
    initializeTester();
    tester.set("x", new StringValue("y")).setWarning("fizzlepop");
    StringValue value = (StringValue) tester.evalAndGet("x");
    assertEquals("y", value.getValue());
    assertContainsEvent(eventCollector, "fizzlepop");
    assertEventCount(1, eventCollector);

    tester.invalidate();
    value = (StringValue) tester.evalAndGet("x");
    assertEquals("y", value.getValue());
    // No new events emitted.
    assertEventCount(1, eventCollector);
  }

  /**
   * We are checking here that we are resilient to a race condition in which a value that is
   * checking its children for dirtiness is signaled by all of its children, putting it in a ready
   * state, before the thread has terminated. Optionally, one of its children may throw an error,
   * shutting down the threadpool. This is similar to
   * {@link ParallelEvaluatorTest#slowChildCleanup}: a child about to throw signals its parent and
   * the parent's builder restarts itself before the exception is thrown. Here, the signaling
   * happens while dirty dependencies are being checked, as opposed to during actual evaluation, but
   * the principle is the same. We control the timing by blocking "top"'s registering itself on its
   * deps.
   */
  private void dirtyChildEnqueuesParentDuringCheckDependencies(final boolean throwError)
      throws Exception {
    // Value to be built. It will be signaled to rebuild before it has finished checking its deps.
    final SkyKey top = GraphTester.toSkyKey("top");
    // Dep that blocks before it acknowledges being added as a dep by top, so the firstKey value has
    // time to signal top. (Importantly its key is alphabetically after 'firstKey').
    final SkyKey slowAddingDep = GraphTester.toSkyKey("slowDep");
    // Don't perform any blocking on the first build.
    final AtomicBoolean delayTopSignaling = new AtomicBoolean(false);
    final CountDownLatch topSignaled = new CountDownLatch(1);
    final CountDownLatch topRestartedBuild = new CountDownLatch(1);
    setGraphForTesting(
        new DeterministicInMemoryGraph(
            new Listener() {
              @Override
              public void accept(
                  SkyKey key, EventType type, Order order, @Nullable Object context) {
                if (!delayTopSignaling.get()) {
                  return;
                }
                if (key.equals(top) && type == EventType.SIGNAL && order == Order.AFTER) {
                  // top is signaled by firstKey (since slowAddingDep is blocking), so slowAddingDep
                  // is now free to acknowledge top as a parent.
                  topSignaled.countDown();
                  return;
                }
                if (key.equals(slowAddingDep)
                    && type == EventType.ADD_REVERSE_DEP
                    && top.equals(context)
                    && order == Order.BEFORE) {
                  // If top is trying to declare a dep on slowAddingDep, wait until firstKey has
                  // signaled top. Then this add dep will return DONE and top will be signaled,
                  // making it ready, so it will be enqueued.
                  TrackingAwaiter.INSTANCE.awaitLatchAndTrackExceptions(
                      topSignaled, "first key didn't signal top in time");
                }
              }
            }));
    // Value that is modified on the second build. Its thread won't finish until it signals top,
    // which will wait for the signal before it enqueues its next dep. We prevent the thread from
    // finishing by having the listener to which it reports its warning block until top's builder
    // starts.
    final SkyKey firstKey = GraphTester.skyKey("first");
    tester.set(firstKey, new StringValue("biding"));
    tester.set(slowAddingDep, new StringValue("dep"));
    final AtomicInteger numTopInvocations = new AtomicInteger(0);
    tester.getOrCreate(top).setBuilder(new NoExtractorFunction() {
      @Override
      public SkyValue compute(SkyKey key, SkyFunction.Environment env) {
        numTopInvocations.incrementAndGet();
        if (delayTopSignaling.get()) {
          // The reporter will be given firstKey's warning to emit when it is requested as a dep
          // below, if firstKey is already built, so we release the reporter's latch beforehand.
          topRestartedBuild.countDown();
        }
        // top's builder just requests both deps in a group.
        env.getValuesOrThrow(ImmutableList.of(firstKey, slowAddingDep), SomeErrorException.class);
        return env.valuesMissing() ? null : new StringValue("top");
      }
    });
    reporter =
        new DelegatingEventHandler(reporter) {
          @Override
          public void handle(Event e) {
            super.handle(e);
            if (e.getKind() == EventKind.WARNING) {
              if (!throwError) {
                TrackingAwaiter.INSTANCE.awaitLatchAndTrackExceptions(
                    topRestartedBuild, "top's builder did not start in time");
              }
            }
          }
        };
    // First build : just prime the graph.
    EvaluationResult<StringValue> result = tester.eval(/*keepGoing=*/false, top);
    assertFalse(result.hasError());
    assertEquals(new StringValue("top"), result.get(top));
    assertEquals(2, numTopInvocations.get());
    // Now dirty the graph, and maybe have firstKey throw an error.
    String warningText = "warning text";
    tester.getOrCreate(firstKey, /*markAsModified=*/true).setHasError(throwError)
        .setWarning(warningText);
    tester.invalidate();
    delayTopSignaling.set(true);
    result = tester.eval(/*keepGoing=*/false, top);
    if (throwError) {
      assertTrue(result.hasError());
      assertThat(result.keyNames()).isEmpty(); // No successfully evaluated values.
      ErrorInfo errorInfo = result.getError(top);
      assertThat(errorInfo.getRootCauses()).containsExactly(firstKey);
      assertEquals("on the incremental build, top's builder should have only been used in error "
          + "bubbling", 3, numTopInvocations.get());
    } else {
      assertEquals(new StringValue("top"), result.get(top));
      assertFalse(result.hasError());
      assertEquals("on the incremental build, top's builder should have only been executed once in "
          + "normal evaluation", 3, numTopInvocations.get());
    }
    assertContainsEvent(eventCollector, warningText);
    assertEquals(0, topSignaled.getCount());
    assertEquals(0, topRestartedBuild.getCount());
  }

  @Test
  public void dirtyChildEnqueuesParentDuringCheckDependencies_ThrowDoesntEnqueue()
      throws Exception {
    dirtyChildEnqueuesParentDuringCheckDependencies(/*throwError=*/true);
  }

  @Test
  public void dirtyChildEnqueuesParentDuringCheckDependencies_NoThrow() throws Exception {
    dirtyChildEnqueuesParentDuringCheckDependencies(/*throwError=*/false);
  }

  @Test
  public void removeReverseDepFromRebuildingNode() throws Exception {
    SkyKey topKey = GraphTester.skyKey("top");
    final SkyKey midKey = GraphTester.skyKey("mid");
    final SkyKey changedKey = GraphTester.skyKey("changed");
    tester.getOrCreate(changedKey).setConstantValue(new StringValue("first"));
    // When top depends on mid,
    tester.getOrCreate(topKey).addDependency(midKey).setComputedValue(CONCATENATE);
    // And mid depends on changed,
    tester.getOrCreate(midKey).addDependency(changedKey).setComputedValue(CONCATENATE);
    final CountDownLatch changedKeyStarted = new CountDownLatch(1);
    final CountDownLatch changedKeyCanFinish = new CountDownLatch(1);
    final AtomicBoolean controlTiming = new AtomicBoolean(false);
    setGraphForTesting(
        new NotifyingInMemoryGraph(
            new Listener() {
              @Override
              public void accept(SkyKey key, EventType type, Order order, Object context) {
                if (!controlTiming.get()) {
                  return;
                }
                if (key.equals(midKey)
                    && type == EventType.CHECK_IF_DONE
                    && order == Order.BEFORE) {
                  TrackingAwaiter.INSTANCE.awaitLatchAndTrackExceptions(
                      changedKeyStarted, "changed key didn't start");
                } else if (key.equals(changedKey)
                    && type == EventType.REMOVE_REVERSE_DEP
                    && order == Order.AFTER
                    && midKey.equals(context)) {
                  changedKeyCanFinish.countDown();
                }
              }
            }));
    // Then top builds as expected.
    assertThat(tester.evalAndGet(/*keepGoing=*/ false, topKey)).isEqualTo(new StringValue("first"));
    // When changed is modified,
    tester
        .getOrCreate(changedKey, /*markAsModified=*/ true)
        .setConstantValue(null)
        .setBuilder(
            // And changed is not allowed to finish building until it is released,
            new ChainedFunction(
                changedKeyStarted,
                changedKeyCanFinish,
                null,
                false,
                new StringValue("second"),
                ImmutableList.<SkyKey>of()));
    // And mid is independently marked as modified,
    tester.getOrCreate(midKey, /*markAsModified=*/ true);
    tester.invalidate();
    SkyKey newTopKey = GraphTester.skyKey("newTop");
    // And changed will start rebuilding independently of midKey, because it's requested directly by
    // newTop
    tester.getOrCreate(newTopKey).addDependency(changedKey).setComputedValue(CONCATENATE);
    // And we control the timing using the graph listener above to make sure that:
    // (1) before we do anything with mid, changed has already started, and
    // (2) changed key can't finish until mid tries to remove its reverse dep from changed,
    controlTiming.set(true);
    // Then this evaluation completes without crashing.
    tester.eval(/*keepGoing=*/ false, newTopKey, topKey);
  }

  @Test
  public void dirtyThenDeleted() throws Exception {
    initializeTester();
    SkyKey topKey = GraphTester.skyKey("top");
    SkyKey leafKey = GraphTester.skyKey("leaf");
    tester.getOrCreate(topKey).addDependency(leafKey).setComputedValue(CONCATENATE);
    tester.set(leafKey, new StringValue("leafy"));
    assertThat(tester.evalAndGet(/*keepGoing=*/false, topKey)).isEqualTo(new StringValue("leafy"));
    tester.getOrCreate(topKey, /*markAsModified=*/true);
    tester.invalidate();
    assertThat(tester.evalAndGet(/*keepGoing=*/false, leafKey))
        .isEqualTo(new StringValue("leafy"));
    tester.delete("top");
    tester.getOrCreate(leafKey, /*markAsModified=*/true);
    tester.invalidate();
    assertThat(tester.evalAndGet(/*keepGoing=*/false, leafKey))
        .isEqualTo(new StringValue("leafy"));
  }

  /**
   * The same dep is requested in two groups, but its value determines what the other dep in the
   * second group is. When it changes, the other dep in the second group should not be requested.
   */
  @Test
  public void sameDepInTwoGroups() throws Exception {
    initializeTester();

    // leaf4 should not built in the second build.
    final SkyKey leaf4 = GraphTester.toSkyKey("leaf4");
    final AtomicBoolean shouldNotBuildLeaf4 = new AtomicBoolean(false);
    setGraphForTesting(
        new NotifyingInMemoryGraph(
            new Listener() {
              @Override
              public void accept(SkyKey key, EventType type, Order order, Object context) {
                if (shouldNotBuildLeaf4.get()
                    && key.equals(leaf4)
                    && type != EventType.REMOVE_REVERSE_DEP) {
                  throw new IllegalStateException(
                      "leaf4 should not have been considered this build: "
                          + type
                          + ", "
                          + order
                          + ", "
                          + context);
                }
              }
            }));
    tester.set(leaf4, new StringValue("leaf4"));

    // Create leaf0, leaf1 and leaf2 values with values "leaf2", "leaf3", "leaf4" respectively.
    // These will be requested as one dependency group. In the second build, leaf2 will have the
    // value "leaf5".
    final List<SkyKey> leaves = new ArrayList<>();
    for (int i = 0; i <= 2; i++) {
      SkyKey leaf = GraphTester.toSkyKey("leaf" + i);
      leaves.add(leaf);
      tester.set(leaf, new StringValue("leaf" + (i + 2)));
    }

    // Create "top" value. It depends on all leaf values in two overlapping dependency groups.
    SkyKey topKey = GraphTester.toSkyKey("top");
    final SkyValue topValue = new StringValue("top");
    tester.getOrCreate(topKey).setBuilder(new NoExtractorFunction() {
      @Override
      public SkyValue compute(SkyKey skyKey, Environment env) throws SkyFunctionException,
          InterruptedException {
        // Request the first group, [leaf0, leaf1, leaf2].
        // In the first build, it has values ["leaf2", "leaf3", "leaf4"].
        // In the second build it has values ["leaf2", "leaf3", "leaf5"]
        Map<SkyKey, SkyValue> values = env.getValues(leaves);
        if (env.valuesMissing()) {
          return null;
        }

        // Request the second group. In the first build it's [leaf2, leaf4].
        // In the second build it's [leaf2, leaf5]
        env.getValues(ImmutableList.of(leaves.get(2),
            GraphTester.toSkyKey(((StringValue) values.get(leaves.get(2))).getValue())));
        if (env.valuesMissing()) {
          return null;
        }

        return topValue;
      }
    });

    // First build: assert we can evaluate "top".
    assertEquals(topValue, tester.evalAndGet(/*keepGoing=*/false, topKey));

    // Second build: replace "leaf4" by "leaf5" in leaf2's value. Assert leaf4 is not requested.
    final SkyKey leaf5 = GraphTester.toSkyKey("leaf5");
    tester.set(leaf5, new StringValue("leaf5"));
    tester.set(leaves.get(2), new StringValue("leaf5"));
    tester.invalidate();
    shouldNotBuildLeaf4.set(true);
    assertEquals(topValue, tester.evalAndGet(/*keepGoing=*/false, topKey));
  }

  @Test
  public void dirtyAndChanged() throws Exception {
    initializeTester();
    SkyKey leaf = GraphTester.toSkyKey("leaf");
    SkyKey mid = GraphTester.toSkyKey("mid");
    SkyKey top = GraphTester.toSkyKey("top");
    tester.getOrCreate(top).addDependency(mid).setComputedValue(COPY);
    tester.getOrCreate(mid).addDependency(leaf).setComputedValue(COPY);
    tester.set(leaf, new StringValue("leafy"));
    // For invalidation.
    tester.set("dummy", new StringValue("dummy"));
    StringValue topValue = (StringValue) tester.evalAndGet("top");
    assertEquals("leafy", topValue.getValue());
    tester.set(leaf, new StringValue("crunchy"));
    tester.invalidate();
    // For invalidation.
    tester.evalAndGet("dummy");
    tester.getOrCreate(mid, /*markAsModified=*/true);
    tester.invalidate();
    topValue = (StringValue) tester.evalAndGet("top");
    assertEquals("crunchy", topValue.getValue());
  }

  /**
   * Test whether a value that was already marked changed will be incorrectly marked dirty, not
   * changed, if another thread tries to mark it just dirty. To exercise this, we need to have a
   * race condition where both threads see that the value is not dirty yet, then the "changed"
   * thread marks the value changed before the "dirty" thread marks the value dirty. To accomplish
   * this, we use a countdown latch to make the "dirty" thread wait until the "changed" thread is
   * done, and another countdown latch to make both of them wait until they have both checked if the
   * value is currently clean.
   */
  @Test
  public void dirtyAndChangedValueIsChanged() throws Exception {
    final SkyKey parent = GraphTester.toSkyKey("parent");
    final AtomicBoolean blockingEnabled = new AtomicBoolean(false);
    final CountDownLatch waitForChanged = new CountDownLatch(1);
    // changed thread checks value entry once (to see if it is changed). dirty thread checks twice,
    // to see if it is changed, and if it is dirty.
    final CountDownLatch threadsStarted = new CountDownLatch(3);
    setGraphForTesting(
        new NotifyingInMemoryGraph(
            new Listener() {
              @Override
              public void accept(SkyKey key, EventType type, Order order, Object context) {
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
                  TrackingAwaiter.INSTANCE.awaitLatchAndTrackExceptions(
                      threadsStarted, "Both threads did not query if value isChanged in time");
                  boolean isChanged = (Boolean) context;
                  if (order == Order.BEFORE && !isChanged) {
                    TrackingAwaiter.INSTANCE.awaitLatchAndTrackExceptions(
                        waitForChanged, "'changed' thread did not mark value changed in time");
                    return;
                  }
                  if (order == Order.AFTER && isChanged) {
                    waitForChanged.countDown();
                  }
                }
              }
            }));
    SkyKey leaf = GraphTester.toSkyKey("leaf");
    tester.set(leaf, new StringValue("leaf"));
    tester.getOrCreate(parent).addDependency(leaf).setComputedValue(CONCATENATE);
    EvaluationResult<StringValue> result;
    result = tester.eval(/*keepGoing=*/false, parent);
    assertEquals("leaf", result.get(parent).getValue());
    // Invalidate leaf, but don't actually change it. It will transitively dirty parent
    // concurrently with parent directly dirtying itself.
    tester.getOrCreate(leaf, /*markAsModified=*/true);
    SkyKey other2 = GraphTester.toSkyKey("other2");
    tester.set(other2, new StringValue("other2"));
    // Invalidate parent, actually changing it.
    tester.getOrCreate(parent, /*markAsModified=*/true).addDependency(other2);
    tester.invalidate();
    blockingEnabled.set(true);
    result = tester.eval(/*keepGoing=*/false, parent);
    assertEquals("leafother2", result.get(parent).getValue());
    assertEquals(0, waitForChanged.getCount());
    assertEquals(0, threadsStarted.getCount());
  }

  @Test
  public void singleValueDependsOnManyDirtyValues() throws Exception {
    initializeTester();
    SkyKey[] values = new SkyKey[TEST_NODE_COUNT];
    StringBuilder expected = new StringBuilder();
    for (int i = 0; i < values.length; i++) {
      String valueName = Integer.toString(i);
      values[i] = GraphTester.toSkyKey(valueName);
      tester.set(values[i], new StringValue(valueName));
      expected.append(valueName);
    }
    SkyKey topKey = toSkyKey("top");
    TestFunction value = tester.getOrCreate(topKey)
        .setComputedValue(CONCATENATE);
    for (int i = 0; i < values.length; i++) {
      value.addDependency(values[i]);
    }

    EvaluationResult<StringValue> result = tester.eval(/*keepGoing=*/false, topKey);
    assertEquals(new StringValue(expected.toString()), result.get(topKey));

    for (int j = 0; j < RUNS; j++) {
      for (int i = 0; i < values.length; i++) {
        tester.getOrCreate(values[i], /*markAsModified=*/true);
      }
      // This value has an error, but we should never discover it because it is not marked changed
      // and all of its dependencies re-evaluate to the same thing.
      tester.getOrCreate(topKey, /*markAsModified=*/false).setHasError(true);
      tester.invalidate();

      result = tester.eval(/*keep_going=*/false, topKey);
      assertEquals(new StringValue(expected.toString()), result.get(topKey));
    }
  }

  /**
   * Tests scenario where we have dirty values in the graph, and then one of them is deleted since
   * its evaluation did not complete before an error was thrown. Can either test the graph via an
   * evaluation of that deleted value, or an invalidation of a child, and can either remove the
   * thrown error or throw it again on that evaluation.
   */
  private void dirtyValueChildrenProperlyRemovedOnEarlyBuildAbort(
      boolean reevaluateMissingValue, boolean removeError) throws Exception {
    initializeTester();
    SkyKey errorKey = GraphTester.toSkyKey("error");
    tester.set(errorKey, new StringValue("biding time"));
    SkyKey slowKey = GraphTester.toSkyKey("slow");
    tester.set(slowKey, new StringValue("slow"));
    SkyKey midKey = GraphTester.toSkyKey("mid");
    tester.getOrCreate(midKey).addDependency(slowKey).setComputedValue(COPY);
    SkyKey lastKey = GraphTester.toSkyKey("last");
    tester.set(lastKey, new StringValue("last"));
    SkyKey motherKey = GraphTester.toSkyKey("mother");
    tester.getOrCreate(motherKey).addDependency(errorKey)
        .addDependency(midKey).addDependency(lastKey).setComputedValue(CONCATENATE);
    SkyKey fatherKey = GraphTester.toSkyKey("father");
    tester.getOrCreate(fatherKey).addDependency(errorKey)
        .addDependency(midKey).addDependency(lastKey).setComputedValue(CONCATENATE);
    EvaluationResult<StringValue> result = tester.eval(/*keepGoing=*/false, motherKey, fatherKey);
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
        new ChainedFunction(/*notifyStart=*/null, /*waitToFinish=*/null,
            /*notifyFinish=*/errorFinish, /*waitForException=*/false, /*value=*/null,
            /*deps=*/ImmutableList.<SkyKey>of()));
    tester.getOrCreate(slowKey).setBuilder(
        new ChainedFunction(/*notifyStart=*/null, /*waitToFinish=*/errorFinish,
            /*notifyFinish=*/null, /*waitForException=*/true, new StringValue("leaf2"),
            /*deps=*/ImmutableList.<SkyKey>of()));
    tester.invalidate();
    // errorKey finishes, written to graph -> leafKey maybe starts+finishes & (Visitor aborts)
    // -> one of mother or father builds. The other one should be cleaned, and no references to it
    // left in the graph.
    result = tester.eval(/*keepGoing=*/false, motherKey, fatherKey);
    assertTrue(result.hasError());
    // Only one of mother or father should be in the graph.
    assertTrue(result.getError(motherKey) + ", " + result.getError(fatherKey),
        (result.getError(motherKey) == null) != (result.getError(fatherKey) == null));
    SkyKey parentKey = (reevaluateMissingValue == (result.getError(motherKey) == null))
        ? motherKey : fatherKey;
    // Give slowKey a nice ordinary builder.
    tester.getOrCreate(slowKey, /*markAsModified=*/false).setBuilder(null)
        .setConstantValue(new StringValue("leaf2"));
    if (removeError) {
      tester.getOrCreate(errorKey, /*markAsModified=*/true).setBuilder(null)
          .setConstantValue(new StringValue("reformed"));
    }
    String lastString = "last";
    if (!reevaluateMissingValue) {
      // Mark the last key modified if we're not trying the absent value again. This invalidation
      // will test if lastKey still has a reference to the absent value.
      lastString = "last2";
      tester.set(lastKey, new StringValue(lastString));
    }
    tester.invalidate();
    result = tester.eval(/*keepGoing=*/false, parentKey);
    if (removeError) {
      assertThat(result.get(parentKey).getValue()).isEqualTo("reformedleaf2" + lastString);
    } else {
      assertNotNull(result.getError(parentKey));
    }
  }

  /**
   * The following four tests (dirtyChildrenProperlyRemovedWith*) test the consistency of the graph
   * after a failed build in which a dirty value should have been deleted from the graph. The
   * consistency is tested via either evaluating the missing value, or the re-evaluating the present
   * value, and either clearing the error or keeping it. To evaluate the present value, we
   * invalidate the error value to force re-evaluation. Related to bug "skyframe m1: graph may not
   * be properly cleaned on interrupt or failure".
   */
  @Test
  public void dirtyChildrenProperlyRemovedWithInvalidateRemoveError() throws Exception {
    dirtyValueChildrenProperlyRemovedOnEarlyBuildAbort(/*reevaluateMissingValue=*/false,
        /*removeError=*/true);
  }

  @Test
  public void dirtyChildrenProperlyRemovedWithInvalidateKeepError() throws Exception {
    dirtyValueChildrenProperlyRemovedOnEarlyBuildAbort(/*reevaluateMissingValue=*/false,
        /*removeError=*/false);
  }

  @Test
  public void dirtyChildrenProperlyRemovedWithReevaluateRemoveError() throws Exception {
    dirtyValueChildrenProperlyRemovedOnEarlyBuildAbort(/*reevaluateMissingValue=*/true,
        /*removeError=*/true);
  }

  @Test
  public void dirtyChildrenProperlyRemovedWithReevaluateKeepError() throws Exception {
    dirtyValueChildrenProperlyRemovedOnEarlyBuildAbort(/*reevaluateMissingValue=*/true,
        /*removeError=*/false);
  }

  /**
   * Regression test: enqueue so many values that some of them won't have started processing, and
   * then either interrupt processing or have a child throw an error. In the latter case, this also
   * tests that a value that hasn't started processing can still have a child error bubble up to it.
   * In both cases, it tests that the graph is properly cleaned of the dirty values and references
   * to them.
   */
  private void manyDirtyValuesClearChildrenOnFail(boolean interrupt) throws Exception {
    SkyKey leafKey = GraphTester.toSkyKey("leaf");
    tester.set(leafKey, new StringValue("leafy"));
    SkyKey lastKey = GraphTester.toSkyKey("last");
    tester.set(lastKey, new StringValue("last"));
    final List<SkyKey> tops = new ArrayList<>();
    // Request far more top-level values than there are threads, so some of them will block until
    // the
    // leaf child is enqueued for processing.
    for (int i = 0; i < 10000; i++) {
      SkyKey topKey = GraphTester.toSkyKey("top" + i);
      tester.getOrCreate(topKey).addDependency(leafKey).addDependency(lastKey)
          .setComputedValue(CONCATENATE);
      tops.add(topKey);
    }
    tester.eval(/*keepGoing=*/false, tops.toArray(new SkyKey[0]));
    final CountDownLatch notifyStart = new CountDownLatch(1);
    tester.set(leafKey, null);
    if (interrupt) {
      // leaf will wait for an interrupt if desired. We cannot use the usual ChainedFunction
      // because we need to actually throw the interrupt.
      final AtomicBoolean shouldSleep = new AtomicBoolean(true);
      tester.getOrCreate(leafKey, /*markAsModified=*/true).setBuilder(
          new NoExtractorFunction() {
            @Override
            public SkyValue compute(SkyKey skyKey, Environment env) throws InterruptedException {
              notifyStart.countDown();
              if (shouldSleep.get()) {
                // Should be interrupted within 5 seconds.
                Thread.sleep(5000);
                throw new AssertionError("leaf was not interrupted");
              }
              return new StringValue("crunchy");
            }
          });
      tester.invalidate();
      TestThread evalThread = new TestThread() {
        @Override
        public void runTest() {
          try {
            tester.eval(/*keepGoing=*/false, tops.toArray(new SkyKey[0]));
            Assert.fail();
          } catch (InterruptedException e) {
            // Expected.
          }
        }
      };
      evalThread.start();
      assertTrue(notifyStart.await(TestUtils.WAIT_TIMEOUT_SECONDS, TimeUnit.SECONDS));
      evalThread.interrupt();
      evalThread.joinAndAssertState(TestUtils.WAIT_TIMEOUT_MILLISECONDS);
      // Free leafKey to compute next time.
      shouldSleep.set(false);
    } else {
      // Non-interrupt case. Just throw an error in the child.
      tester.getOrCreate(leafKey, /*markAsModified=*/true).setHasError(true);
      tester.invalidate();
      // The error thrown may non-deterministically bubble up to a parent that has not yet started
      // processing, but has been enqueued for processing.
      tester.eval(/*keepGoing=*/false, tops.toArray(new SkyKey[0]));
      tester.getOrCreate(leafKey, /*markAsModified=*/true).setHasError(false);
      tester.set(leafKey, new StringValue("crunchy"));
    }
    // lastKey was not touched during the previous build, but its reverse deps on its parents should
    // still be accurate.
    tester.set(lastKey, new StringValue("new last"));
    tester.invalidate();
    EvaluationResult<StringValue> result =
        tester.eval(/*keepGoing=*/false, tops.toArray(new SkyKey[0]));
    for (SkyKey topKey : tops) {
      assertEquals(topKey.toString(), "crunchynew last", result.get(topKey).getValue());
    }
  }

  /**
   * Regression test: make sure that if an evaluation fails before a dirty value starts evaluation
   * (in particular, before it is reset), the graph remains consistent.
   */
  @Test
  public void manyDirtyValuesClearChildrenOnError() throws Exception {
    manyDirtyValuesClearChildrenOnFail(/*interrupt=*/false);
  }

  /**
   * Regression test: Make sure that if an evaluation is interrupted before a dirty value starts
   * evaluation (in particular, before it is reset), the graph remains consistent.
   */
  @Test
  public void manyDirtyValuesClearChildrenOnInterrupt() throws Exception {
    manyDirtyValuesClearChildrenOnFail(/*interrupt=*/true);
  }

  /**
   * Regression test for case where the user requests that we delete nodes that are already in the
   * queue to be dirtied. We should handle that gracefully and not complain.
   */
  @Test
  public void deletingDirtyNodes() throws Exception {
    final Thread thread = Thread.currentThread();
    final AtomicBoolean interruptInvalidation = new AtomicBoolean(false);
    initializeTester(new TrackingInvalidationReceiver() {
      private final AtomicBoolean firstInvalidation = new AtomicBoolean(true);

      @Override
      public void invalidated(SkyKey skyKey, InvalidationState state) {
        if (interruptInvalidation.get() && !firstInvalidation.getAndSet(false)) {
          thread.interrupt();
        }
        super.invalidated(skyKey, state);
      }
    });
    SkyKey key = null;
    // Create a long chain of nodes. Most of them will not actually be dirtied, but the last one to
    // be dirtied will enqueue its parent for dirtying, so it will be in the queue for the next run.
    for (int i = 0; i < TEST_NODE_COUNT; i++) {
      key = GraphTester.toSkyKey("node" + i);
      if (i > 0) {
        tester.getOrCreate(key).addDependency("node" + (i - 1)).setComputedValue(COPY);
      } else {
        tester.set(key, new StringValue("node0"));
      }
    }
    // Seed the graph.
    assertEquals("node0", ((StringValue) tester.evalAndGet(/*keepGoing=*/false, key)).getValue());
    // Start the dirtying process.
    tester.set("node0", new StringValue("new"));
    tester.invalidate();
    interruptInvalidation.set(true);
    try {
      tester.eval(/*keepGoing=*/false, key);
      fail();
    } catch (InterruptedException e) {
      // Expected.
    }
    interruptInvalidation.set(false);
    // Now delete all the nodes. The node that was going to be dirtied is also deleted, which we
    // should handle.
    tester.evaluator.delete(Predicates.<SkyKey>alwaysTrue());
    assertEquals("new", ((StringValue) tester.evalAndGet(/*keepGoing=*/false, key)).getValue());
  }

  @Test
  public void changePruning() throws Exception {
    initializeTester();
    SkyKey leaf = GraphTester.toSkyKey("leaf");
    SkyKey mid = GraphTester.toSkyKey("mid");
    SkyKey top = GraphTester.toSkyKey("top");
    tester.getOrCreate(top).addDependency(mid).setComputedValue(COPY);
    tester.getOrCreate(mid).addDependency(leaf).setComputedValue(COPY);
    tester.set(leaf, new StringValue("leafy"));
    StringValue topValue = (StringValue) tester.evalAndGet("top");
    assertEquals("leafy", topValue.getValue());
    // Mark leaf changed, but don't actually change it.
    tester.getOrCreate(leaf, /*markAsModified=*/true);
    // mid will give an error if re-evaluated, but it shouldn't be because it is not marked changed,
    // and its dirty child will evaluate to the same element.
    tester.getOrCreate(mid, /*markAsModified=*/false).setHasError(true);
    tester.invalidate();
    EvaluationResult<StringValue> result = tester.eval(/*keepGoing=*/false, top);
    assertFalse(result.hasError());
    topValue = result.get(top);
    assertEquals("leafy", topValue.getValue());
    assertThat(tester.getDirtyKeys()).isEmpty();
    assertThat(tester.getDeletedKeys()).isEmpty();
  }

  @Test
  public void changePruningWithDoneValue() throws Exception {
    initializeTester();
    SkyKey leaf = GraphTester.toSkyKey("leaf");
    SkyKey mid = GraphTester.toSkyKey("mid");
    SkyKey top = GraphTester.toSkyKey("top");
    SkyKey suffix = GraphTester.toSkyKey("suffix");
    StringValue suffixValue = new StringValue("suffix");
    tester.set(suffix, suffixValue);
    tester.getOrCreate(top).addDependency(mid).addDependency(suffix).setComputedValue(CONCATENATE);
    tester.getOrCreate(mid).addDependency(leaf).addDependency(suffix).setComputedValue(CONCATENATE);
    SkyValue leafyValue = new StringValue("leafy");
    tester.set(leaf, leafyValue);
    StringValue value = (StringValue) tester.evalAndGet("top");
    assertEquals("leafysuffixsuffix", value.getValue());
    // Mark leaf changed, but don't actually change it.
    tester.getOrCreate(leaf, /*markAsModified=*/true);
    // mid will give an error if re-evaluated, but it shouldn't be because it is not marked changed,
    // and its dirty child will evaluate to the same element.
    tester.getOrCreate(mid, /*markAsModified=*/false).setHasError(true);
    tester.invalidate();
    value = (StringValue) tester.evalAndGet("leaf");
    assertEquals("leafy", value.getValue());
    assertThat(tester.getDirtyKeys()).containsExactly(mid, top);
    assertThat(tester.getDeletedKeys()).isEmpty();
    EvaluationResult<StringValue> result = tester.eval(/*keepGoing=*/false, top);
    assertWithMessage(result.toString()).that(result.hasError()).isFalse();
    value = result.get(top);
    assertEquals("leafysuffixsuffix", value.getValue());
    assertThat(tester.getDirtyKeys()).isEmpty();
    assertThat(tester.getDeletedKeys()).isEmpty();
  }

  @Test
  public void changePruningAfterParentPrunes() throws Exception {
    initializeTester();
    final SkyKey leaf = GraphTester.toSkyKey("leaf");
    SkyKey top = GraphTester.toSkyKey("top");
    tester.set(leaf, new StringValue("leafy"));
    // When top depends on leaf, but always returns the same value,
    final StringValue fixedTopValue = new StringValue("top");
    final AtomicBoolean topEvaluated = new AtomicBoolean(false);
    tester
        .getOrCreate(top)
        .setBuilder(
            new SkyFunction() {
              @Override
              public SkyValue compute(SkyKey skyKey, Environment env) {
                topEvaluated.set(true);
                return env.getValue(leaf) == null ? null : fixedTopValue;
              }

              @Nullable
              @Override
              public String extractTag(SkyKey skyKey) {
                return null;
              }
            });
    // And top is evaluated,
    StringValue topValue = (StringValue) tester.evalAndGet("top");
    // Then top's value is as expected,
    assertEquals(fixedTopValue, topValue);
    // And top was actually evaluated.
    assertThat(topEvaluated.get()).isTrue();
    // When leaf is changed,
    tester.set(leaf, new StringValue("crunchy"));
    tester.invalidate();
    topEvaluated.set(false);
    // And top is evaluated,
    StringValue topValue2 = (StringValue) tester.evalAndGet("top");
    // Then top's value is as expected,
    assertEquals(fixedTopValue, topValue2);
    // And top was actually evaluated.
    assertThat(topEvaluated.get()).isTrue();
    // When leaf is invalidated but not actually changed,
    tester.getOrCreate(leaf, /*markAsModified=*/ true);
    tester.invalidate();
    topEvaluated.set(false);
    // And top is evaluated,
    StringValue topValue3 = (StringValue) tester.evalAndGet("top");
    // Then top's value is as expected,
    assertEquals(fixedTopValue, topValue3);
    // And top was *not* actually evaluated, because change pruning cut off evaluation.
    assertThat(topEvaluated.get()).isFalse();
  }

  @Test
  public void changePruningFromOtherNodeAfterParentPrunes() throws Exception {
    initializeTester();
    final SkyKey leaf = GraphTester.toSkyKey("leaf");
    final SkyKey other = GraphTester.toSkyKey("other");
    SkyKey top = GraphTester.toSkyKey("top");
    tester.set(leaf, new StringValue("leafy"));
    tester.set(other, new StringValue("other"));
    // When top depends on leaf and other, but always returns the same value,
    final StringValue fixedTopValue = new StringValue("top");
    final AtomicBoolean topEvaluated = new AtomicBoolean(false);
    tester
        .getOrCreate(top)
        .setBuilder(
            new SkyFunction() {
              @Override
              public SkyValue compute(SkyKey skyKey, Environment env) {
                topEvaluated.set(true);

                return env.getValue(other) == null || env.getValue(leaf) == null
                    ? null
                    : fixedTopValue;
              }

              @Nullable
              @Override
              public String extractTag(SkyKey skyKey) {
                return null;
              }
            });
    // And top is evaluated,
    StringValue topValue = (StringValue) tester.evalAndGet("top");
    // Then top's value is as expected,
    assertEquals(fixedTopValue, topValue);
    // And top was actually evaluated.
    assertThat(topEvaluated.get()).isTrue();
    // When leaf is changed,
    tester.set(leaf, new StringValue("crunchy"));
    tester.invalidate();
    topEvaluated.set(false);
    // And top is evaluated,
    StringValue topValue2 = (StringValue) tester.evalAndGet("top");
    // Then top's value is as expected,
    assertEquals(fixedTopValue, topValue2);
    // And top was actually evaluated.
    assertThat(topEvaluated.get()).isTrue();
    // When other is invalidated but not actually changed,
    tester.getOrCreate(other, /*markAsModified=*/ true);
    tester.invalidate();
    topEvaluated.set(false);
    // And top is evaluated,
    StringValue topValue3 = (StringValue) tester.evalAndGet("top");
    // Then top's value is as expected,
    assertEquals(fixedTopValue, topValue3);
    // And top was *not* actually evaluated, because change pruning cut off evaluation.
    assertThat(topEvaluated.get()).isFalse();
  }

  @Test
  public void changedChildChangesDepOfParent() throws Exception {
    initializeTester();
    final SkyKey buildFile = GraphTester.toSkyKey("buildFile");
    ValueComputer authorDrink = new ValueComputer() {
      @Override
      public SkyValue compute(Map<SkyKey, SkyValue> deps, SkyFunction.Environment env) {
        String author = ((StringValue) deps.get(buildFile)).getValue();
        StringValue beverage;
        switch (author) {
          case "hemingway":
            beverage = (StringValue) env.getValue(GraphTester.toSkyKey("absinthe"));
            break;
          case "joyce":
            beverage = (StringValue) env.getValue(GraphTester.toSkyKey("whiskey"));
            break;
          default:
              throw new IllegalStateException(author);
        }
        if (beverage == null) {
          return null;
        }
        return new StringValue(author + " drank " + beverage.getValue());
      }
    };

    tester.set(buildFile, new StringValue("hemingway"));
    SkyKey absinthe = GraphTester.toSkyKey("absinthe");
    tester.set(absinthe, new StringValue("absinthe"));
    SkyKey whiskey = GraphTester.toSkyKey("whiskey");
    tester.set(whiskey, new StringValue("whiskey"));
    SkyKey top = GraphTester.toSkyKey("top");
    tester.getOrCreate(top).addDependency(buildFile).setComputedValue(authorDrink);
    StringValue topValue = (StringValue) tester.evalAndGet("top");
    assertEquals("hemingway drank absinthe", topValue.getValue());
    tester.set(buildFile, new StringValue("joyce"));
    // Don't evaluate absinthe successfully anymore.
    tester.getOrCreate(absinthe).setHasError(true);
    tester.invalidate();
    topValue = (StringValue) tester.evalAndGet("top");
    assertEquals("joyce drank whiskey", topValue.getValue());
    assertThat(tester.getDirtyKeys()).containsExactly(buildFile, top);
    assertThat(tester.getDeletedKeys()).isEmpty();
  }

  @Test
  public void dirtyDepIgnoresChildren() throws Exception {
    initializeTester();
    SkyKey leaf = GraphTester.toSkyKey("leaf");
    SkyKey mid = GraphTester.toSkyKey("mid");
    SkyKey top = GraphTester.toSkyKey("top");
    tester.set(mid, new StringValue("ignore"));
    tester.getOrCreate(top).addDependency(mid).setComputedValue(COPY);
    tester.getOrCreate(mid).addDependency(leaf);
    tester.set(leaf, new StringValue("leafy"));
    StringValue topValue = (StringValue) tester.evalAndGet("top");
    assertEquals("ignore", topValue.getValue());
    assertThat(tester.getDirtyKeys()).isEmpty();
    assertThat(tester.getDeletedKeys()).isEmpty();
    // Change leaf.
    tester.set(leaf, new StringValue("crunchy"));
    tester.invalidate();
    topValue = (StringValue) tester.evalAndGet("top");
    assertEquals("ignore", topValue.getValue());
    assertThat(tester.getDirtyKeys()).containsExactly(leaf);
    assertThat(tester.getDeletedKeys()).isEmpty();
    tester.set(leaf, new StringValue("smushy"));
    tester.invalidate();
    topValue = (StringValue) tester.evalAndGet("top");
    assertEquals("ignore", topValue.getValue());
    assertThat(tester.getDirtyKeys()).containsExactly(leaf);
    assertThat(tester.getDeletedKeys()).isEmpty();
  }

  private static final SkyFunction INTERRUPT_BUILDER = new SkyFunction() {

    @Override
    public SkyValue compute(SkyKey skyKey, Environment env) throws SkyFunctionException,
        InterruptedException {
      throw new InterruptedException();
    }

    @Override
    public String extractTag(SkyKey skyKey) {
      throw new UnsupportedOperationException();
    }
  };

  /**
   * Utility function to induce a graph clean of whatever value is requested, by trying to build
   * this value and interrupting the build as soon as this value's function evaluation starts.
   */
  private void failBuildAndRemoveValue(final SkyKey value) {
    tester.set(value, null);
    // Evaluator will think leaf was interrupted because it threw, so it will be cleaned from graph.
    tester.getOrCreate(value, /*markAsModified=*/true).setBuilder(INTERRUPT_BUILDER);
    tester.invalidate();
    try {
      tester.eval(/*keepGoing=*/false, value);
      Assert.fail();
    } catch (InterruptedException e) {
      // Expected.
    }
    tester.getOrCreate(value, /*markAsModified=*/false).setBuilder(null);
  }

  /**
   * Make sure that when a dirty value is building, the fact that a child may no longer exist in the
   * graph doesn't cause problems.
   */
  @Test
  public void dirtyBuildAfterFailedBuild() throws Exception {
    initializeTester();
    final SkyKey leaf = GraphTester.toSkyKey("leaf");
    SkyKey top = GraphTester.toSkyKey("top");
    tester.getOrCreate(top).addDependency(leaf).setComputedValue(COPY);
    tester.set(leaf, new StringValue("leafy"));
    StringValue topValue = (StringValue) tester.evalAndGet("top");
    assertEquals("leafy", topValue.getValue());
    assertThat(tester.getDirtyKeys()).isEmpty();
    assertThat(tester.getDeletedKeys()).isEmpty();
    failBuildAndRemoveValue(leaf);
    // Leaf should no longer exist in the graph. Check that this doesn't cause problems.
    tester.set(leaf, null);
    tester.set(leaf, new StringValue("crunchy"));
    tester.invalidate();
    topValue = (StringValue) tester.evalAndGet("top");
    assertEquals("crunchy", topValue.getValue());
  }

  /**
   * Regression test: error when clearing reverse deps on dirty value about to be rebuilt, because
   * child values were deleted and recreated in interim, forgetting they had reverse dep on dirty
   * value in the first place.
   */
  @Test
  public void changedBuildAfterFailedThenSuccessfulBuild() throws Exception {
    initializeTester();
    final SkyKey leaf = GraphTester.toSkyKey("leaf");
    SkyKey top = GraphTester.toSkyKey("top");
    tester.getOrCreate(top).addDependency(leaf).setComputedValue(COPY);
    tester.set(leaf, new StringValue("leafy"));
    StringValue topValue = (StringValue) tester.evalAndGet("top");
    assertEquals("leafy", topValue.getValue());
    assertThat(tester.getDirtyKeys()).isEmpty();
    assertThat(tester.getDeletedKeys()).isEmpty();
    failBuildAndRemoveValue(leaf);
    tester.set(leaf, new StringValue("crunchy"));
    tester.invalidate();
    tester.eval(/*keepGoing=*/false, leaf);
    // Leaf no longer has reverse dep on top. Check that this doesn't cause problems, even if the
    // top value is evaluated unconditionally.
    tester.getOrCreate(top, /*markAsModified=*/true);
    tester.invalidate();
    topValue = (StringValue) tester.evalAndGet("top");
    assertEquals("crunchy", topValue.getValue());
  }

  /**
   * Regression test: child value that has been deleted since it and its parent were marked dirty no
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
  public void manyDirtyValuesClearChildrenOnSecondFail() throws Exception {
    final SkyKey leafKey = GraphTester.toSkyKey("leaf");
    tester.set(leafKey, new StringValue("leafy"));
    SkyKey lastKey = GraphTester.toSkyKey("last");
    tester.set(lastKey, new StringValue("last"));
    final List<SkyKey> tops = new ArrayList<>();
    // Request far more top-level values than there are threads, so some of them will block until
    // the leaf child is enqueued for processing.
    for (int i = 0; i < 10000; i++) {
      SkyKey topKey = GraphTester.toSkyKey("top" + i);
      tester.getOrCreate(topKey).addDependency(leafKey).addDependency(lastKey)
          .setComputedValue(CONCATENATE);
      tops.add(topKey);
    }
    tester.eval(/*keepGoing=*/false, tops.toArray(new SkyKey[0]));
    failBuildAndRemoveValue(leafKey);
    // Request the tops. Since leaf was deleted from the graph last build, it no longer knows that
    // its parents depend on it. When leaf throws, at least one of its parents (hopefully) will not
    // have re-informed leaf that the parent depends on it, exposing the bug, since the parent
    // should then not try to clean the reverse dep from leaf.
    tester.set(leafKey, null);
    // Evaluator will think leaf was interrupted because it threw, so it will be cleaned from graph.
    tester.getOrCreate(leafKey, /*markAsModified=*/true).setBuilder(INTERRUPT_BUILDER);
    tester.invalidate();
    try {
      tester.eval(/*keepGoing=*/false, tops.toArray(new SkyKey[0]));
      Assert.fail();
    } catch (InterruptedException e) {
      // Expected.
    }
  }

  @Test
  public void failedDirtyBuild() throws Exception {
    initializeTester();
    SkyKey leaf = GraphTester.toSkyKey("leaf");
    SkyKey top = GraphTester.toSkyKey("top");
    tester.getOrCreate(top).addErrorDependency(leaf, new StringValue("recover"))
        .setComputedValue(COPY);
    tester.set(leaf, new StringValue("leafy"));
    StringValue topValue = (StringValue) tester.evalAndGet("top");
    assertEquals("leafy", topValue.getValue());
    assertThat(tester.getDirtyKeys()).isEmpty();
    assertThat(tester.getDeletedKeys()).isEmpty();
    // Change leaf.
    tester.getOrCreate(leaf, /*markAsModified=*/true).setHasError(true);
    tester.getOrCreate(top, /*markAsModified=*/false).setHasError(true);
    tester.invalidate();
    EvaluationResult<StringValue> result = tester.eval(/*keepGoing=*/false, top);
    assertNull("value should not have completed evaluation", result.get(top));
    assertWithMessage(
        "The error thrown by leaf should have been swallowed by the error thrown by top")
        .that(result.getError().getRootCauses()).containsExactly(top);
  }

  @Test
  public void failedDirtyBuildInBuilder() throws Exception {
    initializeTester();
    SkyKey leaf = GraphTester.toSkyKey("leaf");
    SkyKey secondError = GraphTester.toSkyKey("secondError");
    SkyKey top = GraphTester.toSkyKey("top");
    tester.getOrCreate(top).addDependency(leaf)
        .addErrorDependency(secondError, new StringValue("recover")).setComputedValue(CONCATENATE);
    tester.set(secondError, new StringValue("secondError")).addDependency(leaf);
    tester.set(leaf, new StringValue("leafy"));
    StringValue topValue = (StringValue) tester.evalAndGet("top");
    assertEquals("leafysecondError", topValue.getValue());
    assertThat(tester.getDirtyKeys()).isEmpty();
    assertThat(tester.getDeletedKeys()).isEmpty();
    // Invalidate leaf.
    tester.getOrCreate(leaf, /*markAsModified=*/true);
    tester.set(leaf, new StringValue("crunchy"));
    tester.getOrCreate(secondError, /*markAsModified=*/true).setHasError(true);
    tester.getOrCreate(top, /*markAsModified=*/false).setHasError(true);
    tester.invalidate();
    EvaluationResult<StringValue> result = tester.eval(/*keepGoing=*/false, top);
    assertNull("value should not have completed evaluation", result.get(top));
    assertWithMessage(
        "The error thrown by leaf should have been swallowed by the error thrown by top")
        .that(result.getError().getRootCauses()).containsExactly(top);
  }

  @Test
  public void dirtyErrorTransienceValue() throws Exception {
    initializeTester();
    SkyKey error = GraphTester.toSkyKey("error");
    tester.getOrCreate(error).setHasError(true);
    assertNotNull(tester.evalAndGetError(error));
    tester.invalidateTransientErrors();
    SkyKey secondError = GraphTester.toSkyKey("secondError");
    tester.getOrCreate(secondError).setHasError(true);
    // secondError declares a new dependence on ErrorTransienceValue, but not until it has already
    // thrown an error.
    assertNotNull(tester.evalAndGetError(secondError));
  }

  @Test
  public void dirtyDependsOnErrorTurningGood() throws Exception {
    initializeTester();
    SkyKey error = GraphTester.toSkyKey("error");
    tester.getOrCreate(error).setHasError(true);
    SkyKey topKey = GraphTester.toSkyKey("top");
    tester.getOrCreate(topKey).addDependency(error).setComputedValue(COPY);
    EvaluationResult<StringValue> result = tester.eval(/*keepGoing=*/false, topKey);
    assertThat(result.getError(topKey).getRootCauses()).containsExactly(error);
    tester.getOrCreate(error).setHasError(false);
    StringValue val = new StringValue("reformed");
    tester.set(error, val);
    tester.invalidate();
    result = tester.eval(/*keepGoing=*/false, topKey);
    assertEquals(val, result.get(topKey));
    assertFalse(result.hasError());
  }

  /** Regression test for crash bug. */
  @Test
  public void dirtyWithOwnErrorDependsOnTransientErrorTurningGood() throws Exception {
    initializeTester();
    final SkyKey error = GraphTester.toSkyKey("error");
    tester.getOrCreate(error).setHasTransientError(true);
    SkyKey topKey = GraphTester.toSkyKey("top");
    SkyFunction errorFunction = new SkyFunction() {
      @Override
      public SkyValue compute(SkyKey skyKey, Environment env) throws GenericFunctionException,
          InterruptedException {
        try {
          return env.getValueOrThrow(error, SomeErrorException.class);
        } catch (SomeErrorException e) {
          throw new GenericFunctionException(e, Transience.PERSISTENT);
        }
      }

      @Override
      public String extractTag(SkyKey skyKey) {
        throw new UnsupportedOperationException();
      }
    };
    tester.getOrCreate(topKey).setBuilder(errorFunction);
    EvaluationResult<StringValue> result = tester.eval(/*keepGoing=*/false, topKey);
    tester.invalidateTransientErrors();
    assertThat(result.getError(topKey).getRootCauses()).containsExactly(topKey);
    tester.getOrCreate(error).setHasTransientError(false);
    StringValue reformed = new StringValue("reformed");
    tester.set(error, reformed);
    tester.getOrCreate(topKey).setBuilder(null).addDependency(error).setComputedValue(COPY);
    tester.invalidate();
    tester.invalidateTransientErrors();
    result = tester.eval(/*keepGoing=*/false, topKey);
    assertEquals(reformed, result.get(topKey));
    assertFalse(result.hasError());
  }

  /**
   * Make sure that when an error is thrown, it is given for handling only to parents that have
   * already registered a dependence on the value that threw the error.
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
    setGraphForTesting(new DeterministicInMemoryGraph());
    SkyKey errorKey = GraphTester.toSkyKey("error");
    tester.set(errorKey, new StringValue("biding time"));
    SkyKey slowKey = GraphTester.toSkyKey("slow");
    tester.set(slowKey, new StringValue("slow"));
    SkyKey midKey = GraphTester.toSkyKey("mid");
    tester.getOrCreate(midKey).addDependency(slowKey).setComputedValue(COPY);
    SkyKey topErrorFirstKey = GraphTester.toSkyKey("2nd top alphabetically");
    tester.getOrCreate(topErrorFirstKey).addDependency(errorKey).setComputedValue(CONCATENATE);
    SkyKey topBubbleKey = GraphTester.toSkyKey("1st top alphabetically");
    tester.getOrCreate(topBubbleKey).addDependency(midKey).addDependency(errorKey)
        .setComputedValue(CONCATENATE);
    // First error-free evaluation, to put all values in graph.
    EvaluationResult<StringValue> result = tester.eval(/*keepGoing=*/false,
        topErrorFirstKey, topBubbleKey);
    assertEquals("biding time", result.get(topErrorFirstKey).getValue());
    assertEquals("slowbiding time", result.get(topBubbleKey).getValue());
    // Set up timing of child values: slowKey waits to finish until errorKey has thrown an
    // exception that has been caught by the threadpool.
    tester.set(slowKey, null);
    CountDownLatch errorFinish = new CountDownLatch(1);
    tester.set(errorKey, null);
    tester.getOrCreate(errorKey).setBuilder(
        new ChainedFunction(/*notifyStart=*/null, /*waitToFinish=*/null,
            /*notifyFinish=*/errorFinish, /*waitForException=*/false, /*value=*/null,
            /*deps=*/ImmutableList.<SkyKey>of()));
    tester.getOrCreate(slowKey).setBuilder(
        new ChainedFunction(/*notifyStart=*/null, /*waitToFinish=*/errorFinish,
            /*notifyFinish=*/null, /*waitForException=*/true, new StringValue("leaf2"),
            /*deps=*/ImmutableList.<SkyKey>of()));
    tester.invalidate();
    // errorKey finishes, written to graph -> slowKey maybe starts+finishes & (Visitor aborts)
    // -> some top key builds.
    result = tester.eval(/*keepGoing=*/false, topErrorFirstKey, topBubbleKey);
    assertTrue(result.hasError());
    assertWithMessage(result.toString()).that(result.getError(topErrorFirstKey)).isNotNull();
  }

  @Test
  public void dirtyWithRecoveryErrorDependsOnErrorTurningGood() throws Exception {
    initializeTester();
    final SkyKey error = GraphTester.toSkyKey("error");
    tester.getOrCreate(error).setHasError(true);
    SkyKey topKey = GraphTester.toSkyKey("top");
    SkyFunction recoveryErrorFunction = new SkyFunction() {
      @Override
      public SkyValue compute(SkyKey skyKey, Environment env) throws SkyFunctionException,
          InterruptedException {
        try {
          env.getValueOrThrow(error, SomeErrorException.class);
        } catch (SomeErrorException e) {
          throw new GenericFunctionException(e, Transience.PERSISTENT);
        }
        return null;
      }

      @Override
      public String extractTag(SkyKey skyKey) {
        throw new UnsupportedOperationException();
      }
    };
    tester.getOrCreate(topKey).setBuilder(recoveryErrorFunction);
    EvaluationResult<StringValue> result = tester.eval(/*keepGoing=*/false, topKey);
    assertThat(result.getError(topKey).getRootCauses()).containsExactly(topKey);
    tester.getOrCreate(error).setHasError(false);
    StringValue reformed = new StringValue("reformed");
    tester.set(error, reformed);
    tester.getOrCreate(topKey).setBuilder(null).addDependency(error).setComputedValue(COPY);
    tester.invalidate();
    result = tester.eval(/*keepGoing=*/false, topKey);
    assertEquals(reformed, result.get(topKey));
    assertFalse(result.hasError());
  }

  /**
   * Similar to {@link ParallelEvaluatorTest#errorTwoLevelsDeep}, except here we request multiple
   * toplevel values.
   */
  @Test
  public void errorPropagationToTopLevelValues() throws Exception {
    SkyKey topKey = GraphTester.toSkyKey("top");
    SkyKey midKey = GraphTester.toSkyKey("mid");
    SkyKey badKey = GraphTester.toSkyKey("bad");
    tester.getOrCreate(topKey).addDependency(midKey).setComputedValue(CONCATENATE);
    tester.getOrCreate(midKey).addDependency(badKey).setComputedValue(CONCATENATE);
    tester.getOrCreate(badKey).setHasError(true);
    EvaluationResult<SkyValue> result = tester.eval(/*keepGoing=*/ false, topKey, midKey);
    assertThat(result.getError(midKey).getRootCauses()).containsExactly(badKey);
    // Do it again with keepGoing.  We should also see an error for the top key this time.
    result = tester.eval(/*keepGoing=*/ true, topKey, midKey);
    if (rootCausesStored()) {
      assertThat(result.getError(midKey).getRootCauses()).containsExactly(badKey);
      assertThat(result.getError(topKey).getRootCauses()).containsExactly(badKey);
    }
  }

  @Test
  public void breakWithInterruptibleErrorDep() throws Exception {
    SkyKey errorKey = GraphTester.toSkyKey("my_error_value");
    tester.getOrCreate(errorKey).setHasError(true);
    SkyKey parentKey = GraphTester.toSkyKey("parent");
    tester
        .getOrCreate(parentKey)
        .addErrorDependency(errorKey, new StringValue("recovered"))
        .setComputedValue(CONCATENATE);
    // When the error value throws, the propagation will cause an interrupted exception in parent.
    EvaluationResult<StringValue> result = tester.eval(/*keepGoing=*/ false, parentKey);
    assertThat(result.keyNames()).isEmpty();
    Map.Entry<SkyKey, ErrorInfo> error = Iterables.getOnlyElement(result.errorMap().entrySet());
    assertEquals(parentKey, error.getKey());
    assertThat(error.getValue().getRootCauses()).containsExactly(errorKey);
    assertFalse(Thread.interrupted());
    result = tester.eval(/*keepGoing=*/ true, parentKey);
    assertThat(result.errorMap()).isEmpty();
    assertEquals("recovered", result.get(parentKey).getValue());
  }

  /**
   * Regression test: "clearing incomplete values on --keep_going build is racy".
   * Tests that if a value is requested on the first (non-keep-going) build and its child throws
   * an error, when the second (keep-going) build runs, there is not a race that keeps it as a
   * reverse dep of its children.
   */
  @Test
  public void raceClearingIncompleteValues() throws Exception {
    SkyKey topKey = GraphTester.toSkyKey("top");
    final SkyKey midKey = GraphTester.toSkyKey("mid");
    SkyKey badKey = GraphTester.toSkyKey("bad");
    final AtomicBoolean waitForSecondCall = new AtomicBoolean(false);
    final CountDownLatch otherThreadWinning = new CountDownLatch(1);
    final AtomicReference<Thread> firstThread = new AtomicReference<>();
    setGraphForTesting(
        new NotifyingInMemoryGraph(
            new Listener() {
              @Override
              public void accept(SkyKey key, EventType type, Order order, Object context) {
                if (!waitForSecondCall.get()) {
                  return;
                }
                if (key.equals(midKey)) {
                  if (type == EventType.CREATE_IF_ABSENT) {
                    // The first thread to create midKey will not be the first thread to add a
                    // reverse dep to it.
                    firstThread.compareAndSet(null, Thread.currentThread());
                    return;
                  }
                  if (type == EventType.ADD_REVERSE_DEP) {
                    if (order == Order.BEFORE && Thread.currentThread().equals(firstThread.get())) {
                      // If this thread created midKey, block until the other thread adds a dep on
                      // it.
                      TrackingAwaiter.INSTANCE.awaitLatchAndTrackExceptions(
                          otherThreadWinning, "other thread didn't pass this one");
                    } else if (order == Order.AFTER
                        && !Thread.currentThread().equals(firstThread.get())) {
                      // This thread has added a dep. Allow the other thread to proceed.
                      otherThreadWinning.countDown();
                    }
                  }
                }
              }
            }));
    tester.getOrCreate(topKey).addDependency(midKey).setComputedValue(CONCATENATE);
    tester.getOrCreate(midKey).addDependency(badKey).setComputedValue(CONCATENATE);
    tester.getOrCreate(badKey).setHasError(true);
    EvaluationResult<SkyValue> result = tester.eval(/*keepGoing=*/ false, topKey, midKey);
    assertThat(result.getError(midKey).getRootCauses()).containsExactly(badKey);
    waitForSecondCall.set(true);
    result = tester.eval(/*keepGoing=*/ true, topKey, midKey);
    assertNotNull(firstThread.get());
    assertEquals(0, otherThreadWinning.getCount());
    assertThat(result.getError(midKey).getRootCauses()).containsExactly(badKey);
    assertThat(result.getError(topKey).getRootCauses()).containsExactly(badKey);
  }

  @Test
  public void breakWithErrorDep() throws Exception {
    SkyKey errorKey = GraphTester.toSkyKey("my_error_value");
    tester.getOrCreate(errorKey).setHasError(true);
    tester.set("after", new StringValue("after"));
    SkyKey parentKey = GraphTester.toSkyKey("parent");
    tester
        .getOrCreate(parentKey)
        .addErrorDependency(errorKey, new StringValue("recovered"))
        .setComputedValue(CONCATENATE)
        .addDependency("after");
    EvaluationResult<StringValue> result = tester.eval(/*keepGoing=*/ false, parentKey);
    assertThat(result.keyNames()).isEmpty();
    Map.Entry<SkyKey, ErrorInfo> error = Iterables.getOnlyElement(result.errorMap().entrySet());
    assertEquals(parentKey, error.getKey());
    assertThat(error.getValue().getRootCauses()).containsExactly(errorKey);
    result = tester.eval(/*keepGoing=*/ true, parentKey);
    assertThat(result.errorMap()).isEmpty();
    assertEquals("recoveredafter", result.get(parentKey).getValue());
  }

  @Test
  public void raceConditionWithNoKeepGoingErrors_InflightError() throws Exception {
    // Given a graph of two nodes, errorKey and otherErrorKey,
    final SkyKey errorKey = GraphTester.toSkyKey("errorKey");
    final SkyKey otherErrorKey = GraphTester.toSkyKey("otherErrorKey");

    final CountDownLatch errorCommitted = new CountDownLatch(1);

    final CountDownLatch otherStarted = new CountDownLatch(1);

    final CountDownLatch otherDone = new CountDownLatch(1);

    final AtomicInteger numOtherInvocations = new AtomicInteger(0);
    final AtomicReference<String> bogusInvocationMessage = new AtomicReference<>(null);
    final AtomicReference<String> nonNullValueMessage = new AtomicReference<>(null);

    tester
        .getOrCreate(errorKey)
        .setBuilder(
            new SkyFunction() {
              @Override
              public SkyValue compute(SkyKey skyKey, Environment env) throws SkyFunctionException {
                // Given that errorKey waits for otherErrorKey to begin evaluation before completing
                // its evaluation,
                TrackingAwaiter.INSTANCE.awaitLatchAndTrackExceptions(
                    otherStarted, "otherErrorKey's SkyFunction didn't start in time.");
                // And given that errorKey throws an error,
                throw new GenericFunctionException(
                    new SomeErrorException("error"), Transience.PERSISTENT);
              }

              @Override
              public String extractTag(SkyKey skyKey) {
                return null;
              }
            });
    tester
        .getOrCreate(otherErrorKey)
        .setBuilder(
            new SkyFunction() {
              @Override
              public SkyValue compute(SkyKey skyKey, Environment env) throws SkyFunctionException {
                otherStarted.countDown();
                int invocations = numOtherInvocations.incrementAndGet();
                // And given that otherErrorKey waits for errorKey's error to be committed before
                // trying to get errorKey's value,
                TrackingAwaiter.INSTANCE.awaitLatchAndTrackExceptions(
                    errorCommitted, "errorKey's error didn't get committed to the graph in time");
                try {
                  SkyValue value = env.getValueOrThrow(errorKey, SomeErrorException.class);
                  if (value != null) {
                    nonNullValueMessage.set("bogus non-null value " + value);
                  }
                  if (invocations != 1) {
                    bogusInvocationMessage.set("bogus invocation count: " + invocations);
                  }
                  otherDone.countDown();
                  // And given that otherErrorKey throws an error,
                  throw new GenericFunctionException(
                      new SomeErrorException("other"), Transience.PERSISTENT);
                } catch (SomeErrorException e) {
                  fail();
                  return null;
                }
              }

              @Override
              public String extractTag(SkyKey skyKey) {
                return null;
              }
            });
    setGraphForTesting(
        new NotifyingInMemoryGraph(
            new Listener() {
              @Override
              public void accept(SkyKey key, EventType type, Order order, Object context) {
                if (key.equals(errorKey) && type == EventType.SET_VALUE && order == Order.AFTER) {
                  errorCommitted.countDown();
                  TrackingAwaiter.INSTANCE.awaitLatchAndTrackExceptions(
                      otherDone, "otherErrorKey's SkyFunction didn't finish in time.");
                }
              }
            }));

    // When the graph is evaluated in noKeepGoing mode,
    EvaluationResult<StringValue> result =
        tester.eval(/*keepGoing=*/ false, errorKey, otherErrorKey);

    // Then the result reports that an error occurred because of errorKey,
    assertTrue(result.hasError());
    assertEquals(errorKey, result.getError().getRootCauseOfException());

    // And no value is committed for otherErrorKey,
    assertNull(tester.driver.getExistingErrorForTesting(otherErrorKey));
    assertNull(tester.driver.getExistingValueForTesting(otherErrorKey));

    // And no value was committed for errorKey,
    assertNull(nonNullValueMessage.get(), nonNullValueMessage.get());

    // And the SkyFunction for otherErrorKey was evaluated exactly once.
    assertEquals(numOtherInvocations.get(), 1);
    assertNull(bogusInvocationMessage.get(), bogusInvocationMessage.get());

    // NB: The SkyFunction for otherErrorKey gets evaluated exactly once--it does not get
    // re-evaluated during error bubbling. Why? When otherErrorKey throws, it is always the
    // second error encountered, because it waited for errorKey's error to be committed before
    // trying to get it. In fail-fast evaluations only the first failing SkyFunction's
    // newly-discovered-dependencies are registered. Therefore, there won't be a reverse-dep from
    // errorKey to otherErrorKey for the error to bubble through.
  }

  @Test
  public void absentParent() throws Exception {
    initializeTester();
    SkyKey errorKey = GraphTester.toSkyKey("my_error_value");
    tester.set(errorKey, new StringValue("biding time"));
    SkyKey absentParentKey = GraphTester.toSkyKey("absentParent");
    tester.getOrCreate(absentParentKey).addDependency(errorKey).setComputedValue(CONCATENATE);
    assertEquals(new StringValue("biding time"),
        tester.evalAndGet(/*keepGoing=*/false, absentParentKey));
    tester.getOrCreate(errorKey, /*markAsModified=*/true).setHasError(true);
    SkyKey newParent = GraphTester.toSkyKey("newParent");
    tester.getOrCreate(newParent).addDependency(errorKey).setComputedValue(CONCATENATE);
    tester.invalidate();
    EvaluationResult<StringValue> result = tester.eval(/*keepGoing=*/false, newParent);
    ErrorInfo error = result.getError(newParent);
    assertThat(error.getRootCauses()).containsExactly(errorKey);
  }

  @Test
  public void notComparableNotPrunedNoEvent() throws Exception {
    checkNotComparableNotPruned(false);
  }

  @Test
  public void notComparableNotPrunedEvent() throws Exception {
    checkNotComparableNotPruned(true);
  }

  private void checkNotComparableNotPruned(boolean hasEvent) throws Exception {
    initializeTester();
    SkyKey parent = GraphTester.toSkyKey("parent");
    SkyKey child = GraphTester.toSkyKey("child");
    NotComparableStringValue notComparableString = new NotComparableStringValue("not comparable");
    if (hasEvent) {
      tester.getOrCreate(child).setConstantValue(notComparableString).setWarning("shmoop");
    } else {
      tester.getOrCreate(child).setConstantValue(notComparableString);
    }
    final AtomicInteger parentEvaluated = new AtomicInteger();
    final String val = "some val";
    tester
        .getOrCreate(parent)
        .addDependency(child)
        .setComputedValue(new ValueComputer() {
          @Override
          public SkyValue compute(Map<SkyKey, SkyValue> deps, Environment env)
              throws InterruptedException {
            parentEvaluated.incrementAndGet();
            return new StringValue(val);
          }
        });
    assertStringValue(val, tester.evalAndGet( /*keepGoing=*/false, parent));
    assertThat(parentEvaluated.get()).isEqualTo(1);
    if (hasEvent) {
      assertContainsEvent(eventCollector, "shmoop");
    } else {
      assertEventCount(0, eventCollector);
    }

    tester.resetPlayedEvents();
    tester.getOrCreate(child, /*markAsModified=*/true);
    tester.invalidate();
    assertStringValue(val, tester.evalAndGet( /*keepGoing=*/false, parent));
    assertThat(parentEvaluated.get()).isEqualTo(2);
    if (hasEvent) {
      assertContainsEvent(eventCollector, "shmoop");
    } else {
      assertEventCount(0, eventCollector);
    }
  }

  private static void assertStringValue(String expected, SkyValue val) {
    assertThat(((StringValue) val).getValue()).isEqualTo(expected);
  }

  @Test
  public void changePruningWithEvent() throws Exception {
    initializeTester();
    SkyKey parent = GraphTester.toSkyKey("parent");
    SkyKey child = GraphTester.toSkyKey("child");
    tester.getOrCreate(child).setConstantValue(new StringValue("child")).setWarning("bloop");
    // Restart once because child isn't ready.
    CountDownLatch parentEvaluated = new CountDownLatch(3);
    StringValue parentVal = new StringValue("parent");
    tester
        .getOrCreate(parent)
        .setBuilder(
            new ChainedFunction(
                parentEvaluated, null, null, false, parentVal, ImmutableList.of(child)));
    assertThat(tester.evalAndGet( /*keepGoing=*/false, parent)).isEqualTo(parentVal);
    assertThat(parentEvaluated.getCount()).isEqualTo(1);
    assertContainsEvent(eventCollector, "bloop");
    tester.resetPlayedEvents();
    tester.getOrCreate(child, /*markAsModified=*/ true);
    tester.invalidate();
    assertThat(tester.evalAndGet( /*keepGoing=*/false, parent)).isEqualTo(parentVal);
    assertContainsEvent(eventCollector, "bloop");
    assertThat(parentEvaluated.getCount()).isEqualTo(1);
  }

  // Tests that we have a sane implementation of error transience.
  @Test
  public void errorTransienceBug() throws Exception {
    tester.getOrCreate("key").setHasTransientError(true);
    assertNotNull(tester.evalAndGetError("key").getException());
    StringValue value = new StringValue("hi");
    tester.getOrCreate("key").setHasTransientError(false).setConstantValue(value);
    tester.invalidateTransientErrors();
    assertEquals(value, tester.evalAndGet("key"));
    // This works because the version of the ValueEntry for the ErrorTransience value is always
    // increased on each InMemoryMemoizingEvaluator#evaluate call. But that's not the only way to
    // implement error transience; another valid implementation would be to unconditionally mark
    // values depending on the ErrorTransience value as being changed (rather than merely dirtied)
    // during invalidation.
  }

  @Test
  public void transientErrorTurningGoodHasNoError() throws Exception {
    initializeTester();
    SkyKey errorKey = GraphTester.toSkyKey("my_error_value");
    tester.getOrCreate(errorKey).setHasTransientError(true);
    ErrorInfo errorInfo = tester.evalAndGetError(errorKey);
    assertNotNull(errorInfo);
    assertThat(errorInfo.getRootCauses()).containsExactly(errorKey);
    // Re-evaluates to same thing when errors are invalidated
    tester.invalidateTransientErrors();
    errorInfo = tester.evalAndGetError(errorKey);
    assertNotNull(errorInfo);
    StringValue value = new StringValue("reformed");
    assertThat(errorInfo.getRootCauses()).containsExactly(errorKey);
    tester.getOrCreate(errorKey, /*markAsModified=*/false).setHasTransientError(false)
        .setConstantValue(value);
    tester.invalidateTransientErrors();
    StringValue stringValue = (StringValue) tester.evalAndGet(/*keepGoing=*/true, errorKey);
    assertSame(stringValue, value);
    // Value builder will now throw, but we should never get to it because it isn't dirty.
    tester.getOrCreate(errorKey, /*markAsModified=*/false).setHasTransientError(true);
    tester.invalidateTransientErrors();
    stringValue = (StringValue) tester.evalAndGet(/*keepGoing=*/true, errorKey);
    assertThat(stringValue).isEqualTo(value);
  }

  @Test
  public void deleteInvalidatedValue() throws Exception {
    initializeTester();
    SkyKey top = GraphTester.toSkyKey("top");
    SkyKey toDelete = GraphTester.toSkyKey("toDelete");
    // Must be a concatenation -- COPY doesn't actually copy.
    tester.getOrCreate(top).addDependency(toDelete).setComputedValue(CONCATENATE);
    tester.set(toDelete, new StringValue("toDelete"));
    SkyValue value = tester.evalAndGet("top");
    SkyKey forceInvalidation = GraphTester.toSkyKey("forceInvalidation");
    tester.set(forceInvalidation, new StringValue("forceInvalidation"));
    tester.getOrCreate(toDelete, /*markAsModified=*/true);
    tester.invalidate();
    tester.eval(/*keepGoing=*/false, forceInvalidation);
    tester.delete("toDelete");
    WeakReference<SkyValue> ref = new WeakReference<>(value);
    value = null;
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
    SkyKey[] leftValues = new SkyKey[TEST_NODE_COUNT];
    SkyKey[] rightValues = new SkyKey[TEST_NODE_COUNT];
    for (int i = 0; i < TEST_NODE_COUNT; i++) {
      leftValues[i] = GraphTester.toSkyKey("left-" + i);
      rightValues[i] = GraphTester.toSkyKey("right-" + i);
      if (i == 0) {
        tester.getOrCreate(leftValues[i])
              .addDependency("leaf")
              .setComputedValue(COPY);
        tester.getOrCreate(rightValues[i])
              .addDependency("leaf")
              .setComputedValue(COPY);
      } else {
        tester.getOrCreate(leftValues[i])
              .addDependency(leftValues[i - 1])
              .addDependency(rightValues[i - 1])
              .setComputedValue(new PassThroughSelected(leftValues[i - 1]));
        tester.getOrCreate(rightValues[i])
              .addDependency(leftValues[i - 1])
              .addDependency(rightValues[i - 1])
              .setComputedValue(new PassThroughSelected(rightValues[i - 1]));
      }
    }
    tester.set("leaf", new StringValue("leaf"));

    String lastLeft = "left-" + (TEST_NODE_COUNT - 1);
    String lastRight = "right-" + (TEST_NODE_COUNT - 1);

    for (int i = 0; i < TESTED_NODES; i++) {
      try {
        tester.getOrCreate(leftValues[i], /*markAsModified=*/true).setHasError(true);
        tester.invalidate();
        EvaluationResult<StringValue> result = tester.eval(
            /*keep_going=*/false, lastLeft, lastRight);
        assertTrue(result.hasError());
        tester.differencer.invalidate(ImmutableList.of(leftValues[i]));
        tester.invalidate();
        result = tester.eval(/*keep_going=*/false, lastLeft, lastRight);
        assertTrue(result.hasError());
        tester.getOrCreate(leftValues[i], /*markAsModified=*/true).setHasError(false);
        tester.invalidate();
        result = tester.eval(/*keep_going=*/false, lastLeft, lastRight);
        assertEquals(new StringValue("leaf"), result.get(toSkyKey(lastLeft)));
        assertEquals(new StringValue("leaf"), result.get(toSkyKey(lastRight)));
      } catch (Exception e) {
        System.err.println("twoRailLeftRightDependenciesWithFailure exception on run " + i);
        throw e;
      }
    }
  }

  @Test
  public void valueInjection() throws Exception {
    SkyKey key = GraphTester.toSkyKey("new_value");
    SkyValue val = new StringValue("val");

    tester.differencer.inject(ImmutableMap.of(key, val));
    assertEquals(val, tester.evalAndGet("new_value"));
  }

  @Test
  public void valueInjectionOverExistingEntry() throws Exception {
    SkyKey key = GraphTester.toSkyKey("value");
    SkyValue val = new StringValue("val");

    tester.getOrCreate(key).setConstantValue(new StringValue("old_val"));
    tester.differencer.inject(ImmutableMap.of(key, val));
    assertEquals(val, tester.evalAndGet("value"));
  }

  @Test
  public void valueInjectionOverExistingDirtyEntry() throws Exception {
    SkyKey key = GraphTester.toSkyKey("value");
    SkyValue val = new StringValue("val");

    tester.getOrCreate(key).setConstantValue(new StringValue("old_val"));
    tester.differencer.inject(ImmutableMap.of(key, val));
    tester.eval(/*keepGoing=*/false, new SkyKey[0]); // Create the value.

    tester.differencer.invalidate(ImmutableList.of(key));
    tester.eval(/*keepGoing=*/false, new SkyKey[0]); // Mark value as dirty.

    tester.differencer.inject(ImmutableMap.of(key, val));
    tester.eval(/*keepGoing=*/false, new SkyKey[0]); // Inject again.
    assertEquals(val, tester.evalAndGet("value"));
  }

  @Test
  public void valueInjectionOverExistingEntryMarkedForInvalidation() throws Exception {
    SkyKey key = GraphTester.toSkyKey("value");
    SkyValue val = new StringValue("val");

    tester.getOrCreate(key).setConstantValue(new StringValue("old_val"));
    tester.differencer.invalidate(ImmutableList.of(key));
    tester.differencer.inject(ImmutableMap.of(key, val));
    assertEquals(val, tester.evalAndGet("value"));
  }

  @Test
  public void valueInjectionOverExistingEntryMarkedForDeletion() throws Exception {
    SkyKey key = GraphTester.toSkyKey("value");
    SkyValue val = new StringValue("val");

    tester.getOrCreate(key).setConstantValue(new StringValue("old_val"));
    tester.evaluator.delete(Predicates.<SkyKey>alwaysTrue());
    tester.differencer.inject(ImmutableMap.of(key, val));
    assertEquals(val, tester.evalAndGet("value"));
  }

  @Test
  public void valueInjectionOverExistingEqualEntryMarkedForInvalidation() throws Exception {
    SkyKey key = GraphTester.toSkyKey("value");
    SkyValue val = new StringValue("val");

    tester.differencer.inject(ImmutableMap.of(key, val));
    assertEquals(val, tester.evalAndGet("value"));

    tester.differencer.invalidate(ImmutableList.of(key));
    tester.differencer.inject(ImmutableMap.of(key, val));
    assertEquals(val, tester.evalAndGet("value"));
  }

  @Test
  public void valueInjectionOverExistingEqualEntryMarkedForDeletion() throws Exception {
    SkyKey key = GraphTester.toSkyKey("value");
    SkyValue val = new StringValue("val");

    tester.differencer.inject(ImmutableMap.of(key, val));
    assertEquals(val, tester.evalAndGet("value"));

    tester.evaluator.delete(Predicates.<SkyKey>alwaysTrue());
    tester.differencer.inject(ImmutableMap.of(key, val));
    assertEquals(val, tester.evalAndGet("value"));
  }

  @Test
  public void valueInjectionOverValueWithDeps() throws Exception {
    SkyKey key = GraphTester.toSkyKey("value");
    SkyValue val = new StringValue("val");
    StringValue prevVal = new StringValue("foo");

    tester.getOrCreate("other").setConstantValue(prevVal);
    tester.getOrCreate(key).addDependency("other").setComputedValue(COPY);
    assertEquals(prevVal, tester.evalAndGet("value"));
    tester.differencer.inject(ImmutableMap.of(key, val));
    try {
      tester.evalAndGet("value");
      Assert.fail("injection over value with deps should have failed");
    } catch (IllegalStateException e) {
      assertThat(e).hasMessage(
          "existing entry for " + NODE_TYPE.getName() + ":value has deps: "
              + "[" + NODE_TYPE.getName() + ":other]");
    }
  }

  @Test
  public void valueInjectionOverEqualValueWithDeps() throws Exception {
    SkyKey key = GraphTester.toSkyKey("value");
    SkyValue val = new StringValue("val");

    tester.getOrCreate("other").setConstantValue(val);
    tester.getOrCreate(key).addDependency("other").setComputedValue(COPY);
    assertEquals(val, tester.evalAndGet("value"));
    tester.differencer.inject(ImmutableMap.of(key, val));
    try {
      tester.evalAndGet("value");
      Assert.fail("injection over value with deps should have failed");
    } catch (IllegalStateException e) {
      assertThat(e).hasMessage(
          "existing entry for " + NODE_TYPE.getName() + ":value has deps: "
              + "[" + NODE_TYPE.getName() + ":other]");
    }
  }

  @Test
  public void valueInjectionOverValueWithErrors() throws Exception {
    SkyKey key = GraphTester.toSkyKey("value");
    SkyValue val = new StringValue("val");

    tester.getOrCreate(key).setHasError(true);
    tester.evalAndGetError(key);

    tester.differencer.inject(ImmutableMap.of(key, val));
    assertEquals(val, tester.evalAndGet(false, key));
  }

  @Test
  public void valueInjectionInvalidatesReverseDeps() throws Exception {
    SkyKey childKey = GraphTester.toSkyKey("child");
    SkyKey parentKey = GraphTester.toSkyKey("parent");
    StringValue oldVal = new StringValue("old_val");

    tester.getOrCreate(childKey).setConstantValue(oldVal);
    tester.getOrCreate(parentKey).addDependency("child").setComputedValue(COPY);

    EvaluationResult<SkyValue> result = tester.eval(false, parentKey);
    assertFalse(result.hasError());
    assertEquals(oldVal, result.get(parentKey));

    SkyValue val = new StringValue("val");
    tester.differencer.inject(ImmutableMap.of(childKey, val));
    assertEquals(val, tester.evalAndGet("child"));
    // Injecting a new child should have invalidated the parent.
    Assert.assertNull(tester.getExistingValue("parent"));

    tester.eval(false, childKey);
    assertEquals(val, tester.getExistingValue("child"));
    Assert.assertNull(tester.getExistingValue("parent"));
    assertEquals(val, tester.evalAndGet("parent"));
  }

  @Test
  public void valueInjectionOverExistingEqualEntryDoesNotInvalidate() throws Exception {
    SkyKey childKey = GraphTester.toSkyKey("child");
    SkyKey parentKey = GraphTester.toSkyKey("parent");
    SkyValue val = new StringValue("same_val");

    tester.getOrCreate(parentKey).addDependency("child").setComputedValue(COPY);
    tester.getOrCreate(childKey).setConstantValue(new StringValue("same_val"));
    assertEquals(val, tester.evalAndGet("parent"));

    tester.differencer.inject(ImmutableMap.of(childKey, val));
    assertEquals(val, tester.getExistingValue("child"));
    // Since we are injecting an equal value, the parent should not have been invalidated.
    assertEquals(val, tester.getExistingValue("parent"));
  }

  @Test
  public void valueInjectionInterrupt() throws Exception {
    SkyKey key = GraphTester.toSkyKey("key");
    SkyValue val = new StringValue("val");

    tester.differencer.inject(ImmutableMap.of(key, val));
    Thread.currentThread().interrupt();
    try {
      tester.evalAndGet("key");
      fail();
    } catch (InterruptedException expected) {
      // Expected.
    }
    SkyValue newVal = tester.evalAndGet("key");
    assertEquals(val, newVal);
  }

  @Test
  public void persistentErrorsNotRerun() throws Exception {
    SkyKey topKey = GraphTester.toSkyKey("top");
    SkyKey transientErrorKey = GraphTester.toSkyKey("transientError");
    SkyKey persistentErrorKey1 = GraphTester.toSkyKey("persistentError1");
    SkyKey persistentErrorKey2 = GraphTester.toSkyKey("persistentError2");

    tester.getOrCreate(topKey)
          .addErrorDependency(transientErrorKey, new StringValue("doesn't matter"))
          .addErrorDependency(persistentErrorKey1, new StringValue("doesn't matter"))
          .setHasError(true);
    tester.getOrCreate(persistentErrorKey1).setHasError(true);
    tester.getOrCreate(transientErrorKey)
          .addErrorDependency(persistentErrorKey2, new StringValue("doesn't matter"))
          .setHasTransientError(true);
    tester.getOrCreate(persistentErrorKey2).setHasError(true);

    tester.evalAndGetError(topKey);
    assertThat(tester.getEnqueuedValues()).containsExactly(
        topKey, transientErrorKey, persistentErrorKey1, persistentErrorKey2);

    tester.invalidate();
    tester.invalidateTransientErrors();
    tester.evalAndGetError(topKey);
    // TODO(bazel-team): We can do better here once we implement change pruning for errors.
    assertThat(tester.getEnqueuedValues()).containsExactly(topKey, transientErrorKey);
  }

  /**
   * The following two tests check that the evaluator shuts down properly when encountering an error
   * that is marked dirty but later verified to be unchanged from a prior build. In that case, the
   * invariant that its parents are not enqueued for evaluation should be maintained.
   */
  /**
   * Test that a parent of a cached but invalidated error doesn't successfully build. First build
   * the error. Then invalidate the error via a dependency (so it will not actually change) and
   * build two new parents. Parent A will request error and abort since error isn't done yet. error
   * is then revalidated, and A is restarted. If A does not throw upon encountering the error, and
   * instead sets its value, then we throw in parent B, which waits for error to be done before
   * requesting it. Then there will be the impossible situation of a node that was built during this
   * evaluation depending on a node in error.
   */
  @Test
  public void shutDownBuildOnCachedError_Done() throws Exception {
    // errorKey will be invalidated due to its dependence on invalidatedKey, but later revalidated
    // since invalidatedKey re-evaluates to the same value on a subsequent build.
    final SkyKey errorKey = GraphTester.toSkyKey("error");
    SkyKey invalidatedKey = GraphTester.toSkyKey("invalidated-leaf");
    tester.set(invalidatedKey, new StringValue("invalidated-leaf-value"));
    tester.getOrCreate(errorKey).addDependency(invalidatedKey).setHasError(true);
    // Names are alphabetized in reverse deps of errorKey.
    final SkyKey fastToRequestSlowToSetValueKey = GraphTester.toSkyKey("A-slow-set-value-parent");
    final SkyKey failingKey = GraphTester.toSkyKey("B-fast-fail-parent");
    tester.getOrCreate(fastToRequestSlowToSetValueKey).addDependency(errorKey)
        .setComputedValue(CONCATENATE);
    tester.getOrCreate(failingKey).addDependency(errorKey).setComputedValue(CONCATENATE);
    // We only want to force a particular order of operations at some points during evaluation.
    final AtomicBoolean synchronizeThreads = new AtomicBoolean(false);
    // We don't expect slow-set-value to actually be built, but if it is, we wait for it.
    final CountDownLatch slowBuilt = new CountDownLatch(1);
    setGraphForTesting(
        new DeterministicInMemoryGraph(
            new Listener() {
              @Override
              public void accept(SkyKey key, EventType type, Order order, Object context) {
                if (!synchronizeThreads.get()) {
                  return;
                }
                if (type == EventType.IS_DIRTY && key.equals(failingKey)) {
                  // Wait for the build to abort or for the other node to incorrectly build.
                  try {
                    assertTrue(slowBuilt.await(TestUtils.WAIT_TIMEOUT_SECONDS, TimeUnit.SECONDS));
                  } catch (InterruptedException e) {
                    // This is ok, because it indicates the build is shutting down.
                    Thread.currentThread().interrupt();
                  }
                } else if (type == EventType.SET_VALUE
                    && key.equals(fastToRequestSlowToSetValueKey)
                    && order == Order.AFTER) {
                  // This indicates a problem -- this parent shouldn't be built since it depends on
                  // an error.
                  slowBuilt.countDown();
                  // Before this node actually sets its value (and then throws an exception) we wait
                  // for the other node to throw an exception.
                  try {
                    Thread.sleep(TestUtils.WAIT_TIMEOUT_MILLISECONDS);
                    throw new IllegalStateException("uninterrupted in " + key);
                  } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                  }
                }
              }
            }));
    // Initialize graph.
    tester.eval(/*keepGoing=*/true, errorKey);
    tester.getOrCreate(invalidatedKey, /*markAsModified=*/true);
    tester.invalidate();
    synchronizeThreads.set(true);
    tester.eval(/*keepGoing=*/false, fastToRequestSlowToSetValueKey, failingKey);
  }

  /**
   * Test that the invalidated parent of a cached but invalidated error doesn't get marked clean.
   * First build the parent -- it will contain an error. Then invalidate the error via a dependency
   * (so it will not actually change) and then build the parent and another node that depends on the
   * error. The other node will wait to throw until the parent is signaled that all of its
   * dependencies are done, or until it is interrupted. If it throws, the parent will be
   * VERIFIED_CLEAN but not done, which is not a valid state once evaluation shuts down. The
   * evaluator avoids this situation by throwing when the error is encountered, even though the
   * error isn't evaluated or requested by an evaluating node.
   */
  @Test
  public void shutDownBuildOnCachedError_Verified() throws Exception {
    // errorKey will be invalidated due to its dependence on invalidatedKey, but later revalidated
    // since invalidatedKey re-evaluates to the same value on a subsequent build.
    SkyKey errorKey = GraphTester.toSkyKey("error");
    SkyKey invalidatedKey = GraphTester.toSkyKey("invalidated-leaf");
    SkyKey changedKey = GraphTester.toSkyKey("changed-leaf");
    tester.set(invalidatedKey, new StringValue("invalidated-leaf-value"));
    tester.set(changedKey, new StringValue("changed-leaf-value"));
    // Names are alphabetized in reverse deps of errorKey.
    final SkyKey cachedParentKey = GraphTester.toSkyKey("A-cached-parent");
    final SkyKey uncachedParentKey = GraphTester.toSkyKey("B-uncached-parent");
    tester.getOrCreate(errorKey).addDependency(invalidatedKey).setHasError(true);
    tester.getOrCreate(cachedParentKey).addDependency(errorKey)
        .setComputedValue(CONCATENATE);
    tester.getOrCreate(uncachedParentKey).addDependency(changedKey).addDependency(errorKey)
        .setComputedValue(CONCATENATE);
    // We only want to force a particular order of operations at some points during evaluation. In
    // particular, we don't want to force anything during error bubbling.
    final AtomicBoolean synchronizeThreads = new AtomicBoolean(false);
    final CountDownLatch shutdownAwaiterStarted = new CountDownLatch(1);
    setGraphForTesting(
        new DeterministicInMemoryGraph(
            new Listener() {
              private final CountDownLatch cachedSignaled = new CountDownLatch(1);

              @Override
              public void accept(SkyKey key, EventType type, Order order, Object context) {
                if (!synchronizeThreads.get()
                    || order != Order.BEFORE
                    || type != EventType.SIGNAL) {
                  return;
                }
                TrackingAwaiter.INSTANCE.awaitLatchAndTrackExceptions(
                    shutdownAwaiterStarted, "shutdown awaiter not started");
                if (key.equals(uncachedParentKey)) {
                  // When the uncached parent is first signaled by its changed dep, make sure that
                  // we wait until the cached parent is signaled too.
                  try {
                    assertTrue(
                        cachedSignaled.await(TestUtils.WAIT_TIMEOUT_SECONDS, TimeUnit.SECONDS));
                  } catch (InterruptedException e) {
                    // Before the relevant bug was fixed, this code was not interrupted, and the
                    // uncached parent got to build, yielding an inconsistent state at a later point
                    // during evaluation. With the bugfix, the cached parent is never signaled
                    // before the evaluator shuts down, and so the above code is interrupted.
                    Thread.currentThread().interrupt();
                  }
                } else if (key.equals(cachedParentKey)) {
                  // This branch should never be reached by a well-behaved evaluator, since when the
                  // error node is reached, the evaluator should shut down. However, we don't test
                  // for that behavior here because that would be brittle and we expect that such an
                  // evaluator will crash hard later on in any case.
                  cachedSignaled.countDown();
                  try {
                    // Sleep until we're interrupted by the evaluator, so we know it's shutting
                    // down.
                    Thread.sleep(TestUtils.WAIT_TIMEOUT_MILLISECONDS);
                    Thread currentThread = Thread.currentThread();
                    throw new IllegalStateException(
                        "no interruption in time in "
                            + key
                            + " for "
                            + (currentThread.isInterrupted() ? "" : "un")
                            + "interrupted "
                            + currentThread
                            + " with hash "
                            + System.identityHashCode(currentThread)
                            + " at "
                            + System.currentTimeMillis());
                  } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                  }
                }
              }
            }));
    // Initialize graph.
    tester.eval(/*keepGoing=*/true, cachedParentKey, uncachedParentKey);
    tester.getOrCreate(invalidatedKey, /*markAsModified=*/true);
    tester.set(changedKey, new StringValue("new value"));
    tester.invalidate();
    synchronizeThreads.set(true);
    SkyKey waitForShutdownKey = GraphTester.skyKey("wait-for-shutdown");
    tester
        .getOrCreate(waitForShutdownKey)
        .setBuilder(
            new SkyFunction() {
              @Override
              public SkyValue compute(SkyKey skyKey, Environment env) throws InterruptedException {
                shutdownAwaiterStarted.countDown();
                TrackingAwaiter.INSTANCE.awaitLatchAndTrackExceptions(
                    ((ParallelEvaluator.SkyFunctionEnvironment) env).getExceptionLatchForTesting(),
                    "exception not thrown");
                // Threadpool is shutting down. Don't try to synchronize anything in the future
                // during error bubbling.
                synchronizeThreads.set(false);
                throw new InterruptedException();
              }

              @Nullable
              @Override
              public String extractTag(SkyKey skyKey) {
                return null;
              }
            });
    EvaluationResult<StringValue> result =
        tester.eval(/*keepGoing=*/false, cachedParentKey, uncachedParentKey, waitForShutdownKey);
    assertWithMessage(result.toString()).that(result.hasError()).isTrue();
    tester.getOrCreate(invalidatedKey, /*markAsModified=*/true);
    tester.invalidate();
    result = tester.eval(/*keepGoing=*/false, cachedParentKey, uncachedParentKey);
    assertWithMessage(result.toString()).that(result.hasError()).isTrue();
  }

  @Test
  public void cachedChildErrorDepWithSiblingDepOnNoKeepGoingEval() throws Exception {
    SkyKey parent1Key = GraphTester.toSkyKey("parent1");
    SkyKey parent2Key = GraphTester.toSkyKey("parent2");
    final SkyKey errorKey = GraphTester.toSkyKey("error");
    final SkyKey otherKey = GraphTester.toSkyKey("other");
    SkyFunction parentBuilder = new SkyFunction() {
      @Override
      public SkyValue compute(SkyKey skyKey, Environment env) {
        env.getValue(errorKey);
        env.getValue(otherKey);
        if (env.valuesMissing()) {
          return null;
        }
        return new StringValue("parent");
      }

      @Override
      public String extractTag(SkyKey skyKey) {
        return null;
      }
    };
    tester.getOrCreate(parent1Key).setBuilder(parentBuilder);
    tester.getOrCreate(parent2Key).setBuilder(parentBuilder);
    tester.getOrCreate(errorKey).setConstantValue(new StringValue("no error yet"));
    tester.getOrCreate(otherKey).setConstantValue(new StringValue("other"));
    tester.eval(/*keepGoing=*/true, parent1Key);
    tester.eval(/*keepGoing=*/false, parent2Key);
    tester.getOrCreate(errorKey, /*markAsModified=*/true).setHasError(true);
    tester.invalidate();
    tester.eval(/*keepGoing=*/true, parent1Key);
    tester.eval(/*keepGoing=*/false, parent2Key);
  }

  private void setGraphForTesting(NotifyingInMemoryGraph notifyingInMemoryGraph) {
    graph = notifyingInMemoryGraph;
    InMemoryMemoizingEvaluator memoizingEvaluator = (InMemoryMemoizingEvaluator) tester.evaluator;
    memoizingEvaluator.setGraphForTesting(notifyingInMemoryGraph);
  }

  private static final class PassThroughSelected implements ValueComputer {
    private final SkyKey key;

    public PassThroughSelected(SkyKey key) {
      this.key = key;
    }

    @Override
    public SkyValue compute(Map<SkyKey, SkyValue> deps, SkyFunction.Environment env) {
      return Preconditions.checkNotNull(deps.get(key));
    }
  }

  private void removedNodeComesBack() throws Exception {
    SkyKey top = GraphTester.skyKey("top");
    SkyKey mid = GraphTester.skyKey("mid");
    SkyKey leaf = GraphTester.skyKey("leaf");
    // When top depends on mid, which depends on leaf,
    tester.getOrCreate(top).addDependency(mid).setComputedValue(CONCATENATE);
    tester.getOrCreate(mid).addDependency(leaf).setComputedValue(CONCATENATE);
    StringValue leafValue = new StringValue("leaf");
    tester.set(leaf, leafValue);
    // Then when top is evaluated, its value is as expected.
    assertThat(tester.evalAndGet(/*keepGoing=*/ true, top)).isEqualTo(leafValue);
    // When top is changed to no longer depend on mid,
    StringValue topValue = new StringValue("top");
    tester
        .getOrCreate(top, /*markAsModified=*/ true)
        .removeDependency(mid)
        .setComputedValue(null)
        .setConstantValue(topValue);
    // And leaf is invalidated,
    tester.getOrCreate(leaf, /*markAsModified=*/ true);
    // Then when top is evaluated, its value is as expected,
    tester.invalidate();
    assertThat(tester.evalAndGet(/*keepGoing=*/ true, top)).isEqualTo(topValue);
    // And there is no value for mid in the graph,
    assertThat(tester.driver.getExistingValueForTesting(mid)).isNull();
    assertThat(tester.driver.getExistingErrorForTesting(mid)).isNull();
    // Or for leaf.
    assertThat(tester.driver.getExistingValueForTesting(leaf)).isNull();
    assertThat(tester.driver.getExistingErrorForTesting(leaf)).isNull();

    // When top is changed to depend directly on leaf,
    tester
        .getOrCreate(top, /*markAsModified=*/ true)
        .addDependency(leaf)
        .setConstantValue(null)
        .setComputedValue(CONCATENATE);
    // Then when top is evaluated, its value is as expected,
    tester.invalidate();
    assertThat(tester.evalAndGet(/*keepGoing=*/ true, top)).isEqualTo(leafValue);
    // and there is no value for mid in the graph,
    assertThat(tester.driver.getExistingValueForTesting(mid)).isNull();
    assertThat(tester.driver.getExistingErrorForTesting(mid)).isNull();
  }

  // Tests that a removed and then reinstated node doesn't try to invalidate its erstwhile parent
  // when it is invalidated.
  @Test
  public void removedNodeComesBackAndInvalidates() throws Exception {
    removedNodeComesBack();
    // When leaf is invalidated again,
    tester.getOrCreate(GraphTester.skyKey("leaf"), /*markAsModified=*/ true);
    // Then when top is evaluated, its value is as expected.
    tester.invalidate();
    assertThat(tester.evalAndGet(/*keepGoing=*/ true, "top")).isEqualTo(new StringValue("leaf"));
  }

  // Tests that a removed and then reinstated node behaves properly when its parent disappears and
  // then reappears.
  @Test
  public void removedNodeComesBackAndOtherInvalidates() throws Exception {
    removedNodeComesBack();
    SkyKey top = GraphTester.skyKey("top");
    SkyKey mid = GraphTester.skyKey("mid");
    SkyKey leaf = GraphTester.skyKey("leaf");
    // When top is invalidated again,
    tester.getOrCreate(top, /*markAsModified=*/ true).removeDependency(leaf).addDependency(mid);
    // Then when top is evaluated, its value is as expected.
    tester.invalidate();
    assertThat(tester.evalAndGet(/*keepGoing=*/ true, top)).isEqualTo(new StringValue("leaf"));
  }

  // Tests that a removed and then reinstated node doesn't have a reverse dep on a former parent.
  @Test
  public void removedInvalidatedNodeComesBackAndOtherInvalidates() throws Exception {
    SkyKey top = GraphTester.skyKey("top");
    SkyKey leaf = GraphTester.skyKey("leaf");
    // When top depends on leaf,
    tester.getOrCreate(top).addDependency(leaf).setComputedValue(CONCATENATE);
    StringValue leafValue = new StringValue("leaf");
    tester.set(leaf, leafValue);
    // Then when top is evaluated, its value is as expected.
    assertThat(tester.evalAndGet(/*keepGoing=*/ true, top)).isEqualTo(leafValue);
    // When top is changed to no longer depend on leaf,
    StringValue topValue = new StringValue("top");
    tester
        .getOrCreate(top, /*markAsModified=*/ true)
        .removeDependency(leaf)
        .setComputedValue(null)
        .setConstantValue(topValue);
    // And leaf is invalidated,
    tester.getOrCreate(leaf, /*markAsModified=*/ true);
    // Then when top is evaluated, its value is as expected,
    tester.invalidate();
    assertThat(tester.evalAndGet(/*keepGoing=*/ true, top)).isEqualTo(topValue);
    // And there is no value for leaf in the graph.
    assertThat(tester.driver.getExistingValueForTesting(leaf)).isNull();
    assertThat(tester.driver.getExistingErrorForTesting(leaf)).isNull();

    // When leaf is evaluated, so that it is present in the graph again,
    assertThat(tester.evalAndGet(/*keepGoing=*/ true, leaf)).isEqualTo(leafValue);
    // And top is changed to depend on leaf again,
    tester
        .getOrCreate(top, /*markAsModified=*/ true)
        .addDependency(leaf)
        .setConstantValue(null)
        .setComputedValue(CONCATENATE);
    // Then when top is evaluated, its value is as expected.
    tester.invalidate();
    assertThat(tester.evalAndGet(/*keepGoing=*/ true, top)).isEqualTo(leafValue);
  }


  @Test
  public void cleanReverseDepFromDirtyNodeNotInBuild() throws Exception {
    final SkyKey topKey = GraphTester.skyKey("top");
    SkyKey inactiveKey = GraphTester.skyKey("inactive");
    final Thread mainThread = Thread.currentThread();
    final AtomicBoolean shouldInterrupt = new AtomicBoolean(false);
    NotifyingInMemoryGraph graph =
        new NotifyingInMemoryGraph(
            new Listener() {
              @Override
              public void accept(SkyKey key, EventType type, Order order, Object context) {
                if (shouldInterrupt.get()
                    && key.equals(topKey)
                    && type == EventType.IS_READY
                    && order == Order.BEFORE) {
                  mainThread.interrupt();
                  shouldInterrupt.set(false);
                  try {
                    // Make sure threadpool propagates interrupt.
                    Thread.sleep(TestUtils.WAIT_TIMEOUT_MILLISECONDS);
                  } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                  }
                }
              }
            });
    setGraphForTesting(graph);
    // When top depends on inactive,
    tester.getOrCreate(topKey).addDependency(inactiveKey).setComputedValue(COPY);
    StringValue val = new StringValue("inactive");
    // And inactive is constant,
    tester.set(inactiveKey, val);
    // Then top evaluates normally.
    assertThat(tester.evalAndGet(/*keepGoing=*/ true, topKey)).isEqualTo(val);
    // When evaluation will be interrupted as soon as top starts evaluating,
    shouldInterrupt.set(true);
    // And inactive is dirty,
    tester.getOrCreate(inactiveKey, /*markAsModified=*/ true);
    // And so is top,
    tester.getOrCreate(topKey, /*markAsModified=*/ true);
    tester.invalidate();
    try {
      // Then evaluation is interrupted,
      tester.eval(/*keepGoing=*/ false, topKey);
      fail();
    } catch (InterruptedException e) {
      // Expected.
    }
    // But inactive is still present,
    assertThat(graph.get(inactiveKey)).isNotNull();
    // And still dirty,
    assertThat(graph.get(inactiveKey).isDirty()).isTrue();
    // And re-evaluates successfully,
    assertThat(tester.evalAndGet(/*keepGoing=*/ true, inactiveKey)).isEqualTo(val);
    // But top is gone from the graph,
    assertThat(graph.get(topKey)).isNull();
    // And we can successfully invalidate and re-evaluate inactive again.
    tester.getOrCreate(inactiveKey, /*markAsModified=*/ true);
    tester.invalidate();
    assertThat(tester.evalAndGet(/*keepGoing=*/ true, inactiveKey)).isEqualTo(val);
  }

  /** A graph tester that is specific to the memoizing evaluator, with some convenience methods. */
  protected class MemoizingEvaluatorTester extends GraphTester {
    private RecordingDifferencer differencer;
    private MemoizingEvaluator evaluator;
    private BuildDriver driver;
    private TrackingInvalidationReceiver invalidationReceiver = new TrackingInvalidationReceiver();

    public void initialize() {
      this.differencer = new RecordingDifferencer();
      this.evaluator =
          getMemoizingEvaluator(getSkyFunctionMap(), differencer, invalidationReceiver);
      this.driver = getBuildDriver(evaluator);
    }

    public void setInvalidationReceiver(TrackingInvalidationReceiver customInvalidationReceiver) {
      Preconditions.checkState(evaluator == null, "evaluator already initialized");
      invalidationReceiver = customInvalidationReceiver;
    }

    public void invalidate() {
      differencer.invalidate(getModifiedValues());
      clearModifiedValues();
      invalidationReceiver.clear();
    }

    public void invalidateTransientErrors() {
      differencer.invalidateTransientErrors();
    }

    public void delete(String key) {
      evaluator.delete(Predicates.equalTo(GraphTester.skyKey(key)));
    }

    public void resetPlayedEvents() {
      emittedEventState.clear();
    }

    public Set<SkyKey> getDirtyKeys() {
      return invalidationReceiver.dirty;
    }

    public Set<SkyKey> getDeletedKeys() {
      return invalidationReceiver.deleted;
    }

    public Set<SkyKey> getEnqueuedValues() {
      return invalidationReceiver.enqueued;
    }

    public <T extends SkyValue> EvaluationResult<T> eval(
        boolean keepGoing, int numThreads, SkyKey... keys) throws InterruptedException {
      assertThat(getModifiedValues()).isEmpty();
      return driver.evaluate(ImmutableList.copyOf(keys), keepGoing, numThreads, reporter);
    }

    public <T extends SkyValue> EvaluationResult<T> eval(boolean keepGoing, SkyKey... keys)
        throws InterruptedException {
      return eval(keepGoing, 100, keys);
    }

    public <T extends SkyValue> EvaluationResult<T> eval(boolean keepGoing, String... keys)
        throws InterruptedException {
      return eval(keepGoing, toSkyKeys(keys));
    }

    public SkyValue evalAndGet(boolean keepGoing, String key)
        throws InterruptedException {
      return evalAndGet(keepGoing, toSkyKey(key));
    }

    public SkyValue evalAndGet(String key) throws InterruptedException {
      return evalAndGet(/*keepGoing=*/false, key);
    }

    public SkyValue evalAndGet(boolean keepGoing, SkyKey key)
        throws InterruptedException {
      EvaluationResult<StringValue> evaluationResult = eval(keepGoing, key);
      SkyValue result = evaluationResult.get(key);
      assertNotNull(evaluationResult.toString(), result);
      return result;
    }

    public ErrorInfo evalAndGetError(SkyKey key) throws InterruptedException {
      EvaluationResult<StringValue> evaluationResult = eval(/*keepGoing=*/true, key);
      ErrorInfo result = evaluationResult.getError(key);
      assertNotNull(evaluationResult.toString(), result);
      return result;
    }

    public ErrorInfo evalAndGetError(String key) throws InterruptedException {
      return evalAndGetError(toSkyKey(key));
    }

    @Nullable
    public SkyValue getExistingValue(SkyKey key) {
      return driver.getExistingValueForTesting(key);
    }

    @Nullable
    public SkyValue getExistingValue(String key) {
      return getExistingValue(toSkyKey(key));
    }
  }
}
