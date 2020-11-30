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
import static com.google.devtools.build.lib.testutil.EventIterableSubjectFactory.assertThatEvents;
import static com.google.devtools.build.skyframe.ErrorInfoSubjectFactory.assertThatErrorInfo;
import static com.google.devtools.build.skyframe.EvaluationResultSubjectFactory.assertThatEvaluationResult;
import static com.google.devtools.build.skyframe.GraphTester.CONCATENATE;
import static com.google.devtools.build.skyframe.GraphTester.COPY;
import static com.google.devtools.build.skyframe.GraphTester.NODE_TYPE;
import static com.google.devtools.build.skyframe.GraphTester.skyKey;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.fail;

import com.google.auto.value.AutoValue;
import com.google.common.base.Preconditions;
import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Sets;
import com.google.common.eventbus.EventBus;
import com.google.common.testing.GcFinalization;
import com.google.common.truth.IterableSubject;
import com.google.common.util.concurrent.SettableFuture;
import com.google.common.util.concurrent.Uninterruptibles;
import com.google.devtools.build.lib.events.DelegatingEventHandler;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventCollector;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.testutil.TestThread;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.skyframe.GraphTester.NotComparableStringValue;
import com.google.devtools.build.skyframe.GraphTester.StringValue;
import com.google.devtools.build.skyframe.GraphTester.TestFunction;
import com.google.devtools.build.skyframe.GraphTester.ValueComputer;
import com.google.devtools.build.skyframe.NotifyingHelper.EventType;
import com.google.devtools.build.skyframe.NotifyingHelper.Listener;
import com.google.devtools.build.skyframe.NotifyingHelper.Order;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.ThinNodeEntry.DirtyType;
import com.google.devtools.build.skyframe.proto.GraphInconsistency.Inconsistency;
import java.lang.ref.WeakReference;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import javax.annotation.Nullable;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link MemoizingEvaluator}.*/
@RunWith(JUnit4.class)
public class MemoizingEvaluatorTest {

  protected MemoizingEvaluatorTester tester;
  protected EventCollector eventCollector;
  private ExtendedEventHandler reporter;
  protected MemoizingEvaluator.EmittedEventState emittedEventState;

  // Knobs that control the size / duration of larger tests.
  private static final int TEST_NODE_COUNT = 100;
  private static final int TESTED_NODES = 10;
  private static final int RUNS = 10;

  @Before
  public void initializeTester() {
    initializeTester(null);
    initializeReporter();
  }

  private void initializeTester(@Nullable TrackingProgressReceiver customProgressReceiver) {
    emittedEventState = new MemoizingEvaluator.EmittedEventState();
    tester = new MemoizingEvaluatorTester();
    if (customProgressReceiver != null) {
      tester.setProgressReceiver(customProgressReceiver);
    }
    tester.initialize(true);
  }

  protected TrackingProgressReceiver createTrackingProgressReceiver(
      boolean checkEvaluationResults) {
    return new TrackingProgressReceiver(checkEvaluationResults);
  }

  @After
  public void assertNoTrackedErrors() {
    TrackingAwaiter.INSTANCE.assertNoErrors();
  }

  protected RecordingDifferencer getRecordingDifferencer() {
    return new SequencedRecordingDifferencer();
  }

  protected MemoizingEvaluator getMemoizingEvaluator(
      Map<SkyFunctionName, SkyFunction> functions,
      Differencer differencer,
      EvaluationProgressReceiver progressReceiver,
      GraphInconsistencyReceiver graphInconsistencyReceiver,
      EventFilter eventFilter,
      boolean keepEdges) {
    return new InMemoryMemoizingEvaluator(
        functions,
        differencer,
        progressReceiver,
        graphInconsistencyReceiver,
        eventFilter,
        emittedEventState,
        true);
  }

  protected BuildDriver getBuildDriver(MemoizingEvaluator evaluator) {
    return new SequentialBuildDriver(evaluator);
  }

  protected boolean eventsStored() {
    return true;
  }

  protected boolean cyclesDetected() {
    return true;
  }

  protected boolean preciseEvaluationStatusStored() {
    return true;
  }

  protected void initializeReporter() {
    eventCollector = new EventCollector();
    reporter = new Reporter(new EventBus(), eventCollector);
    tester.resetPlayedEvents();
  }

  private static SkyKey toSkyKey(String name) {
    return GraphTester.toSkyKey(name);
  }

  /**
   * Equips {@link #tester} with a {@link GraphInconsistencyReceiver} that tolerates and tracks
   * inconsistencies.
   *
   * <p>Returns a concurrent {@link Set} containing {@link InconsistencyData}s discovered during
   * evaluation. Callers should assert the desired properties on the returned set.
   *
   * <p>Calls {@code tester.initialize} under the hood, so call early in test setup!
   */
  protected Set<InconsistencyData> setupGraphInconsistencyReceiver(boolean allowDuplicates) {
    Set<InconsistencyData> inconsistencies = Sets.newConcurrentHashSet();
    tester.setGraphInconsistencyReceiver(
        restartEnabledInconsistencyReceiver(
            (key, otherKeys, inconsistency) -> {
              if (otherKeys == null) {
                Preconditions.checkState(
                    inconsistencies.add(
                            InconsistencyData.create(key, /*otherKey=*/ null, inconsistency))
                        || allowDuplicates,
                    "Duplicate inconsistency: (%s, %s, %s)\nexisting = %s",
                    key,
                    null,
                    inconsistency,
                    inconsistencies);
              } else {
                for (SkyKey otherKey : otherKeys) {
                  Preconditions.checkState(
                      inconsistencies.add(InconsistencyData.create(key, otherKey, inconsistency))
                          || allowDuplicates,
                      "Duplicate inconsistency: (%s, %s, %s)\nexisting = %s",
                      key,
                      otherKey,
                      inconsistency,
                      inconsistencies);
                }
              }
            }));
    // #initialize must be called after setting the GraphInconsistencyReceiver for the receiver to
    // be registered with the test's memoizing evaluator.
    tester.initialize(/*keepEdges=*/ true);
    return inconsistencies;
  }

  /** Calls {@code tester.initialize} under the hood, so call early in test setup! */
  protected Set<InconsistencyData> setupGraphInconsistencyReceiver() {
    return setupGraphInconsistencyReceiver(/*allowDuplicates=*/ false);
  }

  @Test
  public void smoke() throws Exception {
    tester.set("x", new StringValue("y"));
    StringValue value = (StringValue) tester.evalAndGet("x");
    assertThat(value.getValue()).isEqualTo("y");
  }

  @Test
  public void evaluateEmptySet() throws InterruptedException {
    tester.eval(false, new SkyKey[0]);
    tester.eval(true, new SkyKey[0]);
  }

  @Test
  public void invalidationWithNothingChanged() throws Exception {
    tester.set("x", new StringValue("y")).setWarning("fizzlepop");
    StringValue value = (StringValue) tester.evalAndGet("x");
    assertThat(value.getValue()).isEqualTo("y");
    assertThatEvents(eventCollector).containsExactly("fizzlepop");

    initializeReporter();
    tester.invalidate();
    value = (StringValue) tester.evalAndGet("x");
    assertThat(value.getValue()).isEqualTo("y");
    if (eventsStored()) {
      assertThatEvents(eventCollector).containsExactly("fizzlepop");
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
    assertThatEvaluationResult(result).hasSingletonErrorThat(toSkyKey("top"));
  }

  @Test
  public void testEnvProvidesTemporaryDirectDeps() throws Exception {
    AtomicInteger counter = new AtomicInteger();
    List<SkyKey> deps = Collections.synchronizedList(new ArrayList<>());
    SkyKey topKey = toSkyKey("top");
    SkyKey bottomKey = toSkyKey("bottom");
    SkyValue bottomValue = new StringValue("bottom");
    tester
        .getOrCreate(topKey)
        .setBuilder(
            new NoExtractorFunction() {
              @Override
              public SkyValue compute(SkyKey skyKey, Environment env) throws InterruptedException {
                if (counter.getAndIncrement() > 0) {
                  deps.addAll(env.getTemporaryDirectDeps().get(0));
                } else {
                  assertThat(env.getTemporaryDirectDeps().listSize()).isEqualTo(0);
                }
                return env.getValue(bottomKey);
              }
            });
    tester.getOrCreate(bottomKey).setConstantValue(bottomValue);
    EvaluationResult<StringValue> result = tester.eval(/*keepGoing=*/ true, "top");
    assertThat(result.get(topKey)).isEqualTo(bottomValue);
    assertThat(deps).containsExactly(bottomKey);
  }

  @Test
  public void cachedErrorShutsDownThreadpool() throws Exception {
    // When a node throws an error on the first build,
    SkyKey cachedErrorKey = GraphTester.skyKey("error");
    tester.getOrCreate(cachedErrorKey).setHasError(true);
    assertThat(tester.evalAndGetError(/*keepGoing=*/ true, cachedErrorKey)).isNotNull();
    // And on the second build, it is requested as a dep,
    SkyKey topKey = GraphTester.skyKey("top");
    tester.getOrCreate(topKey).addDependency(cachedErrorKey).setComputedValue(CONCATENATE);
    // And another node throws an error, but waits to throw until the child error is thrown,
    SkyKey newErrorKey = GraphTester.skyKey("newError");
    tester
        .getOrCreate(newErrorKey)
        .setBuilder(
            new ChainedFunction.Builder()
                .setWaitForException(true)
                .setWaitToFinish(new CountDownLatch(0))
                .setValue(null)
                .build());
    // Then when evaluation happens,
    EvaluationResult<StringValue> result = tester.eval(/*keepGoing=*/ false, newErrorKey, topKey);
    // The result has an error,
    assertThatEvaluationResult(result).hasError();
    // But the new error is not persisted to the graph, since the child error shut down evaluation.
    assertThatEvaluationResult(result).hasErrorEntryForKeyThat(newErrorKey).isNull();
  }

  @Test
  public void interruptBitCleared() throws Exception {
    SkyKey interruptKey = GraphTester.skyKey("interrupt");
    tester.getOrCreate(interruptKey).setBuilder(INTERRUPT_BUILDER);
    assertThrows(InterruptedException.class, () -> tester.eval(/*keepGoing=*/ true, interruptKey));
    assertThat(Thread.interrupted()).isFalse();
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
                  throw new RuntimeException("I don't like being woken up!", e);
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

    // When it is interrupted during evaluation (here, caused by the failure of the throwing
    // SkyFunction during a no-keep-going evaluation), then the ParallelEvaluator#evaluate call
    // throws a RuntimeException e where e.getCause() is the RuntimeException thrown by that
    // SkyFunction.
    RuntimeException e =
        assertThrows(
            RuntimeException.class,
            () -> tester.eval(/*keepGoing=*/ false, badInterruptkey, failKey));
    assertThat(e).hasCauseThat().hasMessageThat().isEqualTo("I don't like being woken up!");
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
    // Then the ParallelEvaluator#evaluate call returns an EvaluationResult that has no error for
    // the interrupted SkyFunction.
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
    assertThat(tester.getDeletedKeys()).isEqualTo(ImmutableSet.of(skyKey("d1"), skyKey("top")));
    assertThat(tester.getExistingValue("top")).isNull();
    assertThat(tester.getExistingValue("d1")).isNull();
    assertThat(tester.getExistingValue("d2")).isEqualTo(d2);
    assertThat(tester.getExistingValue("d3")).isEqualTo(d3);
  }

  @Test
  public void deleteOldNodesTest() throws Exception {
    SkyKey d2Key = GraphTester.nonHermeticKey("d2");
    tester
        .getOrCreate("top")
        .setComputedValue(CONCATENATE)
        .addDependency("d1")
        .addDependency(d2Key);
    tester.set("d1", new StringValue("one"));
    tester.set(d2Key, new StringValue("two"));
    tester.eval(true, "top");

    tester.set(d2Key, new StringValue("three"));
    tester.invalidate();
    tester.eval(true, d2Key);

    // The graph now contains the three above nodes (and ERROR_TRANSIENCE).
    assertThat(tester.evaluator.getValues().keySet())
        .containsExactly(skyKey("top"), skyKey("d1"), d2Key, ErrorTransienceValue.KEY);

    String[] noKeys = {};
    tester.evaluator.deleteDirty(2);
    tester.eval(true, noKeys);

    // The top node's value is dirty, but less than two generations old, so it wasn't deleted.
    assertThat(tester.evaluator.getValues().keySet())
        .containsExactly(skyKey("top"), skyKey("d1"), d2Key, ErrorTransienceValue.KEY);

    tester.evaluator.deleteDirty(2);
    tester.eval(true, noKeys);

    // The top node's value was dirty, and was two generations old, so it was deleted.
    assertThat(tester.evaluator.getValues().keySet())
        .containsExactly(skyKey("d1"), d2Key, ErrorTransienceValue.KEY);
  }

  @Test
  public void deleteDirtyCleanedValue() throws Exception {
    SkyKey leafKey = GraphTester.nonHermeticKey("leafKey");
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
    assertThat(tester.getEnqueuedValues())
        .containsExactlyElementsIn(MemoizingEvaluatorTester.toSkyKeys("top1", "d1", "d2"));

    tester.eval(true, "top2");
    assertThat(tester.getEnqueuedValues())
        .containsExactlyElementsIn(MemoizingEvaluatorTester.toSkyKeys(
            "top1", "d1", "d2", "top2", "d3"));
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
        assertThatEvents(eventCollector).containsExactly("warn-d1", "warn-d2");
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
      assertThatEvaluationResult(result)
          .hasSingletonErrorThat(topKey)
          .hasExceptionThat()
          .hasMessageThat()
          .isEqualTo(topKey.toString());
      assertThatEvaluationResult(result)
          .hasSingletonErrorThat(topKey)
          .hasExceptionThat()
          .isInstanceOf(SomeErrorException.class);
      if (i == 0 || eventsStored()) {
        assertThatEvents(eventCollector).containsExactly("warn-dep");
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
      assertThatEvaluationResult(result)
          .hasSingletonErrorThat(topKey)
          .hasExceptionThat()
          .hasMessageThat()
          .isEqualTo(topKey.toString());
      assertThatEvaluationResult(result)
          .hasSingletonErrorThat(topKey)
          .hasExceptionThat()
          .isInstanceOf(SomeErrorException.class);
      if (i == 0 || eventsStored()) {
        assertThatEvents(eventCollector).containsExactly("warning msg");
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
      assertThatEvaluationResult(result)
          .hasSingletonErrorThat(topKey)
          .hasExceptionThat()
          .hasMessageThat()
          .isEqualTo(topKey.toString());
      assertThatEvaluationResult(result)
          .hasSingletonErrorThat(topKey)
          .hasExceptionThat()
          .isInstanceOf(SomeErrorException.class);
      if (i == 0 || eventsStored()) {
        assertThatEvents(eventCollector).containsExactly("warning msg");
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
        assertThatEvents(eventCollector).containsExactly("look both ways before crossing");
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
      tester.evalAndGetError(/*keepGoing=*/ true, "error-value");
      if (i == 0 || eventsStored()) {
        assertThatEvents(eventCollector).containsExactly("don't chew with your mouth open");
      }
    }

    initializeReporter();
    tester.evalAndGet("warning-value");
    if (eventsStored()) {
      assertThatEvents(eventCollector).containsExactly("don't chew with your mouth open");
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
    assertThat(value.getValue()).isEqualTo("y");
    assertThatEvents(eventCollector).containsExactly("fizzlepop", "just letting you know");

    if (eventsStored()) {
      // On the rebuild, we only replay warning messages.
      initializeReporter();
      value = (StringValue) tester.evalAndGet("x");
      assertThat(value.getValue()).isEqualTo("y");
      assertThatEvents(eventCollector).containsExactly("fizzlepop");
    }
  }

  @Test
  public void depMessageBeforeNodeMessageOrNodeValue() throws Exception {
    SkyKey top = skyKey("top");
    AtomicBoolean depWarningEmitted = new AtomicBoolean(false);
    injectGraphListenerForTesting(
        (key, type, order, context) -> {
          if (key.equals(top) && type == EventType.SET_VALUE) {
            assertThat(depWarningEmitted.get()).isTrue();
          }
        },
        /*deterministic=*/ false);
    String depWarning = "dep warning";
    Event topWarning = Event.warn("top warning");
    reporter =
        new DelegatingEventHandler(reporter) {
          @Override
          public void handle(Event e) {
            if (e.getMessage().equals(depWarning)) {
              depWarningEmitted.set(true);
            }
            if (e.equals(topWarning)) {
              assertThat(depWarningEmitted.get()).isTrue();
            }
            super.handle(e);
          }
        };
    SkyKey leaf = skyKey("leaf");
    tester.getOrCreate(leaf).setWarning(depWarning).setConstantValue(new StringValue("leaf"));
    tester
        .getOrCreate(top)
        .setBuilder(
            new NoExtractorFunction() {
              @Nullable
              @Override
              public SkyValue compute(SkyKey skyKey, Environment env) throws InterruptedException {
                SkyValue depValue = env.getValue(leaf);
                if (depValue != null) {
                  // Default GraphTester implementation warns before requesting deps, which doesn't
                  // work
                  // for ordering assertions with memoizing evaluator subclsses that don't store
                  // events
                  // and instead just pass them through directly. By warning after the dep is done
                  // we
                  // avoid that issue.
                  env.getListener().handle(topWarning);
                }
                return depValue;
              }
            });
    EvaluationResult<StringValue> result = tester.eval(/*keepGoing=*/ false, top);
    assertThatEvaluationResult(result).hasEntryThat(top).isEqualTo(new StringValue("leaf"));
    assertThatEvents(eventCollector).containsExactly(depWarning, topWarning.getMessage()).inOrder();
  }

  @Test
  public void invalidationWithChangeAndThenNothingChanged() throws Exception {
    SkyKey bKey = GraphTester.nonHermeticKey("b");
    tester.getOrCreate("a").addDependency(bKey).setComputedValue(COPY);
    tester.set(bKey, new StringValue("y"));
    StringValue original = (StringValue) tester.evalAndGet("a");
    assertThat(original.getValue()).isEqualTo("y");
    tester.set(bKey, new StringValue("z"));
    tester.invalidate();
    StringValue old = (StringValue) tester.evalAndGet("a");
    assertThat(old.getValue()).isEqualTo("z");
    tester.invalidate();
    StringValue current = (StringValue) tester.evalAndGet("a");
    assertThat(current).isEqualTo(old);
  }

  @Test
  public void noKeepGoingErrorAfterKeepGoingError() throws Exception {
    SkyKey topKey = GraphTester.nonHermeticKey("top");
    SkyKey errorKey = GraphTester.skyKey("error");
    tester.getOrCreate(errorKey).setHasError(true);
    tester.getOrCreate(topKey).addDependency(errorKey).setComputedValue(CONCATENATE);
    EvaluationResult<StringValue> result = tester.eval(/*keepGoing=*/ true, topKey);
    assertThatEvaluationResult(result).hasSingletonErrorThat(topKey);
    tester.getOrCreate(topKey, /*markAsModified=*/ true);
    tester.invalidate();
    result = tester.eval(/*keepGoing=*/ false, topKey);
    assertThatEvaluationResult(result).hasSingletonErrorThat(topKey);
  }

  @Test
  public void transientErrorValueInvalidation() throws Exception {
    // Verify that invalidating errors causes all transient error values to be rerun.
    tester.getOrCreate("error-value").setHasTransientError(true).setProgress(
        "just letting you know");

    tester.evalAndGetError(/*keepGoing=*/ true, "error-value");
    assertThatEvents(eventCollector).containsExactly("just letting you know");

    // Change the progress message.
    tester.getOrCreate("error-value").setHasTransientError(true).setProgress(
        "letting you know more");

    // Without invalidating errors, we shouldn't show the new progress message.
    for (int i = 0; i < 2; i++) {
      initializeReporter();
      tester.evalAndGetError(/*keepGoing=*/ true, "error-value");
      assertThatEvents(eventCollector).isEmpty();
    }

    // When invalidating errors, we should show the new progress message.
    initializeReporter();
    tester.invalidateTransientErrors();
    tester.evalAndGetError(/*keepGoing=*/ true, "error-value");
    assertThatEvents(eventCollector).containsExactly("letting you know more");
  }

  @Test
  public void transientPruning() throws Exception {
    SkyKey leaf = GraphTester.nonHermeticKey("leaf");
    tester.getOrCreate("top").setHasTransientError(true).addDependency(leaf);
    tester.set(leaf, new StringValue("leafy"));
    tester.evalAndGetError(/*keepGoing=*/ true, "top");
    tester.getOrCreate(leaf, /*markAsModified=*/true);
    tester.invalidate();
    tester.evalAndGetError(/*keepGoing=*/ true, "top");
  }

  @Test
  public void simpleDependency() throws Exception {
    tester.getOrCreate("ab")
        .addDependency("a")
        .setComputedValue(COPY);
    tester.set("a", new StringValue("me"));
    StringValue value = (StringValue) tester.evalAndGet("ab");
    assertThat(value.getValue()).isEqualTo("me");
  }

  @Test
  public void incrementalSimpleDependency() throws Exception {
    SkyKey aKey = GraphTester.nonHermeticKey("a");
    tester.getOrCreate("ab").addDependency(aKey).setComputedValue(COPY);
    tester.set(aKey, new StringValue("me"));
    tester.evalAndGet("ab");

    tester.set(aKey, new StringValue("other"));
    tester.invalidate();
    StringValue value = (StringValue) tester.evalAndGet("ab");
    assertThat(value.getValue()).isEqualTo("other");
  }

  @Test
  public void diamondDependency() throws Exception {
    SkyKey diamondBase = setupDiamondDependency();
    tester.set(diamondBase, new StringValue("me"));
    StringValue value = (StringValue) tester.evalAndGet("a");
    assertThat(value.getValue()).isEqualTo("meme");
  }

  @Test
  public void incrementalDiamondDependency() throws Exception {
    SkyKey diamondBase = setupDiamondDependency();
    tester.set(diamondBase, new StringValue("me"));
    tester.evalAndGet("a");

    tester.set(diamondBase, new StringValue("other"));
    tester.invalidate();
    StringValue value = (StringValue) tester.evalAndGet("a");
    assertThat(value.getValue()).isEqualTo("otherother");
  }

  private SkyKey setupDiamondDependency() {
    SkyKey diamondBase = GraphTester.nonHermeticKey("d");
    tester.getOrCreate("a")
        .addDependency("b")
        .addDependency("c")
        .setComputedValue(CONCATENATE);
    tester.getOrCreate("b").addDependency(diamondBase).setComputedValue(COPY);
    tester.getOrCreate("c").addDependency(diamondBase).setComputedValue(COPY);
    return diamondBase;
  }

  // ParallelEvaluator notifies ValueProgressReceiver of already-built top-level values in error: we
  // built "top" and "mid" as top-level targets; "mid" contains an error. We make sure "mid" is
  // built as a dependency of "top" before enqueuing mid as a top-level target (by using a latch),
  // so that the top-level enqueuing finds that mid has already been built. The progress receiver
  // should be notified that mid has been built.
  @Test
  public void alreadyAnalyzedBadTarget() throws Exception {
    final SkyKey mid = GraphTester.toSkyKey("zzmid");
    final CountDownLatch valueSet = new CountDownLatch(1);
    injectGraphListenerForTesting(
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
                  TrackingAwaiter.INSTANCE.awaitLatchAndTrackExceptions(valueSet, "value not set");
                }
                break;
              case SET_VALUE:
                valueSet.countDown();
                break;
              default:
                break;
            }
          }
        },
        /*deterministic=*/ true);
    SkyKey top = GraphTester.skyKey("aatop");
    tester.getOrCreate(top).addDependency(mid).setComputedValue(CONCATENATE);
    tester.getOrCreate(mid).setHasError(true);
    tester.eval(/*keepGoing=*/false, top, mid);
    assertThat(valueSet.getCount()).isEqualTo(0L);
    assertThat(tester.progressReceiver.evaluated).containsExactly(mid);
  }

  @Test
  public void receiverToldOfVerifiedValueDependingOnCycle() throws Exception {
    SkyKey leaf = GraphTester.nonHermeticKey("leaf");
    SkyKey cycle = GraphTester.toSkyKey("cycle");
    SkyKey top = GraphTester.toSkyKey("top");
    tester.set(leaf, new StringValue("leaf"));
    tester.getOrCreate(cycle).addDependency(cycle);
    tester.getOrCreate(top).addDependency(leaf).addDependency(cycle);
    tester.eval(/*keepGoing=*/true, top);
    assertThat(tester.progressReceiver.evaluated).containsExactly(leaf, top, cycle);
    tester.progressReceiver.clear();
    tester.getOrCreate(leaf, /*markAsModified=*/true);
    tester.invalidate();
    tester.eval(/*keepGoing=*/true, top);
    if (preciseEvaluationStatusStored()) {
      assertThat(tester.progressReceiver.evaluated).containsExactly(leaf, top);
    }
  }

  @Test
  public void incrementalAddedDependency() throws Exception {
    SkyKey aKey = GraphTester.nonHermeticKey("a");
    SkyKey bKey = GraphTester.nonHermeticKey("b");
    tester.getOrCreate(aKey).addDependency(bKey).setComputedValue(CONCATENATE);
    tester.set(bKey, new StringValue("first"));
    tester.set("c", new StringValue("second"));
    tester.evalAndGet(/*keepGoing=*/ false, aKey);

    tester.getOrCreate(aKey).addDependency("c");
    tester.set(bKey, new StringValue("now"));
    tester.invalidate();
    StringValue value = (StringValue) tester.evalAndGet(/*keepGoing=*/ false, aKey);
    assertThat(value.getValue()).isEqualTo("nowsecond");
  }

  @Test
  public void manyValuesDependOnSingleValue() throws Exception {
    SkyKey leaf = GraphTester.nonHermeticKey("leaf");
    String[] values = new String[TEST_NODE_COUNT];
    for (int i = 0; i < values.length; i++) {
      values[i] = Integer.toString(i);
      tester.getOrCreate(values[i]).addDependency(leaf).setComputedValue(COPY);
    }
    tester.set(leaf, new StringValue("leaf"));

    EvaluationResult<StringValue> result = tester.eval(/* keepGoing= */ false, values);
    for (int i = 0; i < values.length; i++) {
      SkyValue actual = result.get(toSkyKey(values[i]));
      assertThat(actual).isEqualTo(new StringValue("leaf"));
    }

    for (int j = 0; j < TESTED_NODES; j++) {
      tester.set(leaf, new StringValue("other" + j));
      tester.invalidate();
      result = tester.eval(/* keepGoing= */ false, values);
      for (int i = 0; i < values.length; i++) {
        SkyValue actual = result.get(toSkyKey(values[i]));
        assertWithMessage("Run " + j + ", value " + i)
            .that(actual)
            .isEqualTo(new StringValue("other" + j));
      }
    }
  }

  @Test
  public void singleValueDependsOnManyValues() throws Exception {
    SkyKey[] values = new SkyKey[TEST_NODE_COUNT];
    StringBuilder expected = new StringBuilder();
    for (int i = 0; i < values.length; i++) {
      String iString = Integer.toString(i);
      values[i] = GraphTester.nonHermeticKey(iString);
      tester.set(values[i], new StringValue(iString));
      expected.append(iString);
    }
    SkyKey rootKey = toSkyKey("root");
    TestFunction value = tester.getOrCreate(rootKey)
        .setComputedValue(CONCATENATE);
    for (int i = 0; i < values.length; i++) {
      value.addDependency(values[i]);
    }

    EvaluationResult<StringValue> result = tester.eval(/* keepGoing= */ false, rootKey);
    assertThat(result.get(rootKey)).isEqualTo(new StringValue(expected.toString()));

    for (int j = 0; j < 10; j++) {
      expected.setLength(0);
      for (int i = 0; i < values.length; i++) {
        String s = "other" + i + " " + j;
        tester.set(values[i], new StringValue(s));
        expected.append(s);
      }
      tester.invalidate();

      result = tester.eval(/* keepGoing= */ false, rootKey);
      assertThat(result.get(rootKey)).isEqualTo(new StringValue(expected.toString()));
    }
  }

  @Test
  public void twoRailLeftRightDependencies() throws Exception {
    SkyKey leaf = GraphTester.nonHermeticKey("leaf");
    String[] leftValues = new String[TEST_NODE_COUNT];
    String[] rightValues = new String[TEST_NODE_COUNT];
    for (int i = 0; i < leftValues.length; i++) {
      leftValues[i] = "left-" + i;
      rightValues[i] = "right-" + i;
      if (i == 0) {
        tester.getOrCreate(leftValues[i]).addDependency(leaf).setComputedValue(COPY);
        tester.getOrCreate(rightValues[i]).addDependency(leaf).setComputedValue(COPY);
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
    tester.set(leaf, new StringValue("leaf"));

    String lastLeft = "left-" + (TEST_NODE_COUNT - 1);
    String lastRight = "right-" + (TEST_NODE_COUNT - 1);

    EvaluationResult<StringValue> result = tester.eval(/* keepGoing= */ false, lastLeft, lastRight);
    assertThat(result.get(toSkyKey(lastLeft))).isEqualTo(new StringValue("leaf"));
    assertThat(result.get(toSkyKey(lastRight))).isEqualTo(new StringValue("leaf"));

    for (int j = 0; j < TESTED_NODES; j++) {
      String value = "other" + j;
      tester.set(leaf, new StringValue(value));
      tester.invalidate();
      result = tester.eval(/* keepGoing= */ false, lastLeft, lastRight);
      assertThat(result.get(toSkyKey(lastLeft))).isEqualTo(new StringValue(value));
      assertThat(result.get(toSkyKey(lastRight))).isEqualTo(new StringValue(value));
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
    assertThat(result.get(goodKey)).isEqualTo(goodValue);
    assertThat(result.get(topKey)).isNull();
    ErrorInfo errorInfo = result.getError(topKey);
    CycleInfo cycleInfo = Iterables.getOnlyElement(errorInfo.getCycleInfo());
    if (cyclesDetected()) {
      assertThat(cycleInfo.getCycle()).containsExactly(aKey, bKey).inOrder();
      assertThat(cycleInfo.getPathToCycle()).containsExactly(topKey, midKey).inOrder();
    }

    tester.invalidate();
    result = tester.eval(/*keepGoing=*/false, topKey, goodKey);
    assertThat(result.get(topKey)).isNull();
    errorInfo = result.getError(topKey);
    cycleInfo = Iterables.getOnlyElement(errorInfo.getCycleInfo());
    if (cyclesDetected()) {
      assertThat(cycleInfo.getCycle()).containsExactly(aKey, bKey).inOrder();
      assertThat(cycleInfo.getPathToCycle()).containsExactly(topKey, midKey).inOrder();
    }
  }

  @Test
  public void keepGoingCycleAlreadyPresent() throws Exception {
    SkyKey selfEdge = GraphTester.toSkyKey("selfEdge");
    tester.getOrCreate(selfEdge).addDependency(selfEdge).setComputedValue(CONCATENATE);
    EvaluationResult<StringValue> result = tester.eval(/*keepGoing=*/ true, selfEdge);
    assertThatEvaluationResult(result).hasError();
    CycleInfo cycleInfo = Iterables.getOnlyElement(result.getError(selfEdge).getCycleInfo());
    if (cyclesDetected()) {
      CycleInfoSubjectFactory.assertThat(cycleInfo).hasCycleThat().containsExactly(selfEdge);
      CycleInfoSubjectFactory.assertThat(cycleInfo).hasPathToCycleThat().isEmpty();
    }
    SkyKey parent = GraphTester.toSkyKey("parent");
    tester.getOrCreate(parent).addDependency(selfEdge).setComputedValue(CONCATENATE);
    EvaluationResult<StringValue> result2 = tester.eval(/*keepGoing=*/ true, parent);
    assertThatEvaluationResult(result).hasError();
    CycleInfo cycleInfo2 = Iterables.getOnlyElement(result2.getError(parent).getCycleInfo());
    if (cyclesDetected()) {
      CycleInfoSubjectFactory.assertThat(cycleInfo2).hasCycleThat().containsExactly(selfEdge);
      CycleInfoSubjectFactory.assertThat(cycleInfo2).hasPathToCycleThat().containsExactly(parent);
    }
  }

  private void changeCycle(boolean keepGoing) throws Exception {
    SkyKey aKey = GraphTester.toSkyKey("a");
    SkyKey bKey = GraphTester.nonHermeticKey("b");
    SkyKey topKey = GraphTester.toSkyKey("top");
    SkyKey midKey = GraphTester.toSkyKey("mid");
    tester.getOrCreate(topKey).addDependency(midKey).setComputedValue(COPY);
    tester.getOrCreate(midKey).addDependency(aKey).setComputedValue(COPY);
    tester.getOrCreate(aKey).addDependency(bKey).setComputedValue(COPY);
    tester.getOrCreate(bKey).addDependency(aKey);
    EvaluationResult<StringValue> result = tester.eval(keepGoing, topKey);
    assertThat(result.get(topKey)).isNull();
    ErrorInfo errorInfo = result.getError(topKey);
    CycleInfo cycleInfo = Iterables.getOnlyElement(errorInfo.getCycleInfo());
    if (cyclesDetected()) {
      assertThat(cycleInfo.getCycle()).containsExactly(aKey, bKey).inOrder();
      assertThat(cycleInfo.getPathToCycle()).containsExactly(topKey, midKey).inOrder();
    }

    tester.getOrCreate(bKey).removeDependency(aKey);
    tester.set(bKey, new StringValue("bValue"));
    tester.invalidate();
    result = tester.eval(keepGoing, topKey);
    assertThat(result.get(topKey)).isEqualTo(new StringValue("bValue"));
    assertThat(result.getError(topKey)).isNull();
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
    makeGraphDeterministic();
    SkyKey aKey = GraphTester.toSkyKey("a");
    SkyKey bKey = GraphTester.toSkyKey("b");
    SkyKey cKey = GraphTester.nonHermeticKey("c");
    final SkyKey leafKey = GraphTester.nonHermeticKey("leaf");
    // When aKey depends on leafKey and bKey,
    tester
        .getOrCreate(aKey)
        .setBuilder(
            new SkyFunction() {
              @Nullable
              @Override
              public SkyValue compute(SkyKey skyKey, Environment env) throws InterruptedException {
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
    assertThat(result.get(aKey)).isNull();
    if (cyclesDetected()) {
      // And both cycles were found underneath aKey: the (aKey->bKey->cKey) cycle, and the
      // aKey->(bKey->cKey) cycle. This is because cKey depended on aKey and then bKey, so it pushed
      // them down on the stack in that order, so bKey was processed first. It found its cycle, then
      // popped off the stack, and then aKey was processed and found its cycle.
      assertThatEvaluationResult(result)
          .hasErrorEntryForKeyThat(aKey)
          .hasCycleInfoThat()
          .containsExactly(
              new CycleInfo(ImmutableList.of(aKey, bKey, cKey)),
              new CycleInfo(ImmutableList.of(aKey), ImmutableList.of(bKey, cKey)));
    } else {
      assertThatEvaluationResult(result)
          .hasErrorEntryForKeyThat(aKey)
          .hasCycleInfoThat()
          .hasSize(1);
    }
    // When leafKey is changed, so that aKey will be marked as NEEDS_REBUILDING,
    tester.set(leafKey, new StringValue("crunchy"));
    // And cKey is invalidated, so that cycle checking will have to explore the full graph,
    tester.getOrCreate(cKey, /*markAsModified=*/ true);
    tester.invalidate();
    // Then when we evaluate,
    EvaluationResult<StringValue> result2 = tester.eval(/*keepGoing=*/ true, aKey);
    // Things are just as before.
    assertThat(result2.get(aKey)).isNull();
    if (cyclesDetected()) {
      assertThatEvaluationResult(result)
          .hasErrorEntryForKeyThat(aKey)
          .hasCycleInfoThat()
          .containsExactly(
              new CycleInfo(ImmutableList.of(aKey, bKey, cKey)),
              new CycleInfo(ImmutableList.of(aKey), ImmutableList.of(bKey, cKey)));
    } else {
      assertThatEvaluationResult(result)
          .hasErrorEntryForKeyThat(aKey)
          .hasCycleInfoThat()
          .hasSize(1);
    }
  }

  /** Regression test: "crash in cycle checker with dirty values". */
  @Test
  public void cycleAndSelfEdgeWithDirtyValue() throws Exception {
    initializeTester();
    // The cycle detection algorithm non-deterministically traverses into children nodes, so
    // use explicit determinism.
    makeGraphDeterministic();
    SkyKey cycleKey1 = GraphTester.nonHermeticKey("ZcycleKey1");
    SkyKey cycleKey2 = GraphTester.toSkyKey("AcycleKey2");
    tester.getOrCreate(cycleKey1).addDependency(cycleKey2).addDependency(cycleKey1)
    .setComputedValue(CONCATENATE);
    tester.getOrCreate(cycleKey2).addDependency(cycleKey1).setComputedValue(COPY);
    EvaluationResult<StringValue> result = tester.eval(/*keepGoing=*/true, cycleKey1);
    assertThat(result.get(cycleKey1)).isNull();
    ErrorInfo errorInfo = result.getError(cycleKey1);
    CycleInfo cycleInfo = Iterables.getOnlyElement(errorInfo.getCycleInfo());
    if (cyclesDetected()) {
      assertThat(cycleInfo.getCycle()).containsExactly(cycleKey1).inOrder();
      assertThat(cycleInfo.getPathToCycle()).isEmpty();
    }
    tester.getOrCreate(cycleKey1, /*markAsModified=*/true);
    tester.invalidate();
    result = tester.eval(/*keepGoing=*/true, cycleKey1, cycleKey2);
    assertThat(result.get(cycleKey1)).isNull();
    errorInfo = result.getError(cycleKey1);
    cycleInfo = Iterables.getOnlyElement(errorInfo.getCycleInfo());
    if (cyclesDetected()) {
      assertThat(cycleInfo.getCycle()).containsExactly(cycleKey1).inOrder();
      assertThat(cycleInfo.getPathToCycle()).isEmpty();
    }
    cycleInfo =
        Iterables.getOnlyElement(
            tester.driver.getExistingErrorForTesting(cycleKey2).getCycleInfo());
    if (cyclesDetected()) {
      assertThat(cycleInfo.getCycle()).containsExactly(cycleKey1).inOrder();
      assertThat(cycleInfo.getPathToCycle()).containsExactly(cycleKey2).inOrder();
    }
  }

  @Test
  public void cycleAndSelfEdgeWithDirtyValueInSameGroup() throws Exception {
    makeGraphDeterministic();
    final SkyKey cycleKey1 = GraphTester.toSkyKey("ZcycleKey1");
    final SkyKey cycleKey2 = GraphTester.toSkyKey("AcycleKey2");
    tester.getOrCreate(cycleKey2).addDependency(cycleKey2).setComputedValue(CONCATENATE);
    tester
        .getOrCreate(cycleKey1)
        .setBuilder(
            new SkyFunction() {
              @Nullable
              @Override
              public SkyValue compute(SkyKey skyKey, Environment env)
                  throws SkyFunctionException, InterruptedException {
                // The order here is important -- 2 before 1.
                Map<SkyKey, SkyValue> result =
                    env.getValues(ImmutableList.of(cycleKey2, cycleKey1));
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
      assertThat(result.get(cycleKey1)).isNull();
      ErrorInfo errorInfo = result.getError(cycleKey1);
      CycleInfo cycleInfo = Iterables.getOnlyElement(errorInfo.getCycleInfo());
      if (cyclesDetected()) {
        assertThat(cycleInfo.getCycle()).containsExactly(cycleKey1).inOrder();
        assertThat(cycleInfo.getPathToCycle()).isEmpty();
      }
    }
  }

  /** Regression test: "crash in cycle checker with dirty values". */
  @Test
  public void cycleWithDirtyValue() throws Exception {
    SkyKey cycleKey1 = GraphTester.nonHermeticKey("cycleKey1");
    SkyKey cycleKey2 = GraphTester.toSkyKey("cycleKey2");
    tester.getOrCreate(cycleKey1).addDependency(cycleKey2).setComputedValue(COPY);
    tester.getOrCreate(cycleKey2).addDependency(cycleKey1).setComputedValue(COPY);
    EvaluationResult<StringValue> result = tester.eval(/*keepGoing=*/true, cycleKey1);
    assertThat(result.get(cycleKey1)).isNull();
    ErrorInfo errorInfo = result.getError(cycleKey1);
    CycleInfo cycleInfo = Iterables.getOnlyElement(errorInfo.getCycleInfo());
    if (cyclesDetected()) {
      assertThat(cycleInfo.getCycle()).containsExactly(cycleKey1, cycleKey2).inOrder();
      assertThat(cycleInfo.getPathToCycle()).isEmpty();
    }
    tester.getOrCreate(cycleKey1, /*markAsModified=*/true);
    tester.invalidate();
    result = tester.eval(/*keepGoing=*/true, cycleKey1);
    assertThat(result.get(cycleKey1)).isNull();
    errorInfo = result.getError(cycleKey1);
    cycleInfo = Iterables.getOnlyElement(errorInfo.getCycleInfo());
    if (cyclesDetected()) {
      assertThat(cycleInfo.getCycle()).containsExactly(cycleKey1, cycleKey2).inOrder();
      assertThat(cycleInfo.getPathToCycle()).isEmpty();
    }
  }

  /**
   * {@link ParallelEvaluator} can be configured to not store errors alongside recovered values.
   *
   * @param errorsStoredAlongsideValues true if we expect Skyframe to store the error for the
   * cycle in ErrorInfo. If true, supportsTransientExceptions must be true as well.
   * @param supportsTransientExceptions true if we expect Skyframe to mark an ErrorInfo as transient
   * for certain exception types.
   * @param useTransientError true if the test should set the {@link TestFunction} it creates to
   * throw a transient error.
   */
  protected void parentOfCycleAndErrorInternal(
      boolean errorsStoredAlongsideValues,
      boolean supportsTransientExceptions,
      boolean useTransientError)
      throws Exception {
    initializeTester();
    if (errorsStoredAlongsideValues) {
      Preconditions.checkArgument(supportsTransientExceptions);
    }
    SkyKey cycleKey1 = GraphTester.toSkyKey("cycleKey1");
    SkyKey cycleKey2 = GraphTester.toSkyKey("cycleKey2");
    SkyKey mid = GraphTester.toSkyKey("mid");
    SkyKey errorKey = GraphTester.toSkyKey("errorKey");
    tester.getOrCreate(cycleKey1).addDependency(cycleKey2).setComputedValue(COPY);
    tester.getOrCreate(cycleKey2).addDependency(cycleKey1).setComputedValue(COPY);
    TestFunction errorFunction = tester.getOrCreate(errorKey);
    if (useTransientError) {
      errorFunction.setHasTransientError(true);
    } else {
      errorFunction.setHasError(true);
    }
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
      if (useTransientError) {
        // The parent should be transitively transient, since it transitively depends on a transient
        // error.
        assertThat(errorInfo.isTransitivelyTransient()).isTrue();
      } else {
        assertThatErrorInfo(errorInfo).isNotTransient();
      }
      assertThat(errorInfo.getException())
          .hasMessageThat()
          .isEqualTo(NODE_TYPE.getName() + ":errorKey");
    } else {
      // When errors are not stored alongside values, transient errors that are recovered from do
      // not make the parent transient
      if (supportsTransientExceptions) {
        assertThatErrorInfo(errorInfo).isTransient();
        assertThatErrorInfo(errorInfo).hasExceptionThat().isNotNull();
      } else {
        assertThatErrorInfo(errorInfo).isNotTransient();
        assertThatErrorInfo(errorInfo).hasExceptionThat().isNull();
      }
    }
    if (cyclesDetected()) {
      assertThatErrorInfo(errorInfo)
          .hasCycleInfoThat()
          .containsExactly(
              new CycleInfo(ImmutableList.of(top), ImmutableList.of(cycleKey1, cycleKey2)));
    } else {
      assertThatErrorInfo(errorInfo).hasCycleInfoThat().hasSize(1);
    }
    // But the parent itself shouldn't have a direct dep on the special error transience node.
    verifyParentDepsForParentOfCycleAndError(
        assertThatEvaluationResult(evalResult).hasDirectDepsInGraphThat(top));
  }

  protected void verifyParentDepsForParentOfCycleAndError(IterableSubject parentDeps) {
    parentDeps.doesNotContain(ErrorTransienceValue.KEY);
  }

  @Test
  public void parentOfCycleAndError() throws Exception {
    parentOfCycleAndErrorInternal(
        /*errorsStoredAlongsideValues=*/ true,
        /*supportsTransientExceptions=*/ true,
        /*useTransientError=*/ true);
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
    injectGraphListenerForTesting(
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
        },
        /*deterministic=*/ true);
    SkyKey dep1 = GraphTester.nonHermeticKey("dep1");
    tester.set(dep1, new StringValue("dep1"));
    final SkyKey dep2 = GraphTester.toSkyKey("dep2");
    tester.set(dep2, new StringValue("dep2"));
    // otherTop should request the deps one at a time, so that it can be in the CHECK_DEPENDENCIES
    // state even after one dep is re-evaluated.
    tester
        .getOrCreate(otherTop)
        .setBuilder(
            new NoExtractorFunction() {
              @Override
              public SkyValue compute(SkyKey skyKey, Environment env) throws InterruptedException {
                env.getValue(dep1);
                if (env.valuesMissing()) {
                  return null;
                }
                env.getValue(dep2);
                return env.valuesMissing() ? null : new StringValue("otherTop");
              }
            });
    // Prime the graph with otherTop, so we can dirty it next build.
    assertThat(tester.evalAndGet(/*keepGoing=*/ false, otherTop))
        .isEqualTo(new StringValue("otherTop"));
    // Mark dep1 changed, so otherTop will be dirty and request re-evaluation of dep1.
    tester.getOrCreate(dep1, /*markAsModified=*/true);
    SkyKey topKey = GraphTester.toSkyKey("top");
    // Note that since DeterministicHelper alphabetizes reverse deps, it is important that
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
    Iterable<CycleInfo> cycleInfos = result.getError(topKey).getCycleInfo();
    assertWithMessage(result.toString()).that(cycleInfos).isNotEmpty();
    CycleInfo cycleInfo = Iterables.getOnlyElement(cycleInfos);
    if (cyclesDetected()) {
      assertThat(result.errorMap().keySet()).containsExactly(topKey);
      assertThat(cycleInfo.getPathToCycle()).containsExactly(topKey);
      assertThat(cycleInfo.getCycle()).containsExactly(cycle1Key, cycle2Key);
    }
  }

  @Test
  public void breakCycle() throws Exception {
    SkyKey aKey = GraphTester.nonHermeticKey("a");
    SkyKey bKey = GraphTester.nonHermeticKey("b");
    // When aKey and bKey depend on each other,
    tester.getOrCreate(aKey).addDependency(bKey);
    tester.getOrCreate(bKey).addDependency(aKey);
    // And they are evaluated,
    EvaluationResult<StringValue> result = tester.eval(/*keepGoing=*/ true, aKey, bKey);
    // Then the evaluation is in error,
    assertThatEvaluationResult(result).hasError();
    // And each node has the expected cycle.
    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(aKey)
        .hasCycleInfoThat()
        .isNotEmpty();
    CycleInfo aCycleInfo = Iterables.getOnlyElement(result.getError(aKey).getCycleInfo());
    if (cyclesDetected()) {
      assertThat(aCycleInfo.getCycle()).containsExactly(aKey, bKey).inOrder();
      assertThat(aCycleInfo.getPathToCycle()).isEmpty();
    }
    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(bKey)
        .hasCycleInfoThat()
        .isNotEmpty();
    CycleInfo bCycleInfo = Iterables.getOnlyElement(result.getError(bKey).getCycleInfo());
    if (cyclesDetected()) {
      assertThat(bCycleInfo.getCycle()).containsExactly(bKey, aKey).inOrder();
      assertThat(bCycleInfo.getPathToCycle()).isEmpty();
    }

    // When both dependencies are broken,
    tester.getOrCreate(bKey).removeDependency(aKey);
    tester.set(bKey, new StringValue("bValue"));
    tester.getOrCreate(aKey).removeDependency(bKey);
    tester.set(aKey, new StringValue("aValue"));
    tester.invalidate();
    // And the nodes are re-evaluated,
    result = tester.eval(/*keepGoing=*/ true, aKey, bKey);
    // Then evaluation is successful and the nodes have the expected values.
    assertThatEvaluationResult(result).hasEntryThat(aKey).isEqualTo(new StringValue("aValue"));
    assertThatEvaluationResult(result).hasEntryThat(bKey).isEqualTo(new StringValue("bValue"));
  }

  @Test
  public void nodeInvalidatedThenDoubleCycle() throws InterruptedException {
    makeGraphDeterministic();
    // When topKey depends on depKey, and both are top-level nodes in the graph,
    SkyKey topKey = GraphTester.nonHermeticKey("bKey");
    SkyKey depKey = GraphTester.nonHermeticKey("aKey");
    tester.getOrCreate(topKey).addDependency(depKey).setConstantValue(new StringValue("a"));
    tester.getOrCreate(depKey).setConstantValue(new StringValue("b"));
    // Then evaluation is as expected.
    EvaluationResult<StringValue> result1 = tester.eval(/*keepGoing=*/ true, topKey, depKey);
    assertThatEvaluationResult(result1).hasEntryThat(topKey).isEqualTo(new StringValue("a"));
    assertThatEvaluationResult(result1).hasEntryThat(depKey).isEqualTo(new StringValue("b"));
    assertThatEvaluationResult(result1).hasNoError();
    // When both nodes acquire self-edges, with topKey still also depending on depKey, in the same
    // group,
    tester.getOrCreate(depKey, /*markAsModified=*/ true).addDependency(depKey);
    tester
        .getOrCreate(topKey, /*markAsModified=*/ true)
        .setConstantValue(null)
        .removeDependency(depKey)
        .setBuilder(
            new SkyFunction() {
              @Nullable
              @Override
              public SkyValue compute(SkyKey skyKey, Environment env)
                  throws SkyFunctionException, InterruptedException {
                env.getValues(ImmutableList.of(topKey, depKey));
                assertThat(env.valuesMissing()).isTrue();
                return null;
              }

              @Nullable
              @Override
              public String extractTag(SkyKey skyKey) {
                return null;
              }
            });
    tester.invalidate();
    // Then evaluation is as expected -- topKey has removed its dep on depKey (since depKey was not
    // done when topKey found its cycle), and both topKey and depKey have cycles.
    EvaluationResult<StringValue> result2 = tester.eval(/*keepGoing=*/ true, topKey, depKey);
    if (cyclesDetected()) {
      assertThatEvaluationResult(result2)
          .hasErrorEntryForKeyThat(topKey)
          .hasCycleInfoThat()
          .containsExactly(new CycleInfo(ImmutableList.of(topKey)));
      assertThatEvaluationResult(result2).hasDirectDepsInGraphThat(topKey).containsExactly(topKey);
      assertThatEvaluationResult(result2)
          .hasErrorEntryForKeyThat(depKey)
          .hasCycleInfoThat()
          .containsExactly(new CycleInfo(ImmutableList.of(depKey)));
    } else {
      assertThatEvaluationResult(result2)
          .hasErrorEntryForKeyThat(topKey)
          .hasCycleInfoThat()
          .hasSize(1);
      assertThatEvaluationResult(result2)
          .hasErrorEntryForKeyThat(depKey)
          .hasCycleInfoThat()
          .hasSize(1);
    }
    // When the nodes return to their original, error-free state,
    tester
        .getOrCreate(topKey, /*markAsModified=*/ true)
        .setBuilder(null)
        .addDependency(depKey)
        .setConstantValue(new StringValue("a"));
    tester.getOrCreate(depKey, /*markAsModified=*/ true).removeDependency(depKey);
    tester.invalidate();
    // Then evaluation is as expected.
    EvaluationResult<StringValue> result3 = tester.eval(/*keepGoing=*/ true, topKey, depKey);
    assertThatEvaluationResult(result3).hasEntryThat(topKey).isEqualTo(new StringValue("a"));
    assertThatEvaluationResult(result3).hasEntryThat(depKey).isEqualTo(new StringValue("b"));
    assertThatEvaluationResult(result3).hasNoError();
  }

  @SuppressWarnings("PreferJavaTimeOverload")
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
      tester
          .getOrCreate("subKey" + i)
          .setComputedValue(
              new ValueComputer() {
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
    assertThat(result.hasError()).isFalse();
    assertThat(maxValue[0]).isEqualTo(5);
  }

  @Test
  public void nodeIsChangedWithoutBeingEvaluated() throws Exception {
    SkyKey buildFile = GraphTester.nonHermeticKey("buildfile");
    SkyKey top = GraphTester.skyKey("top");
    SkyKey dep = GraphTester.nonHermeticKey("dep");
    tester.set(buildFile, new StringValue("depend on dep"));
    StringValue depVal = new StringValue("this is dep");
    tester.set(dep, depVal);
    tester
        .getOrCreate(top)
        .setBuilder(
            new SkyFunction() {
              @Nullable
              @Override
              public SkyValue compute(SkyKey skyKey, Environment env)
                  throws SkyFunctionException, InterruptedException {
                StringValue val = (StringValue) env.getValue(buildFile);
                if (env.valuesMissing()) {
                  return null;
                }
                if (val.getValue().equals("depend on dep")) {
                  StringValue result = (StringValue) env.getValue(dep);
                  return env.valuesMissing() ? null : result;
                }
                throw new GenericFunctionException(
                    new SomeErrorException("bork"), Transience.PERSISTENT);
              }

              @Nullable
              @Override
              public String extractTag(SkyKey skyKey) {
                return null;
              }
            });
    assertThat(tester.evalAndGet("top")).isEqualTo(depVal);
    StringValue newDepVal = new StringValue("this is new dep");
    tester.set(dep, newDepVal);
    tester.set(buildFile, new StringValue("don't depend on dep"));
    tester.invalidate();
    tester.eval(/*keepGoing=*/ false, top);
    tester.set(buildFile, new StringValue("depend on dep"));
    tester.invalidate();
    assertThat(tester.evalAndGet("top")).isEqualTo(newDepVal);
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
    CountDownLatch slowStart = new CountDownLatch(1);
    CountDownLatch errorFinish = new CountDownLatch(1);
    SkyKey errorKey = GraphTester.nonHermeticKey("error");
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
    assertThatEvaluationResult(result).hasSingletonErrorThat(topKey);
    // Make sure midKey didn't finish building.
    assertThat(tester.getExistingValue(midKey)).isNull();
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
    assertThatEvaluationResult(result).hasSingletonErrorThat(topKey);
    // Make sure midKey didn't finish building.
    assertThat(tester.getExistingValue(midKey)).isNull();
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
  private void passThenFailToBuild(boolean successFirst) throws Exception {
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

    EvaluationResult<StringValue> result;
    if (successFirst) {
      result = tester.eval(/*keepGoing=*/ false, successKey, slowFailKey);
    } else {
      result = tester.eval(/*keepGoing=*/ false, slowFailKey, successKey);
    }
    assertThatEvaluationResult(result).hasErrorEntryForKeyThat(slowFailKey);
    assertThat(result.values()).containsExactly(new StringValue("yippee"));
  }

  @Test
  public void passThenFailToBuild() throws Exception {
    passThenFailToBuild(true);
  }

  @Test
  public void passThenFailToBuildAlternateOrder() throws Exception {
    passThenFailToBuild(false);
  }

  @Test
  public void incompleteDirectDepsForDirtyValue() throws Exception {
    SkyKey topKey = GraphTester.nonHermeticKey("top");
    tester.set(topKey, new StringValue("initial"));
    // Put topKey into graph so it will be dirtied on next run.
    assertThat(tester.evalAndGet(/*keepGoing=*/ false, topKey))
        .isEqualTo(new StringValue("initial"));
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
    assertThatEvaluationResult(result).hasSingletonErrorThat(topKey);
    // Make sure midKey didn't finish building.
    assertThat(tester.getExistingValue(midKey)).isNull();
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
    SkyKey afterKey = GraphTester.nonHermeticKey("after");
    SkyKey errorKey = GraphTester.toSkyKey("my_error_value");
    tester.getOrCreate(errorKey).setHasError(true);
    tester.set(afterKey, new StringValue("after"));
    SkyKey parentKey = GraphTester.toSkyKey("parent");
    tester
        .getOrCreate(parentKey)
        .addErrorDependency(errorKey, new StringValue("recovered"))
        .setComputedValue(CONCATENATE)
        .addDependency(afterKey);
    EvaluationResult<StringValue> result = tester.eval(/*keepGoing=*/true, parentKey);
    assertThat(result.errorMap()).isEmpty();
    assertThat(result.get(parentKey).getValue()).isEqualTo("recoveredafter");
    tester.set(afterKey, new StringValue("before"));
    tester.invalidate();
    result = tester.eval(/*keepGoing=*/true, parentKey);
    assertThat(result.errorMap()).isEmpty();
    assertThat(result.get(parentKey).getValue()).isEqualTo("recoveredbefore");
  }

  @Test
  public void continueWithErrorDepTurnedGood() throws Exception {
    SkyKey errorKey = GraphTester.nonHermeticKey("my_error_value");
    tester.getOrCreate(errorKey).setHasError(true);
    tester.set("after", new StringValue("after"));
    SkyKey parentKey = GraphTester.toSkyKey("parent");
    tester.getOrCreate(parentKey).addErrorDependency(errorKey, new StringValue("recovered"))
        .setComputedValue(CONCATENATE).addDependency("after");
    EvaluationResult<StringValue> result = tester.eval(/*keepGoing=*/true, parentKey);
    assertThat(result.errorMap()).isEmpty();
    assertThat(result.get(parentKey).getValue()).isEqualTo("recoveredafter");
    tester.set(errorKey, new StringValue("reformed")).setHasError(false);
    tester.invalidate();
    result = tester.eval(/*keepGoing=*/true, parentKey);
    assertThat(result.errorMap()).isEmpty();
    assertThat(result.get(parentKey).getValue()).isEqualTo("reformedafter");
  }

  @Test
  public void errorDepAlreadyThereThenTurnedGood() throws Exception {
    SkyKey errorKey = GraphTester.nonHermeticKey("my_error_value");
    tester.getOrCreate(errorKey).setHasError(true);
    SkyKey parentKey = GraphTester.toSkyKey("parent");
    tester.getOrCreate(parentKey).addErrorDependency(errorKey, new StringValue("recovered"))
        .setHasError(true);
    // Prime the graph by putting the error value in it beforehand.
    assertThat(tester.evalAndGetError(/*keepGoing=*/ true, errorKey)).isNotNull();
    // Request the parent.
    EvaluationResult<StringValue> result = tester.eval(/*keepGoing=*/ false, parentKey);
    assertThatEvaluationResult(result).hasSingletonErrorThat(parentKey);
    // Change the error value to no longer throw.
    tester.set(errorKey, new StringValue("reformed")).setHasError(false);
    tester.getOrCreate(parentKey, /*markAsModified=*/false).setHasError(false)
        .setComputedValue(COPY);
    tester.differencer.invalidate(ImmutableList.of(errorKey));
    tester.invalidate();
    // Request the parent again. This time it should succeed.
    result = tester.eval(/*keepGoing=*/false, parentKey);
    assertThat(result.errorMap()).isEmpty();
    assertThat(result.get(parentKey).getValue()).isEqualTo("reformed");
    // Confirm that the parent no longer depends on the error transience value -- make it
    // unbuildable again, but without invalidating it, and invalidate transient errors. The parent
    // should not be rebuilt.
    tester.getOrCreate(parentKey, /*markAsModified=*/false).setHasError(true);
    tester.invalidateTransientErrors();
    result = tester.eval(/*keepGoing=*/false, parentKey);
    assertThat(result.errorMap()).isEmpty();
    assertThat(result.get(parentKey).getValue()).isEqualTo("reformed");
  }

  /**
   * Regression test for 2014 bug: error transience value is registered before newly requested deps.
   * A value requests a child, gets it back immediately, and then throws, causing the error
   * transience value to be registered as a dep. The following build, the error is invalidated via
   * that child.
   */
  @Test
  public void doubleDepOnErrorTransienceValue() throws Exception {
    SkyKey leafKey = GraphTester.nonHermeticKey("leaf");
    tester.set(leafKey, new StringValue("leaf"));
    // Prime the graph by putting leaf in beforehand.
    assertThat(tester.evalAndGet(/*keepGoing=*/ false, leafKey)).isEqualTo(new StringValue("leaf"));
    SkyKey topKey = GraphTester.toSkyKey("top");
    tester.getOrCreate(topKey).addDependency(leafKey).setHasError(true);
    // Build top -- it has an error.
    tester.evalAndGetError(/*keepGoing=*/ true, topKey);
    // Invalidate top via leaf, and rebuild.
    tester.set(leafKey, new StringValue("leaf2"));
    tester.invalidate();
    tester.evalAndGetError(/*keepGoing=*/ true, topKey);
  }

  /** Regression test for crash bug. */
  @Test
  public void errorTransienceDepCleared() throws Exception {
    SkyKey top = GraphTester.toSkyKey("top");
    SkyKey leaf = GraphTester.nonHermeticKey("leaf");
    tester.set(leaf, new StringValue("leaf"));
    tester.getOrCreate(top).addDependency(leaf).setHasTransientError(true);
    EvaluationResult<StringValue> result = tester.eval(/*keepGoing=*/false, top);
    assertWithMessage(result.toString()).that(result.hasError()).isTrue();
    tester.getOrCreate(leaf, /*markAsModified=*/true);
    tester.invalidate();
    SkyKey irrelevant = GraphTester.toSkyKey("irrelevant");
    tester.set(irrelevant, new StringValue("irrelevant"));
    tester.eval(/*keepGoing=*/true, irrelevant);
    tester.invalidateTransientErrors();
    result = tester.eval(/*keepGoing=*/true, top);
    assertWithMessage(result.toString()).that(result.hasError()).isTrue();
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
    // because it was only called during the bubbling-up phase.
    EvaluationResult<StringValue> result = tester.eval(/*keepGoing=*/ false, midKey);
    assertThat(result.get(midKey)).isNull();
    assertThatEvaluationResult(result).hasSingletonErrorThat(midKey);
    // In a keepGoing build, midKey should be re-evaluated.
    assertThat(((StringValue) tester.evalAndGet(/*keepGoing=*/ true, parentKey)).getValue())
        .isEqualTo("recovered");
  }

  /**
   * "top" requests a dependency group in which the first value, called "error", throws an
   * exception, so "mid" and "mid2", which depend on "slow", never get built.
   */
  @Test
  public void errorInDependencyGroup() throws Exception {
    SkyKey topKey = GraphTester.toSkyKey("top");
    CountDownLatch slowStart = new CountDownLatch(1);
    CountDownLatch errorFinish = new CountDownLatch(1);
    SkyKey errorKey = GraphTester.nonHermeticKey("error");
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
    assertThat(result.hasError()).isTrue();
    assertThatEvaluationResult(result).hasErrorEntryForKeyThat(topKey);

    // Ensure that evaluation succeeds if errorKey does not throw an error.
    tester.getOrCreate(errorKey).setBuilder(null);
    tester.set(errorKey, new StringValue("ok"));
    tester.invalidate();
    assertThat(tester.evalAndGet("top")).isEqualTo(new StringValue("top"));
  }

  /**
   * Regression test -- if value top requests {depA, depB}, depC, with depA and depC there and depB
   * absent, and then throws an exception, the stored deps should be depA, depC (in different
   * groups), not {depA, depC} (same group).
   */
  @Test
  public void valueInErrorWithGroups() throws Exception {
    SkyKey topKey = GraphTester.toSkyKey("top");
    final SkyKey groupDepA = GraphTester.nonHermeticKey("groupDepA");
    final SkyKey groupDepB = GraphTester.toSkyKey("groupDepB");
    SkyKey depC = GraphTester.nonHermeticKey("depC");
    tester.set(groupDepA, new SkyKeyValue(depC));
    tester.set(groupDepB, new StringValue(""));
    tester.getOrCreate(depC).setHasError(true);
    tester
        .getOrCreate(topKey)
        .setBuilder(
            new NoExtractorFunction() {
              @Override
              public SkyValue compute(SkyKey skyKey, Environment env)
                  throws SkyFunctionException, InterruptedException {
                SkyKeyValue val =
                    ((SkyKeyValue)
                        env.getValues(ImmutableList.of(groupDepA, groupDepB)).get(groupDepA));
                if (env.valuesMissing()) {
                  return null;
                }
                try {
                  env.getValueOrThrow(val.key, SomeErrorException.class);
                } catch (SomeErrorException e) {
                  throw new GenericFunctionException(e, Transience.PERSISTENT);
                }
                return env.valuesMissing() ? null : new StringValue("top");
              }
            });

    EvaluationResult<SkyValue> evaluationResult = tester.eval(/*keepGoing=*/ true, groupDepA, depC);
    assertThat(((SkyKeyValue) evaluationResult.get(groupDepA)).key).isEqualTo(depC);
    assertThatEvaluationResult(evaluationResult).hasErrorEntryForKeyThat(depC);
    evaluationResult = tester.eval(/*keepGoing=*/false, topKey);
    assertThatEvaluationResult(evaluationResult).hasErrorEntryForKeyThat(topKey);

    tester.set(groupDepA, new SkyKeyValue(groupDepB));
    tester.getOrCreate(depC, /*markAsModified=*/true);
    tester.invalidate();
    evaluationResult = tester.eval(/*keepGoing=*/false, topKey);
    assertWithMessage(evaluationResult.toString()).that(evaluationResult.hasError()).isFalse();
    assertThat(evaluationResult.get(topKey)).isEqualTo(new StringValue("top"));
  }

  private static class SkyKeyValue implements SkyValue {
    private final SkyKey key;

    private SkyKeyValue(SkyKey key) {
      this.key = key;
    }
  }

  @Test
  public void errorOnlyEmittedOnce() throws Exception {
    initializeTester();
    tester.set("x", new StringValue("y")).setWarning("fizzlepop");
    StringValue value = (StringValue) tester.evalAndGet("x");
    assertThat(value.getValue()).isEqualTo("y");
    assertThatEvents(eventCollector).containsExactly("fizzlepop");

    tester.invalidate();
    value = (StringValue) tester.evalAndGet("x");
    assertThat(value.getValue()).isEqualTo("y");
    // No new events emitted.
  }

  /**
   * We are checking here that we are resilient to a race condition in which a value that is
   * checking its children for dirtiness is signaled by all of its children, putting it in a ready
   * state, before the thread has terminated. Optionally, one of its children may throw an error,
   * shutting down the threadpool. The essential race is that a child about to throw signals its
   * parent and the parent's builder restarts itself before the exception is thrown. Here, the
   * signaling happens while dirty dependencies are being checked. We control the timing by blocking
   * "top"'s registering itself on its deps.
   */
  private void dirtyChildEnqueuesParentDuringCheckDependencies(final boolean throwError)
      throws Exception {
    // Value to be built. It will be signaled to rebuild before it has finished checking its deps.
    final SkyKey top = GraphTester.toSkyKey("a_top");
    // otherTop is alphabetically after top.
    SkyKey otherTop = skyKey("z_otherTop");
    // Dep that blocks before it acknowledges being added as a dep by top, so the firstKey value has
    // time to signal top. (Importantly its key is alphabetically after 'firstKey').
    final SkyKey slowAddingDep = GraphTester.toSkyKey("slowDep");
    // Value that is modified on the second build. Its thread won't finish until it signals top,
    // which will wait for the signal before it enqueues its next dep. We prevent the thread from
    // finishing by having the graph listener block on the second reverse dep to signal.
    SkyKey firstKey = GraphTester.nonHermeticKey("first");
    tester.set(firstKey, new StringValue("biding"));
    // Don't perform any blocking on the first build.
    final AtomicBoolean delayTopSignaling = new AtomicBoolean(false);
    final CountDownLatch topSignaled = new CountDownLatch(1);
    final CountDownLatch topRequestedDepOrRestartedBuild = new CountDownLatch(1);
    final CountDownLatch parentsRequested = new CountDownLatch(2);
    injectGraphListenerForTesting(
        (key, type, order, context) -> {
          if (!delayTopSignaling.get()) {
            return;
          }
          if (key.equals(otherTop) && type == EventType.SIGNAL) {
            TrackingAwaiter.INSTANCE.awaitLatchAndTrackExceptions(
                topRequestedDepOrRestartedBuild, "top's builder did not start in time");
            return;
          }
          if (key.equals(firstKey) && type == EventType.ADD_REVERSE_DEP && order == Order.AFTER) {
            parentsRequested.countDown();
            return;
          }
          if (key.equals(firstKey) && type == EventType.CHECK_IF_DONE && order == Order.AFTER) {
            parentsRequested.countDown();
            if (throwError) {
              topRequestedDepOrRestartedBuild.countDown();
            }
            return;
          }
          if (key.equals(top) && type == EventType.SIGNAL && order == Order.AFTER) {
            // top is signaled by firstKey (since slowAddingDep is blocking), so slowAddingDep
            // is now free to acknowledge top as a parent.
            topSignaled.countDown();
            return;
          }
          if (key.equals(firstKey) && type == EventType.SET_VALUE && order == Order.BEFORE) {
            // Make sure both parents add themselves as rdeps.
            TrackingAwaiter.INSTANCE.awaitLatchAndTrackExceptions(
                parentsRequested, "parents did not request dep in time");
          }
          if (key.equals(slowAddingDep)
              && type == EventType.CHECK_IF_DONE
              && top.equals(context)
              && order == Order.BEFORE) {
            // If top is trying to declare a dep on slowAddingDep, wait until firstKey has
            // signaled top. Then this add dep will return DONE and top will be signaled,
            // making it ready, so it will be enqueued.
            TrackingAwaiter.INSTANCE.awaitLatchAndTrackExceptions(
                topSignaled, "first key didn't signal top in time");
          }
        },
        /*deterministic=*/ true);
    tester.set(slowAddingDep, new StringValue("dep"));
    final AtomicInteger numTopInvocations = new AtomicInteger(0);
    tester
        .getOrCreate(top)
        .setBuilder(
            new NoExtractorFunction() {
              @Override
              public SkyValue compute(SkyKey key, SkyFunction.Environment env)
                  throws InterruptedException {
                numTopInvocations.incrementAndGet();
                if (delayTopSignaling.get()) {
                  // The graph listener will block on firstKey's signaling of otherTop above until
                  // this thread starts running.
                  topRequestedDepOrRestartedBuild.countDown();
                }
                // top's builder just requests both deps in a group.
                env.getValuesOrThrow(
                    ImmutableList.of(firstKey, slowAddingDep), SomeErrorException.class);
                return env.valuesMissing() ? null : new StringValue("top");
              }
            });
    // First build : just prime the graph.
    EvaluationResult<StringValue> result = tester.eval(/*keepGoing=*/false, top);
    assertThat(result.hasError()).isFalse();
    assertThat(result.get(top)).isEqualTo(new StringValue("top"));
    assertThat(numTopInvocations.get()).isEqualTo(2);
    // Now dirty the graph, and maybe have firstKey throw an error.
    if (throwError) {
      tester
          .getOrCreate(firstKey, /*markAsModified=*/ true)
          .setConstantValue(null)
          .setBuilder(
              new NoExtractorFunction() {
                @Nullable
                @Override
                public SkyValue compute(SkyKey skyKey, Environment env)
                    throws SkyFunctionException {
                  TrackingAwaiter.INSTANCE.awaitLatchAndTrackExceptions(
                      parentsRequested, "both parents didn't request in time");
                  throw new GenericFunctionException(
                      new SomeErrorException(firstKey.toString()), Transience.PERSISTENT);
                }
              });
    } else {
      tester
          .getOrCreate(firstKey, /*markAsModified=*/ true)
          .setConstantValue(new StringValue("new"));
    }
    tester.getOrCreate(otherTop).addDependency(firstKey).setComputedValue(CONCATENATE);
    tester.invalidate();
    delayTopSignaling.set(true);
    result = tester.eval(/*keepGoing=*/ false, top, otherTop);
    if (throwError) {
      assertThatEvaluationResult(result).hasError();
      assertThat(result.keyNames()).isEmpty(); // No successfully evaluated values.
      assertThatEvaluationResult(result).hasErrorEntryForKeyThat(top);
      assertWithMessage(
              "on the incremental build, top's builder should have only been used in error "
                  + "bubbling")
          .that(numTopInvocations.get())
          .isEqualTo(3);
    } else {
      assertThatEvaluationResult(result).hasEntryThat(top).isEqualTo(new StringValue("top"));
      assertThatEvaluationResult(result).hasNoError();
      assertWithMessage(
              "on the incremental build, top's builder should have only been executed once in "
                  + "normal evaluation")
          .that(numTopInvocations.get())
          .isEqualTo(3);
    }
    assertThat(topSignaled.getCount()).isEqualTo(0);
    assertThat(topRequestedDepOrRestartedBuild.getCount()).isEqualTo(0);
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
    final SkyKey midKey = GraphTester.nonHermeticKey("mid");
    final SkyKey changedKey = GraphTester.nonHermeticKey("changed");
    tester.getOrCreate(changedKey).setConstantValue(new StringValue("first"));
    // When top depends on mid,
    tester.getOrCreate(topKey).addDependency(midKey).setComputedValue(CONCATENATE);
    // And mid depends on changed,
    tester.getOrCreate(midKey).addDependency(changedKey).setComputedValue(CONCATENATE);
    final CountDownLatch changedKeyStarted = new CountDownLatch(1);
    final CountDownLatch changedKeyCanFinish = new CountDownLatch(1);
    final AtomicBoolean controlTiming = new AtomicBoolean(false);
    injectGraphListenerForTesting(
        new Listener() {
          @Override
          public void accept(SkyKey key, EventType type, Order order, Object context) {
            if (!controlTiming.get()) {
              return;
            }
            if (key.equals(midKey) && type == EventType.CHECK_IF_DONE && order == Order.BEFORE) {
              TrackingAwaiter.INSTANCE.awaitLatchAndTrackExceptions(
                  changedKeyStarted, "changed key didn't start");
            } else if (key.equals(changedKey)
                && type == EventType.REMOVE_REVERSE_DEP
                && order == Order.AFTER
                && midKey.equals(context)) {
              changedKeyCanFinish.countDown();
            }
          }
        },
        /*deterministic=*/ false);
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
    tester
        .getOrCreate(midKey, /*markAsModified=*/ true)
        .removeDependency(changedKey)
        .setComputedValue(null)
        .setConstantValue(new StringValue("mid"));
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
    SkyKey topKey = GraphTester.nonHermeticKey("top");
    SkyKey leafKey = GraphTester.nonHermeticKey("leaf");
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

  @Test
  public void resetNodeOnRequest_smoke() throws Exception {
    SkyKey restartingKey = GraphTester.nonHermeticKey("restart");
    StringValue expectedValue = new StringValue("done");
    AtomicInteger numInconsistencyCalls = new AtomicInteger(0);
    tester.setGraphInconsistencyReceiver(
        restartEnabledInconsistencyReceiver(
            (key, otherKey, inconsistency) -> {
              Preconditions.checkState(otherKey == null, otherKey);
              Preconditions.checkState(
                  inconsistency == Inconsistency.RESET_REQUESTED, inconsistency);
              Preconditions.checkState(restartingKey.equals(key), key);
              numInconsistencyCalls.incrementAndGet();
            }));
    tester.initialize(/*keepEdges=*/ true);
    AtomicInteger numFunctionCalls = new AtomicInteger(0);
    tester
        .getOrCreate(restartingKey)
        .setBuilder(
            new SkyFunction() {

              @Override
              public SkyValue compute(SkyKey skyKey, Environment env) {
                if (numFunctionCalls.getAndIncrement() < 2) {
                  return Restart.SELF;
                }
                return expectedValue;
              }

              @Nullable
              @Override
              public String extractTag(SkyKey skyKey) {
                return null;
              }
            });
    assertThat(tester.evalAndGet(/*keepGoing=*/ false, restartingKey)).isEqualTo(expectedValue);
    assertThat(numInconsistencyCalls.get()).isEqualTo(2);
    assertThat(numFunctionCalls.get()).isEqualTo(3);
    tester.getOrCreate(restartingKey, /*markAsModified=*/ true);
    tester.invalidate();
    numInconsistencyCalls.set(0);
    numFunctionCalls.set(0);
    assertThat(tester.evalAndGet(/*keepGoing=*/ false, restartingKey)).isEqualTo(expectedValue);
    assertThat(numInconsistencyCalls.get()).isEqualTo(2);
    assertThat(numFunctionCalls.get()).isEqualTo(3);
  }

  /** Mode in which to run {@link #runResetNodeOnRequest_withDeps}. */
  protected enum RunResetNodeOnRequestWithDepsMode {
    /** Do a second evaluation of the top node. */
    REEVALUATE_TOP_NODE,
    /**
     * On the second evaluation, only evaluate a leaf node. This will detect reverse dep
     * inconsistencies in that node from the clean evaluation, but does not require handling
     * resetting dirty nodes.
     */
    REEVALUATE_LEAF_NODE_TO_FORCE_DIRTY,
    /** Run the clean build without keeping edges. Incremental builds are therefore not possible. */
    NO_KEEP_EDGES_SO_NO_REEVALUATION
  }

  protected void runResetNodeOnRequest_withDeps(RunResetNodeOnRequestWithDepsMode mode)
      throws Exception {
    SkyKey restartingKey = GraphTester.skyKey("restart");
    AtomicInteger numInconsistencyCalls = new AtomicInteger(0);
    tester.setGraphInconsistencyReceiver(
        restartEnabledInconsistencyReceiver(
            (key, otherKey, inconsistency) -> {
              Preconditions.checkState(otherKey == null, otherKey);
              Preconditions.checkState(
                  inconsistency == Inconsistency.RESET_REQUESTED, inconsistency);
              Preconditions.checkState(restartingKey.equals(key), key);
              numInconsistencyCalls.incrementAndGet();
            }));
    tester.initialize(mode != RunResetNodeOnRequestWithDepsMode.NO_KEEP_EDGES_SO_NO_REEVALUATION);
    StringValue expectedValue = new StringValue("done");
    SkyKey alreadyRequestedDep = GraphTester.skyKey("alreadyRequested");
    SkyKey newlyRequestedNotDoneDep = GraphTester.skyKey("newlyRequestedNotDone");
    SkyKey newlyRequestedDoneDep = GraphTester.skyKey("newlyRequestedDone");
    tester
        .getOrCreate(newlyRequestedDoneDep)
        .setConstantValue(new StringValue("newlyRequestedDone"));
    tester
        .getOrCreate(alreadyRequestedDep)
        .addDependency(newlyRequestedDoneDep)
        .setConstantValue(new StringValue("alreadyRequested"));
    tester
        .getOrCreate(newlyRequestedNotDoneDep)
        .setConstantValue(new StringValue("newlyRequestedNotDone"));
    AtomicInteger numFunctionCalls = new AtomicInteger(0);
    AtomicBoolean cleanBuild = new AtomicBoolean(true);
    tester
        .getOrCreate(restartingKey)
        .setBuilder(
            new SkyFunction() {
              @Override
              public SkyValue compute(SkyKey skyKey, Environment env) throws InterruptedException {
                numFunctionCalls.getAndIncrement();
                SkyValue dep1 = env.getValue(alreadyRequestedDep);
                if (dep1 == null) {
                  return null;
                }
                env.getValues(ImmutableList.of(newlyRequestedDoneDep, newlyRequestedNotDoneDep));
                if (numFunctionCalls.get() < 4) {
                  return Restart.SELF;
                } else if (numFunctionCalls.get() == 4) {
                  if (cleanBuild.get()) {
                    Preconditions.checkState(
                        env.valuesMissing(), "Not done dep should never have been enqueued");
                  }
                  return null;
                }
                return expectedValue;
              }

              @Nullable
              @Override
              public String extractTag(SkyKey skyKey) {
                return null;
              }
            });
    assertThat(tester.evalAndGet(/*keepGoing=*/ false, restartingKey)).isEqualTo(expectedValue);
    assertThat(numInconsistencyCalls.get()).isEqualTo(2);
    assertThat(numFunctionCalls.get()).isEqualTo(5);
    switch (mode) {
      case REEVALUATE_TOP_NODE:
        tester
            .getOrCreate(newlyRequestedDoneDep, /*markAsModified=*/ true)
            .setConstantValue(new StringValue("new value"));
        tester.invalidate();
        numInconsistencyCalls.set(0);
        // The dirty restartingKey's deps are checked for a changed dep before it is actually
        // evaluated. It picks up dep1 as a dep during that checking phase, so to keep the
        // SkyFunction in sync, we start numFunctionCalls at 1 instead of 0.
        numFunctionCalls.set(1);
        cleanBuild.set(false);
        assertThat(tester.evalAndGet(/*keepGoing=*/ false, restartingKey)).isEqualTo(expectedValue);
        assertThat(numInconsistencyCalls.get()).isEqualTo(2);
        assertThat(numFunctionCalls.get()).isEqualTo(5);
        return;
      case REEVALUATE_LEAF_NODE_TO_FORCE_DIRTY:
        // Confirm that when a node is marked dirty and its reverse deps are consolidated, we don't
        // crash due to inconsistencies.
        tester.getOrCreate(alreadyRequestedDep, /*markAsModified=*/ true);
        tester.invalidate();
        assertThat(tester.evalAndGet(/*keepGoing=*/ false, alreadyRequestedDep))
            .isEqualTo(new StringValue("alreadyRequested"));
        return;
      case NO_KEEP_EDGES_SO_NO_REEVALUATION:
        return;
    }
    throw new IllegalStateException("Unknown mode " + mode);
  }

  // TODO(mschaller): Enable test with other modes.
  // TODO(janakr): Test would actually pass if there was no invalidation/subsequent re-evaluation
  // because duplicate reverse deps aren't detected until the child is dirtied, which isn't awesome.
  // REEVALUATE_LEAF_NODE_TO_FORCE_DIRTY allows us to check that even clean evaluations with
  // keepEdges are still poisoned.
  @Test
  public void resetNodeOnRequest_withDeps() throws Exception {
    runResetNodeOnRequest_withDeps(
        RunResetNodeOnRequestWithDepsMode.NO_KEEP_EDGES_SO_NO_REEVALUATION);
  }

  private void missingDirtyChild(boolean sameGroup) throws Exception {
    SkyKey topKey = GraphTester.skyKey("top");
    SkyKey missingChild = GraphTester.skyKey("missing");
    AtomicInteger numInconsistencyCalls = new AtomicInteger(0);
    tester.setGraphInconsistencyReceiver(
        restartEnabledInconsistencyReceiver(
            (key, otherKeys, inconsistency) -> {
              Preconditions.checkState(otherKeys.size() == 1, otherKeys);
              Preconditions.checkState(
                  missingChild.equals(Iterables.getOnlyElement(otherKeys)),
                  "%s %s",
                  missingChild,
                  otherKeys);
              Preconditions.checkState(
                  inconsistency == Inconsistency.DIRTY_PARENT_HAD_MISSING_CHILD, inconsistency);
              Preconditions.checkState(topKey.equals(key), key);
              numInconsistencyCalls.incrementAndGet();
            }));
    tester.initialize(/*keepEdges=*/ true);
    tester.getOrCreate(missingChild).setConstantValue(new StringValue("will go missing"));
    SkyKey presentChild = GraphTester.nonHermeticKey("present");
    tester.getOrCreate(presentChild).setConstantValue(new StringValue("present"));
    StringValue topValue = new StringValue("top");
    tester
        .getOrCreate(topKey)
        .setBuilder(
            new SkyFunction() {
              @Nullable
              @Override
              public SkyValue compute(SkyKey skyKey, Environment env) throws InterruptedException {
                if (sameGroup) {
                  env.getValues(ImmutableSet.of(presentChild, missingChild));
                } else {
                  env.getValue(presentChild);
                  if (env.valuesMissing()) {
                    return null;
                  }
                  env.getValue(missingChild);
                }
                return env.valuesMissing() ? null : topValue;
              }

              @Nullable
              @Override
              public String extractTag(SkyKey skyKey) {
                return null;
              }
            });
    assertThat(tester.evalAndGet(/*keepGoing=*/ false, topKey)).isEqualTo(topValue);
    deleteKeyFromGraph(missingChild);
    tester
        .getOrCreate(presentChild, /*markAsModified=*/ true)
        .setConstantValue(new StringValue("changed"));
    tester.invalidate();
    assertThat(tester.evalAndGet(/*keepGoing=*/ false, topKey)).isEqualTo(topValue);
    assertThat(numInconsistencyCalls.get()).isEqualTo(1);
  }

  protected void deleteKeyFromGraph(SkyKey key) {
    ((InMemoryMemoizingEvaluator) tester.evaluator).getGraphForTesting().remove(key);
  }

  @Test
  public void missingDirtyChild_SameGroup() throws Exception {
    missingDirtyChild(/*sameGroup=*/ true);
  }

  @Test
  public void missingDirtyChild_DifferentGroup() throws Exception {
    missingDirtyChild(/*sameGroup=*/ false);
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
    injectGraphListenerForTesting(
        new Listener() {
          @Override
          public void accept(SkyKey key, EventType type, Order order, Object context) {
            if (shouldNotBuildLeaf4.get()
                && key.equals(leaf4)
                && type != EventType.REMOVE_REVERSE_DEP
                && type != EventType.GET_BATCH) {
              throw new IllegalStateException(
                  "leaf4 should not have been considered this build: "
                      + type
                      + ", "
                      + order
                      + ", "
                      + context);
            }
          }
        },
        /*deterministic=*/ false);
    tester.set(leaf4, new StringValue("leaf4"));

    // Create leaf0, leaf1 and leaf2 values with values "leaf2", "leaf3", "leaf4" respectively.
    // These will be requested as one dependency group. In the second build, leaf2 will have the
    // value "leaf5".
    final List<SkyKey> leaves = new ArrayList<>();
    for (int i = 0; i <= 2; i++) {
      SkyKey leaf =
          i == 2 ? GraphTester.nonHermeticKey("leaf" + i) : GraphTester.toSkyKey("leaf" + i);
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
    assertThat(tester.evalAndGet(/*keepGoing=*/ false, topKey)).isEqualTo(topValue);

    // Second build: replace "leaf4" by "leaf5" in leaf2's value. Assert leaf4 is not requested.
    final SkyKey leaf5 = GraphTester.toSkyKey("leaf5");
    tester.set(leaf5, new StringValue("leaf5"));
    tester.set(leaves.get(2), new StringValue("leaf5"));
    tester.invalidate();
    shouldNotBuildLeaf4.set(true);
    assertThat(tester.evalAndGet(/*keepGoing=*/ false, topKey)).isEqualTo(topValue);
  }

  @Test
  public void dirtyAndChanged() throws Exception {
    SkyKey leaf = GraphTester.nonHermeticKey("leaf");
    SkyKey mid = GraphTester.nonHermeticKey("mid");
    SkyKey top = GraphTester.toSkyKey("top");
    tester.getOrCreate(top).addDependency(mid).setComputedValue(COPY);
    tester.getOrCreate(mid).addDependency(leaf).setComputedValue(COPY);
    tester.set(leaf, new StringValue("leafy"));
    // For invalidation.
    tester.set("dummy", new StringValue("dummy"));
    StringValue topValue = (StringValue) tester.evalAndGet("top");
    assertThat(topValue.getValue()).isEqualTo("leafy");
    tester.set(leaf, new StringValue("crunchy"));
    tester.invalidate();
    // For invalidation.
    tester.evalAndGet("dummy");
    tester.getOrCreate(mid, /*markAsModified=*/true);
    tester.invalidate();
    topValue = (StringValue) tester.evalAndGet("top");
    assertThat(topValue.getValue()).isEqualTo("crunchy");
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
    SkyKey parent = GraphTester.nonHermeticKey("parent");
    final AtomicBoolean blockingEnabled = new AtomicBoolean(false);
    final CountDownLatch waitForChanged = new CountDownLatch(1);
    // changed thread checks value entry once (to see if it is changed). dirty thread checks twice,
    // to see if it is changed, and if it is dirty.
    final CountDownLatch threadsStarted = new CountDownLatch(3);
    injectGraphListenerForTesting(
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
              if (order == Order.BEFORE) {
                DirtyType dirtyType = (DirtyType) context;
                if (dirtyType.equals(DirtyType.DIRTY)) {
                  TrackingAwaiter.INSTANCE.awaitLatchAndTrackExceptions(
                      waitForChanged, "'changed' thread did not mark value changed in time");
                  return;
                }
              }
              if (order == Order.AFTER) {
                DirtyType dirtyType = ((NotifyingHelper.MarkDirtyAfterContext) context).dirtyType();
                if (dirtyType.equals(DirtyType.CHANGE)) {
                  waitForChanged.countDown();
                }
              }
            }
          }
        },
        /*deterministic=*/ false);
    SkyKey leaf = GraphTester.nonHermeticKey("leaf");
    tester.set(leaf, new StringValue("leaf"));
    tester.getOrCreate(parent).addDependency(leaf).setComputedValue(CONCATENATE);
    EvaluationResult<StringValue> result;
    result = tester.eval(/*keepGoing=*/false, parent);
    assertThat(result.get(parent).getValue()).isEqualTo("leaf");
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
    assertThat(result.get(parent).getValue()).isEqualTo("leafother2");
    assertThat(waitForChanged.getCount()).isEqualTo(0);
    assertThat(threadsStarted.getCount()).isEqualTo(0);
  }

  @Test
  public void hermeticSkyFunctionCanThrowTransientErrorThenRecover() throws Exception {
    SkyKey leaf = skyKey("leaf");
    SkyKey top = skyKey("top");
    // When top depends on leaf, but throws a transient error,
    tester
        .getOrCreate(top)
        .addDependency(leaf)
        .setHasTransientError(true)
        .setComputedValue(CONCATENATE);
    StringValue value = new StringValue("value");
    tester.getOrCreate(leaf).setConstantValue(value);
    // And the first build throws a transient error (as expected),
    EvaluationResult<StringValue> result = tester.eval(/*keepGoing=*/ true, top);
    assertThatEvaluationResult(result).hasError();
    assertThatEvaluationResult(result).hasErrorEntryForKeyThat(top).hasExceptionThat().isNotNull();
    // And then top's transient error is removed,
    tester.getOrCreate(top, /*markAsModified=*/ false).setHasTransientError(false);
    tester.invalidateTransientErrors();
    // Then top evaluates successfully, even though it was hermetic and didn't give the same result
    // on successive evaluations with the same inputs.
    result = tester.eval(/*keepGoing=*/ true, top);
    assertThatEvaluationResult(result).hasNoError();
    assertThatEvaluationResult(result).hasEntryThat(top).isEqualTo(value);
  }

  @Test
  public void singleValueDependsOnManyDirtyValues() throws Exception {
    SkyKey[] values = new SkyKey[TEST_NODE_COUNT];
    StringBuilder expected = new StringBuilder();
    for (int i = 0; i < values.length; i++) {
      String valueName = Integer.toString(i);
      values[i] = GraphTester.nonHermeticKey(valueName);
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
    assertThat(result.get(topKey)).isEqualTo(new StringValue(expected.toString()));

    for (int j = 0; j < RUNS; j++) {
      for (int i = 0; i < values.length; i++) {
        tester.getOrCreate(values[i], /*markAsModified=*/true);
      }
      // This value has an error, but we should never discover it because it is not marked changed
      // and all of its dependencies re-evaluate to the same thing.
      tester.getOrCreate(topKey, /*markAsModified=*/false).setHasError(true);
      tester.invalidate();

      result = tester.eval(/* keepGoing= */ false, topKey);
      assertThat(result.get(topKey)).isEqualTo(new StringValue(expected.toString()));
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
    SkyKey errorKey = GraphTester.nonHermeticKey("error");
    tester.set(errorKey, new StringValue("biding time"));
    SkyKey slowKey = GraphTester.nonHermeticKey("slow");
    tester.set(slowKey, new StringValue("slow"));
    SkyKey midKey = GraphTester.toSkyKey("mid");
    tester.getOrCreate(midKey).addDependency(slowKey).setComputedValue(COPY);
    SkyKey lastKey = GraphTester.nonHermeticKey("last");
    tester.set(lastKey, new StringValue("last"));
    SkyKey motherKey = GraphTester.toSkyKey("mother");
    tester.getOrCreate(motherKey).addDependency(errorKey)
        .addDependency(midKey).addDependency(lastKey).setComputedValue(CONCATENATE);
    SkyKey fatherKey = GraphTester.toSkyKey("father");
    tester.getOrCreate(fatherKey).addDependency(errorKey)
        .addDependency(midKey).addDependency(lastKey).setComputedValue(CONCATENATE);
    EvaluationResult<StringValue> result = tester.eval(/*keepGoing=*/false, motherKey, fatherKey);
    assertThat(result.get(motherKey).getValue()).isEqualTo("biding timeslowlast");
    assertThat(result.get(fatherKey).getValue()).isEqualTo("biding timeslowlast");
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
    assertThat(result.hasError()).isTrue();
    // Only one of mother or father should be in the graph.
    assertWithMessage(result.getError(motherKey) + ", " + result.getError(fatherKey))
        .that((result.getError(motherKey) == null) != (result.getError(fatherKey) == null))
        .isTrue();
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
      assertThat(result.getError(parentKey)).isNotNull();
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
    SkyKey leafKey = GraphTester.nonHermeticKey("leaf");
    tester.set(leafKey, new StringValue("leafy"));
    SkyKey lastKey = GraphTester.nonHermeticKey("last");
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
      TestThread evalThread =
          new TestThread(
              () ->
                  assertThrows(
                      InterruptedException.class,
                      () -> tester.eval(/*keepGoing=*/ false, tops.toArray(new SkyKey[0]))));
      evalThread.start();
      assertThat(notifyStart.await(TestUtils.WAIT_TIMEOUT_SECONDS, TimeUnit.SECONDS)).isTrue();
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
      assertWithMessage(topKey.toString())
          .that(result.get(topKey).getValue())
          .isEqualTo("crunchynew last");
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

  private SkyKey makeTestKey(SkyKey node0) {
    SkyKey key = null;
    // Create a long chain of nodes. Most of them will not actually be dirtied, but the last one to
    // be dirtied will enqueue its parent for dirtying, so it will be in the queue for the next run.
    for (int i = 0; i < TEST_NODE_COUNT; i++) {
      key = i == 0 ? node0 : GraphTester.toSkyKey("node" + i);
      if (i > 1) {
        tester.getOrCreate(key).addDependency("node" + (i - 1)).setComputedValue(COPY);
      } else if (i == 1) {
        tester.getOrCreate(key).addDependency(node0).setComputedValue(COPY);
      } else {
        tester.set(key, new StringValue("node0"));
      }
    }
    return key;
  }

  /**
   * Regression test for case where the user requests that we delete nodes that are already in the
   * queue to be dirtied. We should handle that gracefully and not complain.
   */
  @Test
  public void deletingDirtyNodes() throws Exception {
    SkyKey node0 = GraphTester.nonHermeticKey("node0");
    SkyKey key = makeTestKey(node0);
    // Seed the graph.
    assertThat(((StringValue) tester.evalAndGet(/*keepGoing=*/ false, key)).getValue())
        .isEqualTo("node0");
    // Start the dirtying process.
    tester.set(node0, new StringValue("new"));
    tester.invalidate();

    // Interrupt current thread on a next invalidate call
    final Thread thread = Thread.currentThread();
    tester.progressReceiver.setNextInvalidationCallback(thread::interrupt);

    assertThrows(InterruptedException.class, () -> tester.eval(/*keepGoing=*/ false, key));

    // Cleanup + paranoid check
    tester.progressReceiver.setNextInvalidationCallback(null);
    // Now delete all the nodes. The node that was going to be dirtied is also deleted, which we
    // should handle.
    tester.evaluator.delete(Predicates.alwaysTrue());
    assertThat(((StringValue) tester.evalAndGet(/*keepGoing=*/ false, key)).getValue())
        .isEqualTo("new");
  }

  @Test
  public void changePruning() throws Exception {
    SkyKey leaf = GraphTester.nonHermeticKey("leaf");
    SkyKey mid = GraphTester.toSkyKey("mid");
    SkyKey top = GraphTester.toSkyKey("top");
    tester.getOrCreate(top).addDependency(mid).setComputedValue(COPY);
    tester.getOrCreate(mid).addDependency(leaf).setComputedValue(COPY);
    tester.set(leaf, new StringValue("leafy"));
    StringValue topValue = (StringValue) tester.evalAndGet("top");
    assertThat(topValue.getValue()).isEqualTo("leafy");
    // Mark leaf changed, but don't actually change it.
    tester.getOrCreate(leaf, /*markAsModified=*/true);
    // mid will give an error if re-evaluated, but it shouldn't be because it is not marked changed,
    // and its dirty child will evaluate to the same element.
    tester.getOrCreate(mid, /*markAsModified=*/false).setHasError(true);
    tester.invalidate();
    EvaluationResult<StringValue> result = tester.eval(/*keepGoing=*/false, top);
    assertThat(result.hasError()).isFalse();
    topValue = result.get(top);
    assertThat(topValue.getValue()).isEqualTo("leafy");
    assertThat(tester.getDirtyKeys()).isEmpty();
    assertThat(tester.getDeletedKeys()).isEmpty();
  }

  @Test
  public void changePruningWithDoneValue() throws Exception {
    SkyKey leaf = GraphTester.nonHermeticKey("leaf");
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
    assertThat(value.getValue()).isEqualTo("leafysuffixsuffix");
    // Mark leaf changed, but don't actually change it.
    tester.getOrCreate(leaf, /*markAsModified=*/true);
    // mid will give an error if re-evaluated, but it shouldn't be because it is not marked changed,
    // and its dirty child will evaluate to the same element.
    tester.getOrCreate(mid, /*markAsModified=*/false).setHasError(true);
    tester.invalidate();
    value = (StringValue) tester.evalAndGet(/*keepGoing=*/ false, leaf);
    assertThat(value.getValue()).isEqualTo("leafy");
    assertThat(tester.getDirtyKeys()).containsExactly(mid, top);
    assertThat(tester.getDeletedKeys()).isEmpty();
    EvaluationResult<StringValue> result = tester.eval(/*keepGoing=*/false, top);
    assertWithMessage(result.toString()).that(result.hasError()).isFalse();
    value = result.get(top);
    assertThat(value.getValue()).isEqualTo("leafysuffixsuffix");
    assertThat(tester.getDirtyKeys()).isEmpty();
    assertThat(tester.getDeletedKeys()).isEmpty();
  }

  @Test
  public void changePruningAfterParentPrunes() throws Exception {
    SkyKey leaf = GraphTester.nonHermeticKey("leaf");
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
              public SkyValue compute(SkyKey skyKey, Environment env) throws InterruptedException {
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
    assertThat(topValue).isEqualTo(fixedTopValue);
    // And top was actually evaluated.
    assertThat(topEvaluated.get()).isTrue();
    // When leaf is changed,
    tester.set(leaf, new StringValue("crunchy"));
    tester.invalidate();
    topEvaluated.set(false);
    // And top is evaluated,
    StringValue topValue2 = (StringValue) tester.evalAndGet("top");
    // Then top's value is as expected,
    assertThat(topValue2).isEqualTo(fixedTopValue);
    // And top was actually evaluated.
    assertThat(topEvaluated.get()).isTrue();
    // When leaf is invalidated but not actually changed,
    tester.getOrCreate(leaf, /*markAsModified=*/ true);
    tester.invalidate();
    topEvaluated.set(false);
    // And top is evaluated,
    StringValue topValue3 = (StringValue) tester.evalAndGet("top");
    // Then top's value is as expected,
    assertThat(topValue3).isEqualTo(fixedTopValue);
    // And top was *not* actually evaluated, because change pruning cut off evaluation.
    assertThat(topEvaluated.get()).isFalse();
  }

  @Test
  public void changePruningFromOtherNodeAfterParentPrunes() throws Exception {
    SkyKey leaf = GraphTester.nonHermeticKey("leaf");
    SkyKey other = GraphTester.nonHermeticKey("other");
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
              public SkyValue compute(SkyKey skyKey, Environment env) throws InterruptedException {
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
    assertThat(topValue).isEqualTo(fixedTopValue);
    // And top was actually evaluated.
    assertThat(topEvaluated.get()).isTrue();
    // When leaf is changed,
    tester.set(leaf, new StringValue("crunchy"));
    tester.invalidate();
    topEvaluated.set(false);
    // And top is evaluated,
    StringValue topValue2 = (StringValue) tester.evalAndGet("top");
    // Then top's value is as expected,
    assertThat(topValue2).isEqualTo(fixedTopValue);
    // And top was actually evaluated.
    assertThat(topEvaluated.get()).isTrue();
    // When other is invalidated but not actually changed,
    tester.getOrCreate(other, /*markAsModified=*/ true);
    tester.invalidate();
    topEvaluated.set(false);
    // And top is evaluated,
    StringValue topValue3 = (StringValue) tester.evalAndGet("top");
    // Then top's value is as expected,
    assertThat(topValue3).isEqualTo(fixedTopValue);
    // And top was *not* actually evaluated, because change pruning cut off evaluation.
    assertThat(topEvaluated.get()).isFalse();
  }

  @Test
  public void changedChildChangesDepOfParent() throws Exception {
    SkyKey buildFile = GraphTester.nonHermeticKey("buildFile");
    ValueComputer authorDrink =
        new ValueComputer() {
          @Override
          public SkyValue compute(Map<SkyKey, SkyValue> deps, SkyFunction.Environment env)
              throws InterruptedException {
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
    assertThat(topValue.getValue()).isEqualTo("hemingway drank absinthe");
    tester.set(buildFile, new StringValue("joyce"));
    // Don't evaluate absinthe successfully anymore.
    tester.getOrCreate(absinthe, /*markAsModified=*/ false).setHasError(true);
    tester.invalidate();
    topValue = (StringValue) tester.evalAndGet("top");
    assertThat(topValue.getValue()).isEqualTo("joyce drank whiskey");
    if (preciseEvaluationStatusStored()) {
      assertThat(tester.getDirtyKeys()).containsExactly(buildFile, top);
      assertThat(tester.getDeletedKeys()).isEmpty();
    }
  }

  @Test
  public void dirtyDepIgnoresChildren() throws Exception {
    SkyKey leaf = GraphTester.nonHermeticKey("leaf");
    SkyKey mid = GraphTester.toSkyKey("mid");
    SkyKey top = GraphTester.toSkyKey("top");
    tester.set(mid, new StringValue("ignore"));
    tester.getOrCreate(top).addDependency(mid).setComputedValue(COPY);
    tester.getOrCreate(mid).addDependency(leaf);
    tester.set(leaf, new StringValue("leafy"));
    StringValue topValue = (StringValue) tester.evalAndGet("top");
    assertThat(topValue.getValue()).isEqualTo("ignore");
    assertThat(tester.getDirtyKeys()).isEmpty();
    assertThat(tester.getDeletedKeys()).isEmpty();
    // Change leaf.
    tester.set(leaf, new StringValue("crunchy"));
    tester.invalidate();
    topValue = (StringValue) tester.evalAndGet("top");
    assertThat(topValue.getValue()).isEqualTo("ignore");
    if (preciseEvaluationStatusStored()) {
      assertThat(tester.getDirtyKeys()).containsExactly(leaf);
      assertThat(tester.getDeletedKeys()).isEmpty();
    }
    tester.set(leaf, new StringValue("smushy"));
    tester.invalidate();
    topValue = (StringValue) tester.evalAndGet("top");
    assertThat(topValue.getValue()).isEqualTo("ignore");
    if (preciseEvaluationStatusStored()) {
      assertThat(tester.getDirtyKeys()).containsExactly(leaf);
      assertThat(tester.getDeletedKeys()).isEmpty();
    }
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
    assertThrows(InterruptedException.class, () -> tester.eval(/*keepGoing=*/ false, value));
    tester.getOrCreate(value, /*markAsModified=*/false).setBuilder(null);
  }

  /**
   * Make sure that when a dirty value is building, the fact that a child may no longer exist in the
   * graph doesn't cause problems.
   */
  @Test
  public void dirtyBuildAfterFailedBuild() throws Exception {
    SkyKey leaf = GraphTester.nonHermeticKey("leaf");
    SkyKey top = GraphTester.toSkyKey("top");
    tester.getOrCreate(top).addDependency(leaf).setComputedValue(COPY);
    tester.set(leaf, new StringValue("leafy"));
    StringValue topValue = (StringValue) tester.evalAndGet("top");
    assertThat(topValue.getValue()).isEqualTo("leafy");
    assertThat(tester.getDirtyKeys()).isEmpty();
    assertThat(tester.getDeletedKeys()).isEmpty();
    failBuildAndRemoveValue(leaf);
    // Leaf should no longer exist in the graph. Check that this doesn't cause problems.
    tester.set(leaf, null);
    tester.set(leaf, new StringValue("crunchy"));
    tester.invalidate();
    topValue = (StringValue) tester.evalAndGet("top");
    assertThat(topValue.getValue()).isEqualTo("crunchy");
  }

  /**
   * Regression test: error when clearing reverse deps on dirty value about to be rebuilt, because
   * child values were deleted and recreated in interim, forgetting they had reverse dep on dirty
   * value in the first place.
   */
  @Test
  public void changedBuildAfterFailedThenSuccessfulBuild() throws Exception {
    SkyKey leaf = GraphTester.nonHermeticKey("leaf");
    SkyKey top = GraphTester.nonHermeticKey("top");
    tester.getOrCreate(top).addDependency(leaf).setComputedValue(COPY);
    tester.set(leaf, new StringValue("leafy"));
    StringValue topValue = (StringValue) tester.evalAndGet(/*keepGoing=*/ false, top);
    assertThat(topValue.getValue()).isEqualTo("leafy");
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
    topValue = (StringValue) tester.evalAndGet(/*keepGoing=*/ false, top);
    assertThat(topValue.getValue()).isEqualTo("crunchy");
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
    SkyKey leafKey = GraphTester.nonHermeticKey("leaf");
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
    assertThrows(
        InterruptedException.class,
        () -> tester.eval(/*keepGoing=*/ false, tops.toArray(new SkyKey[0])));
  }

  @Test
  public void failedDirtyBuild() throws Exception {
    SkyKey leaf = GraphTester.nonHermeticKey("leaf");
    SkyKey top = GraphTester.toSkyKey("top");
    tester.getOrCreate(top).addErrorDependency(leaf, new StringValue("recover"))
        .setComputedValue(COPY);
    tester.set(leaf, new StringValue("leafy"));
    StringValue topValue = (StringValue) tester.evalAndGet("top");
    assertThat(topValue.getValue()).isEqualTo("leafy");
    assertThat(tester.getDirtyKeys()).isEmpty();
    assertThat(tester.getDeletedKeys()).isEmpty();
    // Change leaf.
    tester.getOrCreate(leaf, /*markAsModified=*/true).setHasError(true);
    tester.getOrCreate(top, /*markAsModified=*/false).setHasError(true);
    tester.invalidate();
    EvaluationResult<StringValue> result = tester.eval(/*keepGoing=*/false, top);
    assertThatEvaluationResult(result).hasSingletonErrorThat(top);
  }

  @Test
  public void failedDirtyBuildInBuilder() throws Exception {
    SkyKey leaf = GraphTester.nonHermeticKey("leaf");
    SkyKey secondError = GraphTester.nonHermeticKey("secondError");
    SkyKey top = GraphTester.toSkyKey("top");
    tester.getOrCreate(top).addDependency(leaf)
        .addErrorDependency(secondError, new StringValue("recover")).setComputedValue(CONCATENATE);
    tester.set(secondError, new StringValue("secondError")).addDependency(leaf);
    tester.set(leaf, new StringValue("leafy"));
    StringValue topValue = (StringValue) tester.evalAndGet("top");
    assertThat(topValue.getValue()).isEqualTo("leafysecondError");
    assertThat(tester.getDirtyKeys()).isEmpty();
    assertThat(tester.getDeletedKeys()).isEmpty();
    // Invalidate leaf.
    tester.getOrCreate(leaf, /*markAsModified=*/true);
    tester.set(leaf, new StringValue("crunchy"));
    tester.getOrCreate(secondError, /*markAsModified=*/true).setHasError(true);
    tester.getOrCreate(top, /*markAsModified=*/false).setHasError(true);
    tester.invalidate();
    EvaluationResult<StringValue> result = tester.eval(/*keepGoing=*/false, top);
    assertThatEvaluationResult(result).hasSingletonErrorThat(top);
  }

  @Test
  public void dirtyErrorTransienceValue() throws Exception {
    initializeTester();
    SkyKey error = GraphTester.toSkyKey("error");
    tester.getOrCreate(error).setHasError(true);
    assertThat(tester.evalAndGetError(/*keepGoing=*/ true, error)).isNotNull();
    tester.invalidateTransientErrors();
    SkyKey secondError = GraphTester.toSkyKey("secondError");
    tester.getOrCreate(secondError).setHasError(true);
    // secondError declares a new dependence on ErrorTransienceValue, but not until it has already
    // thrown an error.
    assertThat(tester.evalAndGetError(/*keepGoing=*/ true, secondError)).isNotNull();
  }

  @Test
  public void dirtyDependsOnErrorTurningGood() throws Exception {
    SkyKey error = GraphTester.nonHermeticKey("error");
    tester.getOrCreate(error).setHasError(true);
    SkyKey topKey = GraphTester.toSkyKey("top");
    tester.getOrCreate(topKey).addDependency(error).setComputedValue(COPY);
    EvaluationResult<StringValue> result = tester.eval(/*keepGoing=*/false, topKey);
    assertThatEvaluationResult(result).hasSingletonErrorThat(topKey);
    tester.getOrCreate(error).setHasError(false);
    StringValue val = new StringValue("reformed");
    tester.set(error, val);
    tester.invalidate();
    result = tester.eval(/*keepGoing=*/false, topKey);
    assertThatEvaluationResult(result).hasEntryThat(topKey).isEqualTo(val);
    assertThatEvaluationResult(result).hasNoError();
  }

  /** Regression test for crash bug. */
  @Test
  public void dirtyWithOwnErrorDependsOnTransientErrorTurningGood() throws Exception {
    SkyKey error = GraphTester.nonHermeticKey("error");
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
    assertThatEvaluationResult(result).hasSingletonErrorThat(topKey);
    tester.getOrCreate(error).setHasTransientError(false);
    StringValue reformed = new StringValue("reformed");
    tester.set(error, reformed);
    tester
        .getOrCreate(topKey, /*markAsModified=*/ false)
        .setBuilder(null)
        .addDependency(error)
        .setComputedValue(COPY);
    tester.invalidate();
    tester.invalidateTransientErrors();
    result = tester.eval(/*keepGoing=*/false, topKey);
    assertThatEvaluationResult(result).hasEntryThat(topKey).isEqualTo(reformed);
    assertThatEvaluationResult(result).hasNoError();
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
    makeGraphDeterministic();
    SkyKey errorKey = GraphTester.nonHermeticKey("error");
    tester.set(errorKey, new StringValue("biding time"));
    SkyKey slowKey = GraphTester.nonHermeticKey("slow");
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
    assertThat(result.get(topErrorFirstKey).getValue()).isEqualTo("biding time");
    assertThat(result.get(topBubbleKey).getValue()).isEqualTo("slowbiding time");
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
    assertThat(result.hasError()).isTrue();
    assertWithMessage(result.toString()).that(result.getError(topErrorFirstKey)).isNotNull();
  }

  @Test
  public void dirtyWithRecoveryErrorDependsOnErrorTurningGood() throws Exception {
    final SkyKey error = GraphTester.nonHermeticKey("error");
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
    assertThatEvaluationResult(result).hasSingletonErrorThat(topKey);
    tester.getOrCreate(error).setHasError(false);
    StringValue reformed = new StringValue("reformed");
    tester.set(error, reformed);
    tester.getOrCreate(topKey).setBuilder(null).addDependency(error).setComputedValue(COPY);
    tester.invalidate();
    result = tester.eval(/*keepGoing=*/false, topKey);
    assertThatEvaluationResult(result).hasEntryThat(topKey).isEqualTo(reformed);
    assertThatEvaluationResult(result).hasNoError();
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
    assertThatEvaluationResult(result).hasErrorEntryForKeyThat(midKey);
    // Do it again with keepGoing.  We should also see an error for the top key this time.
    result = tester.eval(/*keepGoing=*/ true, topKey, midKey);
    assertThatEvaluationResult(result).hasErrorEntryForKeyThat(midKey);
    assertThatEvaluationResult(result).hasErrorEntryForKeyThat(topKey);
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
    assertThatEvaluationResult(result).hasErrorMapThat().hasSize(1);
    assertThatEvaluationResult(result).hasErrorMapThat().containsKey(parentKey);
    assertThat(Thread.interrupted()).isFalse();
    result = tester.eval(/*keepGoing=*/ true, parentKey);
    assertThatEvaluationResult(result).hasNoError();
    assertThatEvaluationResult(result)
        .hasEntryThat(parentKey)
        .isEqualTo(new StringValue("recovered"));
  }

  /**
   * Regression test: "clearing incomplete values on --keep_going build is racy".
   * Tests that if a value is requested on the first (non-keep-going) build and its child throws
   * an error, when the second (keep-going) build runs, there is not a race that keeps it as a
   * reverse dep of its children.
   */
  @Test
  public void raceClearingIncompleteValues() throws Exception {
    // Make sure top is enqueued before mid, to avoid a deadlock.
    SkyKey topKey = GraphTester.toSkyKey("aatop");
    final SkyKey midKey = GraphTester.toSkyKey("zzmid");
    SkyKey badKey = GraphTester.toSkyKey("bad");
    final AtomicBoolean waitForSecondCall = new AtomicBoolean(false);
    final CountDownLatch otherThreadWinning = new CountDownLatch(1);
    final AtomicReference<Thread> firstThread = new AtomicReference<>();
    injectGraphListenerForTesting(
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
        },
        /*deterministic=*/ true);
    tester.getOrCreate(topKey).addDependency(midKey).setComputedValue(CONCATENATE);
    tester.getOrCreate(midKey).addDependency(badKey).setComputedValue(CONCATENATE);
    tester.getOrCreate(badKey).setHasError(true);
    EvaluationResult<SkyValue> result = tester.eval(/*keepGoing=*/ false, topKey, midKey);
    assertThatEvaluationResult(result).hasErrorEntryForKeyThat(midKey);
    waitForSecondCall.set(true);
    result = tester.eval(/*keepGoing=*/ true, topKey, midKey);
    assertThat(firstThread.get()).isNotNull();
    assertThat(otherThreadWinning.getCount()).isEqualTo(0);
    assertThatEvaluationResult(result).hasErrorEntryForKeyThat(midKey);
    assertThatEvaluationResult(result).hasErrorEntryForKeyThat(topKey);
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
    assertThatEvaluationResult(result).hasSingletonErrorThat(parentKey);
    result = tester.eval(/*keepGoing=*/ true, parentKey);
    assertThatEvaluationResult(result).hasNoError();
    assertThatEvaluationResult(result)
        .hasEntryThat(parentKey)
        .isEqualTo(new StringValue("recoveredafter"));
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
              public SkyValue compute(SkyKey skyKey, Environment env)
                  throws SkyFunctionException, InterruptedException {
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
    injectGraphListenerForTesting(
        new Listener() {
          @Override
          public void accept(SkyKey key, EventType type, Order order, Object context) {
            if (key.equals(errorKey) && type == EventType.SET_VALUE && order == Order.AFTER) {
              errorCommitted.countDown();
              TrackingAwaiter.INSTANCE.awaitLatchAndTrackExceptions(
                  otherDone, "otherErrorKey's SkyFunction didn't finish in time.");
            }
          }
        },
        /*deterministic=*/ false);

    // When the graph is evaluated in noKeepGoing mode,
    EvaluationResult<StringValue> result =
        tester.eval(/*keepGoing=*/ false, errorKey, otherErrorKey);

    // Then the result reports that an error occurred,
    assertThat(result.hasError()).isTrue();

    // And no value is committed for otherErrorKey,
    assertThat(tester.driver.getExistingErrorForTesting(otherErrorKey)).isNull();
    assertThat(tester.driver.getExistingValueForTesting(otherErrorKey)).isNull();

    // And no value was committed for errorKey,
    assertWithMessage(nonNullValueMessage.get()).that(nonNullValueMessage.get()).isNull();

    // And the SkyFunction for otherErrorKey was evaluated exactly once.
    assertThat(numOtherInvocations.get()).isEqualTo(1);
    assertWithMessage(bogusInvocationMessage.get()).that(bogusInvocationMessage.get()).isNull();

    // NB: The SkyFunction for otherErrorKey gets evaluated exactly once--it does not get
    // re-evaluated during error bubbling. Why? When otherErrorKey throws, it is always the
    // second error encountered, because it waited for errorKey's error to be committed before
    // trying to get it. In fail-fast evaluations only the first failing SkyFunction's
    // newly-discovered-dependencies are registered. Therefore, there won't be a reverse-dep from
    // errorKey to otherErrorKey for the error to bubble through.
  }

  @Test
  public void absentParent() throws Exception {
    SkyKey errorKey = GraphTester.nonHermeticKey("my_error_value");
    tester.set(errorKey, new StringValue("biding time"));
    SkyKey absentParentKey = GraphTester.toSkyKey("absentParent");
    tester.getOrCreate(absentParentKey).addDependency(errorKey).setComputedValue(CONCATENATE);
    assertThat(tester.evalAndGet(/*keepGoing=*/ false, absentParentKey))
        .isEqualTo(new StringValue("biding time"));
    tester.getOrCreate(errorKey, /*markAsModified=*/true).setHasError(true);
    SkyKey newParent = GraphTester.toSkyKey("newParent");
    tester.getOrCreate(newParent).addDependency(errorKey).setComputedValue(CONCATENATE);
    tester.invalidate();
    EvaluationResult<StringValue> result = tester.eval(/*keepGoing=*/false, newParent);
    assertThatEvaluationResult(result).hasSingletonErrorThat(newParent);
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
    SkyKey parent = GraphTester.toSkyKey("parent");
    SkyKey child = GraphTester.nonHermeticKey("child");
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
        .setComputedValue(
            (deps, env) -> {
              parentEvaluated.incrementAndGet();
              return new StringValue(val);
            });
    assertStringValue(val, tester.evalAndGet( /*keepGoing=*/false, parent));
    assertThat(parentEvaluated.get()).isEqualTo(1);
    if (hasEvent) {
      assertThatEvents(eventCollector).containsExactly("shmoop");
    } else {
      assertThatEvents(eventCollector).isEmpty();
    }
    eventCollector.clear();

    tester.resetPlayedEvents();
    tester.getOrCreate(child, /*markAsModified=*/true);
    tester.invalidate();
    assertStringValue(val, tester.evalAndGet( /*keepGoing=*/false, parent));
    assertThat(parentEvaluated.get()).isEqualTo(2);
    if (hasEvent) {
      assertThatEvents(eventCollector).containsExactly("shmoop");
    } else {
      assertThatEvents(eventCollector).isEmpty();
    }
  }

  private static void assertStringValue(String expected, SkyValue val) {
    assertThat(((StringValue) val).getValue()).isEqualTo(expected);
  }

  @Test
  public void changePruningWithEvent() throws Exception {
    SkyKey parent = GraphTester.toSkyKey("parent");
    SkyKey child = GraphTester.nonHermeticKey("child");
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
    assertThatEvents(eventCollector).containsExactly("bloop");
    eventCollector.clear();
    tester.resetPlayedEvents();
    tester.getOrCreate(child, /*markAsModified=*/ true);
    tester.invalidate();
    assertThat(tester.evalAndGet( /*keepGoing=*/false, parent)).isEqualTo(parentVal);
    assertThatEvents(eventCollector).containsExactly("bloop");
    assertThat(parentEvaluated.getCount()).isEqualTo(1);
  }

  @Test
  public void changePruningWithIntermittentEvent() throws Exception {
    String parentEvent = "parent_event";
    String waitEvent = "wait_event";
    String childEvent = "child_event";
    SkyKey wait = GraphTester.toSkyKey("wait_key");
    SkyKey parent = GraphTester.toSkyKey("parent_key");
    SkyKey child = GraphTester.nonHermeticKey("child_key");
    StringValue parentStringValue = new StringValue("parent_value");
    StringValue waitStringValue = new StringValue("wait_value");
    CountDownLatch parentEvaluated = new CountDownLatch(2);

    reporter =
        new DelegatingEventHandler(reporter) {
          @Override
          public void handle(Event e) {
            super.handle(e);
            // Release the CountDownLatch every time the parent node fires the event
            if (e.getMessage().equals(parentEvent)) {
              parentEvaluated.countDown();
            }
          }
        };

    tester
        .getOrCreate(wait)
        .setBuilder(
            new SkyFunction() {
              @Override
              public SkyValue compute(SkyKey skyKey, Environment env) throws InterruptedException {
                // Wait for the parent and child actions to complete before computing wait node
                parentEvaluated.await(TestUtils.WAIT_TIMEOUT_SECONDS, TimeUnit.SECONDS);
                assertThatEvents(eventCollector).containsExactly(childEvent, parentEvent);

                env.getListener().handle(Event.progress(waitEvent));
                return waitStringValue;
              }

              @Nullable
              @Override
              public String extractTag(SkyKey skyKey) {
                return null;
              }
            });
    tester
        .getOrCreate(child)
        .setConstantValue(new StringValue("child_value"))
        .setWarning(childEvent);
    tester
        .getOrCreate(parent)
        .addDependency(child)
        .setConstantValue(parentStringValue)
        .setWarning(parentEvent);

    assertThat(tester.evalAndGet(/*keepGoing=*/ false, parent)).isEqualTo(parentStringValue);
    assertThatEvents(eventCollector).containsExactly(childEvent, parentEvent);
    assertThat(parentEvaluated.getCount()).isEqualTo(1);

    // Reset the event collector and mark the child as modified without actually changing values
    eventCollector.clear();
    tester.resetPlayedEvents();
    tester.getOrCreate(child, /*markAsModified=*/ true);
    tester.invalidate();

    EvaluationResult<StringValue> result = tester.eval(false, parent, wait);
    assertThat(result.values()).containsExactly(parentStringValue, waitStringValue);

    // These assertions are to check that all events fired at the end of evaluation.
    assertThat(parentEvaluated.getCount()).isEqualTo(0);
    assertThatEvents(eventCollector).containsExactly(childEvent, parentEvent, waitEvent);
  }

  @Test
  public void depEventPredicate() throws Exception {
    if (!eventsStored()) {
      return;
    }
    SkyKey parent = GraphTester.toSkyKey("parent");
    SkyKey excludedDep = GraphTester.toSkyKey("excludedDep");
    SkyKey includedDep = GraphTester.toSkyKey("includedDep");
    tester.setEventFilter(
        new EventFilter() {
          @Override
          public boolean storeEventsAndPosts() {
            return true;
          }

          @Override
          public boolean apply(Event input) {
            return true;
          }

          @Override
          public Predicate<SkyKey> depEdgeFilterForEventsAndPosts(SkyKey primaryKey) {
            if (primaryKey.equals(parent)) {
              return includedDep::equals;
            } else {
              return Predicates.alwaysTrue();
            }
          }
        });
    tester.initialize(/*keepEdges=*/ true);
    tester
        .getOrCreate(parent)
        .addDependency(excludedDep)
        .addDependency(includedDep)
        .setComputedValue(CONCATENATE);
    tester
        .getOrCreate(excludedDep)
        .setWarning("excludedDep warning")
        .setConstantValue(new StringValue("excludedDep"));
    tester
        .getOrCreate(includedDep)
        .setWarning("includedDep warning")
        .setConstantValue(new StringValue("includedDep"));
    tester.eval(/*keepGoing=*/ false, includedDep, excludedDep);
    assertThatEvents(eventCollector).containsExactly("excludedDep warning", "includedDep warning");
    eventCollector.clear();
    emittedEventState.clear();
    tester.eval(/*keepGoing=*/ true, parent);
    assertThatEvents(eventCollector).containsExactly("includedDep warning");
    assertThat(
            ValueWithMetadata.getEvents(
                    tester.driver.getEntryForTesting(parent).getValueMaybeWithMetadata())
                .toList())
        .containsExactly(
            new TaggedEvents(null, ImmutableList.of(Event.warn("includedDep warning"))));
  }

  // Tests that we have a sane implementation of error transience.
  @Test
  public void errorTransienceBug() throws Exception {
    tester.getOrCreate("key").setHasTransientError(true);
    assertThat(tester.evalAndGetError(/*keepGoing=*/ true, "key").getException()).isNotNull();
    StringValue value = new StringValue("hi");
    tester.getOrCreate("key").setHasTransientError(false).setConstantValue(value);
    tester.invalidateTransientErrors();
    assertThat(tester.evalAndGet("key")).isEqualTo(value);
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
    ErrorInfo errorInfo = tester.evalAndGetError(/*keepGoing=*/ true, errorKey);
    assertThat(errorInfo).isNotNull();
    // Re-evaluates to same thing when errors are invalidated
    tester.invalidateTransientErrors();
    errorInfo = tester.evalAndGetError(/*keepGoing=*/ true, errorKey);
    assertThat(errorInfo).isNotNull();
    StringValue value = new StringValue("reformed");
    tester.getOrCreate(errorKey, /*markAsModified=*/false).setHasTransientError(false)
        .setConstantValue(value);
    tester.invalidateTransientErrors();
    StringValue stringValue = (StringValue) tester.evalAndGet(/*keepGoing=*/true, errorKey);
    assertThat(value).isSameInstanceAs(stringValue);
    // Value builder will now throw, but we should never get to it because it isn't dirty.
    tester.getOrCreate(errorKey, /*markAsModified=*/false).setHasTransientError(true);
    tester.invalidateTransientErrors();
    stringValue = (StringValue) tester.evalAndGet(/*keepGoing=*/true, errorKey);
    assertThat(stringValue).isEqualTo(value);
  }

  @Test
  public void transientErrorTurnsGoodOnSecondTry() throws Exception {
    SkyKey leafKey = toSkyKey("leaf");
    SkyKey errorKey = toSkyKey("error");
    SkyKey topKey = toSkyKey("top");
    StringValue value = new StringValue("val");
    tester.getOrCreate(topKey).addDependency(errorKey).setConstantValue(value);
    tester
        .getOrCreate(errorKey)
        .addDependency(leafKey)
        .setConstantValue(value)
        .setHasTransientError(true);
    tester.getOrCreate(leafKey).setConstantValue(new StringValue("leaf"));
    ErrorInfo errorInfo = tester.evalAndGetError(/*keepGoing=*/ true, topKey);
    assertThat(errorInfo).isNotNull();
    assertThatErrorInfo(errorInfo).isTransient();
    tester.invalidateTransientErrors();
    errorInfo = tester.evalAndGetError(/*keepGoing=*/ true, topKey);
    assertThat(errorInfo).isNotNull();
    assertThatErrorInfo(errorInfo).isTransient();
    tester.invalidateTransientErrors();
    tester.getOrCreate(errorKey, /*markAsModified=*/ false).setHasTransientError(false);
    assertThat(tester.evalAndGet(/*keepGoing=*/ true, topKey)).isEqualTo(value);
  }

  @Test
  public void deleteInvalidatedValue() throws Exception {
    SkyKey top = GraphTester.toSkyKey("top");
    SkyKey toDelete = GraphTester.nonHermeticKey("toDelete");
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
      leftValues[i] = GraphTester.nonHermeticKey("left-" + i);
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

    SkyKey lastLeft = GraphTester.nonHermeticKey("left-" + (TEST_NODE_COUNT - 1));
    SkyKey lastRight = GraphTester.skyKey("right-" + (TEST_NODE_COUNT - 1));

    for (int i = 0; i < TESTED_NODES; i++) {
      try {
        tester.getOrCreate(leftValues[i], /*markAsModified=*/true).setHasError(true);
        tester.invalidate();
        EvaluationResult<StringValue> result =
            tester.eval(/* keepGoing= */ false, lastLeft, lastRight);
        assertThat(result.hasError()).isTrue();
        tester.differencer.invalidate(ImmutableList.of(leftValues[i]));
        tester.invalidate();
        result = tester.eval(/* keepGoing= */ false, lastLeft, lastRight);
        assertThat(result.hasError()).isTrue();
        tester.getOrCreate(leftValues[i], /*markAsModified=*/true).setHasError(false);
        tester.invalidate();
        result = tester.eval(/* keepGoing= */ false, lastLeft, lastRight);
        assertThat(result.get(lastLeft)).isEqualTo(new StringValue("leaf"));
        assertThat(result.get(lastRight)).isEqualTo(new StringValue("leaf"));
      } catch (Exception e) {
        System.err.println("twoRailLeftRightDependenciesWithFailure exception on run " + i);
        throw e;
      }
    }
  }

  @Test
  public void valueInjection() throws Exception {
    SkyKey key = GraphTester.nonHermeticKey("new_value");
    SkyValue val = new StringValue("val");

    tester.differencer.inject(ImmutableMap.of(key, val));
    assertThat(tester.evalAndGet(/*keepGoing=*/ false, key)).isEqualTo(val);
  }

  @Test
  public void valueInjectionOverExistingEntry() throws Exception {
    SkyKey key = GraphTester.nonHermeticKey("key");
    SkyValue val = new StringValue("val");

    tester.getOrCreate(key).setConstantValue(new StringValue("old_val"));
    tester.differencer.inject(ImmutableMap.of(key, val));
    assertThat(tester.evalAndGet(/*keepGoing=*/ false, key)).isEqualTo(val);
  }

  @Test
  public void valueInjectionOverExistingDirtyEntry() throws Exception {
    SkyKey key = GraphTester.nonHermeticKey("key");
    SkyValue val = new StringValue("val");

    tester.getOrCreate(key).setConstantValue(new StringValue("old_val"));
    tester.differencer.inject(ImmutableMap.of(key, val));
    tester.eval(/*keepGoing=*/false, new SkyKey[0]); // Create the value.

    tester.differencer.invalidate(ImmutableList.of(key));
    tester.eval(/*keepGoing=*/false, new SkyKey[0]); // Mark value as dirty.

    tester.differencer.inject(ImmutableMap.of(key, val));
    tester.eval(/*keepGoing=*/false, new SkyKey[0]); // Inject again.
    assertThat(tester.evalAndGet(/*keepGoing=*/ false, key)).isEqualTo(val);
  }

  @Test
  public void valueInjectionOverExistingEntryMarkedForInvalidation() throws Exception {
    SkyKey key = GraphTester.nonHermeticKey("key");
    SkyValue val = new StringValue("val");

    tester.getOrCreate(key).setConstantValue(new StringValue("old_val"));
    tester.differencer.invalidate(ImmutableList.of(key));
    tester.differencer.inject(ImmutableMap.of(key, val));
    assertThat(tester.evalAndGet(/*keepGoing=*/ false, key)).isEqualTo(val);
  }

  @Test
  public void valueInjectionOverExistingEntryMarkedForDeletion() throws Exception {
    SkyKey key = GraphTester.nonHermeticKey("key");
    SkyValue val = new StringValue("val");

    tester.getOrCreate(key).setConstantValue(new StringValue("old_val"));
    tester.evaluator.delete(Predicates.<SkyKey>alwaysTrue());
    tester.differencer.inject(ImmutableMap.of(key, val));
    assertThat(tester.evalAndGet(/*keepGoing=*/ false, key)).isEqualTo(val);
  }

  @Test
  public void valueInjectionOverExistingEqualEntryMarkedForInvalidation() throws Exception {
    SkyKey key = GraphTester.nonHermeticKey("key");
    SkyValue val = new StringValue("val");

    tester.differencer.inject(ImmutableMap.of(key, val));
    assertThat(tester.evalAndGet(/*keepGoing=*/ false, key)).isEqualTo(val);

    tester.differencer.invalidate(ImmutableList.of(key));
    tester.differencer.inject(ImmutableMap.of(key, val));
    assertThat(tester.evalAndGet(/*keepGoing=*/ false, key)).isEqualTo(val);
  }

  @Test
  public void valueInjectionOverExistingEqualEntryMarkedForDeletion() throws Exception {
    SkyKey key = GraphTester.nonHermeticKey("key");
    SkyValue val = new StringValue("val");

    tester.differencer.inject(ImmutableMap.of(key, val));
    assertThat(tester.evalAndGet(/*keepGoing=*/ false, key)).isEqualTo(val);

    tester.evaluator.delete(Predicates.<SkyKey>alwaysTrue());
    tester.differencer.inject(ImmutableMap.of(key, val));
    assertThat(tester.evalAndGet(/*keepGoing=*/ false, key)).isEqualTo(val);
  }

  @Test
  public void valueInjectionOverValueWithDeps() throws Exception {
    SkyKey key = GraphTester.nonHermeticKey("key");
    SkyKey otherKey = GraphTester.nonHermeticKey("other");
    SkyValue val = new StringValue("val");
    StringValue prevVal = new StringValue("foo");

    tester.getOrCreate(otherKey).setConstantValue(prevVal);
    tester.getOrCreate(key).addDependency(otherKey).setComputedValue(COPY);
    assertThat(tester.evalAndGet(/*keepGoing=*/ false, key)).isEqualTo(prevVal);
    tester.differencer.inject(ImmutableMap.of(key, val));
    StringValue depVal = new StringValue("newfoo");
    tester.getOrCreate(otherKey).setConstantValue(depVal);
    tester.differencer.invalidate(ImmutableList.of(otherKey));
    // Injected value is ignored for value with deps.
    assertThat(tester.evalAndGet(/*keepGoing=*/ false, key)).isEqualTo(depVal);
  }

  @Test
  public void valueInjectionOverEqualValueWithDeps() throws Exception {
    SkyKey key = GraphTester.nonHermeticKey("key");
    SkyValue val = new StringValue("val");

    tester.getOrCreate("other").setConstantValue(val);
    tester.getOrCreate(key).addDependency("other").setComputedValue(COPY);
    assertThat(tester.evalAndGet(/*keepGoing=*/ false, key)).isEqualTo(val);
    tester.differencer.inject(ImmutableMap.of(key, val));
    assertThat(tester.evalAndGet(/*keepGoing=*/ false, key)).isEqualTo(val);
  }

  @Test
  public void valueInjectionOverValueWithErrors() throws Exception {
    SkyKey key = GraphTester.nonHermeticKey("key");
    SkyValue val = new StringValue("val");

    tester.getOrCreate(key).setHasError(true);
    tester.evalAndGetError(/*keepGoing=*/ true, key);

    tester.differencer.inject(ImmutableMap.of(key, val));
    assertThat(tester.evalAndGet(false, key)).isEqualTo(val);
  }

  @Test
  public void valueInjectionInvalidatesReverseDeps() throws Exception {
    SkyKey childKey = GraphTester.nonHermeticKey("child");
    SkyKey parentKey = GraphTester.toSkyKey("parent");
    StringValue oldVal = new StringValue("old_val");

    tester.getOrCreate(childKey).setConstantValue(oldVal);
    tester.getOrCreate(parentKey).addDependency(childKey).setComputedValue(COPY);

    EvaluationResult<SkyValue> result = tester.eval(false, parentKey);
    assertThat(result.hasError()).isFalse();
    assertThat(result.get(parentKey)).isEqualTo(oldVal);

    SkyValue val = new StringValue("val");
    tester.differencer.inject(ImmutableMap.of(childKey, val));
    assertThat(tester.evalAndGet(/*keepGoing=*/ false, childKey)).isEqualTo(val);
    // Injecting a new child should have invalidated the parent.
    assertThat(tester.getExistingValue("parent")).isNull();

    tester.eval(false, childKey);
    assertThat(tester.getExistingValue(childKey)).isEqualTo(val);
    assertThat(tester.getExistingValue("parent")).isNull();
    assertThat(tester.evalAndGet("parent")).isEqualTo(val);
  }

  @Test
  public void valueInjectionOverExistingEqualEntryDoesNotInvalidate() throws Exception {
    SkyKey childKey = GraphTester.nonHermeticKey("child");
    SkyKey parentKey = GraphTester.toSkyKey("parent");
    SkyValue val = new StringValue("same_val");

    tester.getOrCreate(parentKey).addDependency(childKey).setComputedValue(COPY);
    tester.getOrCreate(childKey).setConstantValue(new StringValue("same_val"));
    assertThat(tester.evalAndGet("parent")).isEqualTo(val);

    tester.differencer.inject(ImmutableMap.of(childKey, val));
    assertThat(tester.getExistingValue(childKey)).isEqualTo(val);
    // Since we are injecting an equal value, the parent should not have been invalidated.
    assertThat(tester.getExistingValue("parent")).isEqualTo(val);
  }

  @Test
  public void valueInjectionInterrupt() throws Exception {
    SkyKey key = GraphTester.nonHermeticKey("key");
    SkyValue val = new StringValue("val");

    tester.differencer.inject(ImmutableMap.of(key, val));
    Thread.currentThread().interrupt();
    assertThrows(InterruptedException.class, () -> tester.evalAndGet(/*keepGoing=*/ false, key));
    SkyValue newVal = tester.evalAndGet(/*keepGoing=*/ false, key);
    assertThat(newVal).isEqualTo(val);
  }

  protected void runTestPersistentErrorsNotRerun(boolean includeTransientError) throws Exception {
    SkyKey topKey = GraphTester.toSkyKey("top");
    SkyKey transientErrorKey = GraphTester.toSkyKey("transientError");
    SkyKey persistentErrorKey1 = GraphTester.toSkyKey("persistentError1");
    SkyKey persistentErrorKey2 = GraphTester.toSkyKey("persistentError2");

    TestFunction topFunction = tester.getOrCreate(topKey)
          .addErrorDependency(persistentErrorKey1, new StringValue("doesn't matter"))
          .setHasError(true);
    tester.getOrCreate(persistentErrorKey1).setHasError(true);
    if (includeTransientError) {
      topFunction.addErrorDependency(transientErrorKey, new StringValue("doesn't matter"));
      tester.getOrCreate(transientErrorKey)
            .addErrorDependency(persistentErrorKey2, new StringValue("doesn't matter"))
            .setHasTransientError(true);
    }
    tester.getOrCreate(persistentErrorKey2).setHasError(true);

    tester.evalAndGetError(/*keepGoing=*/ true, topKey);
    if (includeTransientError) {
      assertThat(tester.getEnqueuedValues()).containsExactly(
          topKey, transientErrorKey, persistentErrorKey1, persistentErrorKey2);
    } else {
      assertThat(tester.getEnqueuedValues()).containsExactly(topKey, persistentErrorKey1);
    }

    tester.invalidate();
    tester.invalidateTransientErrors();
    tester.evalAndGetError(/*keepGoing=*/ true, topKey);
    if (includeTransientError) {
      // TODO(bazel-team): We can do better here once we implement change pruning for errors.
      assertThat(tester.getEnqueuedValues()).containsExactly(topKey, transientErrorKey);
    } else {
      assertThat(tester.getEnqueuedValues()).isEmpty();
    }
  }

  @Test
  public void persistentErrorsNotRerun() throws Exception {
    runTestPersistentErrorsNotRerun(/*includeTransientError=*/ true);
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
    SkyKey errorKey = GraphTester.toSkyKey("error");
    SkyKey invalidatedKey = GraphTester.nonHermeticKey("invalidated-leaf");
    tester.set(invalidatedKey, new StringValue("invalidated-leaf-value"));
    tester.getOrCreate(errorKey).addDependency(invalidatedKey).setHasError(true);
    // Names are alphabetized in reverse deps of errorKey.
    SkyKey fastToRequestSlowToSetValueKey = GraphTester.toSkyKey("A-slow-set-value-parent");
    SkyKey failingKey = GraphTester.toSkyKey("B-fast-fail-parent");
    tester.getOrCreate(fastToRequestSlowToSetValueKey).addDependency(errorKey)
        .setComputedValue(CONCATENATE);
    tester.getOrCreate(failingKey).addDependency(errorKey).setComputedValue(CONCATENATE);
    // We only want to force a particular order of operations at some points during evaluation.
    AtomicBoolean synchronizeThreads = new AtomicBoolean(false);
    // We don't expect slow-set-value to actually be built, but if it is, we wait for it.
    CountDownLatch slowBuilt = new CountDownLatch(1);
    injectGraphListenerForTesting(
        (key, type, order, context) -> {
          if (!synchronizeThreads.get()) {
            return;
          }
          if (type == EventType.GET_DIRTY_STATE && key.equals(failingKey)) {
            // Wait for the build to abort or for the other node to incorrectly build.
            try {
              assertThat(slowBuilt.await(TestUtils.WAIT_TIMEOUT_SECONDS, TimeUnit.SECONDS))
                  .isTrue();
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
        },
        /*deterministic=*/ true);
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
    // TrackingProgressReceiver does unnecessary examination of node values.
    initializeTester(createTrackingProgressReceiver(/* checkEvaluationResults= */ false));
    // errorKey will be invalidated due to its dependence on invalidatedKey, but later revalidated
    // since invalidatedKey re-evaluates to the same value on a subsequent build.
    SkyKey errorKey = GraphTester.toSkyKey("error");
    SkyKey invalidatedKey = GraphTester.nonHermeticKey("invalidated-leaf");
    SkyKey changedKey = GraphTester.nonHermeticKey("changed-leaf");
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
    injectGraphListenerForTesting(
        new Listener() {
          private final CountDownLatch cachedSignaled = new CountDownLatch(1);

          @Override
          public void accept(SkyKey key, EventType type, Order order, Object context) {
            if (!synchronizeThreads.get() || order != Order.BEFORE || type != EventType.SIGNAL) {
              return;
            }
            TrackingAwaiter.INSTANCE.awaitLatchAndTrackExceptions(
                shutdownAwaiterStarted, "shutdown awaiter not started");
            if (key.equals(uncachedParentKey)) {
              // When the uncached parent is first signaled by its changed dep, make sure that
              // we wait until the cached parent is signaled too.
              try {
                assertThat(cachedSignaled.await(TestUtils.WAIT_TIMEOUT_SECONDS, TimeUnit.SECONDS))
                    .isTrue();
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
        },
        /*deterministic=*/ true);
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
                    ((SkyFunctionEnvironment) env).getExceptionLatchForTesting(),
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

  /**
   * Tests that a race between a node being marked clean and another node requesting it is benign.
   * Here, we first evaluate errorKey, depending on invalidatedKey. Then we invalidate
   * invalidatedKey (without actually changing it) and evaluate errorKey and topKey together.
   * Through forced synchronization, we make sure that the following sequence of events happens:
   *
   * <ol>
   * <li>topKey requests errorKey;
   * <li>errorKey is marked clean;
   * <li>topKey finishes its first evaluation and registers its deps;
   * <li>topKey restarts, since it sees that its only dep, errorKey, is done;
   * <li>topKey sees the error thrown by errorKey and throws the error, shutting down the
   *     threadpool;
   * </ol>
   */
  @Test
  public void cachedErrorCausesRestart() throws Exception {
    // TrackingProgressReceiver does unnecessary examination of node values.
    initializeTester(createTrackingProgressReceiver(/* checkEvaluationResults= */ false));
    final SkyKey errorKey = GraphTester.toSkyKey("error");
    SkyKey invalidatedKey = GraphTester.nonHermeticKey("invalidated");
    final SkyKey topKey = GraphTester.toSkyKey("top");
    tester.getOrCreate(errorKey).addDependency(invalidatedKey).setHasError(true);
    tester.getOrCreate(invalidatedKey).setConstantValue(new StringValue("constant"));
    final CountDownLatch topSecondEval = new CountDownLatch(2);
    final CountDownLatch topRequestedError = new CountDownLatch(1);
    final CountDownLatch errorMarkedClean = new CountDownLatch(1);
    injectGraphListenerForTesting(
        new Listener() {
          @Override
          public void accept(SkyKey key, EventType type, Order order, Object context) {
            if (errorKey.equals(key) && type == EventType.MARK_CLEAN) {
              if (order == Order.BEFORE) {
                TrackingAwaiter.INSTANCE.awaitLatchAndTrackExceptions(
                    topRequestedError, "top didn't request");
              } else {
                errorMarkedClean.countDown();
                TrackingAwaiter.INSTANCE.awaitLatchAndTrackExceptions(
                    topSecondEval, "top didn't restart");
                // Make sure that the other thread notices the error and interrupts this thread.
                try {
                  Thread.sleep(TestUtils.WAIT_TIMEOUT_MILLISECONDS);
                } catch (InterruptedException e) {
                  Thread.currentThread().interrupt();
                }
              }
            }
          }
        },
        /*deterministic=*/ false);
    EvaluationResult<StringValue> result = tester.eval(/*keepGoing=*/ false, errorKey);
    assertThatEvaluationResult(result).hasError();
    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(errorKey)
        .hasExceptionThat()
        .isNotNull();
    tester
        .getOrCreate(topKey)
        .setBuilder(
            new SkyFunction() {
              @Nullable
              @Override
              public SkyValue compute(SkyKey skyKey, Environment env)
                  throws SkyFunctionException, InterruptedException {
                topSecondEval.countDown();
                env.getValue(errorKey);
                topRequestedError.countDown();
                assertThat(env.valuesMissing()).isTrue();
                TrackingAwaiter.INSTANCE.awaitLatchAndTrackExceptions(
                    errorMarkedClean, "error not marked clean");
                return null;
              }

              @Nullable
              @Override
              public String extractTag(SkyKey skyKey) {
                return null;
              }
            });
    tester.getOrCreate(invalidatedKey, /*markAsModified=*/ true);
    tester.invalidate();
    EvaluationResult<StringValue> result2 = tester.eval(/*keepGoing=*/ false, errorKey, topKey);
    assertThatEvaluationResult(result2).hasError();
    assertThatEvaluationResult(result2)
        .hasErrorEntryForKeyThat(errorKey)
        .hasExceptionThat()
        .isNotNull();
    assertThatEvaluationResult(result2)
        .hasErrorEntryForKeyThat(topKey)
        .hasExceptionThat()
        .isNotNull();
  }

  @Test
  public void cachedChildErrorDepWithSiblingDepOnNoKeepGoingEval() throws Exception {
    SkyKey parent1Key = GraphTester.toSkyKey("parent1");
    SkyKey parent2Key = GraphTester.toSkyKey("parent2");
    SkyKey errorKey = GraphTester.nonHermeticKey("error");
    final SkyKey otherKey = GraphTester.toSkyKey("other");
    SkyFunction parentBuilder =
        new SkyFunction() {
          @Override
          public SkyValue compute(SkyKey skyKey, Environment env) throws InterruptedException {
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

  private void injectGraphListenerForTesting(Listener listener, boolean deterministic) {
    tester.evaluator.injectGraphTransformerForTesting(
        DeterministicHelper.makeTransformer(listener, deterministic));
  }

  private void makeGraphDeterministic() {
    tester.evaluator.injectGraphTransformerForTesting(DeterministicHelper.MAKE_DETERMINISTIC);
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
    SkyKey top = GraphTester.nonHermeticKey("top");
    SkyKey mid = GraphTester.skyKey("mid");
    SkyKey leaf = GraphTester.nonHermeticKey("leaf");
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
    if (preciseEvaluationStatusStored()) {
      // And there is no value for mid in the graph,
      assertThat(tester.driver.getExistingValueForTesting(mid)).isNull();
      assertThat(tester.driver.getExistingErrorForTesting(mid)).isNull();
      // Or for leaf.
      assertThat(tester.driver.getExistingValueForTesting(leaf)).isNull();
      assertThat(tester.driver.getExistingErrorForTesting(leaf)).isNull();
    }

    // When top is changed to depend directly on leaf,
    tester
        .getOrCreate(top, /*markAsModified=*/ true)
        .addDependency(leaf)
        .setConstantValue(null)
        .setComputedValue(CONCATENATE);
    // Then when top is evaluated, its value is as expected,
    tester.invalidate();
    assertThat(tester.evalAndGet(/*keepGoing=*/ true, top)).isEqualTo(leafValue);
    if (preciseEvaluationStatusStored()) {
      // and there is no value for mid in the graph,
      assertThat(tester.driver.getExistingValueForTesting(mid)).isNull();
      assertThat(tester.driver.getExistingErrorForTesting(mid)).isNull();
    }
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
    assertThat(tester.evalAndGet(/*keepGoing=*/ true, GraphTester.nonHermeticKey("top")))
        .isEqualTo(new StringValue("leaf"));
  }

  // Tests that a removed and then reinstated node behaves properly when its parent disappears and
  // then reappears.
  @Test
  public void removedNodeComesBackAndOtherInvalidates() throws Exception {
    removedNodeComesBack();
    SkyKey top = GraphTester.nonHermeticKey("top");
    SkyKey mid = GraphTester.skyKey("mid");
    SkyKey leaf = GraphTester.nonHermeticKey("leaf");
    // When top is invalidated again,
    tester.getOrCreate(top, /*markAsModified=*/ true).removeDependency(leaf).addDependency(mid);
    // Then when top is evaluated, its value is as expected.
    tester.invalidate();
    assertThat(tester.evalAndGet(/*keepGoing=*/ true, top)).isEqualTo(new StringValue("leaf"));
  }

  // Tests that a removed and then reinstated node doesn't have a reverse dep on a former parent.
  @Test
  public void removedInvalidatedNodeComesBackAndOtherInvalidates() throws Exception {
    SkyKey top = GraphTester.nonHermeticKey("top");
    SkyKey leaf = GraphTester.nonHermeticKey("leaf");
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
    if (preciseEvaluationStatusStored()) {
      // And there is no value for leaf in the graph.
      assertThat(tester.driver.getExistingValueForTesting(leaf)).isNull();
      assertThat(tester.driver.getExistingErrorForTesting(leaf)).isNull();
    }
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
    SkyKey topKey = GraphTester.nonHermeticKey("top");
    SkyKey inactiveKey = GraphTester.nonHermeticKey("inactive");
    Thread mainThread = Thread.currentThread();
    AtomicBoolean shouldInterrupt = new AtomicBoolean(false);
    injectGraphListenerForTesting(
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
        },
        /*deterministic=*/ false);
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
    assertThrows(InterruptedException.class, () -> tester.eval(/*keepGoing=*/ false, topKey));
    // But inactive is still present,
    assertThat(tester.driver.getEntryForTesting(inactiveKey)).isNotNull();
    // And still dirty,
    assertThat(tester.driver.getEntryForTesting(inactiveKey).isDirty()).isTrue();
    // And re-evaluates successfully,
    assertThat(tester.evalAndGet(/*keepGoing=*/ true, inactiveKey)).isEqualTo(val);
    // But top is gone from the graph,
    assertThat(tester.driver.getEntryForTesting(topKey)).isNull();
    // And we can successfully invalidate and re-evaluate inactive again.
    tester.getOrCreate(inactiveKey, /*markAsModified=*/ true);
    tester.invalidate();
    assertThat(tester.evalAndGet(/*keepGoing=*/ true, inactiveKey)).isEqualTo(val);
  }

  @Test
  public void errorChanged() throws Exception {
    SkyKey error = GraphTester.nonHermeticKey("error");
    tester.getOrCreate(error).setHasError(true);
    assertThatErrorInfo(tester.evalAndGetError(/*keepGoing=*/ true, error))
        .hasExceptionThat().isNotNull();
    tester.getOrCreate(error, /*markAsModified=*/ true);
    tester.invalidate();
    assertThatErrorInfo(tester.evalAndGetError(/*keepGoing=*/ true, error))
        .hasExceptionThat().isNotNull();
  }

  @Test
  public void duplicateUnfinishedDeps_NoKeepGoing() throws Exception {
    runTestDuplicateUnfinishedDeps(/*keepGoing=*/ false);
  }

  @Test
  public void duplicateUnfinishedDeps_KeepGoing() throws Exception {
    runTestDuplicateUnfinishedDeps(/*keepGoing=*/ true);
  }

  @Test
  public void externalDep() throws Exception {
    externalDep(1, 0);
    externalDep(2, 0);
    externalDep(1, 1);
    externalDep(1, 2);
    externalDep(2, 1);
    externalDep(2, 2);
  }

  private void externalDep(int firstPassCount, int secondPassCount) throws Exception {
    final SkyKey parentKey = toSkyKey("parentKey");
    final CountDownLatch firstPassLatch = new CountDownLatch(1);
    final CountDownLatch secondPassLatch = new CountDownLatch(1);
    tester
        .getOrCreate(parentKey)
        .setBuilder(
            new SkyFunction() {
              // Skyframe doesn't have native support for continuations, so we use fields here. A
              // simple continuation API in Skyframe could be Environment providing a
              // setContinuation(SkyContinuation) method, where SkyContinuation provides a compute
              // method similar to SkyFunction. When restarting the node, Skyframe would then call
              // the continuation rather than the original SkyFunction. If we do that, we should
              // consider only allowing calls to dependOnFuture in combination with setContinuation.
              private List<SettableFuture<SkyValue>> firstPass;
              private List<SettableFuture<SkyValue>> secondPass;

              @Override
              public SkyValue compute(SkyKey skyKey, Environment env) {
                if (firstPass == null) {
                  firstPass = new ArrayList<>();
                  for (int i = 0; i < firstPassCount; i++) {
                    SettableFuture<SkyValue> future = SettableFuture.create();
                    firstPass.add(future);
                    env.dependOnFuture(future);
                  }
                  assertThat(env.valuesMissing()).isTrue();
                  Thread helper =
                      new Thread(
                          () -> {
                            try {
                              firstPassLatch.await();
                              for (int i = 0; i < firstPassCount; i++) {
                                firstPass.get(i).set(new StringValue("value1"));
                              }
                            } catch (InterruptedException e) {
                              throw new RuntimeException(e);
                            }
                          });
                  helper.start();
                  return null;
                } else if (secondPass == null && secondPassCount > 0) {
                  for (int i = 0; i < firstPassCount; i++) {
                    assertThat(firstPass.get(i).isDone()).isTrue();
                  }
                  secondPass = new ArrayList<>();
                  for (int i = 0; i < secondPassCount; i++) {
                    SettableFuture<SkyValue> future = SettableFuture.create();
                    secondPass.add(future);
                    env.dependOnFuture(future);
                  }
                  assertThat(env.valuesMissing()).isTrue();
                  Thread helper =
                      new Thread(
                          () -> {
                            try {
                              secondPassLatch.await();
                              for (int i = 0; i < secondPassCount; i++) {
                                secondPass.get(i).set(new StringValue("value2"));
                              }
                            } catch (InterruptedException e) {
                              throw new RuntimeException(e);
                            }
                          });
                  helper.start();
                  return null;
                }
                for (int i = 0; i < secondPassCount; i++) {
                  assertThat(secondPass.get(i).isDone()).isTrue();
                }
                return new StringValue("done!");
              }

              @Override
              public String extractTag(SkyKey skyKey) {
                return null;
              }
            });
    tester.evaluator.injectGraphTransformerForTesting(
        NotifyingHelper.makeNotifyingTransformer(
            new Listener() {
              private boolean firstPassDone;

              @Override
              public void accept(SkyKey key, EventType type, Order order, Object context) {
                // NodeEntry.addExternalDep is called as part of bookkeeping at the end of
                // AbstractParallelEvaluator.Evaluate#run.
                if (key == parentKey && type == EventType.ADD_EXTERNAL_DEP) {
                  if (!firstPassDone) {
                    firstPassLatch.countDown();
                    firstPassDone = true;
                  } else {
                    secondPassLatch.countDown();
                  }
                }
              }
            }));
    EvaluationResult<StringValue> result = tester.eval(/*keepGoing=*/ false, parentKey);
    assertThat(result.hasError()).isFalse();
    assertThat(result.get(parentKey)).isEqualTo(new StringValue("done!"));
  }

  private void runTestDuplicateUnfinishedDeps(boolean keepGoing) throws Exception {
    SkyKey parentKey = GraphTester.skyKey("parent");
    SkyKey childKey = GraphTester.skyKey("child");
    SkyValue childValue = new StringValue("child");
    tester
        .getOrCreate(childKey)
        .setBuilder(
            new SkyFunction() {
              @Nullable
              @Override
              public SkyValue compute(SkyKey skyKey, Environment env)
                  throws SkyFunctionException, InterruptedException {
                if (keepGoing) {
                  return childValue;
                } else {
                  throw new IllegalStateException("shouldn't get here");
                }
              }

              @Nullable
              @Override
              public String extractTag(SkyKey skyKey) {
                return null;
              }
            });
    SomeErrorException parentExn = new SomeErrorException("bad");
    AtomicInteger numParentComputeCalls = new AtomicInteger(0);
    tester
        .getOrCreate(parentKey)
        .setBuilder(
            new SkyFunction() {
              @Nullable
              @Override
              public SkyValue compute(SkyKey skyKey, Environment env)
                  throws SkyFunctionException, InterruptedException {
                numParentComputeCalls.incrementAndGet();
                if (!keepGoing || numParentComputeCalls.get() == 1) {
                  Preconditions.checkState(env.getValue(childKey) == null);
                  Preconditions.checkState(env.getValue(childKey) == null);
                } else {
                  Preconditions.checkState(env.getValue(childKey).equals(childValue));
                  Preconditions.checkState(env.getValue(childKey).equals(childValue));
                }
                throw new GenericFunctionException(parentExn, Transience.PERSISTENT);
              }

              @Nullable
              @Override
              public String extractTag(SkyKey skyKey) {
                return null;
              }
            });

    Exception exception = tester.evalAndGetError(keepGoing, parentKey).getException();
    assertThat(exception).isInstanceOf(SomeErrorException.class);
    assertThat(exception).hasMessageThat().isEqualTo("bad");
  }

  /** Data encapsulating a graph inconsistency found during evaluation. */
  @AutoValue
  public abstract static class InconsistencyData {
    public abstract SkyKey key();

    @Nullable
    public abstract SkyKey otherKey();

    public abstract Inconsistency inconsistency();

    public static InconsistencyData create(
        SkyKey key, @Nullable SkyKey otherKey, Inconsistency inconsistency) {
      return new AutoValue_MemoizingEvaluatorTest_InconsistencyData(key, otherKey, inconsistency);
    }
  }

  /** A graph tester that is specific to the memoizing evaluator, with some convenience methods. */
  protected class MemoizingEvaluatorTester extends GraphTester {
    private RecordingDifferencer differencer;
    private MemoizingEvaluator evaluator;
    private BuildDriver driver;
    private TrackingProgressReceiver progressReceiver =
        createTrackingProgressReceiver(/* checkEvaluationResults= */ true);
    private GraphInconsistencyReceiver graphInconsistencyReceiver =
        GraphInconsistencyReceiver.THROWING;
    private EventFilter eventFilter = InMemoryMemoizingEvaluator.DEFAULT_STORED_EVENT_FILTER;

    /** Constructs a new {@link #evaluator}, so call before injecting a transformer into it! */
    public void initialize(boolean keepEdges) {
      this.differencer = getRecordingDifferencer();
      this.evaluator =
          getMemoizingEvaluator(
              getSkyFunctionMap(),
              differencer,
              progressReceiver,
              graphInconsistencyReceiver,
              eventFilter,
              keepEdges);
      this.driver = getBuildDriver(evaluator);
    }

    /**
     * Sets the {@link #progressReceiver}. {@link #initialize} must be called after this to have any
     * effect.
     */
    public void setProgressReceiver(TrackingProgressReceiver progressReceiver) {
      this.progressReceiver = progressReceiver;
    }

    /**
     * Sets the {@link #eventFilter}. {@link #initialize} must be called after this to have any
     * effect.
     */
    public void setEventFilter(EventFilter eventFilter) {
      this.eventFilter = eventFilter;
    }

    /**
     * Sets the {@link #graphInconsistencyReceiver}. {@link #initialize} must be called after this
     * to have any effect.
     */
    public void setGraphInconsistencyReceiver(
        GraphInconsistencyReceiver graphInconsistencyReceiver) {
      this.graphInconsistencyReceiver = graphInconsistencyReceiver;
    }

    public MemoizingEvaluator getEvaluator() {
      return evaluator;
    }

    public void invalidate() {
      differencer.invalidate(getModifiedValues());
      clearModifiedValues();
      progressReceiver.clear();
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
      return progressReceiver.dirty;
    }

    public Set<SkyKey> getDeletedKeys() {
      return progressReceiver.deleted;
    }

    public Set<SkyKey> getEnqueuedValues() {
      return progressReceiver.enqueued;
    }

    public <T extends SkyValue> EvaluationResult<T> eval(
        boolean keepGoing, int numThreads, SkyKey... keys) throws InterruptedException {
      assertThat(getModifiedValues()).isEmpty();
      EvaluationContext evaluationContext =
          EvaluationContext.newBuilder()
              .setKeepGoing(keepGoing)
              .setNumThreads(numThreads)
              .setEventHandler(reporter)
              .build();
      return driver.evaluate(ImmutableList.copyOf(keys), evaluationContext);
    }

    public <T extends SkyValue> EvaluationResult<T> eval(boolean keepGoing, SkyKey... keys)
        throws InterruptedException {
      return eval(keepGoing, 100, keys);
    }

    public <T extends SkyValue> EvaluationResult<T> eval(boolean keepGoing, String... keys)
        throws InterruptedException {
      return eval(keepGoing, toSkyKeys(keys).toArray(new SkyKey[0]));
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
      assertWithMessage(evaluationResult.toString()).that(result).isNotNull();
      return result;
    }

    public ErrorInfo evalAndGetError(boolean keepGoing, SkyKey key) throws InterruptedException {
      EvaluationResult<StringValue> evaluationResult = eval(keepGoing, key);
      assertThatEvaluationResult(evaluationResult).hasErrorEntryForKeyThat(key);
      return evaluationResult.getError(key);
    }

    public ErrorInfo evalAndGetError(boolean keepGoing, String key) throws InterruptedException {
      return evalAndGetError(keepGoing, toSkyKey(key));
    }

    @Nullable
    public SkyValue getExistingValue(SkyKey key) throws InterruptedException {
      return driver.getExistingValueForTesting(key);
    }

    @Nullable
    public SkyValue getExistingValue(String key) throws InterruptedException {
      return getExistingValue(toSkyKey(key));
    }
  }

  /** {@link SkyFunction} with no tag extraction for easier lambda-izing. */
  protected interface TaglessSkyFunction extends SkyFunction {
    @Nullable
    @Override
    default String extractTag(SkyKey skyKey) {
      return null;
    }
  }

  private static GraphInconsistencyReceiver restartEnabledInconsistencyReceiver(
      GraphInconsistencyReceiver delegate) {
    return new GraphInconsistencyReceiver() {
      @Override
      public void noteInconsistencyAndMaybeThrow(
          SkyKey key, @Nullable Collection<SkyKey> otherKeys, Inconsistency inconsistency) {
        delegate.noteInconsistencyAndMaybeThrow(key, otherKeys, inconsistency);
      }

      @Override
      public boolean restartPermitted() {
        return true;
      }
    };
  }
}
