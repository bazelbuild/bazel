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

import static com.google.common.collect.Iterables.getOnlyElement;
import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertContainsEvent;
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertEventCount;
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertNoEvents;
import static com.google.devtools.build.skyframe.EvaluationResultSubjectFactory.assertThatEvaluationResult;
import static com.google.devtools.build.skyframe.GraphTester.CONCATENATE;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import com.google.common.base.Supplier;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import com.google.common.util.concurrent.Uninterruptibles;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventCollector;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.testutil.TestThread;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.skyframe.GraphTester.StringValue;
import com.google.devtools.build.skyframe.NotifyingInMemoryGraph.EventType;
import com.google.devtools.build.skyframe.NotifyingInMemoryGraph.Listener;
import com.google.devtools.build.skyframe.NotifyingInMemoryGraph.Order;
import com.google.devtools.build.skyframe.ParallelEvaluator.EventFilter;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;

import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

import javax.annotation.Nullable;

/**
 * Tests for {@link ParallelEvaluator}.
 */
@RunWith(JUnit4.class)
public class ParallelEvaluatorTest {
  protected ProcessableGraph graph;
  protected IntVersion graphVersion = new IntVersion(0);
  protected GraphTester tester = new GraphTester();

  private EventCollector eventCollector;

  private EvaluationProgressReceiver revalidationReceiver;

  @Before
  public void initializeReporter() {
    eventCollector = new EventCollector();
  }

  @After
  public void assertNoTrackedErrors() {
    TrackingAwaiter.INSTANCE.assertNoErrors();
    if (graph instanceof NotifyingInMemoryGraph) {
      ((NotifyingInMemoryGraph) graph).assertNoExceptions();
    }
  }

  private ParallelEvaluator makeEvaluator(
      ProcessableGraph graph,
      ImmutableMap<SkyFunctionName, ? extends SkyFunction> builders,
      boolean keepGoing,
      EventFilter storedEventFilter) {
    Version oldGraphVersion = graphVersion;
    graphVersion = graphVersion.next();
    return new ParallelEvaluator(graph,
        oldGraphVersion,
        builders,
        eventCollector,
        new MemoizingEvaluator.EmittedEventState(),
        storedEventFilter,
        keepGoing,
        150,
        revalidationReceiver,
        new DirtyKeyTrackerImpl(),
        new ParallelEvaluator.Receiver<Collection<SkyKey>>() {
          @Override
          public void accept(Collection<SkyKey> object) {
            // ignore
          }
        });
  }

  private ParallelEvaluator makeEvaluator(ProcessableGraph graph,
      ImmutableMap<SkyFunctionName, ? extends SkyFunction> builders, boolean keepGoing) {
    return makeEvaluator(graph, builders, keepGoing,
        InMemoryMemoizingEvaluator.DEFAULT_STORED_EVENT_FILTER);
  }

  /** Convenience method for eval-ing a single value. */
  protected SkyValue eval(boolean keepGoing, SkyKey key) throws InterruptedException {
    return eval(keepGoing, ImmutableList.of(key)).get(key);
  }

  protected ErrorInfo evalValueInError(SkyKey key) throws InterruptedException {
    return eval(true, ImmutableList.of(key)).getError(key);
  }

  protected <T extends SkyValue> EvaluationResult<T> eval(boolean keepGoing, SkyKey... keys)
      throws InterruptedException {
    return eval(keepGoing, ImmutableList.copyOf(keys));
  }

  protected <T extends SkyValue> EvaluationResult<T> eval(boolean keepGoing, Iterable<SkyKey> keys)
      throws InterruptedException {
    ParallelEvaluator evaluator = makeEvaluator(graph, tester.getSkyFunctionMap(), keepGoing);
    return evaluator.eval(keys);
  }

  protected GraphTester.TestFunction set(String name, String value) {
    return tester.set(name, new StringValue(value));
  }

  @Test
  public void smoke() throws Exception {
    graph = new InMemoryGraph();
    set("a", "a");
    set("b", "b");
    tester.getOrCreate("ab").addDependency("a").addDependency("b").setComputedValue(CONCATENATE);
    StringValue value = (StringValue) eval(false, GraphTester.toSkyKey("ab"));
    assertEquals("ab", value.getValue());
    assertNoEvents(eventCollector);
  }

  /**
   * Test interruption handling when a long-running SkyFunction gets interrupted.
   */
  @Test
  public void interruptedFunction() throws Exception {
    runInterruptionTest(new SkyFunctionFactory() {
      @Override
      public SkyFunction create(final Semaphore threadStarted, final String[] errorMessage) {
        return new SkyFunction() {
          @Override
          public SkyValue compute(SkyKey key, Environment env) throws InterruptedException {
            // Signal the waiting test thread that the evaluator thread has really started.
            threadStarted.release();

            // Simulate a SkyFunction that runs for 10 seconds (this number was chosen arbitrarily).
            // The main thread should interrupt it shortly after it got started.
            Thread.sleep(10 * 1000);

            // Set an error message to indicate that the expected interruption didn't happen.
            // We can't use Assert.fail(String) on an async thread.
            errorMessage[0] = "SkyFunction should have been interrupted";
            return null;
          }

          @Nullable
          @Override
          public String extractTag(SkyKey skyKey) {
            return null;
          }
        };
      }
    });
  }

  /**
   * Test interruption handling when the Evaluator is in-between running SkyFunctions.
   *
   * <p>This is the point in time after a SkyFunction requested a dependency which is not yet built
   * so the builder returned null to the Evaluator, and the latter is about to schedule evaluation
   * of the missing dependency but gets interrupted before the dependency's SkyFunction could start.
   */
  @Test
  public void interruptedEvaluatorThread() throws Exception {
    runInterruptionTest(new SkyFunctionFactory() {
      @Override
      public SkyFunction create(final Semaphore threadStarted, final String[] errorMessage) {
        return new SkyFunction() {
          // No need to synchronize access to this field; we always request just one more
          // dependency, so it's only one SkyFunction running at any time.
          private int valueIdCounter = 0;

          @Override
          public SkyValue compute(SkyKey key, Environment env) {
            // Signal the waiting test thread that the Evaluator thread has really started.
            threadStarted.release();

            // Keep the evaluator busy until the test's thread gets scheduled and can
            // interrupt the Evaluator's thread.
            env.getValue(GraphTester.toSkyKey("a" + valueIdCounter++));

            // This method never throws InterruptedException, therefore it's the responsibility
            // of the Evaluator to detect the interrupt and avoid calling subsequent SkyFunctions.
            return null;
          }

          @Nullable
          @Override
          public String extractTag(SkyKey skyKey) {
            return null;
          }
        };
      }
    });
  }

  private void runPartialResultOnInterruption(boolean buildFastFirst) throws Exception {
    graph = new InMemoryGraph();
    // Two runs for fastKey's builder and one for the start of waitKey's builder.
    final CountDownLatch allValuesReady = new CountDownLatch(3);
    final SkyKey waitKey = GraphTester.toSkyKey("wait");
    final SkyKey fastKey = GraphTester.toSkyKey("fast");
    SkyKey leafKey = GraphTester.toSkyKey("leaf");
    tester.getOrCreate(waitKey).setBuilder(new SkyFunction() {
          @Override
          public SkyValue compute(SkyKey skyKey, Environment env) throws InterruptedException {
            allValuesReady.countDown();
            Thread.sleep(10000);
            throw new AssertionError("Should have been interrupted");
          }

          @Override
          public String extractTag(SkyKey skyKey) {
            return null;
          }
        });
    tester.getOrCreate(fastKey).setBuilder(new ChainedFunction(null, null, allValuesReady, false,
        new StringValue("fast"), ImmutableList.of(leafKey)));
    tester.set(leafKey, new StringValue("leaf"));
    if (buildFastFirst) {
      eval(/*keepGoing=*/false, fastKey);
    }
    final Set<SkyKey> receivedValues = Sets.newConcurrentHashSet();
    revalidationReceiver = new EvaluationProgressReceiver() {
      @Override
      public void invalidated(SkyKey skyKey, InvalidationState state) {}

      @Override
      public void enqueueing(SkyKey key) {}

      @Override
      public void computed(SkyKey skyKey, long elapsedTimeNanos) {}

      @Override
      public void evaluated(SkyKey skyKey, Supplier<SkyValue> skyValueSupplier,
          EvaluationState state) {
        receivedValues.add(skyKey);
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
    assertTrue(allValuesReady.await(TestUtils.WAIT_TIMEOUT_SECONDS, TimeUnit.SECONDS));
    evalThread.interrupt();
    evalThread.join(TestUtils.WAIT_TIMEOUT_MILLISECONDS);
    assertFalse(evalThread.isAlive());
    if (buildFastFirst) {
      // If leafKey was already built, it is not reported to the receiver.
      assertThat(receivedValues).containsExactly(fastKey);
    } else {
      // On first time being built, leafKey is registered too.
      assertThat(receivedValues).containsExactly(fastKey, leafKey);
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
   * Factory for SkyFunctions for interruption testing (see {@link #runInterruptionTest}).
   */
  private interface SkyFunctionFactory {
    /**
     * Creates a SkyFunction suitable for a specific test scenario.
     *
     * @param threadStarted a latch which the returned SkyFunction must
     *     {@link Semaphore#release() release} once it started (otherwise the test won't work)
     * @param errorMessage a single-element array; the SkyFunction can put a error message in it
     *     to indicate that an assertion failed (calling {@code fail} from async thread doesn't
     *     work)
     */
    SkyFunction create(final Semaphore threadStarted, final String[] errorMessage);
  }

  /**
   * Test that we can handle the Evaluator getting interrupted at various points.
   *
   * <p>This method creates an Evaluator with the specified SkyFunction for GraphTested.NODE_TYPE,
   * then starts a thread, requests evaluation and asserts that evaluation started. It then
   * interrupts the Evaluator thread and asserts that it acknowledged the interruption.
   *
   * @param valueBuilderFactory creates a SkyFunction which may or may not handle interruptions
   *     (depending on the test)
   */
  private void runInterruptionTest(SkyFunctionFactory valueBuilderFactory) throws Exception {
    final Semaphore threadStarted = new Semaphore(0);
    final Semaphore threadInterrupted = new Semaphore(0);
    final String[] wasError = new String[] { null };
    final ParallelEvaluator evaluator = makeEvaluator(new InMemoryGraph(),
        ImmutableMap.of(GraphTester.NODE_TYPE, valueBuilderFactory.create(threadStarted, wasError)),
        false);

    Thread t = new Thread(new Runnable() {
        @Override
        public void run() {
          try {
            evaluator.eval(ImmutableList.of(GraphTester.toSkyKey("a")));

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
    t.start();
    assertTrue(threadStarted.tryAcquire(TestUtils.WAIT_TIMEOUT_MILLISECONDS,
        TimeUnit.MILLISECONDS));

    // Interrupt the thread and wait for a semaphore. This ensures that the thread was really
    // interrupted and this fact was acknowledged.
    t.interrupt();
    assertTrue(threadInterrupted.tryAcquire(TestUtils.WAIT_TIMEOUT_MILLISECONDS,
        TimeUnit.MILLISECONDS));

    // The SkyFunction may have reported an error.
    if (wasError[0] != null) {
      fail(wasError[0]);
    }

    // Wait for the thread to finish.
    t.join(TestUtils.WAIT_TIMEOUT_MILLISECONDS);
  }

  @Test
  public void unrecoverableError() throws Exception {
    class CustomRuntimeException extends RuntimeException {}
    final CustomRuntimeException expected = new CustomRuntimeException();

    final SkyFunction builder = new SkyFunction() {
      @Override
      @Nullable
      public SkyValue compute(SkyKey skyKey, Environment env)
          throws SkyFunctionException, InterruptedException {
        throw expected;
      }

      @Override
      @Nullable
      public String extractTag(SkyKey skyKey) {
        return null;
      }
    };

    final ParallelEvaluator evaluator = makeEvaluator(new InMemoryGraph(),
        ImmutableMap.of(GraphTester.NODE_TYPE, builder),
        false);

    SkyKey valueToEval = GraphTester.toSkyKey("a");
    try {
      evaluator.eval(ImmutableList.of(valueToEval));
    } catch (RuntimeException re) {
      assertThat(re.getMessage())
          .contains("Unrecoverable error while evaluating node '" + valueToEval.toString() + "'");
      assertThat(re.getCause()).isInstanceOf(CustomRuntimeException.class);
    }
  }

  @Test
  public void simpleWarning() throws Exception {
    graph = new InMemoryGraph();
    set("a", "a").setWarning("warning on 'a'");
    StringValue value = (StringValue) eval(false, GraphTester.toSkyKey("a"));
    assertEquals("a", value.getValue());
    assertContainsEvent(eventCollector, "warning on 'a'");
    assertEventCount(1, eventCollector);
  }

  /** Regression test: events from already-done value not replayed. */
  @Test
  public void eventFromDoneChildRecorded() throws Exception {
    graph = new InMemoryGraph();
    set("a", "a").setWarning("warning on 'a'");
    SkyKey a = GraphTester.toSkyKey("a");
    SkyKey top = GraphTester.toSkyKey("top");
    tester.getOrCreate(top).addDependency(a).setComputedValue(CONCATENATE);
    // Build a so that it is already in the graph.
    eval(false, a);
    assertEventCount(1, eventCollector);
    eventCollector.clear();
    // Build top. The warning from a should be reprinted.
    eval(false, top);
    assertEventCount(1, eventCollector);
    eventCollector.clear();
    // Build top again. The warning should have been stored in the value.
    eval(false, top);
    assertEventCount(1, eventCollector);
  }

  @Test
  public void storedEventFilter() throws Exception {
    graph = new InMemoryGraph();
    SkyKey a = GraphTester.toSkyKey("a");
    final AtomicBoolean evaluated = new AtomicBoolean(false);
    tester.getOrCreate(a).setBuilder(new SkyFunction() {
      @Nullable
      @Override
      public SkyValue compute(SkyKey skyKey, Environment env) {
        evaluated.set(true);
        env.getListener().handle(Event.error(null, "boop"));
        env.getListener().handle(Event.warn(null, "beep"));
        return new StringValue("a");
      }

      @Nullable
      @Override
      public String extractTag(SkyKey skyKey) {
        return null;
      }
    });
    ParallelEvaluator evaluator =
        makeEvaluator(
            graph,
            tester.getSkyFunctionMap(),
            /*keepGoing=*/ false,
            new EventFilter() {
              @Override
              public boolean apply(Event event) {
                return event.getKind() == EventKind.ERROR;
              }

              @Override
              public boolean storeEvents() {
                return true;
              }
            });
    evaluator.eval(ImmutableList.of(a));
    assertTrue(evaluated.get());
    assertEventCount(2, eventCollector);
    assertContainsEvent(eventCollector, "boop");
    assertContainsEvent(eventCollector, "beep");
    eventCollector.clear();
    evaluator = makeEvaluator(graph, tester.getSkyFunctionMap(), /*keepGoing=*/ false);
    evaluated.set(false);
    evaluator.eval(ImmutableList.of(a));
    assertFalse(evaluated.get());
    assertEventCount(1, eventCollector);
    assertContainsEvent(eventCollector, "boop");
  }

  @Test
  public void shouldCreateErrorValueWithRootCause() throws Exception {
    graph = new InMemoryGraph();
    set("a", "a");
    SkyKey parentErrorKey = GraphTester.toSkyKey("parent");
    SkyKey errorKey = GraphTester.toSkyKey("error");
    tester.getOrCreate(parentErrorKey).addDependency("a").addDependency(errorKey)
    .setComputedValue(CONCATENATE);
    tester.getOrCreate(errorKey).setHasError(true);
    ErrorInfo error = evalValueInError(parentErrorKey);
    assertThat(error.getRootCauses()).containsExactly(errorKey);
  }

  @Test
  public void shouldBuildOneTarget() throws Exception {
    graph = new InMemoryGraph();
    set("a", "a");
    set("b", "b");
    SkyKey parentErrorKey = GraphTester.toSkyKey("parent");
    SkyKey errorFreeKey = GraphTester.toSkyKey("ab");
    SkyKey errorKey = GraphTester.toSkyKey("error");
    tester.getOrCreate(parentErrorKey).addDependency(errorKey).addDependency("a")
    .setComputedValue(CONCATENATE);
    tester.getOrCreate(errorKey).setHasError(true);
    tester.getOrCreate(errorFreeKey).addDependency("a").addDependency("b")
    .setComputedValue(CONCATENATE);
    EvaluationResult<StringValue> result = eval(true, parentErrorKey, errorFreeKey);
    ErrorInfo error = result.getError(parentErrorKey);
    assertThat(error.getRootCauses()).containsExactly(errorKey);
    StringValue abValue = result.get(errorFreeKey);
    assertEquals("ab", abValue.getValue());
  }

  @Test
  public void catastropheHaltsBuild_KeepGoing_KeepEdges() throws Exception {
    catastrophicBuild(true, true);
  }

  @Test
  public void catastropheHaltsBuild_KeepGoing_NoKeepEdges() throws Exception {
    catastrophicBuild(true, false);
  }

  @Test
  public void catastropheInBuild_NoKeepGoing_KeepEdges() throws Exception {
    catastrophicBuild(false, true);
  }

  private void catastrophicBuild(boolean keepGoing, boolean keepEdges) throws Exception {
    graph = new InMemoryGraph(keepEdges);

    SkyKey catastropheKey = GraphTester.toSkyKey("catastrophe");
    SkyKey otherKey = GraphTester.toSkyKey("someKey");

    final Exception catastrophe = new SomeErrorException("bad");
    tester
        .getOrCreate(catastropheKey)
        .setBuilder(
            new SkyFunction() {
              @Nullable
              @Override
              public SkyValue compute(SkyKey skyKey, Environment env) throws SkyFunctionException {
                throw new SkyFunctionException(catastrophe, Transience.PERSISTENT) {
                  @Override
                  public boolean isCatastrophic() {
                    return true;
                  }
                };
              }

              @Nullable
              @Override
              public String extractTag(SkyKey skyKey) {
                return null;
              }
            });

    tester.getOrCreate(otherKey).setBuilder(new SkyFunction() {
      @Nullable
      @Override
      public SkyValue compute(SkyKey skyKey, Environment env) throws InterruptedException {
        new CountDownLatch(1).await();
        throw new RuntimeException("can't get here");
      }

      @Nullable
      @Override
      public String extractTag(SkyKey skyKey) {
        return null;
      }
    });

    SkyKey topKey = GraphTester.toSkyKey("top");
    tester.getOrCreate(topKey).addDependency(catastropheKey).setComputedValue(CONCATENATE);
    EvaluationResult<StringValue> result = eval(keepGoing, topKey, otherKey);
    if (!keepGoing) {
      ErrorInfo error = result.getError(topKey);
      assertThat(error.getRootCauses()).containsExactly(catastropheKey);
    } else {
      assertTrue(result.hasError());
      assertThat(result.errorMap()).isEmpty();
      assertThat(result.getCatastrophe()).isSameAs(catastrophe);
    }
  }

  @Test
  public void parentFailureDoesntAffectChild() throws Exception {
    graph = new InMemoryGraph();
    SkyKey parentKey = GraphTester.toSkyKey("parent");
    tester.getOrCreate(parentKey).setHasError(true);
    SkyKey childKey = GraphTester.toSkyKey("child");
    set("child", "onions");
    tester.getOrCreate(parentKey).addDependency(childKey).setComputedValue(CONCATENATE);
    EvaluationResult<StringValue> result = eval(/*keepGoing=*/true, parentKey, childKey);
    // Child is guaranteed to complete successfully before parent can run (and fail),
    // since parent depends on it.
    StringValue childValue = result.get(childKey);
    Assert.assertNotNull(childValue);
    assertEquals("onions", childValue.getValue());
    ErrorInfo error = result.getError(parentKey);
    Assert.assertNotNull(error);
    assertThat(error.getRootCauses()).containsExactly(parentKey);
  }

  @Test
  public void newParentOfErrorShouldHaveError() throws Exception {
    graph = new InMemoryGraph();
    SkyKey errorKey = GraphTester.toSkyKey("error");
    tester.getOrCreate(errorKey).setHasError(true);
    ErrorInfo error = evalValueInError(errorKey);
    assertThat(error.getRootCauses()).containsExactly(errorKey);
    SkyKey parentKey = GraphTester.toSkyKey("parent");
    tester.getOrCreate(parentKey).addDependency("error").setComputedValue(CONCATENATE);
    error = evalValueInError(parentKey);
    assertThat(error.getRootCauses()).containsExactly(errorKey);
  }

  @Test
  public void errorTwoLevelsDeep() throws Exception {
    graph = new InMemoryGraph();
    SkyKey parentKey = GraphTester.toSkyKey("parent");
    SkyKey errorKey = GraphTester.toSkyKey("error");
    tester.getOrCreate(errorKey).setHasError(true);
    tester.getOrCreate("mid").addDependency(errorKey).setComputedValue(CONCATENATE);
    tester.getOrCreate(parentKey).addDependency("mid").setComputedValue(CONCATENATE);
    ErrorInfo error = evalValueInError(parentKey);
    assertThat(error.getRootCauses()).containsExactly(errorKey);
  }

  @Test
  public void valueNotUsedInFailFastErrorRecovery() throws Exception {
    graph = new InMemoryGraph();
    SkyKey topKey = GraphTester.toSkyKey("top");
    SkyKey recoveryKey = GraphTester.toSkyKey("midRecovery");
    SkyKey badKey = GraphTester.toSkyKey("bad");

    tester.getOrCreate(topKey).addDependency(recoveryKey).setComputedValue(CONCATENATE);
    tester.getOrCreate(recoveryKey).addErrorDependency(badKey, new StringValue("i recovered"))
        .setComputedValue(CONCATENATE);
    tester.getOrCreate(badKey).setHasError(true);

    EvaluationResult<SkyValue> result = eval(/*keepGoing=*/true, ImmutableList.of(recoveryKey));
    assertThat(result.errorMap()).isEmpty();
    assertThatEvaluationResult(result).hasNoError();
    assertEquals(new StringValue("i recovered"), result.get(recoveryKey));

    result = eval(/*keepGoing=*/false, ImmutableList.of(topKey));
    assertThatEvaluationResult(result).hasError();
    assertThat(result.keyNames()).isEmpty();
    assertEquals(1, result.errorMap().size());
    assertNotNull(result.getError(topKey).getException());
  }

  @Test
  public void multipleRootCauses() throws Exception {
    graph = new InMemoryGraph();
    SkyKey parentKey = GraphTester.toSkyKey("parent");
    SkyKey errorKey = GraphTester.toSkyKey("error");
    SkyKey errorKey2 = GraphTester.toSkyKey("error2");
    SkyKey errorKey3 = GraphTester.toSkyKey("error3");
    tester.getOrCreate(errorKey).setHasError(true);
    tester.getOrCreate(errorKey2).setHasError(true);
    tester.getOrCreate(errorKey3).setHasError(true);
    tester.getOrCreate("mid").addDependency(errorKey).addDependency(errorKey2)
      .setComputedValue(CONCATENATE);
    tester.getOrCreate(parentKey)
      .addDependency("mid").addDependency(errorKey2).addDependency(errorKey3)
      .setComputedValue(CONCATENATE);
    ErrorInfo error = evalValueInError(parentKey);
    assertThat(error.getRootCauses()).containsExactly(errorKey, errorKey2, errorKey3);
  }

  @Test
  public void rootCauseWithNoKeepGoing() throws Exception {
    graph = new InMemoryGraph();
    SkyKey parentKey = GraphTester.toSkyKey("parent");
    SkyKey errorKey = GraphTester.toSkyKey("error");
    tester.getOrCreate(errorKey).setHasError(true);
    tester.getOrCreate("mid").addDependency(errorKey).setComputedValue(CONCATENATE);
    tester.getOrCreate(parentKey).addDependency("mid").setComputedValue(CONCATENATE);
    EvaluationResult<StringValue> result = eval(false, ImmutableList.of(parentKey));
    Map.Entry<SkyKey, ErrorInfo> error = Iterables.getOnlyElement(result.errorMap().entrySet());
    assertEquals(parentKey, error.getKey());
    assertThat(error.getValue().getRootCauses()).containsExactly(errorKey);
  }

  @Test
  public void errorBubblesToParentsOfTopLevelValue() throws Exception {
    graph = new InMemoryGraph();
    SkyKey parentKey = GraphTester.toSkyKey("parent");
    final SkyKey errorKey = GraphTester.toSkyKey("error");
    final CountDownLatch latch = new CountDownLatch(1);
    tester.getOrCreate(errorKey).setBuilder(new ChainedFunction(null, /*waitToFinish=*/latch, null,
        false, /*value=*/null, ImmutableList.<SkyKey>of()));
    tester.getOrCreate(parentKey).setBuilder(new ChainedFunction(/*notifyStart=*/latch, null, null,
        false, new StringValue("unused"), ImmutableList.of(errorKey)));
    EvaluationResult<StringValue> result = eval( /*keepGoing=*/false,
        ImmutableList.of(parentKey, errorKey));
    assertEquals(result.toString(), 2, result.errorMap().size());
  }

  @Test
  public void noKeepGoingAfterKeepGoingFails() throws Exception {
    graph = new InMemoryGraph();
    SkyKey errorKey = GraphTester.toSkyKey("my_error_value");
    tester.getOrCreate(errorKey).setHasError(true);
    SkyKey parentKey = GraphTester.toSkyKey("parent");
    tester.getOrCreate(parentKey).addDependency(errorKey);
    ErrorInfo error = evalValueInError(parentKey);
    assertThat(error.getRootCauses()).containsExactly(errorKey);
    SkyKey[] list = { parentKey };
    EvaluationResult<StringValue> result = eval(false, list);
    ErrorInfo errorInfo = result.getError();
    assertEquals(errorKey, Iterables.getOnlyElement(errorInfo.getRootCauses()));
    assertEquals(errorKey.toString(), errorInfo.getException().getMessage());
  }

  @Test
  public void twoErrors() throws Exception {
    graph = new InMemoryGraph();
    SkyKey firstError = GraphTester.toSkyKey("error1");
    SkyKey secondError = GraphTester.toSkyKey("error2");
    CountDownLatch firstStart = new CountDownLatch(1);
    CountDownLatch secondStart = new CountDownLatch(1);
    tester.getOrCreate(firstError).setBuilder(new ChainedFunction(firstStart, secondStart,
        /*notifyFinish=*/null, /*waitForException=*/false, /*value=*/null,
        ImmutableList.<SkyKey>of()));
    tester.getOrCreate(secondError).setBuilder(new ChainedFunction(secondStart, firstStart,
        /*notifyFinish=*/null, /*waitForException=*/false, /*value=*/null,
        ImmutableList.<SkyKey>of()));
    EvaluationResult<StringValue> result = eval(/*keepGoing=*/false, firstError, secondError);
    assertTrue(result.toString(), result.hasError());
    // With keepGoing=false, the eval call will terminate with exactly one error (the first one
    // thrown). But the first one thrown here is non-deterministic since we synchronize the
    // builders so that they run at roughly the same time.
    assertThat(ImmutableSet.of(firstError, secondError)).contains(
        Iterables.getOnlyElement(result.errorMap().keySet()));
  }

  @Test
  public void simpleCycle() throws Exception {
    graph = new InMemoryGraph();
    SkyKey aKey = GraphTester.toSkyKey("a");
    SkyKey bKey = GraphTester.toSkyKey("b");
    tester.getOrCreate(aKey).addDependency(bKey);
    tester.getOrCreate(bKey).addDependency(aKey);
    ErrorInfo errorInfo = eval(false, ImmutableList.of(aKey)).getError();
    assertEquals(null, errorInfo.getException());
    CycleInfo cycleInfo = Iterables.getOnlyElement(errorInfo.getCycleInfo());
    assertThat(cycleInfo.getCycle()).containsExactly(aKey, bKey).inOrder();
    assertTrue(cycleInfo.getPathToCycle().isEmpty());
  }

  @Test
  public void cycleWithHead() throws Exception {
    graph = new InMemoryGraph();
    SkyKey aKey = GraphTester.toSkyKey("a");
    SkyKey bKey = GraphTester.toSkyKey("b");
    SkyKey topKey = GraphTester.toSkyKey("top");
    SkyKey midKey = GraphTester.toSkyKey("mid");
    tester.getOrCreate(topKey).addDependency(midKey);
    tester.getOrCreate(midKey).addDependency(aKey);
    tester.getOrCreate(aKey).addDependency(bKey);
    tester.getOrCreate(bKey).addDependency(aKey);
    ErrorInfo errorInfo = eval(false, ImmutableList.of(topKey)).getError();
    assertEquals(null, errorInfo.getException());
    CycleInfo cycleInfo = Iterables.getOnlyElement(errorInfo.getCycleInfo());
    assertThat(cycleInfo.getCycle()).containsExactly(aKey, bKey).inOrder();
    assertThat(cycleInfo.getPathToCycle()).containsExactly(topKey, midKey).inOrder();
  }

  @Test
  public void selfEdgeWithHead() throws Exception {
    graph = new InMemoryGraph();
    SkyKey aKey = GraphTester.toSkyKey("a");
    SkyKey topKey = GraphTester.toSkyKey("top");
    SkyKey midKey = GraphTester.toSkyKey("mid");
    tester.getOrCreate(topKey).addDependency(midKey);
    tester.getOrCreate(midKey).addDependency(aKey);
    tester.getOrCreate(aKey).addDependency(aKey);
    ErrorInfo errorInfo = eval(false, ImmutableList.of(topKey)).getError();
    assertEquals(null, errorInfo.getException());
    CycleInfo cycleInfo = Iterables.getOnlyElement(errorInfo.getCycleInfo());
    assertThat(cycleInfo.getCycle()).containsExactly(aKey).inOrder();
    assertThat(cycleInfo.getPathToCycle()).containsExactly(topKey, midKey).inOrder();
  }

  @Test
  public void cycleWithKeepGoing() throws Exception {
    graph = new InMemoryGraph();
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
    EvaluationResult<StringValue> result = eval(true, topKey, goodKey);
    assertEquals(goodValue, result.get(goodKey));
    assertEquals(null, result.get(topKey));
    ErrorInfo errorInfo = result.getError(topKey);
    CycleInfo cycleInfo = Iterables.getOnlyElement(errorInfo.getCycleInfo());
    assertThat(cycleInfo.getCycle()).containsExactly(aKey, bKey).inOrder();
    assertThat(cycleInfo.getPathToCycle()).containsExactly(topKey, midKey).inOrder();
  }

  @Test
  public void twoCycles() throws Exception {
    graph = new InMemoryGraph();
    SkyKey aKey = GraphTester.toSkyKey("a");
    SkyKey bKey = GraphTester.toSkyKey("b");
    SkyKey cKey = GraphTester.toSkyKey("c");
    SkyKey dKey = GraphTester.toSkyKey("d");
    SkyKey topKey = GraphTester.toSkyKey("top");
    tester.getOrCreate(topKey).addDependency(aKey).addDependency(cKey);
    tester.getOrCreate(aKey).addDependency(bKey);
    tester.getOrCreate(bKey).addDependency(aKey);
    tester.getOrCreate(cKey).addDependency(dKey);
    tester.getOrCreate(dKey).addDependency(cKey);
    EvaluationResult<StringValue> result = eval(false, ImmutableList.of(topKey));
    assertEquals(null, result.get(topKey));
    ErrorInfo errorInfo = result.getError(topKey);
    Iterable<CycleInfo> cycles = CycleInfo.prepareCycles(topKey,
        ImmutableList.of(new CycleInfo(ImmutableList.of(aKey, bKey)),
        new CycleInfo(ImmutableList.of(cKey, dKey))));
    assertThat(cycles).contains(getOnlyElement(errorInfo.getCycleInfo()));
  }


  @Test
  public void twoCyclesKeepGoing() throws Exception {
    graph = new InMemoryGraph();
    SkyKey aKey = GraphTester.toSkyKey("a");
    SkyKey bKey = GraphTester.toSkyKey("b");
    SkyKey cKey = GraphTester.toSkyKey("c");
    SkyKey dKey = GraphTester.toSkyKey("d");
    SkyKey topKey = GraphTester.toSkyKey("top");
    tester.getOrCreate(topKey).addDependency(aKey).addDependency(cKey);
    tester.getOrCreate(aKey).addDependency(bKey);
    tester.getOrCreate(bKey).addDependency(aKey);
    tester.getOrCreate(cKey).addDependency(dKey);
    tester.getOrCreate(dKey).addDependency(cKey);
    EvaluationResult<StringValue> result = eval(true, ImmutableList.of(topKey));
    assertEquals(null, result.get(topKey));
    ErrorInfo errorInfo = result.getError(topKey);
    CycleInfo aCycle = new CycleInfo(ImmutableList.of(topKey), ImmutableList.of(aKey, bKey));
    CycleInfo cCycle = new CycleInfo(ImmutableList.of(topKey), ImmutableList.of(cKey, dKey));
    assertThat(errorInfo.getCycleInfo()).containsExactly(aCycle, cCycle);
  }

  @Test
  public void triangleBelowHeadCycle() throws Exception {
    graph = new InMemoryGraph();
    SkyKey aKey = GraphTester.toSkyKey("a");
    SkyKey bKey = GraphTester.toSkyKey("b");
    SkyKey cKey = GraphTester.toSkyKey("c");
    SkyKey topKey = GraphTester.toSkyKey("top");
    tester.getOrCreate(topKey).addDependency(aKey);
    tester.getOrCreate(aKey).addDependency(bKey).addDependency(cKey);
    tester.getOrCreate(bKey).addDependency(cKey);
    tester.getOrCreate(cKey).addDependency(topKey);
    EvaluationResult<StringValue> result = eval(true, ImmutableList.of(topKey));
    assertEquals(null, result.get(topKey));
    ErrorInfo errorInfo = result.getError(topKey);
    CycleInfo topCycle = new CycleInfo(ImmutableList.of(topKey, aKey, cKey));
    assertThat(errorInfo.getCycleInfo()).containsExactly(topCycle);
  }

  @Test
  public void longCycle() throws Exception {
    graph = new InMemoryGraph();
    SkyKey aKey = GraphTester.toSkyKey("a");
    SkyKey bKey = GraphTester.toSkyKey("b");
    SkyKey cKey = GraphTester.toSkyKey("c");
    SkyKey topKey = GraphTester.toSkyKey("top");
    tester.getOrCreate(topKey).addDependency(aKey);
    tester.getOrCreate(aKey).addDependency(bKey);
    tester.getOrCreate(bKey).addDependency(cKey);
    tester.getOrCreate(cKey).addDependency(topKey);
    EvaluationResult<StringValue> result = eval(true, ImmutableList.of(topKey));
    assertEquals(null, result.get(topKey));
    ErrorInfo errorInfo = result.getError(topKey);
    CycleInfo topCycle = new CycleInfo(ImmutableList.of(topKey, aKey, bKey, cKey));
    assertThat(errorInfo.getCycleInfo()).containsExactly(topCycle);
  }

  @Test
  public void cycleWithTail() throws Exception {
    graph = new InMemoryGraph();
    SkyKey aKey = GraphTester.toSkyKey("a");
    SkyKey bKey = GraphTester.toSkyKey("b");
    SkyKey cKey = GraphTester.toSkyKey("c");
    SkyKey topKey = GraphTester.toSkyKey("top");
    tester.getOrCreate(topKey).addDependency(aKey).addDependency(cKey);
    tester.getOrCreate(aKey).addDependency(bKey);
    tester.getOrCreate(bKey).addDependency(aKey).addDependency(cKey);
    tester.getOrCreate(cKey);
    tester.set(cKey, new StringValue("cValue"));
    EvaluationResult<StringValue> result = eval(false, ImmutableList.of(topKey));
    assertEquals(null, result.get(topKey));
    ErrorInfo errorInfo = result.getError(topKey);
    CycleInfo cycleInfo = Iterables.getOnlyElement(errorInfo.getCycleInfo());
    assertThat(cycleInfo.getCycle()).containsExactly(aKey, bKey).inOrder();
    assertThat(cycleInfo.getPathToCycle()).containsExactly(topKey).inOrder();
  }

  /** Regression test: "value cannot be ready in a cycle". */
  @Test
  public void selfEdgeWithExtraChildrenUnderCycle() throws Exception {
    graph = new InMemoryGraph();
    SkyKey aKey = GraphTester.toSkyKey("a");
    SkyKey bKey = GraphTester.toSkyKey("b");
    SkyKey cKey = GraphTester.toSkyKey("c");
    tester.getOrCreate(aKey).addDependency(bKey);
    tester.getOrCreate(bKey).addDependency(cKey).addDependency(bKey);
    tester.getOrCreate(cKey).addDependency(aKey);
    EvaluationResult<StringValue> result = eval(/*keepGoing=*/true, ImmutableList.of(aKey));
    assertEquals(null, result.get(aKey));
    ErrorInfo errorInfo = result.getError(aKey);
    CycleInfo cycleInfo = Iterables.getOnlyElement(errorInfo.getCycleInfo());
    assertThat(cycleInfo.getCycle()).containsExactly(bKey).inOrder();
    assertThat(cycleInfo.getPathToCycle()).containsExactly(aKey).inOrder();
  }

  /** Regression test: "value cannot be ready in a cycle". */
  @Test
  public void cycleWithExtraChildrenUnderCycle() throws Exception {
    graph = new InMemoryGraph();
    SkyKey aKey = GraphTester.toSkyKey("a");
    SkyKey bKey = GraphTester.toSkyKey("b");
    SkyKey cKey = GraphTester.toSkyKey("c");
    SkyKey dKey = GraphTester.toSkyKey("d");
    tester.getOrCreate(aKey).addDependency(bKey);
    tester.getOrCreate(bKey).addDependency(cKey).addDependency(dKey);
    tester.getOrCreate(cKey).addDependency(aKey);
    tester.getOrCreate(dKey).addDependency(bKey);
    EvaluationResult<StringValue> result = eval(/*keepGoing=*/true, ImmutableList.of(aKey));
    assertEquals(null, result.get(aKey));
    ErrorInfo errorInfo = result.getError(aKey);
    CycleInfo cycleInfo = Iterables.getOnlyElement(errorInfo.getCycleInfo());
    assertThat(cycleInfo.getCycle()).containsExactly(bKey, dKey).inOrder();
    assertThat(cycleInfo.getPathToCycle()).containsExactly(aKey).inOrder();
  }

  /** Regression test: "value cannot be ready in a cycle". */
  @Test
  public void cycleAboveIndependentCycle() throws Exception {
    graph = new InMemoryGraph();
    SkyKey aKey = GraphTester.toSkyKey("a");
    SkyKey bKey = GraphTester.toSkyKey("b");
    SkyKey cKey = GraphTester.toSkyKey("c");
    tester.getOrCreate(aKey).addDependency(bKey);
    tester.getOrCreate(bKey).addDependency(cKey);
    tester.getOrCreate(cKey).addDependency(aKey).addDependency(bKey);
    EvaluationResult<StringValue> result = eval(/*keepGoing=*/true, ImmutableList.of(aKey));
    assertEquals(null, result.get(aKey));
    assertThat(result.getError(aKey).getCycleInfo()).containsExactly(
        new CycleInfo(ImmutableList.of(aKey, bKey, cKey)),
        new CycleInfo(ImmutableList.of(aKey), ImmutableList.of(bKey, cKey)));
  }

  public void valueAboveCycleAndExceptionReportsException() throws Exception {
    graph = new InMemoryGraph();
    SkyKey aKey = GraphTester.toSkyKey("a");
    SkyKey errorKey = GraphTester.toSkyKey("error");
    SkyKey bKey = GraphTester.toSkyKey("b");
    tester.getOrCreate(aKey).addDependency(bKey).addDependency(errorKey);
    tester.getOrCreate(bKey).addDependency(bKey);
    tester.getOrCreate(errorKey).setHasError(true);
    EvaluationResult<StringValue> result = eval(/*keepGoing=*/true, ImmutableList.of(aKey));
    assertEquals(null, result.get(aKey));
    assertNotNull(result.getError(aKey).getException());
    CycleInfo cycleInfo = Iterables.getOnlyElement(result.getError(aKey).getCycleInfo());
    assertThat(cycleInfo.getCycle()).containsExactly(bKey).inOrder();
    assertThat(cycleInfo.getPathToCycle()).containsExactly(aKey).inOrder();
  }

  @Test
  public void errorValueStored() throws Exception {
    graph = new InMemoryGraph();
    SkyKey errorKey = GraphTester.toSkyKey("my_error_value");
    tester.getOrCreate(errorKey).setHasError(true);
    EvaluationResult<StringValue> result = eval(false, ImmutableList.of(errorKey));
    assertThat(result.keyNames()).isEmpty();
    assertThat(result.errorMap().keySet()).containsExactly(errorKey);
    ErrorInfo errorInfo = result.getError();
    assertThat(errorInfo.getRootCauses()).containsExactly(errorKey);
    // Update value. But builder won't rebuild it.
    tester.getOrCreate(errorKey).setHasError(false);
    tester.set(errorKey, new StringValue("no error?"));
    result = eval(false, ImmutableList.of(errorKey));
    assertThat(result.keyNames()).isEmpty();
    assertThat(result.errorMap().keySet()).containsExactly(errorKey);
    errorInfo = result.getError();
    assertThat(errorInfo.getRootCauses()).containsExactly(errorKey);
  }

  /**
   * Regression test: "OOM in Skyframe cycle detection".
   * We only store the first 20 cycles found below any given root value.
   */
  @Test
  public void manyCycles() throws Exception {
    graph = new InMemoryGraph();
    SkyKey topKey = GraphTester.toSkyKey("top");
    for (int i = 0; i < 100; i++) {
      SkyKey dep = GraphTester.toSkyKey(Integer.toString(i));
      tester.getOrCreate(topKey).addDependency(dep);
      tester.getOrCreate(dep).addDependency(dep);
    }
    EvaluationResult<StringValue> result = eval(/*keepGoing=*/true, ImmutableList.of(topKey));
    assertEquals(null, result.get(topKey));
    assertManyCycles(result.getError(topKey), topKey, /*selfEdge=*/false);
  }

  /**
   * Regression test: "OOM in Skyframe cycle detection".
   * We filter out multiple paths to a cycle that go through the same child value.
   */
  @Test
  public void manyPathsToCycle() throws Exception {
    graph = new InMemoryGraph();
    SkyKey topKey = GraphTester.toSkyKey("top");
    SkyKey midKey = GraphTester.toSkyKey("mid");
    SkyKey cycleKey = GraphTester.toSkyKey("cycle");
    tester.getOrCreate(topKey).addDependency(midKey);
    tester.getOrCreate(cycleKey).addDependency(cycleKey);
    for (int i = 0; i < 100; i++) {
      SkyKey dep = GraphTester.toSkyKey(Integer.toString(i));
      tester.getOrCreate(midKey).addDependency(dep);
      tester.getOrCreate(dep).addDependency(cycleKey);
    }
    EvaluationResult<StringValue> result = eval(/*keepGoing=*/true, ImmutableList.of(topKey));
    assertEquals(null, result.get(topKey));
    CycleInfo cycleInfo = Iterables.getOnlyElement(result.getError(topKey).getCycleInfo());
    assertEquals(1, cycleInfo.getCycle().size());
    assertEquals(3, cycleInfo.getPathToCycle().size());
    assertThat(cycleInfo.getPathToCycle().subList(0, 2)).containsExactly(topKey, midKey).inOrder();
  }

  /**
   * Checks that errorInfo has many self-edge cycles, and that one of them is a self-edge of
   * topKey, if {@code selfEdge} is true.
   */
  private static void assertManyCycles(ErrorInfo errorInfo, SkyKey topKey, boolean selfEdge) {
    assertThat(Iterables.size(errorInfo.getCycleInfo())).isGreaterThan(1);
    assertThat(Iterables.size(errorInfo.getCycleInfo())).isLessThan(50);
    boolean foundSelfEdge = false;
    for (CycleInfo cycle : errorInfo.getCycleInfo()) {
      assertEquals(1, cycle.getCycle().size()); // Self-edge.
      if (!Iterables.isEmpty(cycle.getPathToCycle())) {
        assertThat(cycle.getPathToCycle()).containsExactly(topKey).inOrder();
      } else {
        assertThat(cycle.getCycle()).containsExactly(topKey).inOrder();
        foundSelfEdge = true;
      }
    }
    assertEquals(errorInfo + ", " + topKey, selfEdge, foundSelfEdge);
  }

  @Test
  public void manyUnprocessedValuesInCycle() throws Exception {
    graph = new InMemoryGraph();
    SkyKey lastSelfKey = GraphTester.toSkyKey("lastSelf");
    SkyKey firstSelfKey = GraphTester.toSkyKey("firstSelf");
    SkyKey midSelfKey = GraphTester.toSkyKey("midSelf");
    // We add firstSelf first so that it is processed last in cycle detection (LIFO), meaning that
    // none of the dep values have to be cleared from firstSelf.
    tester.getOrCreate(firstSelfKey).addDependency(firstSelfKey);
    for (int i = 0; i < 100; i++) {
      SkyKey firstDep = GraphTester.toSkyKey("first" + i);
      SkyKey midDep = GraphTester.toSkyKey("mid" + i);
      SkyKey lastDep = GraphTester.toSkyKey("last" + i);
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
    EvaluationResult<StringValue> result = eval(/*keepGoing=*/true,
        ImmutableList.of(lastSelfKey, firstSelfKey, midSelfKey));
    assertWithMessage(result.toString()).that(result.keyNames()).isEmpty();
    assertThat(result.errorMap().keySet()).containsExactly(lastSelfKey, firstSelfKey, midSelfKey);

    // Check lastSelfKey.
    ErrorInfo errorInfo = result.getError(lastSelfKey);
    assertEquals(errorInfo.toString(), 1, Iterables.size(errorInfo.getCycleInfo()));
    CycleInfo cycleInfo = Iterables.getOnlyElement(errorInfo.getCycleInfo());
    assertThat(cycleInfo.getCycle()).containsExactly(lastSelfKey);
    assertThat(cycleInfo.getPathToCycle()).isEmpty();

    // Check firstSelfKey. It should not have discovered its own self-edge, because there were too
    // many other values before it in the queue.
    assertManyCycles(result.getError(firstSelfKey), firstSelfKey, /*selfEdge=*/false);

    // Check midSelfKey. It should have discovered its own self-edge.
    assertManyCycles(result.getError(midSelfKey), midSelfKey, /*selfEdge=*/true);
  }

  @Test
  public void errorValueStoredWithKeepGoing() throws Exception {
    graph = new InMemoryGraph();
    SkyKey errorKey = GraphTester.toSkyKey("my_error_value");
    tester.getOrCreate(errorKey).setHasError(true);
    EvaluationResult<StringValue> result = eval(true, ImmutableList.of(errorKey));
    assertThat(result.keyNames()).isEmpty();
    assertThat(result.errorMap().keySet()).containsExactly(errorKey);
    ErrorInfo errorInfo = result.getError();
    assertThat(errorInfo.getRootCauses()).containsExactly(errorKey);
    // Update value. But builder won't rebuild it.
    tester.getOrCreate(errorKey).setHasError(false);
    tester.set(errorKey, new StringValue("no error?"));
    result = eval(true, ImmutableList.of(errorKey));
    assertThat(result.keyNames()).isEmpty();
    assertThat(result.errorMap().keySet()).containsExactly(errorKey);
    errorInfo = result.getError();
    assertThat(errorInfo.getRootCauses()).containsExactly(errorKey);
  }

  @Test
  public void continueWithErrorDep() throws Exception {
    graph = new InMemoryGraph();
    SkyKey errorKey = GraphTester.toSkyKey("my_error_value");
    tester.getOrCreate(errorKey).setHasError(true);
    tester.set("after", new StringValue("after"));
    SkyKey parentKey = GraphTester.toSkyKey("parent");
    tester.getOrCreate(parentKey).addErrorDependency(errorKey, new StringValue("recovered"))
        .setComputedValue(CONCATENATE).addDependency("after");
    EvaluationResult<StringValue> result = eval(/*keepGoing=*/true, ImmutableList.of(parentKey));
    assertThat(result.errorMap()).isEmpty();
    assertEquals("recoveredafter", result.get(parentKey).getValue());
    result = eval(/*keepGoing=*/false, ImmutableList.of(parentKey));
    assertThat(result.keyNames()).isEmpty();
    Map.Entry<SkyKey, ErrorInfo> error = Iterables.getOnlyElement(result.errorMap().entrySet());
    assertEquals(parentKey, error.getKey());
    assertThat(error.getValue().getRootCauses()).containsExactly(errorKey);
  }

  @Test
  public void transformErrorDep() throws Exception {
    graph = new InMemoryGraph();
    SkyKey errorKey = GraphTester.toSkyKey("my_error_value");
    tester.getOrCreate(errorKey).setHasError(true);
    SkyKey parentErrorKey = GraphTester.toSkyKey("parent");
    tester.getOrCreate(parentErrorKey).addErrorDependency(errorKey, new StringValue("recovered"))
        .setHasError(true);
    EvaluationResult<StringValue> result = eval(
        /*keepGoing=*/false, ImmutableList.of(parentErrorKey));
    assertThat(result.keyNames()).isEmpty();
    Map.Entry<SkyKey, ErrorInfo> error = Iterables.getOnlyElement(result.errorMap().entrySet());
    assertEquals(parentErrorKey, error.getKey());
    assertThat(error.getValue().getRootCauses()).containsExactly(parentErrorKey);
  }

  @Test
  public void transformErrorDepKeepGoing() throws Exception {
    graph = new InMemoryGraph();
    SkyKey errorKey = GraphTester.toSkyKey("my_error_value");
    tester.getOrCreate(errorKey).setHasError(true);
    SkyKey parentErrorKey = GraphTester.toSkyKey("parent");
    tester.getOrCreate(parentErrorKey).addErrorDependency(errorKey, new StringValue("recovered"))
        .setHasError(true);
    EvaluationResult<StringValue> result = eval(
        /*keepGoing=*/true, ImmutableList.of(parentErrorKey));
    assertThat(result.keyNames()).isEmpty();
    Map.Entry<SkyKey, ErrorInfo> error = Iterables.getOnlyElement(result.errorMap().entrySet());
    assertEquals(parentErrorKey, error.getKey());
    assertThat(error.getValue().getRootCauses()).containsExactly(parentErrorKey);
  }

  @Test
  public void transformErrorDepOneLevelDownKeepGoing() throws Exception {
    graph = new InMemoryGraph();
    SkyKey errorKey = GraphTester.toSkyKey("my_error_value");
    tester.getOrCreate(errorKey).setHasError(true);
    tester.set("after", new StringValue("after"));
    SkyKey parentErrorKey = GraphTester.toSkyKey("parent");
    tester.getOrCreate(parentErrorKey).addErrorDependency(errorKey, new StringValue("recovered"));
    tester.set(parentErrorKey, new StringValue("parent value"));
    SkyKey topKey = GraphTester.toSkyKey("top");
    tester.getOrCreate(topKey).addDependency(parentErrorKey).addDependency("after")
        .setComputedValue(CONCATENATE);
    EvaluationResult<StringValue> result = eval(/*keepGoing=*/true, ImmutableList.of(topKey));
    assertThat(ImmutableList.<String>copyOf(result.<String>keyNames())).containsExactly("top");
    assertEquals("parent valueafter", result.get(topKey).getValue());
    assertThat(result.errorMap()).isEmpty();
  }

  @Test
  public void transformErrorDepOneLevelDownNoKeepGoing() throws Exception {
    graph = new InMemoryGraph();
    SkyKey errorKey = GraphTester.toSkyKey("my_error_value");
    tester.getOrCreate(errorKey).setHasError(true);
    tester.set("after", new StringValue("after"));
    SkyKey parentErrorKey = GraphTester.toSkyKey("parent");
    tester.getOrCreate(parentErrorKey).addErrorDependency(errorKey, new StringValue("recovered"));
    tester.set(parentErrorKey, new StringValue("parent value"));
    SkyKey topKey = GraphTester.toSkyKey("top");
    tester.getOrCreate(topKey).addDependency(parentErrorKey).addDependency("after")
        .setComputedValue(CONCATENATE);
    EvaluationResult<StringValue> result = eval(/*keepGoing=*/false, ImmutableList.of(topKey));
    assertThat(result.keyNames()).isEmpty();
    Map.Entry<SkyKey, ErrorInfo> error = Iterables.getOnlyElement(result.errorMap().entrySet());
    assertEquals(topKey, error.getKey());
    assertThat(error.getValue().getRootCauses()).containsExactly(errorKey);
  }

  /**
   * Make sure that multiple unfinished children can be cleared from a cycle value.
   */
  @Test
  public void cycleWithMultipleUnfinishedChildren() throws Exception {
    graph = new InMemoryGraph();
    tester = new GraphTester();
    SkyKey cycleKey = GraphTester.toSkyKey("cycle");
    SkyKey midKey = GraphTester.toSkyKey("mid");
    SkyKey topKey = GraphTester.toSkyKey("top");
    SkyKey selfEdge1 = GraphTester.toSkyKey("selfEdge1");
    SkyKey selfEdge2 = GraphTester.toSkyKey("selfEdge2");
    tester.getOrCreate(topKey).addDependency(midKey).setComputedValue(CONCATENATE);
    // selfEdge* come before cycleKey, so cycleKey's path will be checked first (LIFO), and the
    // cycle with mid will be detected before the selfEdge* cycles are.
    tester.getOrCreate(midKey).addDependency(selfEdge1).addDependency(selfEdge2)
        .addDependency(cycleKey)
    .setComputedValue(CONCATENATE);
    tester.getOrCreate(cycleKey).addDependency(midKey);
    tester.getOrCreate(selfEdge1).addDependency(selfEdge1);
    tester.getOrCreate(selfEdge2).addDependency(selfEdge2);
    EvaluationResult<StringValue> result = eval(/*keepGoing=*/true, ImmutableSet.of(topKey));
    assertThat(result.errorMap().keySet()).containsExactly(topKey);
    Iterable<CycleInfo> cycleInfos = result.getError(topKey).getCycleInfo();
    CycleInfo cycleInfo = Iterables.getOnlyElement(cycleInfos);
    assertThat(cycleInfo.getPathToCycle()).containsExactly(topKey);
    assertThat(cycleInfo.getCycle()).containsExactly(midKey, cycleKey);
  }

  /**
   * Regression test: "value in cycle depends on error".
   * The mid value will have two parents -- top and cycle. Error bubbles up from mid to cycle, and
   * we should detect cycle.
   */
  private void cycleAndErrorInBubbleUp(boolean keepGoing) throws Exception {
    graph = new DeterministicInMemoryGraph();
    tester = new GraphTester();
    SkyKey errorKey = GraphTester.toSkyKey("error");
    SkyKey cycleKey = GraphTester.toSkyKey("cycle");
    SkyKey midKey = GraphTester.toSkyKey("mid");
    SkyKey topKey = GraphTester.toSkyKey("top");
    tester.getOrCreate(topKey).addDependency(midKey).setComputedValue(CONCATENATE);
    tester.getOrCreate(midKey).addDependency(errorKey).addDependency(cycleKey)
        .setComputedValue(CONCATENATE);

    // We need to ensure that cycle value has finished his work, and we have recorded dependencies
    CountDownLatch cycleFinish = new CountDownLatch(1);
    tester.getOrCreate(cycleKey).setBuilder(new ChainedFunction(null,
        null, cycleFinish, false, new StringValue(""), ImmutableSet.<SkyKey>of(midKey)));
    tester.getOrCreate(errorKey).setBuilder(new ChainedFunction(null, cycleFinish,
        null, /*waitForException=*/false, null, ImmutableSet.<SkyKey>of()));

    EvaluationResult<StringValue> result = eval(keepGoing, ImmutableSet.of(topKey));
    assertThat(result.errorMap().keySet()).containsExactly(topKey);
    Iterable<CycleInfo> cycleInfos = result.getError(topKey).getCycleInfo();
    if (keepGoing) {
      // The error thrown will only be recorded in keep_going mode.
      assertThat(result.getError().getRootCauses()).containsExactly(errorKey);
    }
    assertThat(cycleInfos).isNotEmpty();
    CycleInfo cycleInfo = Iterables.getOnlyElement(cycleInfos);
    assertThat(cycleInfo.getPathToCycle()).containsExactly(topKey);
    assertThat(cycleInfo.getCycle()).containsExactly(midKey, cycleKey);
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
   * Regression test: "value in cycle depends on error".
   * We add another value that won't finish building before the threadpool shuts down, to check that
   * the cycle detection can handle unfinished values.
   */
  @Test
  public void cycleAndErrorAndOtherInBubbleUp() throws Exception {
    graph = new DeterministicInMemoryGraph();
    tester = new GraphTester();
    SkyKey errorKey = GraphTester.toSkyKey("error");
    SkyKey cycleKey = GraphTester.toSkyKey("cycle");
    SkyKey midKey = GraphTester.toSkyKey("mid");
    SkyKey topKey = GraphTester.toSkyKey("top");
    tester.getOrCreate(topKey).addDependency(midKey).setComputedValue(CONCATENATE);
    // We should add cycleKey first and errorKey afterwards. Otherwise there is a chance that
    // during error propagation cycleKey will not be processed, and we will not detect the cycle.
    tester.getOrCreate(midKey).addDependency(errorKey).addDependency(cycleKey)
        .setComputedValue(CONCATENATE);
    SkyKey otherTop = GraphTester.toSkyKey("otherTop");
    CountDownLatch topStartAndCycleFinish = new CountDownLatch(2);
    // In nokeep_going mode, otherTop will wait until the threadpool has received an exception,
    // then request its own dep. This guarantees that there is a value that is not finished when
    // cycle detection happens.
    tester.getOrCreate(otherTop).setBuilder(new ChainedFunction(topStartAndCycleFinish,
        new CountDownLatch(0), null, /*waitForException=*/true, new StringValue("never returned"),
        ImmutableSet.<SkyKey>of(GraphTester.toSkyKey("dep that never builds"))));

    tester.getOrCreate(cycleKey).setBuilder(new ChainedFunction(null, null,
        topStartAndCycleFinish, /*waitForException=*/false, new StringValue(""),
        ImmutableSet.<SkyKey>of(midKey)));
    // error waits until otherTop starts and cycle finishes, to make sure otherTop will request
    // its dep before the threadpool shuts down.
    tester.getOrCreate(errorKey).setBuilder(new ChainedFunction(null, topStartAndCycleFinish,
        null, /*waitForException=*/false, null,
        ImmutableSet.<SkyKey>of()));
    EvaluationResult<StringValue> result =
        eval(/*keepGoing=*/false, ImmutableSet.of(topKey, otherTop));
    assertThat(result.errorMap().keySet()).containsExactly(topKey);
    Iterable<CycleInfo> cycleInfos = result.getError(topKey).getCycleInfo();
    assertThat(cycleInfos).isNotEmpty();
    CycleInfo cycleInfo = Iterables.getOnlyElement(cycleInfos);
    assertThat(cycleInfo.getPathToCycle()).containsExactly(topKey);
    assertThat(cycleInfo.getCycle()).containsExactly(midKey, cycleKey);
  }

  /**
   * Regression test: "value in cycle depends on error".
   * Here, we add an additional top-level key in error, just to mix it up.
   */
  private void cycleAndErrorAndError(boolean keepGoing) throws Exception {
    graph = new DeterministicInMemoryGraph();
    tester = new GraphTester();
    SkyKey errorKey = GraphTester.toSkyKey("error");
    SkyKey cycleKey = GraphTester.toSkyKey("cycle");
    SkyKey midKey = GraphTester.toSkyKey("mid");
    SkyKey topKey = GraphTester.toSkyKey("top");
    tester.getOrCreate(topKey).addDependency(midKey).setComputedValue(CONCATENATE);
    tester.getOrCreate(midKey).addDependency(errorKey).addDependency(cycleKey)
        .setComputedValue(CONCATENATE);
    SkyKey otherTop = GraphTester.toSkyKey("otherTop");
    CountDownLatch topStartAndCycleFinish = new CountDownLatch(2);
    // In nokeep_going mode, otherTop will wait until the threadpool has received an exception,
    // then throw its own exception. This guarantees that its exception will not be the one
    // bubbling up, but that there is a top-level value with an exception by the time the bubbling
    // up starts.
    tester.getOrCreate(otherTop).setBuilder(new ChainedFunction(topStartAndCycleFinish,
        new CountDownLatch(0), null, /*waitForException=*/!keepGoing, null,
        ImmutableSet.<SkyKey>of()));
    // error waits until otherTop starts and cycle finishes, to make sure otherTop will request
    // its dep before the threadpool shuts down.
    tester.getOrCreate(errorKey).setBuilder(new ChainedFunction(null, topStartAndCycleFinish,
        null, /*waitForException=*/false, null,
        ImmutableSet.<SkyKey>of()));
    tester.getOrCreate(cycleKey).setBuilder(new ChainedFunction(null, null,
        topStartAndCycleFinish, /*waitForException=*/false, new StringValue(""),
        ImmutableSet.<SkyKey>of(midKey)));
    EvaluationResult<StringValue> result =
        eval(keepGoing, ImmutableSet.of(topKey, otherTop));
    if (keepGoing) {
      assertThat(result.errorMap().keySet()).containsExactly(otherTop, topKey);
      assertThat(result.getError(otherTop).getRootCauses()).containsExactly(otherTop);
      // The error thrown will only be recorded in keep_going mode.
      assertThat(result.getError(topKey).getRootCauses()).containsExactly(errorKey);
    }
    Iterable<CycleInfo> cycleInfos = result.getError(topKey).getCycleInfo();
    assertThat(cycleInfos).isNotEmpty();
    CycleInfo cycleInfo = Iterables.getOnlyElement(cycleInfos);
    assertThat(cycleInfo.getPathToCycle()).containsExactly(topKey);
    assertThat(cycleInfo.getCycle()).containsExactly(midKey, cycleKey);
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
  public void testFunctionCrashTrace() throws Exception {
    final SkyFunctionName childType = SkyFunctionName.create("child");
    final SkyFunctionName parentType = SkyFunctionName.create("parent");

    class ChildFunction implements SkyFunction {
      @Override
      public SkyValue compute(SkyKey skyKey, Environment env) {
        throw new IllegalStateException("I WANT A PONY!!!");
      }

      @Override public String extractTag(SkyKey skyKey) { return null; }
    }

    class ParentFunction implements SkyFunction {
      @Override
      public SkyValue compute(SkyKey skyKey, Environment env) {
        SkyValue dep = env.getValue(new SkyKey(childType, "billy the kid"));
        if (dep == null) {
          return null;
        }
        throw new IllegalStateException();  // Should never get here.
      }

      @Override public String extractTag(SkyKey skyKey) { return null; }
    }

    ImmutableMap<SkyFunctionName, SkyFunction> skyFunctions = ImmutableMap.of(
        childType, new ChildFunction(),
        parentType, new ParentFunction());
    ParallelEvaluator evaluator = makeEvaluator(new InMemoryGraph(),
        skyFunctions, false);

    try {
      evaluator.eval(ImmutableList.of(new SkyKey(parentType, "octodad")));
      fail();
    } catch (RuntimeException e) {
      assertEquals("I WANT A PONY!!!", e.getCause().getMessage());
      assertEquals("Unrecoverable error while evaluating node 'child:billy the kid' "
          + "(requested by nodes 'parent:octodad')", e.getMessage());
    }
  }

  private static class SomeOtherErrorException extends Exception {
    public SomeOtherErrorException(String msg) {
      super(msg);
    }
  }

  private void unexpectedErrorDep(boolean keepGoing) throws Exception {
    graph = new InMemoryGraph();
    SkyKey errorKey = GraphTester.toSkyKey("my_error_value");
    final SomeOtherErrorException exception = new SomeOtherErrorException("error exception");
    tester.getOrCreate(errorKey).setBuilder(new SkyFunction() {
      @Override
      public SkyValue compute(SkyKey skyKey, Environment env) throws SkyFunctionException {
        throw new SkyFunctionException(exception, Transience.PERSISTENT) {};
      }

      @Override
      public String extractTag(SkyKey skyKey) {
        throw new UnsupportedOperationException();
      }
    });
    SkyKey topKey = GraphTester.toSkyKey("top");
    tester.getOrCreate(topKey).addErrorDependency(errorKey, new StringValue("recovered"))
        .setComputedValue(CONCATENATE);
    EvaluationResult<StringValue> result = eval(keepGoing, ImmutableList.of(topKey));
    assertThat(result.keyNames()).isEmpty();
    assertSame(exception, result.getError(topKey).getException());
    assertThat(result.getError(topKey).getRootCauses()).containsExactly(errorKey);
  }

  /**
   * This and the following three tests are in response a bug: "Skyframe error propagation model is
   * problematic". They ensure that exceptions a child throws that a value does not specify it can
   * handle in getValueOrThrow do not cause a crash.
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
    SkyKey errorKey = GraphTester.toSkyKey("my_error_value");
    final SomeErrorException exception = new SomeErrorException("error exception");
    final SomeErrorException topException = new SomeErrorException("top exception");
    final StringValue topValue = new StringValue("top");
    tester.getOrCreate(errorKey).setBuilder(new SkyFunction() {
      @Override
      public SkyValue compute(SkyKey skyKey, Environment env) throws GenericFunctionException {
        throw new GenericFunctionException(exception, Transience.PERSISTENT);
      }

      @Override
      public String extractTag(SkyKey skyKey) {
        throw new UnsupportedOperationException();
      }
    });
    SkyKey topKey = GraphTester.toSkyKey("top");
    final SkyKey parentKey = GraphTester.toSkyKey("parent");
    tester.getOrCreate(parentKey).addDependency(errorKey).setComputedValue(CONCATENATE);
    tester.getOrCreate(topKey).setBuilder(new SkyFunction() {
      @Override
      public SkyValue compute(SkyKey skyKey, Environment env) throws GenericFunctionException {
        try {
          if (env.getValueOrThrow(parentKey, SomeErrorException.class) == null) {
            return null;
          }
        } catch (SomeErrorException e) {
          assertEquals(e.toString(), exception, e);
        }
        if (keepGoing) {
          return topValue;
        } else {
          throw new GenericFunctionException(topException, Transience.PERSISTENT);
        }
      }
      @Override
      public String extractTag(SkyKey skyKey) {
        throw new UnsupportedOperationException();
      }
    });
    tester.getOrCreate(topKey).addErrorDependency(errorKey, new StringValue("recovered"))
        .setComputedValue(CONCATENATE);
    EvaluationResult<StringValue> result = eval(keepGoing, ImmutableList.of(topKey));
    if (!keepGoing) {
      assertThat(result.keyNames()).isEmpty();
      assertEquals(topException, result.getError(topKey).getException());
      assertThat(result.getError(topKey).getRootCauses()).containsExactly(topKey);
      assertThatEvaluationResult(result).hasError();
    } else {
      assertThatEvaluationResult(result).hasNoError();
      assertSame(topValue, result.get(topKey));
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
   * @param valuesOrThrow whether the deps should be requested using getValuesOrThrow.
   */
  private void sameDepInTwoGroups(final boolean sameFirst, final boolean twoCalls,
      final boolean valuesOrThrow) throws Exception {
    graph = new InMemoryGraph();
    SkyKey topKey = GraphTester.toSkyKey("top");
    final List<SkyKey> leaves = new ArrayList<>();
    for (int i = 1; i <= 3; i++) {
      SkyKey leaf = GraphTester.toSkyKey("leaf" + i);
      leaves.add(leaf);
      tester.set(leaf, new StringValue("leaf" + i));
    }
    final SkyKey leaf4 = GraphTester.toSkyKey("leaf4");
    tester.set(leaf4, new StringValue("leaf" + 4));
    tester.getOrCreate(topKey).setBuilder(new SkyFunction() {
      @Override
      public SkyValue compute(SkyKey skyKey, Environment env) throws SkyFunctionException,
          InterruptedException {
        if (valuesOrThrow) {
          env.getValuesOrThrow(leaves, SomeErrorException.class);
        } else {
          env.getValues(leaves);
        }
        if (twoCalls && env.valuesMissing()) {
          return null;
        }
        SkyKey first = sameFirst ? leaves.get(0) : leaf4;
        SkyKey second = sameFirst ? leaf4 : leaves.get(2);
        List<SkyKey> secondRequest = ImmutableList.of(first, second);
        if (valuesOrThrow) {
          env.getValuesOrThrow(secondRequest, SomeErrorException.class);
        } else {
          env.getValues(secondRequest);
        }
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
    eval(/*keepGoing=*/false, topKey);
    assertEquals(new StringValue("top"), eval(/*keepGoing=*/false, topKey));
  }

  @Test
  public void sameDepInTwoGroups_Same_Two_Throw() throws Exception {
    sameDepInTwoGroups(/*sameFirst=*/true, /*twoCalls=*/true, /*valuesOrThrow=*/true);
  }

  @Test
  public void sameDepInTwoGroups_Same_Two_Deps() throws Exception {
    sameDepInTwoGroups(/*sameFirst=*/true, /*twoCalls=*/true, /*valuesOrThrow=*/false);
  }

  @Test
  public void sameDepInTwoGroups_Same_One_Throw() throws Exception {
    sameDepInTwoGroups(/*sameFirst=*/true, /*twoCalls=*/false, /*valuesOrThrow=*/true);
  }

  @Test
  public void sameDepInTwoGroups_Same_One_Deps() throws Exception {
    sameDepInTwoGroups(/*sameFirst=*/true, /*twoCalls=*/false, /*valuesOrThrow=*/false);
  }

  @Test
  public void sameDepInTwoGroups_Different_Two_Throw() throws Exception {
    sameDepInTwoGroups(/*sameFirst=*/false, /*twoCalls=*/true, /*valuesOrThrow=*/true);
  }

  @Test
  public void sameDepInTwoGroups_Different_Two_Deps() throws Exception {
    sameDepInTwoGroups(/*sameFirst=*/false, /*twoCalls=*/true, /*valuesOrThrow=*/false);
  }

  @Test
  public void sameDepInTwoGroups_Different_One_Throw() throws Exception {
    sameDepInTwoGroups(/*sameFirst=*/false, /*twoCalls=*/false, /*valuesOrThrow=*/true);
  }

  @Test
  public void sameDepInTwoGroups_Different_One_Deps() throws Exception {
    sameDepInTwoGroups(/*sameFirst=*/false, /*twoCalls=*/false, /*valuesOrThrow=*/false);
  }

  private void getValuesOrThrowWithErrors(boolean keepGoing) throws Exception {
    graph = new InMemoryGraph();
    SkyKey parentKey = GraphTester.toSkyKey("parent");
    final SkyKey errorDep = GraphTester.toSkyKey("errorChild");
    final SomeErrorException childExn = new SomeErrorException("child error");
    tester.getOrCreate(errorDep).setBuilder(new SkyFunction() {
      @Override
      public SkyValue compute(SkyKey skyKey, Environment env) throws SkyFunctionException {
        throw new GenericFunctionException(childExn, Transience.PERSISTENT);
      }

      @Override
      public String extractTag(SkyKey skyKey) {
        return null;
      }
    });
    final List<SkyKey> deps = new ArrayList<>();
    for (int i = 1; i <= 3; i++) {
      SkyKey dep = GraphTester.toSkyKey("child" + i);
      deps.add(dep);
      tester.set(dep, new StringValue("child" + i));
    }
    final SomeErrorException parentExn = new SomeErrorException("parent error");
    tester.getOrCreate(parentKey).setBuilder(new SkyFunction() {
      @Override
      public SkyValue compute(SkyKey skyKey, Environment env) throws SkyFunctionException {
        try {
          SkyValue value = env.getValueOrThrow(errorDep, SomeErrorException.class);
          if (value == null) {
            return null;
          }
        } catch (SomeErrorException e) {
          // Recover from the child error.
        }
        env.getValues(deps);
        if (env.valuesMissing()) {
          return null;
        }
        throw new GenericFunctionException(parentExn, Transience.PERSISTENT);
      }

      @Override
      public String extractTag(SkyKey skyKey) {
        return null;
      }
    });
    EvaluationResult<StringValue> evaluationResult = eval(keepGoing, ImmutableList.of(parentKey));
    assertTrue(evaluationResult.hasError());
    assertEquals(keepGoing ? parentExn : childExn, evaluationResult.getError().getException());
  }

  @Test
  public void getValuesOrThrowWithErrors_NoKeepGoing() throws Exception {
    getValuesOrThrowWithErrors(/*keepGoing=*/false);
  }

  @Test
  public void getValuesOrThrowWithErrors_KeepGoing() throws Exception {
    getValuesOrThrowWithErrors(/*keepGoing=*/true);
  }

  @Test
  public void duplicateCycles() throws Exception {
    graph = new InMemoryGraph();
    SkyKey grandparentKey = GraphTester.toSkyKey("grandparent");
    SkyKey parentKey1 = GraphTester.toSkyKey("parent1");
    SkyKey parentKey2 = GraphTester.toSkyKey("parent2");
    SkyKey loopKey1 = GraphTester.toSkyKey("loop1");
    SkyKey loopKey2 = GraphTester.toSkyKey("loop2");
    tester.getOrCreate(loopKey1).addDependency(loopKey2);
    tester.getOrCreate(loopKey2).addDependency(loopKey1);
    tester.getOrCreate(parentKey1).addDependency(loopKey1);
    tester.getOrCreate(parentKey2).addDependency(loopKey2);
    tester.getOrCreate(grandparentKey).addDependency(parentKey1);
    tester.getOrCreate(grandparentKey).addDependency(parentKey2);

    ErrorInfo errorInfo = evalValueInError(grandparentKey);
    List<ImmutableList<SkyKey>> cycles = Lists.newArrayList();
    for (CycleInfo cycleInfo : errorInfo.getCycleInfo()) {
      cycles.add(cycleInfo.getCycle());
    }
    // Skyframe doesn't automatically dedupe cycles that are the same except for entry point.
    assertEquals(2, cycles.size());
    int numUniqueCycles = 0;
    CycleDeduper<SkyKey> cycleDeduper = new CycleDeduper<SkyKey>();
    for (ImmutableList<SkyKey> cycle : cycles) {
      if (cycleDeduper.seen(cycle)) {
        numUniqueCycles++;
      }
    }
    assertEquals(1, numUniqueCycles);
  }

  @Test
  public void signalValueEnqueuedAndEvaluated() throws Exception {
    final Set<SkyKey> enqueuedValues = Sets.newConcurrentHashSet();
    final Set<SkyKey> evaluatedValues = Sets.newConcurrentHashSet();
    EvaluationProgressReceiver progressReceiver = new EvaluationProgressReceiver() {
      @Override
      public void invalidated(SkyKey skyKey, InvalidationState state) {
        throw new IllegalStateException();
      }

      @Override
      public void enqueueing(SkyKey skyKey) {
        enqueuedValues.add(skyKey);
      }

      @Override
      public void computed(SkyKey skyKey, long elapsedTimeNanos) {}

      @Override
      public void evaluated(SkyKey skyKey, Supplier<SkyValue> skyValueSupplier,
          EvaluationState state) {
        evaluatedValues.add(skyKey);
      }
    };

    EventHandler reporter = new EventHandler() {
      @Override
      public void handle(Event e) {
        throw new IllegalStateException();
      }
    };

    MemoizingEvaluator aug = new InMemoryMemoizingEvaluator(
        ImmutableMap.of(GraphTester.NODE_TYPE, tester.getFunction()), new RecordingDifferencer(),
        progressReceiver);
    SequentialBuildDriver driver = new SequentialBuildDriver(aug);

    tester.getOrCreate("top1").setComputedValue(CONCATENATE)
        .addDependency("d1").addDependency("d2");
    tester.getOrCreate("top2").setComputedValue(CONCATENATE).addDependency("d3");
    tester.getOrCreate("top3");
    assertThat(enqueuedValues).isEmpty();
    assertThat(evaluatedValues).isEmpty();

    tester.set("d1", new StringValue("1"));
    tester.set("d2", new StringValue("2"));
    tester.set("d3", new StringValue("3"));

    driver.evaluate(ImmutableList.of(GraphTester.toSkyKey("top1")), false, 200, reporter);
    assertThat(enqueuedValues)
        .containsExactlyElementsIn(Arrays.asList(GraphTester.toSkyKeys("top1", "d1", "d2")));
    assertThat(evaluatedValues)
        .containsExactlyElementsIn(Arrays.asList(GraphTester.toSkyKeys("top1", "d1", "d2")));
    enqueuedValues.clear();
    evaluatedValues.clear();

    driver.evaluate(ImmutableList.of(GraphTester.toSkyKey("top2")), false, 200, reporter);
    assertThat(enqueuedValues)
        .containsExactlyElementsIn(Arrays.asList(GraphTester.toSkyKeys("top2", "d3")));
    assertThat(evaluatedValues)
        .containsExactlyElementsIn(Arrays.asList(GraphTester.toSkyKeys("top2", "d3")));
    enqueuedValues.clear();
    evaluatedValues.clear();

    driver.evaluate(ImmutableList.of(GraphTester.toSkyKey("top1")), false, 200, reporter);
    assertThat(enqueuedValues).isEmpty();
    assertThat(evaluatedValues)
        .containsExactlyElementsIn(Arrays.asList(GraphTester.toSkyKeys("top1")));
  }

  public void runDepOnErrorHaltsNoKeepGoingBuildEagerly(boolean childErrorCached,
      final boolean handleChildError) throws Exception {
    graph = new InMemoryGraph();
    SkyKey parentKey = GraphTester.toSkyKey("parent");
    final SkyKey childKey = GraphTester.toSkyKey("child");
    tester.getOrCreate(childKey).setHasError(/*hasError=*/true);
    // The parent should be built exactly twice: once during normal evaluation and once
    // during error bubbling.
    final AtomicInteger numParentInvocations = new AtomicInteger(0);
    tester.getOrCreate(parentKey).setBuilder(new SkyFunction() {
      @Override
      public SkyValue compute(SkyKey skyKey, Environment env) throws SkyFunctionException {
        int invocations = numParentInvocations.incrementAndGet();
        if (handleChildError) {
          try {
            SkyValue value = env.getValueOrThrow(childKey, SomeErrorException.class);
            // On the first invocation, either the child error should already be cached and not
            // propagated, or it should be computed freshly and not propagated. On the second build
            // (error bubbling), the child error should be propagated.
            assertTrue("bogus non-null value " + value, value == null);
            assertEquals("parent incorrectly re-computed during normal evaluation", 1, invocations);
            assertFalse("child error not propagated during error bubbling",
                env.inErrorBubblingForTesting());
            return value;
          } catch (SomeErrorException e) {
            assertTrue("child error propagated during normal evaluation",
                env.inErrorBubblingForTesting());
            assertEquals(2, invocations);
            return null;
          }
        } else {
          if (invocations == 1) {
            assertFalse("parent's first computation should be during normal evaluation",
                env.inErrorBubblingForTesting());
            return env.getValue(childKey);
          } else {
            assertEquals(2, invocations);
            assertTrue("parent incorrectly re-computed during normal evaluation",
                env.inErrorBubblingForTesting());
            return env.getValue(childKey);
          }
        }
      }

      @Override
      public String extractTag(SkyKey skyKey) {
        return null;
      }
    });
    if (childErrorCached) {
      // Ensure that the child is already in the graph.
      evalValueInError(childKey);
    }
    EvaluationResult<StringValue> result = eval(/*keepGoing=*/false, ImmutableList.of(parentKey));
    assertEquals(2, numParentInvocations.get());
    assertTrue(result.hasError());
    assertEquals(childKey, result.getError().getRootCauseOfException());
  }

  @Test
  public void depOnErrorHaltsNoKeepGoingBuildEagerly_ChildErrorCachedAndHandled()
      throws Exception {
    runDepOnErrorHaltsNoKeepGoingBuildEagerly(/*childErrorCached=*/true,
        /*handleChildError=*/true);
  }

  @Test
  public void depOnErrorHaltsNoKeepGoingBuildEagerly_ChildErrorCachedAndNotHandled()
      throws Exception {
    runDepOnErrorHaltsNoKeepGoingBuildEagerly(/*childErrorCached=*/true,
        /*handleChildError=*/false);
  }

  @Test
  public void depOnErrorHaltsNoKeepGoingBuildEagerly_ChildErrorFreshAndHandled() throws Exception {
    runDepOnErrorHaltsNoKeepGoingBuildEagerly(/*childErrorCached=*/false,
        /*handleChildError=*/true);
  }

  @Test
  public void depOnErrorHaltsNoKeepGoingBuildEagerly_ChildErrorFreshAndNotHandled()
      throws Exception {
    runDepOnErrorHaltsNoKeepGoingBuildEagerly(/*childErrorCached=*/false,
        /*handleChildError=*/false);
  }

  @Test
  public void raceConditionWithNoKeepGoingErrors_FutureError() throws Exception {
    final CountDownLatch errorCommitted = new CountDownLatch(1);
    final CountDownLatch otherStarted = new CountDownLatch(1);
    final CountDownLatch otherParentSignaled = new CountDownLatch(1);
    final SkyKey errorParentKey = GraphTester.toSkyKey("errorParentKey");
    final SkyKey errorKey = GraphTester.toSkyKey("errorKey");
    final SkyKey otherParentKey = GraphTester.toSkyKey("otherParentKey");
    final SkyKey otherKey = GraphTester.toSkyKey("otherKey");
    final AtomicInteger numOtherParentInvocations = new AtomicInteger(0);
    final AtomicInteger numErrorParentInvocations = new AtomicInteger(0);
    tester.getOrCreate(otherParentKey).setBuilder(new SkyFunction() {
      @Override
      public SkyValue compute(SkyKey skyKey, Environment env) throws SkyFunctionException {
        int invocations = numOtherParentInvocations.incrementAndGet();
        assertEquals("otherParentKey should not be restarted", 1, invocations);
        return env.getValue(otherKey);
      }

      @Override
      public String extractTag(SkyKey skyKey) {
        return null;
      }
    });
    tester
        .getOrCreate(otherKey)
        .setBuilder(
            new SkyFunction() {
              @Override
              public SkyValue compute(SkyKey skyKey, Environment env) throws SkyFunctionException {
                otherStarted.countDown();
                TrackingAwaiter.INSTANCE.awaitLatchAndTrackExceptions(
                    errorCommitted, "error didn't get committed to the graph in time");
                return new StringValue("other");
              }

              @Override
              public String extractTag(SkyKey skyKey) {
                return null;
              }
            });
    tester
        .getOrCreate(errorKey)
        .setBuilder(
            new SkyFunction() {
              @Override
              public SkyValue compute(SkyKey skyKey, Environment env) throws SkyFunctionException {
                TrackingAwaiter.INSTANCE.awaitLatchAndTrackExceptions(
                    otherStarted, "other didn't start in time");
                throw new GenericFunctionException(
                    new SomeErrorException("error"), Transience.PERSISTENT);
              }

              @Override
              public String extractTag(SkyKey skyKey) {
                return null;
              }
            });
    tester.getOrCreate(errorParentKey).setBuilder(new SkyFunction() {
      @Override
      public SkyValue compute(SkyKey skyKey, Environment env) throws SkyFunctionException {
        int invocations = numErrorParentInvocations.incrementAndGet();
        try {
          SkyValue value = env.getValueOrThrow(errorKey, SomeErrorException.class);
          assertTrue("bogus non-null value " + value, value == null);
          if (invocations == 1) {
            return null;
          } else {
            assertFalse(env.inErrorBubblingForTesting());
            fail("RACE CONDITION: errorParentKey was restarted!");
            return null;
          }
        } catch (SomeErrorException e) {
          assertTrue("child error propagated during normal evaluation",
              env.inErrorBubblingForTesting());
          assertEquals(2, invocations);
          return null;
        }
      }

      @Override
      public String extractTag(SkyKey skyKey) {
        return null;
      }
    });
    graph =
        new NotifyingInMemoryGraph(
            new Listener() {
              @Override
              public void accept(SkyKey key, EventType type, Order order, Object context) {
                if (key.equals(errorKey) && type == EventType.SET_VALUE && order == Order.AFTER) {
                  errorCommitted.countDown();
                  TrackingAwaiter.INSTANCE.awaitLatchAndTrackExceptions(
                      otherParentSignaled, "otherParent didn't get signaled in time");
                  // We try to give some time for ParallelEvaluator to incorrectly re-evaluate
                  // 'otherParentKey'. This test case is testing for a real race condition and the
                  // 10ms time was chosen experimentally to give a true positive rate of 99.8%
                  // (without a sleep it has a 1% true positive rate). There's no good way to do
                  // this without sleeping. We *could* introspect ParallelEvaulator's
                  // AbstractQueueVisitor to see if the re-evaluation has been enqueued, but that's
                  // relying on pretty low-level implementation details.
                  Uninterruptibles.sleepUninterruptibly(10, TimeUnit.MILLISECONDS);
                }
                if (key.equals(otherParentKey)
                    && type == EventType.SIGNAL
                    && order == Order.AFTER) {
                  otherParentSignaled.countDown();
                }
              }
            });
    EvaluationResult<StringValue> result = eval(/*keepGoing=*/false,
        ImmutableList.of(otherParentKey, errorParentKey));
    assertTrue(result.hasError());
    assertEquals(errorKey, result.getError().getRootCauseOfException());
  }

  @Test
  public void cachedErrorsFromKeepGoingUsedOnNoKeepGoing() throws Exception {
    graph = new DeterministicInMemoryGraph();
    tester = new GraphTester();
    SkyKey errorKey = GraphTester.toSkyKey("error");
    SkyKey parent1Key = GraphTester.toSkyKey("parent1");
    SkyKey parent2Key = GraphTester.toSkyKey("parent2");
    tester.getOrCreate(parent1Key).addDependency(errorKey).setConstantValue(
        new StringValue("parent1"));
    tester.getOrCreate(parent2Key).addDependency(errorKey).setConstantValue(
        new StringValue("parent2"));
    tester.getOrCreate(errorKey).setHasError(true);
    EvaluationResult<StringValue> result = eval(/*keepGoing=*/true, ImmutableList.of(parent1Key));
    assertTrue(result.hasError());
    assertEquals(errorKey, result.getError().getRootCauseOfException());
    result = eval(/*keepGoing=*/false, ImmutableList.of(parent2Key));
    assertTrue(result.hasError());
    assertEquals(errorKey, result.getError(parent2Key).getRootCauseOfException());
  }

  @Test
  public void cachedTopLevelErrorsShouldHaltNoKeepGoingBuildEarly() throws Exception {
    graph = new DeterministicInMemoryGraph();
    tester = new GraphTester();
    SkyKey errorKey = GraphTester.toSkyKey("error");
    tester.getOrCreate(errorKey).setHasError(true);
    EvaluationResult<StringValue> result = eval(/*keepGoing=*/true, ImmutableList.of(errorKey));
    assertTrue(result.hasError());
    assertEquals(errorKey, result.getError().getRootCauseOfException());
    SkyKey rogueKey = GraphTester.toSkyKey("rogue");
    tester.getOrCreate(rogueKey).setBuilder(new SkyFunction() {
      @Override
      public SkyValue compute(SkyKey skyKey, Environment env) {
        // This SkyFunction could do an arbitrarily bad computation, e.g. loop-forever. So we want
        // to make sure that it is never run when we want to fail-fast anyway.
        fail("eval call should have already terminated");
        return null;
      }

      @Override
      public String extractTag(SkyKey skyKey) {
        return null;
      }
    });
    result = eval(/*keepGoing=*/false, ImmutableList.of(errorKey, rogueKey));
    assertTrue(result.hasError());
    assertEquals(errorKey, result.getError(errorKey).getRootCauseOfException());
    assertFalse(result.errorMap().containsKey(rogueKey));
  }

  private void runUnhandledTransitiveErrors(boolean keepGoing,
      final boolean explicitlyPropagateError) throws Exception {
    graph = new DeterministicInMemoryGraph();
    tester = new GraphTester();
    SkyKey grandparentKey = GraphTester.toSkyKey("grandparent");
    final SkyKey parentKey = GraphTester.toSkyKey("parent");
    final SkyKey childKey = GraphTester.toSkyKey("child");
    final AtomicBoolean errorPropagated = new AtomicBoolean(false);
    tester.getOrCreate(grandparentKey).setBuilder(new SkyFunction() {
      @Override
      public SkyValue compute(SkyKey skyKey, Environment env) throws SkyFunctionException {
        try {
          return env.getValueOrThrow(parentKey, SomeErrorException.class);
        } catch (SomeErrorException e) {
          errorPropagated.set(true);
          throw new GenericFunctionException(e, Transience.PERSISTENT);
        }
      }

      @Override
      public String extractTag(SkyKey skyKey) {
        return null;
      }
    });
    tester.getOrCreate(parentKey).setBuilder(new SkyFunction() {
      @Override
      public SkyValue compute(SkyKey skyKey, Environment env) throws SkyFunctionException {
        if (explicitlyPropagateError) {
          try {
            return env.getValueOrThrow(childKey, SomeErrorException.class);
          } catch (SomeErrorException e) {
            throw new GenericFunctionException(e, childKey);
          }
        } else {
          return env.getValue(childKey);
        }
      }

      @Override
      public String extractTag(SkyKey skyKey) {
        return null;
      }
    });
    tester.getOrCreate(childKey).setHasError(/*hasError=*/true);
    EvaluationResult<StringValue> result = eval(keepGoing, ImmutableList.of(grandparentKey));
    assertTrue(result.hasError());
    assertTrue(errorPropagated.get());
    assertEquals(grandparentKey, result.getError().getRootCauseOfException());
  }

  @Test
  public void unhandledTransitiveErrorsDuringErrorBubbling_ImplicitPropagation() throws Exception {
    runUnhandledTransitiveErrors(/*keepGoing=*/false, /*explicitlyPropagateError=*/false);
  }

  @Test
  public void unhandledTransitiveErrorsDuringErrorBubbling_ExplicitPropagation() throws Exception {
    runUnhandledTransitiveErrors(/*keepGoing=*/false, /*explicitlyPropagateError=*/true);
  }

  @Test
  public void unhandledTransitiveErrorsDuringNormalEvaluation_ImplicitPropagation()
      throws Exception {
    runUnhandledTransitiveErrors(/*keepGoing=*/true, /*explicitlyPropagateError=*/false);
  }

  @Test
  public void unhandledTransitiveErrorsDuringNormalEvaluation_ExplicitPropagation()
      throws Exception {
    runUnhandledTransitiveErrors(/*keepGoing=*/true, /*explicitlyPropagateError=*/true);
  }
}
