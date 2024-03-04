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

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.collect.Iterables.getOnlyElement;
import static com.google.common.collect.Streams.stream;
import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;
import static com.google.devtools.build.lib.testutil.EventIterableSubjectFactory.assertThatEvents;
import static com.google.devtools.build.skyframe.EvaluationResultSubjectFactory.assertThatEvaluationResult;
import static com.google.devtools.build.skyframe.GraphTester.CONCATENATE;
import static com.google.devtools.build.skyframe.GraphTester.skyKey;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.fail;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoMoreInteractions;

import com.google.common.base.MoreObjects;
import com.google.common.base.Preconditions;
import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Interner;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import com.google.common.eventbus.EventBus;
import com.google.common.testing.GcFinalization;
import com.google.common.util.concurrent.AtomicLongMap;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.ListeningExecutorService;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.common.util.concurrent.SettableFuture;
import com.google.common.util.concurrent.Uninterruptibles;
import com.google.devtools.build.lib.bugreport.BugReporter;
import com.google.devtools.build.lib.concurrent.AbstractQueueVisitor;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.concurrent.QuiescingExecutor;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.events.ExtendedEventHandler.Postable;
import com.google.devtools.build.lib.events.Reportable;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.testutil.TestThread;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.skyframe.EvaluationContext.UnnecessaryTemporaryStateDropper;
import com.google.devtools.build.skyframe.EvaluationContext.UnnecessaryTemporaryStateDropperReceiver;
import com.google.devtools.build.skyframe.GraphTester.StringValue;
import com.google.devtools.build.skyframe.NotifyingHelper.EventType;
import com.google.devtools.build.skyframe.NotifyingHelper.Order;
import com.google.devtools.build.skyframe.PartialReevaluationMailbox.Kind;
import com.google.devtools.build.skyframe.PartialReevaluationMailbox.Mail;
import com.google.devtools.build.skyframe.SkyFunction.Environment.ClassToInstanceMapSkyKeyComputeState;
import com.google.devtools.build.skyframe.SkyFunction.Environment.SkyKeyComputeState;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.io.IOException;
import java.lang.ref.WeakReference;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import javax.annotation.Nullable;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Tests for {@link ParallelEvaluator}. */
@RunWith(TestParameterInjector.class)
public class ParallelEvaluatorTest {
  private static final SkyFunctionName CHILD_TYPE = SkyFunctionName.createHermetic("child");
  private static final SkyFunctionName PARENT_TYPE = SkyFunctionName.createHermetic("parent");

  @TestParameter private boolean useQueryDep;

  protected ProcessableGraph graph;
  protected IntVersion graphVersion = IntVersion.of(0);
  protected GraphTester tester = new GraphTester();

  private final StoredEventHandler reportedEvents = new StoredEventHandler();

  private DirtyAndInflightTrackingProgressReceiver revalidationReceiver =
      new DirtyAndInflightTrackingProgressReceiver(EvaluationProgressReceiver.NULL);

  @Before
  public void configureTesterUseLookup() {
    tester.setUseQueryDep(useQueryDep);
  }

  @After
  public void assertNoTrackedErrors() {
    TrackingAwaiter.INSTANCE.assertNoErrors();
  }

  private ParallelEvaluator makeEvaluator(
      ProcessableGraph graph,
      ImmutableMap<SkyFunctionName, SkyFunction> builders,
      boolean keepGoing,
      EventFilter storedEventFilter,
      Version evaluationVersion) {
    return new ParallelEvaluator(
        graph,
        evaluationVersion,
        Version.minimal(),
        builders,
        reportedEvents,
        new EmittedEventState(),
        storedEventFilter,
        ErrorInfoManager.UseChildErrorInfoIfNecessary.INSTANCE,
        keepGoing,
        revalidationReceiver,
        GraphInconsistencyReceiver.THROWING,
        AbstractQueueVisitor.create("test-pool", 200, ParallelEvaluatorErrorClassifier.instance()),
        new SimpleCycleDetector(),
        UnnecessaryTemporaryStateDropperReceiver.NULL);
  }

  private ParallelEvaluator makeEvaluator(
      ProcessableGraph graph,
      ImmutableMap<SkyFunctionName, SkyFunction> builders,
      boolean keepGoing,
      EventFilter storedEventFilter) {
    Version oldGraphVersion = graphVersion;
    graphVersion = graphVersion.next();
    return makeEvaluator(graph, builders, keepGoing, storedEventFilter, oldGraphVersion);
  }

  private ParallelEvaluator makeEvaluator(
      ProcessableGraph graph,
      ImmutableMap<SkyFunctionName, SkyFunction> builders,
      boolean keepGoing) {
    return makeEvaluator(graph, builders, keepGoing, EventFilter.FULL_STORAGE);
  }

  /** Convenience method for eval-ing a single value. */
  protected SkyValue eval(boolean keepGoing, SkyKey key) throws InterruptedException {
    return eval(keepGoing, ImmutableList.of(key)).get(key);
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

  private ErrorInfo evalValueInError(SkyKey key) throws InterruptedException {
    return eval(true, ImmutableList.of(key)).getError(key);
  }

  protected GraphTester.TestFunction set(String name, String value) {
    return tester.set(name, new StringValue(value));
  }

  @Test
  public void smoke() throws Exception {
    graph = new InMemoryGraphImpl();
    set("a", "a");
    set("b", "b");
    tester.getOrCreate("ab").addDependency("a").addDependency("b").setComputedValue(CONCATENATE);
    StringValue value = (StringValue) eval(false, skyKey("ab"));
    assertThat(value.getValue()).isEqualTo("ab");
    assertThat(reportedEvents.getEvents()).isEmpty();
    assertThat(reportedEvents.getPosts()).isEmpty();
  }

  @Test
  public void enqueueDoneFuture() throws Exception {
    SkyKey parentKey = skyKey("parentKey");
    tester
        .getOrCreate(parentKey)
        .setBuilder(
            (skyKey, env) -> {
              SettableFuture<SkyValue> future = SettableFuture.create();
              future.set(new StringValue("good"));
              env.dependOnFuture(future);
              assertThat(env.valuesMissing()).isFalse();
              try {
                return future.get();
              } catch (ExecutionException e) {
                throw new RuntimeException(e);
              }
            });
    graph = new InMemoryGraphImpl();
    EvaluationResult<StringValue> result =
        eval(/* keepGoing= */ false, ImmutableList.of(parentKey));
    assertThat(result.hasError()).isFalse();
    assertThat(result.get(parentKey)).isEqualTo(new StringValue("good"));
  }

  @Test
  public void enqueueBadFuture() throws Exception {
    SkyKey parentKey = skyKey("parentKey");
    CountDownLatch doneLatch = new CountDownLatch(1);
    ListeningExecutorService executor =
        MoreExecutors.listeningDecorator(Executors.newFixedThreadPool(1));
    tester
        .getOrCreate(parentKey)
        .setBuilder(
            new SkyFunction() {
              private ListenableFuture<SkyValue> future;

              @Override
              public SkyValue compute(SkyKey skyKey, Environment env) {
                if (future == null) {
                  future =
                      executor.submit(
                          () -> {
                            doneLatch.await();
                            throw new UnsupportedOperationException();
                          });
                  env.dependOnFuture(future);
                  assertThat(env.valuesMissing()).isTrue();
                  return null;
                }
                assertThat(future.isDone()).isTrue();
                ExecutionException expected =
                    assertThrows(ExecutionException.class, () -> future.get());
                assertThat(expected.getCause()).isInstanceOf(UnsupportedOperationException.class);
                return new StringValue("Caught!");
              }
            });
    graph =
        NotifyingHelper.makeNotifyingTransformer(
                (key, type, order, context) -> {
                  // NodeEntry.addExternalDep is called as part of bookkeeping at the end of
                  // AbstractParallelEvaluator.Evaluate#run.
                  if (key == parentKey && type == EventType.ADD_EXTERNAL_DEP) {
                    doneLatch.countDown();
                  }
                })
            .transform(new InMemoryGraphImpl());
    EvaluationResult<StringValue> result =
        eval(/* keepGoing= */ false, ImmutableList.of(parentKey));
    assertThat(result.hasError()).isFalse();
    assertThat(result.get(parentKey)).isEqualTo(new StringValue("Caught!"));
  }

  @Test
  public void dependsOnKeyAndFuture() throws Exception {
    SkyKey parentKey = skyKey("parentKey");
    SkyKey childKey = skyKey("childKey");
    CountDownLatch doneLatch = new CountDownLatch(1);
    tester.getOrCreate(childKey).setConstantValue(new StringValue("child"));
    tester
        .getOrCreate(parentKey)
        .setBuilder(
            new SkyFunction() {
              private SettableFuture<SkyValue> future;

              @Override
              public SkyValue compute(SkyKey skyKey, Environment env) throws InterruptedException {
                SkyValue child = env.getValue(childKey);
                if (future == null) {
                  assertThat(child).isNull();
                  future = SettableFuture.create();
                  env.dependOnFuture(future);
                  assertThat(env.valuesMissing()).isTrue();
                  new Thread(
                          () -> {
                            try {
                              doneLatch.await();
                            } catch (InterruptedException e) {
                              throw new RuntimeException(e);
                            }
                            future.set(new StringValue("future"));
                          })
                      .start();
                  return null;
                }
                assertThat(child).isEqualTo(new StringValue("child"));
                assertThat(future.isDone()).isTrue();
                try {
                  assertThat(future.get()).isEqualTo(new StringValue("future"));
                } catch (ExecutionException e) {
                  throw new RuntimeException(e);
                }
                return new StringValue("All done!");
              }
            });
    graph =
        NotifyingHelper.makeNotifyingTransformer(
                (key, type, order, context) -> {
                  if (key == childKey && type == EventType.SET_VALUE) {
                    doneLatch.countDown();
                  }
                })
            .transform(new InMemoryGraphImpl());
    EvaluationResult<StringValue> result =
        eval(/* keepGoing= */ false, ImmutableList.of(parentKey));
    assertThat(result.hasError()).isFalse();
    assertThat(result.get(parentKey)).isEqualTo(new StringValue("All done!"));
  }

  /** Test interruption handling when a long-running SkyFunction gets interrupted. */
  @Test
  public void interruptedFunction() throws Exception {
    runInterruptionTest(
        (threadStarted, errorMessage) ->
            (key, env) -> {
              // Signal the waiting test thread that the evaluator thread has really started.
              threadStarted.release();

              // Simulate a SkyFunction that runs for 10 seconds (this number was chosen
              // arbitrarily). The main thread should interrupt it shortly after it got started.
              Thread.sleep(10 * 1000);

              // Set an error message to indicate that the expected interruption didn't happen.
              // We can't use Assert.fail(String) on an async thread.
              errorMessage[0] = "SkyFunction should have been interrupted";
              return null;
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
    runInterruptionTest(
        (threadStarted, errorMessage) ->
            new SkyFunction() {
              // No need to synchronize access to this field; we always request just one more
              // dependency, so it's only one SkyFunction running at any time.
              private int valueIdCounter = 0;

              @Override
              public SkyValue compute(SkyKey key, Environment env) throws InterruptedException {
                // Signal the waiting test thread that the Evaluator thread has really started.
                threadStarted.release();

                // Keep the evaluator busy until the test's thread gets scheduled and can
                // interrupt the Evaluator's thread.
                env.getValue(skyKey("a" + valueIdCounter++));

                // This method never throws InterruptedException, therefore it's the responsibility
                // of the Evaluator to detect the interrupt and avoid calling subsequent
                // SkyFunctions.
                return null;
              }
            });
  }

  @Test
  public void interruptedEvaluatorThreadAfterEnqueueBeforeWaitForCompletionAndConstructResult()
      throws InterruptedException {
    // This is a regression test for a crash bug in
    // AbstractExceptionalParallelEvaluator#doMutatingEvaluation in a very specific window of time
    // between enqueueing one top-level node for evaluation and checking if another top-level node
    // is done.

    // When we have two top-level nodes, A and B,
    SkyKey keyA = skyKey("a");
    SkyKey keyB = skyKey("b");

    // And rig the graph and node entries, such that B's addReverseDepAndCheckIfDone waits for A to
    // start computing and then tries to observe an interrupt (which will happen on the calling
    // thread, aka the main Skyframe evaluation thread),
    CountDownLatch keyAStartedComputingLatch = new CountDownLatch(1);
    CountDownLatch keyBAddReverseDepAndCheckIfDoneLatch = new CountDownLatch(1);
    InMemoryNodeEntry nodeEntryB = spy(new IncrementalInMemoryNodeEntry(keyB));
    AtomicBoolean keyBAddReverseDepAndCheckIfDoneInterrupted = new AtomicBoolean(false);
    doAnswer(
            invocation -> {
              keyAStartedComputingLatch.await();
              keyBAddReverseDepAndCheckIfDoneLatch.countDown();
              try {
                Thread.sleep(TestUtils.WAIT_TIMEOUT_MILLISECONDS);
                throw new IllegalStateException("shouldn't get here");
              } catch (InterruptedException e) {
                keyBAddReverseDepAndCheckIfDoneInterrupted.set(true);
                throw e;
              }
            })
        .when(nodeEntryB)
        .addReverseDepAndCheckIfDone(eq(null));
    graph =
        new InMemoryGraphImpl() {
          @Override
          protected InMemoryNodeEntry newNodeEntry(SkyKey key) {
            return key.equals(keyB) ? nodeEntryB : super.newNodeEntry(key);
          }
        };
    // And A's SkyFunction tries to observe an interrupt after it starts computing,
    AtomicBoolean keyAComputeInterrupted = new AtomicBoolean(false);
    tester
        .getOrCreate(keyA)
        .setBuilder(
            (skyKey, env) -> {
              keyAStartedComputingLatch.countDown();
              try {
                Thread.sleep(TestUtils.WAIT_TIMEOUT_MILLISECONDS);
                throw new IllegalStateException("shouldn't get here");
              } catch (InterruptedException e) {
                keyAComputeInterrupted.set(true);
                throw e;
              }
            });

    // And we have a dedicated thread that kicks off the evaluation of A and B together (in that
    // order).
    TestThread evalThread =
        new TestThread(
            () ->
                assertThrows(
                    InterruptedException.class, () -> eval(/* keepGoing= */ true, keyA, keyB)));

    // Then when we start that thread,
    evalThread.start();
    // We (the thread running the test) are able to observe that B's addReverseDepAndCheckIfDone has
    // just been called (implying that A has started to be computed).
    assertThat(
            keyBAddReverseDepAndCheckIfDoneLatch.await(
                TestUtils.WAIT_TIMEOUT_SECONDS, TimeUnit.SECONDS))
        .isTrue();
    // Then when we interrupt the evaluation thread,
    evalThread.interrupt();
    // The evaluation thread eventually terminates.
    evalThread.joinAndAssertState(TestUtils.WAIT_TIMEOUT_MILLISECONDS);
    // And we are able to verify both that A's SkyFunction had observed an interrupt,
    assertThat(keyAComputeInterrupted.get()).isTrue();
    // And also that B's addReverseDepAndCheckIfDoneInterrupted had observed an interrupt.
    assertThat(keyBAddReverseDepAndCheckIfDoneInterrupted.get()).isTrue();
  }

  @Test
  public void runPartialResultOnInterruption(@TestParameter boolean buildFastFirst)
      throws Exception {
    graph = new InMemoryGraphImpl();
    // Two runs for fastKey's builder and one for the start of waitKey's builder.
    CountDownLatch allValuesReady = new CountDownLatch(3);
    SkyKey waitKey = skyKey("wait");
    SkyKey fastKey = skyKey("fast");
    SkyKey leafKey = skyKey("leaf");
    tester
        .getOrCreate(waitKey)
        .setBuilder(
            (skyKey, env) -> {
              allValuesReady.countDown();
              Thread.sleep(10000);
              throw new AssertionError("Should have been interrupted");
            });
    tester
        .getOrCreate(fastKey)
        .setBuilder(
            new ChainedFunction(
                null,
                null,
                allValuesReady,
                false,
                new StringValue("fast"),
                ImmutableList.of(leafKey)));
    tester.set(leafKey, new StringValue("leaf"));
    if (buildFastFirst) {
      eval(/* keepGoing= */ false, fastKey);
    }
    Set<SkyKey> receivedValues = Sets.newConcurrentHashSet();
    revalidationReceiver =
        new DirtyAndInflightTrackingProgressReceiver(
            new EvaluationProgressReceiver() {
              @Override
              public void evaluated(
                  SkyKey skyKey,
                  EvaluationState state,
                  @Nullable SkyValue newValue,
                  @Nullable ErrorInfo newError,
                  @Nullable GroupedDeps directDeps) {
                receivedValues.add(skyKey);
              }
            });
    TestThread evalThread =
        new TestThread(
            () ->
                assertThrows(
                    InterruptedException.class,
                    () -> eval(/* keepGoing= */ true, waitKey, fastKey)));
    evalThread.start();
    assertThat(allValuesReady.await(TestUtils.WAIT_TIMEOUT_SECONDS, TimeUnit.SECONDS)).isTrue();
    evalThread.interrupt();
    evalThread.join(TestUtils.WAIT_TIMEOUT_MILLISECONDS);
    assertThat(evalThread.isAlive()).isFalse();
    if (buildFastFirst) {
      // If leafKey was already built, it is not reported to the receiver.
      assertThat(receivedValues).containsExactly(fastKey);
    } else {
      // On first time being built, leafKey is registered too.
      assertThat(receivedValues).containsExactly(fastKey, leafKey);
    }
  }

  /** Factory for SkyFunctions for interruption testing (see {@link #runInterruptionTest}). */
  private interface SkyFunctionFactory {
    /**
     * Creates a SkyFunction suitable for a specific test scenario.
     *
     * @param threadStarted a latch which the returned SkyFunction must {@link Semaphore#release()
     *     release} once it started (otherwise the test won't work)
     * @param errorMessage a single-element array; the SkyFunction can put a error message in it to
     *     indicate that an assertion failed (calling {@code fail} from async thread doesn't work)
     */
    SkyFunction create(Semaphore threadStarted, String[] errorMessage);
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
    Semaphore threadStarted = new Semaphore(0);
    Semaphore threadInterrupted = new Semaphore(0);
    String[] wasError = new String[] {null};
    ParallelEvaluator evaluator =
        makeEvaluator(
            new InMemoryGraphImpl(),
            ImmutableMap.of(
                GraphTester.NODE_TYPE, valueBuilderFactory.create(threadStarted, wasError)),
            false);

    Thread t =
        new Thread(
            () -> {
              try {
                evaluator.eval(ImmutableList.of(skyKey("a")));

                // There's no real need to set an error here. If the thread is not interrupted then
                // threadInterrupted is not released and the test thread will fail to acquire it.
                wasError[0] = "evaluation should have been interrupted";
              } catch (InterruptedException e) {
                // This is the interrupt we are waiting for. It should come straight from the
                // evaluator (more precisely, the AbstractQueueVisitor).
                // Signal the waiting test thread that the interrupt was acknowledged.
                threadInterrupted.release();
              }
            });

    // Start the thread and wait for a semaphore. This ensures that the thread was really started.
    t.start();
    assertThat(threadStarted.tryAcquire(TestUtils.WAIT_TIMEOUT_MILLISECONDS, TimeUnit.MILLISECONDS))
        .isTrue();

    // Interrupt the thread and wait for a semaphore. This ensures that the thread was really
    // interrupted and this fact was acknowledged.
    t.interrupt();
    assertThat(
            threadInterrupted.tryAcquire(
                TestUtils.WAIT_TIMEOUT_MILLISECONDS, TimeUnit.MILLISECONDS))
        .isTrue();

    // The SkyFunction may have reported an error.
    if (wasError[0] != null) {
      fail(wasError[0]);
    }

    // Wait for the thread to finish.
    t.join(TestUtils.WAIT_TIMEOUT_MILLISECONDS);
  }

  @Test
  public void unrecoverableError() {
    class CustomRuntimeException extends RuntimeException {}
    CustomRuntimeException expected = new CustomRuntimeException();

    SkyFunction builder =
        new SkyFunction() {
          @Override
          @Nullable
          public SkyValue compute(SkyKey skyKey, Environment env) {
            throw expected;
          }
        };

    ParallelEvaluator evaluator =
        makeEvaluator(
            new InMemoryGraphImpl(), ImmutableMap.of(GraphTester.NODE_TYPE, builder), false);

    SkyKey valueToEval = skyKey("a");
    RuntimeException re =
        assertThrows(RuntimeException.class, () -> evaluator.eval(ImmutableList.of(valueToEval)));
    assertThat(re)
        .hasMessageThat()
        .contains("Unrecoverable error while evaluating node '" + valueToEval + "'");
    assertThat(re).hasCauseThat().isInstanceOf(CustomRuntimeException.class);
  }

  @Test
  public void simpleWarning() throws Exception {
    graph = new InMemoryGraphImpl();
    set("a", "a").setWarning("warning on 'a'");
    StringValue value = (StringValue) eval(false, skyKey("a"));
    assertThat(value.getValue()).isEqualTo("a");
    assertThatEvents(reportedEvents.getEvents()).containsExactly("warning on 'a'");
  }

  @Test
  public void errorOfTopLevelTargetReported() throws Exception {
    graph = new InMemoryGraphImpl();
    SkyKey a = skyKey("a");
    SkyKey b = skyKey("b");
    tester.getOrCreate(b).setHasError(true);
    Event errorEvent = Event.error("foobar");
    tester
        .getOrCreate(a)
        .setBuilder(
            (key, env) -> {
              try {
                if (env.getValueOrThrow(b, SomeErrorException.class) == null) {
                  return null;
                }
              } catch (SomeErrorException ignored) {
                // Continue silently.
              }
              env.getListener().handle(errorEvent);
              throw new SkyFunctionException(
                  new SomeErrorException("bazbar"), Transience.PERSISTENT) {};
            });
    eval(false, a);
    assertThat(reportedEvents.getEvents()).containsExactly(errorEvent);
  }

  private static final class ExamplePost implements Postable {
    private final boolean storeForReplay;

    ExamplePost(boolean storeForReplay) {
      this.storeForReplay = storeForReplay;
    }

    @Override
    public boolean storeForReplay() {
      return storeForReplay;
    }

    @Override
    public String toString() {
      return MoreObjects.toStringHelper(this).add("storeForReplay", storeForReplay).toString();
    }
  }

  private enum SkyframeEventType {
    EVENT {
      @Override
      Event createUnstored() {
        return Event.progress("analyzing");
      }

      @Override
      Reportable createStored() {
        return Event.error("broken");
      }

      @Override
      ImmutableList<Event> getResults(StoredEventHandler reportedEvents) {
        return reportedEvents.getEvents();
      }
    },
    POST {
      @Override
      Postable createUnstored() {
        return new ExamplePost(false);
      }

      @Override
      Postable createStored() {
        return new ExamplePost(true);
      }

      @Override
      ImmutableList<Postable> getResults(StoredEventHandler reportedEvents) {
        return reportedEvents.getPosts();
      }
    };

    abstract Reportable createUnstored();

    abstract Reportable createStored();

    abstract ImmutableList<? extends Reportable> getResults(StoredEventHandler reportedEvents);
  }

  @Test
  public void fullEventStorage_unstoredEvent_reportedImmediately_notReplayed(
      @TestParameter SkyframeEventType eventType) throws Exception {
    graph = new InMemoryGraphImpl();
    SkyKey key = skyKey("key");
    AtomicBoolean evaluated = new AtomicBoolean(false);
    Reportable unstoredEvent = eventType.createUnstored();
    assertThat(unstoredEvent.storeForReplay()).isFalse();
    tester
        .getOrCreate(key)
        .setBuilder(
            (skyKey, env) -> {
              evaluated.set(true);
              unstoredEvent.reportTo(env.getListener());
              assertThat(eventType.getResults(reportedEvents)).containsExactly(unstoredEvent);
              return new StringValue("value");
            });
    ParallelEvaluator evaluator =
        makeEvaluator(
            graph, tester.getSkyFunctionMap(), /* keepGoing= */ false, EventFilter.FULL_STORAGE);
    evaluator.eval(ImmutableList.of(key));
    assertThat(evaluated.get()).isTrue();
    assertThat(eventType.getResults(reportedEvents)).containsExactly(unstoredEvent);

    reportedEvents.clear();
    evaluated.set(false);

    evaluator =
        makeEvaluator(
            graph, tester.getSkyFunctionMap(), /* keepGoing= */ false, EventFilter.FULL_STORAGE);
    evaluator.eval(ImmutableList.of(key));
    assertThat(evaluated.get()).isFalse();
    assertThat(eventType.getResults(reportedEvents)).isEmpty();
  }

  @Test
  public void fullEventStorage_storedEvent_reportedAfterSkyFunctionCompletes_replayed(
      @TestParameter SkyframeEventType eventType) throws Exception {
    graph = new InMemoryGraphImpl();
    SkyKey top = skyKey("top");
    SkyKey mid = skyKey("mid");
    SkyKey bottom = skyKey("bottom");
    String tag = "this is the tag";
    AtomicBoolean evaluatedMid = new AtomicBoolean(false);
    Reportable storedEvent = eventType.createStored();
    assertThat(storedEvent.storeForReplay()).isTrue();
    Reportable taggedEvent = storedEvent.withTag(tag);
    tester
        .getOrCreate(top)
        .setBuilder(
            (skyKey, env) -> {
              SkyValue midValue = env.getValue(mid);
              if (midValue == null) {
                return null;
              }
              assertThat(eventType.getResults(reportedEvents)).containsExactly(taggedEvent);
              return new StringValue("topValue");
            });
    tester
        .getOrCreate(mid)
        .setBuilder(
            new SkyFunction() {
              @Nullable
              @Override
              public SkyValue compute(SkyKey skyKey, Environment env) throws InterruptedException {
                evaluatedMid.set(true);
                storedEvent.reportTo(env.getListener());
                assertThat(eventType.getResults(reportedEvents)).isEmpty();
                SkyValue bottomValue = env.getValue(bottom);
                if (bottomValue == null) {
                  return null;
                }
                return new StringValue("midValue");
              }

              @Override
              public String extractTag(SkyKey skyKey) {
                assertThat(skyKey).isEqualTo(mid);
                return tag;
              }
            });
    tester.getOrCreate(bottom).setConstantValue(new StringValue("depValue"));
    ParallelEvaluator evaluator =
        makeEvaluator(
            graph, tester.getSkyFunctionMap(), /* keepGoing= */ false, EventFilter.FULL_STORAGE);
    evaluator.eval(ImmutableList.of(top));
    assertThat(evaluatedMid.get()).isTrue();
    assertThat(eventType.getResults(reportedEvents)).containsExactly(taggedEvent);

    reportedEvents.clear();
    evaluatedMid.set(false);

    evaluator =
        makeEvaluator(
            graph, tester.getSkyFunctionMap(), /* keepGoing= */ false, EventFilter.FULL_STORAGE);
    evaluator.eval(ImmutableList.of(top));
    assertThat(evaluatedMid.get()).isFalse();
    assertThat(eventType.getResults(reportedEvents)).containsExactly(taggedEvent);
  }

  @Test
  public void noEventStorage_unstoredEvent_reportedImmediately_notReplayed(
      @TestParameter SkyframeEventType eventType) throws Exception {
    graph = new InMemoryGraphImpl();
    SkyKey key = skyKey("key");
    AtomicBoolean evaluated = new AtomicBoolean(false);
    Reportable unstoredEvent = eventType.createUnstored();
    assertThat(unstoredEvent.storeForReplay()).isFalse();
    tester
        .getOrCreate(key)
        .setBuilder(
            (skyKey, env) -> {
              evaluated.set(true);
              unstoredEvent.reportTo(env.getListener());
              assertThat(eventType.getResults(reportedEvents)).containsExactly(unstoredEvent);
              return new StringValue("value");
            });
    ParallelEvaluator evaluator =
        makeEvaluator(
            graph, tester.getSkyFunctionMap(), /* keepGoing= */ false, EventFilter.NO_STORAGE);
    evaluator.eval(ImmutableList.of(key));
    assertThat(evaluated.get()).isTrue();
    assertThat(eventType.getResults(reportedEvents)).containsExactly(unstoredEvent);

    reportedEvents.clear();
    evaluated.set(false);

    evaluator =
        makeEvaluator(
            graph, tester.getSkyFunctionMap(), /* keepGoing= */ false, EventFilter.NO_STORAGE);
    evaluator.eval(ImmutableList.of(key));
    assertThat(evaluated.get()).isFalse();
    assertThat(eventType.getResults(reportedEvents)).isEmpty();
  }

  @Test
  public void noEventStorage_storedEvent_reportedAfterSkyFunctionCompletes_notReplayed(
      @TestParameter SkyframeEventType eventType) throws Exception {
    graph = new InMemoryGraphImpl();
    SkyKey top = skyKey("top");
    SkyKey mid = skyKey("mid");
    SkyKey bottom = skyKey("bottom");
    String tag = "this is the tag";
    AtomicBoolean evaluatedMid = new AtomicBoolean(false);
    Reportable storedEvent = eventType.createStored();
    assertThat(storedEvent.storeForReplay()).isTrue();
    Reportable taggedEvent = storedEvent.withTag(tag);
    tester
        .getOrCreate(top)
        .setBuilder(
            (skyKey, env) -> {
              SkyValue midValue = env.getValue(mid);
              if (midValue == null) {
                return null;
              }
              assertThat(eventType.getResults(reportedEvents)).containsExactly(taggedEvent);
              return new StringValue("topValue");
            });
    tester
        .getOrCreate(mid)
        .setBuilder(
            new SkyFunction() {
              @Nullable
              @Override
              public SkyValue compute(SkyKey skyKey, Environment env) throws InterruptedException {
                evaluatedMid.set(true);
                storedEvent.reportTo(env.getListener());
                assertThat(eventType.getResults(reportedEvents)).isEmpty();
                SkyValue bottomValue = env.getValue(bottom);
                if (bottomValue == null) {
                  return null;
                }
                return new StringValue("midValue");
              }

              @Override
              public String extractTag(SkyKey skyKey) {
                assertThat(skyKey).isEqualTo(mid);
                return tag;
              }
            });
    tester.getOrCreate(bottom).setConstantValue(new StringValue("depValue"));
    ParallelEvaluator evaluator =
        makeEvaluator(
            graph, tester.getSkyFunctionMap(), /* keepGoing= */ false, EventFilter.NO_STORAGE);
    evaluator.eval(ImmutableList.of(top));
    assertThat(evaluatedMid.get()).isTrue();
    assertThat(eventType.getResults(reportedEvents)).containsExactly(taggedEvent);

    reportedEvents.clear();
    evaluatedMid.set(false);

    evaluator =
        makeEvaluator(
            graph, tester.getSkyFunctionMap(), /* keepGoing= */ false, EventFilter.NO_STORAGE);
    evaluator.eval(ImmutableList.of(top));
    assertThat(evaluatedMid.get()).isFalse();
    assertThat(eventType.getResults(reportedEvents)).isEmpty();
  }

  @Test
  public void shouldCreateErrorValueWithRootCause() throws Exception {
    graph = new InMemoryGraphImpl();
    set("a", "a");
    SkyKey parentErrorKey = skyKey("parent");
    SkyKey errorKey = skyKey("error");
    tester
        .getOrCreate(parentErrorKey)
        .addDependency("a")
        .addDependency(errorKey)
        .setComputedValue(CONCATENATE);
    tester.getOrCreate(errorKey).setHasError(true);
    evalValueInError(parentErrorKey);
  }

  @Test
  public void shouldBuildOneTarget() throws Exception {
    graph = new InMemoryGraphImpl();
    set("a", "a");
    set("b", "b");
    SkyKey parentErrorKey = skyKey("parent");
    SkyKey errorFreeKey = skyKey("ab");
    SkyKey errorKey = skyKey("error");
    tester
        .getOrCreate(parentErrorKey)
        .addDependency(errorKey)
        .addDependency("a")
        .setComputedValue(CONCATENATE);
    tester.getOrCreate(errorKey).setHasError(true);
    tester
        .getOrCreate(errorFreeKey)
        .addDependency("a")
        .addDependency("b")
        .setComputedValue(CONCATENATE);
    EvaluationResult<StringValue> result = eval(true, parentErrorKey, errorFreeKey);
    assertThatEvaluationResult(result).hasErrorEntryForKeyThat(parentErrorKey);
    assertThatEvaluationResult(result).hasEntryThat(errorFreeKey).isEqualTo(new StringValue("ab"));
  }

  @Test
  public void catastrophicBuild(@TestParameter boolean keepGoing, @TestParameter boolean keepEdges)
      throws Exception {
    Assume.assumeTrue(keepGoing || keepEdges);

    graph =
        keepEdges
            ? InMemoryGraph.create(/* usePooledInterning= */ true)
            : InMemoryGraph.createEdgeless(/* usePooledInterning= */ true);

    SkyKey catastropheKey = skyKey("catastrophe");
    SkyKey otherKey = skyKey("someKey");

    Exception catastrophe = new SomeErrorException("bad");
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
            });

    tester
        .getOrCreate(otherKey)
        .setBuilder(
            new SkyFunction() {
              @Nullable
              @Override
              public SkyValue compute(SkyKey skyKey, Environment env) throws InterruptedException {
                new CountDownLatch(1).await();
                throw new RuntimeException("can't get here");
              }
            });

    SkyKey topKey = skyKey("top");
    tester.getOrCreate(topKey).addDependency(catastropheKey).setComputedValue(CONCATENATE);

    ParallelEvaluator evaluator =
        makeEvaluator(
            graph,
            tester.getSkyFunctionMap(),
            keepGoing,
            EventFilter.FULL_STORAGE,
            keepEdges ? graphVersion : Version.constant());

    EvaluationResult<StringValue> result = evaluator.eval(ImmutableList.of(topKey, otherKey));
    assertThatEvaluationResult(result).hasErrorEntryForKeyThat(topKey);
    if (keepGoing) {
      assertThat(result.getCatastrophe()).isSameInstanceAs(catastrophe);
    }
  }

  @Test
  public void topCatastrophe() throws Exception {
    graph = new InMemoryGraphImpl();
    SkyKey catastropheKey = skyKey("catastrophe");
    Exception catastrophe = new SomeErrorException("bad");
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
            });

    EvaluationResult<StringValue> result =
        eval(/* keepGoing= */ true, ImmutableList.of(catastropheKey));
    assertThat(result.getCatastrophe()).isEqualTo(catastrophe);
  }

  @Test
  public void catastropheBubblesIntoNonCatastrophe() throws Exception {
    graph = new InMemoryGraphImpl();
    SkyKey catastropheKey = skyKey("catastrophe");
    Exception catastrophe = new SomeErrorException("bad");
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
            });
    SkyKey topKey = skyKey("top");
    tester
        .getOrCreate(topKey)
        .setBuilder(
            new SkyFunction() {
              @Nullable
              @Override
              public SkyValue compute(SkyKey skyKey, Environment env)
                  throws SkyFunctionException, InterruptedException {
                try {
                  env.getValueOrThrow(catastropheKey, SomeErrorException.class);
                } catch (SomeErrorException e) {
                  throw new SkyFunctionException(
                      new SomeErrorException("We got: " + e.getMessage()),
                      Transience.PERSISTENT) {};
                }
                return null;
              }
            });
    EvaluationResult<StringValue> result = eval(/* keepGoing= */ true, ImmutableList.of(topKey));

    assertThat(result.getError(topKey).getException()).isInstanceOf(SomeErrorException.class);
    assertThat(result.getError(topKey).getException()).hasMessageThat().isEqualTo("We got: bad");
    assertThat(result.getCatastrophe()).isNotNull();
  }

  @Test
  public void incrementalCycleWithCatastropheAndFailedBubbleUp() throws Exception {
    SkyKey topKey = skyKey("top");
    // Comes alphabetically before "top".
    SkyKey cycleKey = skyKey("cycle");
    SkyKey catastropheKey = skyKey("catastrophe");
    graph = new DeterministicHelper.DeterministicProcessableGraph(new InMemoryGraphImpl());
    StringValue topValue = new StringValue("top");
    tester
        .getOrCreate(topKey)
        .setBuilder(
            new SkyFunction() {
              @Nullable
              @Override
              public SkyValue compute(SkyKey skyKey, Environment env) throws InterruptedException {
                env.getValue(cycleKey);
                return env.valuesMissing() ? null : topValue;
              }
            });
    tester
        .getOrCreate(cycleKey)
        .setBuilder(
            new SkyFunction() {
              @Nullable
              @Override
              public SkyValue compute(SkyKey skyKey, Environment env) throws InterruptedException {
                env.getValuesAndExceptions(ImmutableList.of(cycleKey, catastropheKey));
                Preconditions.checkState(env.valuesMissing());
                return null;
              }
            });
    tester
        .getOrCreate(catastropheKey)
        .setBuilder(
            new SkyFunction() {
              @Nullable
              @Override
              public SkyValue compute(SkyKey skyKey, Environment env) throws SkyFunctionException {
                throw new SkyFunctionException(
                    new SomeErrorException("catastrophe"), Transience.TRANSIENT) {
                  @Override
                  public boolean isCatastrophic() {
                    return true;
                  }
                };
              }
            });
    EvaluationResult<StringValue> result = eval(/* keepGoing= */ true, ImmutableList.of(topKey));
    assertThatEvaluationResult(result).hasError();
    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(topKey)
        .hasCycleInfoThat()
        .containsExactly(new CycleInfo(ImmutableList.of(topKey), ImmutableList.of(cycleKey)));
  }

  @Test
  public void parentFailureDoesntAffectChild() throws Exception {
    graph = new InMemoryGraphImpl();
    SkyKey parentKey = skyKey("parent");
    tester.getOrCreate(parentKey).setHasError(true);
    SkyKey childKey = skyKey("child");
    set("child", "onions");
    tester.getOrCreate(parentKey).addDependency(childKey).setComputedValue(CONCATENATE);
    EvaluationResult<StringValue> result = eval(/* keepGoing= */ true, parentKey, childKey);
    // Child is guaranteed to complete successfully before parent can run (and fail),
    // since parent depends on it.
    assertThatEvaluationResult(result).hasEntryThat(childKey).isEqualTo(new StringValue("onions"));
    assertThatEvaluationResult(result).hasErrorEntryForKeyThat(parentKey);
  }

  @Test
  public void newParentOfErrorShouldHaveError() throws Exception {
    graph = new InMemoryGraphImpl();
    SkyKey errorKey = skyKey("error");
    tester.getOrCreate(errorKey).setHasError(true);
    evalValueInError(errorKey);
    SkyKey parentKey = skyKey("parent");
    tester.getOrCreate(parentKey).addDependency("error").setComputedValue(CONCATENATE);
    evalValueInError(parentKey);
  }

  @Test
  public void errorTwoLevelsDeep() throws Exception {
    graph = new InMemoryGraphImpl();
    SkyKey parentKey = skyKey("parent");
    SkyKey errorKey = skyKey("error");
    tester.getOrCreate(errorKey).setHasError(true);
    tester.getOrCreate("mid").addDependency(errorKey).setComputedValue(CONCATENATE);
    tester.getOrCreate(parentKey).addDependency("mid").setComputedValue(CONCATENATE);
    evalValueInError(parentKey);
  }

  @Test
  public void valueNotUsedInFailFastErrorRecovery() throws Exception {
    graph = new InMemoryGraphImpl();
    SkyKey topKey = skyKey("top");
    SkyKey recoveryKey = skyKey("midRecovery");
    SkyKey badKey = skyKey("bad");

    tester.getOrCreate(topKey).addDependency(recoveryKey).setComputedValue(CONCATENATE);
    tester
        .getOrCreate(recoveryKey)
        .addErrorDependency(badKey, new StringValue("i recovered"))
        .setComputedValue(CONCATENATE);
    tester.getOrCreate(badKey).setHasError(true);

    EvaluationResult<SkyValue> result = eval(/* keepGoing= */ true, ImmutableList.of(recoveryKey));
    assertThat(result.errorMap()).isEmpty();
    assertThatEvaluationResult(result).hasNoError();
    assertThat(result.get(recoveryKey)).isEqualTo(new StringValue("i recovered"));

    result = eval(/* keepGoing= */ false, ImmutableList.of(topKey));
    assertThatEvaluationResult(result).hasError();
    assertThat(result.keyNames()).isEmpty();
    assertThat(result.errorMap()).hasSize(1);
    assertThat(result.getError(topKey).getException()).isNotNull();
  }

  @Test
  public void multipleRootCauses() throws Exception {
    graph = new InMemoryGraphImpl();
    SkyKey parentKey = skyKey("parent");
    SkyKey errorKey = skyKey("error");
    SkyKey errorKey2 = skyKey("error2");
    SkyKey errorKey3 = skyKey("error3");
    tester.getOrCreate(errorKey).setHasError(true);
    tester.getOrCreate(errorKey2).setHasError(true);
    tester.getOrCreate(errorKey3).setHasError(true);
    tester
        .getOrCreate("mid")
        .addDependency(errorKey)
        .addDependency(errorKey2)
        .setComputedValue(CONCATENATE);
    tester
        .getOrCreate(parentKey)
        .addDependency("mid")
        .addDependency(errorKey2)
        .addDependency(errorKey3)
        .setComputedValue(CONCATENATE);
    evalValueInError(parentKey);
  }

  @Test
  public void rootCauseWithNoKeepGoing() throws Exception {
    graph = new InMemoryGraphImpl();
    SkyKey parentKey = skyKey("parent");
    SkyKey errorKey = skyKey("error");
    tester.getOrCreate(errorKey).setHasError(true);
    tester.getOrCreate("mid").addDependency(errorKey).setComputedValue(CONCATENATE);
    tester.getOrCreate(parentKey).addDependency("mid").setComputedValue(CONCATENATE);
    EvaluationResult<StringValue> result = eval(false, ImmutableList.of(parentKey));
    assertThatEvaluationResult(result).hasSingletonErrorThat(parentKey);
  }

  @Test
  public void errorBubblesToParentsOfTopLevelValue() throws Exception {
    SkyKey parentKey = skyKey("parent");
    SkyKey errorKey = skyKey("error");
    CountDownLatch latch = new CountDownLatch(1);
    graph =
        new NotifyingHelper.NotifyingProcessableGraph(
            new InMemoryGraphImpl(),
            (key, type, order, context) -> {
              if (key.equals(errorKey)
                  && parentKey.equals(context)
                  && type == EventType.ADD_REVERSE_DEP
                  && order == Order.AFTER) {
                latch.countDown();
              }
            });
    tester
        .getOrCreate(errorKey)
        .setBuilder(
            new ChainedFunction(
                null,
                /* waitToFinish= */ latch,
                null,
                false,
                /* value= */ null,
                ImmutableList.of()));
    tester.getOrCreate(parentKey).addDependency(errorKey).setComputedValue(CONCATENATE);
    EvaluationResult<StringValue> result =
        eval(/* keepGoing= */ false, ImmutableList.of(parentKey, errorKey));
    assertWithMessage(result.toString()).that(result.errorMap().size()).isEqualTo(2);
  }

  @Test
  public void noKeepGoingAfterKeepGoingFails() throws Exception {
    graph = new InMemoryGraphImpl();
    SkyKey errorKey = skyKey("my_error_value");
    tester.getOrCreate(errorKey).setHasError(true);
    SkyKey parentKey = skyKey("parent");
    tester.getOrCreate(parentKey).addDependency(errorKey);
    evalValueInError(parentKey);
    SkyKey[] list = {parentKey};
    EvaluationResult<StringValue> result = eval(false, list);
    assertThatEvaluationResult(result)
        .hasSingletonErrorThat(parentKey)
        .hasExceptionThat()
        .hasMessageThat()
        .isEqualTo(errorKey.toString());
  }

  @Test
  public void twoErrors() throws Exception {
    graph = new InMemoryGraphImpl();
    SkyKey firstError = skyKey("error1");
    SkyKey secondError = skyKey("error2");
    CountDownLatch firstStart = new CountDownLatch(1);
    CountDownLatch secondStart = new CountDownLatch(1);
    tester
        .getOrCreate(firstError)
        .setBuilder(
            new ChainedFunction(
                firstStart,
                secondStart,
                /* notifyFinish= */ null,
                /* waitForException= */ false,
                /* value= */ null,
                ImmutableList.of()));
    tester
        .getOrCreate(secondError)
        .setBuilder(
            new ChainedFunction(
                secondStart,
                firstStart,
                /* notifyFinish= */ null,
                /* waitForException= */ false,
                /* value= */ null,
                ImmutableList.of()));
    EvaluationResult<StringValue> result = eval(/* keepGoing= */ false, firstError, secondError);
    assertWithMessage(result.toString()).that(result.hasError()).isTrue();
    // With keepGoing=false, the eval call will terminate with exactly one error (the first one
    // thrown). But the first one thrown here is non-deterministic since we synchronize the
    // builders so that they run at roughly the same time.
    assertThat(ImmutableSet.of(firstError, secondError))
        .contains(Iterables.getOnlyElement(result.errorMap().keySet()));
  }

  @Test
  public void simpleCycle() throws Exception {
    graph = new InMemoryGraphImpl();
    SkyKey aKey = skyKey("a");
    SkyKey bKey = skyKey("b");
    tester.getOrCreate(aKey).addDependency(bKey);
    tester.getOrCreate(bKey).addDependency(aKey);
    ErrorInfo errorInfo = eval(false, ImmutableList.of(aKey)).getError();
    assertThat(errorInfo.getException()).isNull();
    CycleInfo cycleInfo = Iterables.getOnlyElement(errorInfo.getCycleInfo());
    assertThat(cycleInfo.getCycle()).containsExactly(aKey, bKey).inOrder();
    assertThat(cycleInfo.getPathToCycle()).isEmpty();
  }

  @Test
  public void cycleWithHead() throws Exception {
    graph = new InMemoryGraphImpl();
    SkyKey aKey = skyKey("a");
    SkyKey bKey = skyKey("b");
    SkyKey topKey = skyKey("top");
    SkyKey midKey = skyKey("mid");
    tester.getOrCreate(topKey).addDependency(midKey);
    tester.getOrCreate(midKey).addDependency(aKey);
    tester.getOrCreate(aKey).addDependency(bKey);
    tester.getOrCreate(bKey).addDependency(aKey);
    ErrorInfo errorInfo = eval(false, ImmutableList.of(topKey)).getError();
    assertThat(errorInfo.getException()).isNull();
    CycleInfo cycleInfo = Iterables.getOnlyElement(errorInfo.getCycleInfo());
    assertThat(cycleInfo.getCycle()).containsExactly(aKey, bKey).inOrder();
    assertThat(cycleInfo.getPathToCycle()).containsExactly(topKey, midKey).inOrder();
  }

  @Test
  public void selfEdgeWithHead() throws Exception {
    graph = new InMemoryGraphImpl();
    SkyKey aKey = skyKey("a");
    SkyKey topKey = skyKey("top");
    SkyKey midKey = skyKey("mid");
    tester.getOrCreate(topKey).addDependency(midKey);
    tester.getOrCreate(midKey).addDependency(aKey);
    tester.getOrCreate(aKey).addDependency(aKey);
    ErrorInfo errorInfo = eval(false, ImmutableList.of(topKey)).getError();
    assertThat(errorInfo.getException()).isNull();
    CycleInfo cycleInfo = Iterables.getOnlyElement(errorInfo.getCycleInfo());
    assertThat(cycleInfo.getCycle()).containsExactly(aKey).inOrder();
    assertThat(cycleInfo.getPathToCycle()).containsExactly(topKey, midKey).inOrder();
  }

  @Test
  public void cycleWithKeepGoing() throws Exception {
    graph = new InMemoryGraphImpl();
    SkyKey aKey = skyKey("a");
    SkyKey bKey = skyKey("b");
    SkyKey topKey = skyKey("top");
    SkyKey midKey = skyKey("mid");
    SkyKey goodKey = skyKey("good");
    StringValue goodValue = new StringValue("good");
    tester.set(goodKey, goodValue);
    tester.getOrCreate(topKey).addDependency(midKey);
    tester.getOrCreate(midKey).addDependency(aKey);
    tester.getOrCreate(aKey).addDependency(bKey);
    tester.getOrCreate(bKey).addDependency(aKey);
    EvaluationResult<StringValue> result = eval(true, topKey, goodKey);
    assertThat(result.get(goodKey)).isEqualTo(goodValue);
    assertThat(result.get(topKey)).isNull();
    ErrorInfo errorInfo = result.getError(topKey);
    CycleInfo cycleInfo = Iterables.getOnlyElement(errorInfo.getCycleInfo());
    assertThat(cycleInfo.getCycle()).containsExactly(aKey, bKey).inOrder();
    assertThat(cycleInfo.getPathToCycle()).containsExactly(topKey, midKey).inOrder();
  }

  @Test
  public void twoCycles() throws Exception {
    graph = new InMemoryGraphImpl();
    SkyKey aKey = skyKey("a");
    SkyKey bKey = skyKey("b");
    SkyKey cKey = skyKey("c");
    SkyKey dKey = skyKey("d");
    SkyKey topKey = skyKey("top");
    tester.getOrCreate(topKey).addDependency(aKey).addDependency(cKey);
    tester.getOrCreate(aKey).addDependency(bKey);
    tester.getOrCreate(bKey).addDependency(aKey);
    tester.getOrCreate(cKey).addDependency(dKey);
    tester.getOrCreate(dKey).addDependency(cKey);
    EvaluationResult<StringValue> result = eval(false, ImmutableList.of(topKey));
    assertThat(result.get(topKey)).isNull();
    ErrorInfo errorInfo = result.getError(topKey);
    Iterable<CycleInfo> cycles =
        CycleInfo.prepareCycles(
            topKey,
            ImmutableList.of(
                new CycleInfo(ImmutableList.of(aKey, bKey)),
                new CycleInfo(ImmutableList.of(cKey, dKey))));
    assertThat(cycles).contains(getOnlyElement(errorInfo.getCycleInfo()));
  }

  @Test
  public void twoCyclesKeepGoing() throws Exception {
    graph = new InMemoryGraphImpl();
    SkyKey aKey = skyKey("a");
    SkyKey bKey = skyKey("b");
    SkyKey cKey = skyKey("c");
    SkyKey dKey = skyKey("d");
    SkyKey topKey = skyKey("top");
    tester.getOrCreate(topKey).addDependency(aKey).addDependency(cKey);
    tester.getOrCreate(aKey).addDependency(bKey);
    tester.getOrCreate(bKey).addDependency(aKey);
    tester.getOrCreate(cKey).addDependency(dKey);
    tester.getOrCreate(dKey).addDependency(cKey);
    EvaluationResult<StringValue> result = eval(true, ImmutableList.of(topKey));
    assertThat(result.get(topKey)).isNull();
    ErrorInfo errorInfo = result.getError(topKey);
    CycleInfo aCycle = new CycleInfo(ImmutableList.of(topKey), ImmutableList.of(aKey, bKey));
    CycleInfo cCycle = new CycleInfo(ImmutableList.of(topKey), ImmutableList.of(cKey, dKey));
    assertThat(errorInfo.getCycleInfo()).containsExactly(aCycle, cCycle);
  }

  @Test
  public void triangleBelowHeadCycle() throws Exception {
    graph = new DeterministicHelper.DeterministicProcessableGraph(new InMemoryGraphImpl());
    SkyKey aKey = skyKey("a");
    SkyKey bKey = skyKey("b");
    SkyKey cKey = skyKey("c");
    SkyKey topKey = skyKey("top");
    tester.getOrCreate(topKey).addDependency(aKey);
    tester.getOrCreate(aKey).addDependency(bKey).addDependency(cKey);
    tester.getOrCreate(bKey).addDependency(cKey);
    tester.getOrCreate(cKey).addDependency(topKey);
    EvaluationResult<StringValue> result = eval(true, ImmutableList.of(topKey));
    assertThat(result.get(topKey)).isNull();
    ErrorInfo errorInfo = result.getError(topKey);
    CycleInfo topCycle = new CycleInfo(ImmutableList.of(topKey, aKey, cKey));
    assertThat(errorInfo.getCycleInfo()).containsExactly(topCycle);
  }

  @Test
  public void longCycle() throws Exception {
    graph = new InMemoryGraphImpl();
    SkyKey aKey = skyKey("a");
    SkyKey bKey = skyKey("b");
    SkyKey cKey = skyKey("c");
    SkyKey topKey = skyKey("top");
    tester.getOrCreate(topKey).addDependency(aKey);
    tester.getOrCreate(aKey).addDependency(bKey);
    tester.getOrCreate(bKey).addDependency(cKey);
    tester.getOrCreate(cKey).addDependency(topKey);
    EvaluationResult<StringValue> result = eval(true, ImmutableList.of(topKey));
    assertThat(result.get(topKey)).isNull();
    ErrorInfo errorInfo = result.getError(topKey);
    CycleInfo topCycle = new CycleInfo(ImmutableList.of(topKey, aKey, bKey, cKey));
    assertThat(errorInfo.getCycleInfo()).containsExactly(topCycle);
  }

  @Test
  public void cycleWithTail() throws Exception {
    graph = new InMemoryGraphImpl();
    SkyKey aKey = skyKey("a");
    SkyKey bKey = skyKey("b");
    SkyKey cKey = skyKey("c");
    SkyKey topKey = skyKey("top");
    tester.getOrCreate(topKey).addDependency(aKey).addDependency(cKey);
    tester.getOrCreate(aKey).addDependency(bKey);
    tester.getOrCreate(bKey).addDependency(aKey).addDependency(cKey);
    tester.getOrCreate(cKey);
    tester.set(cKey, new StringValue("cValue"));
    EvaluationResult<StringValue> result = eval(false, ImmutableList.of(topKey));
    assertThat(result.get(topKey)).isNull();
    ErrorInfo errorInfo = result.getError(topKey);
    CycleInfo cycleInfo = Iterables.getOnlyElement(errorInfo.getCycleInfo());
    assertThat(cycleInfo.getCycle()).containsExactly(aKey, bKey).inOrder();
    assertThat(cycleInfo.getPathToCycle()).containsExactly(topKey).inOrder();
  }

  /** Regression test: "value cannot be ready in a cycle". */
  @Test
  public void selfEdgeWithExtraChildrenUnderCycle() throws Exception {
    graph = new DeterministicHelper.DeterministicProcessableGraph(new InMemoryGraphImpl());
    SkyKey aKey = skyKey("a");
    SkyKey zKey = skyKey("z");
    SkyKey cKey = skyKey("c");
    tester.getOrCreate(aKey).addDependency(zKey);
    tester.getOrCreate(zKey).addDependency(cKey).addDependency(zKey);
    tester.getOrCreate(cKey).addDependency(aKey);
    EvaluationResult<StringValue> result = eval(/* keepGoing= */ true, ImmutableList.of(aKey));
    assertThat(result.get(aKey)).isNull();
    ErrorInfo errorInfo = result.getError(aKey);
    CycleInfo cycleInfo = Iterables.getOnlyElement(errorInfo.getCycleInfo());
    assertThat(cycleInfo.getCycle()).containsExactly(zKey).inOrder();
    assertThat(cycleInfo.getPathToCycle()).containsExactly(aKey).inOrder();
  }

  /** Regression test: "value cannot be ready in a cycle". */
  @Test
  public void cycleWithExtraChildrenUnderCycle() throws Exception {
    graph = new DeterministicHelper.DeterministicProcessableGraph(new InMemoryGraphImpl());
    SkyKey aKey = skyKey("a");
    SkyKey bKey = skyKey("b");
    SkyKey cKey = skyKey("c");
    SkyKey dKey = skyKey("d");
    tester.getOrCreate(aKey).addDependency(bKey);
    tester.getOrCreate(bKey).addDependency(cKey).addDependency(dKey);
    tester.getOrCreate(cKey).addDependency(aKey);
    tester.getOrCreate(dKey).addDependency(bKey);
    EvaluationResult<StringValue> result = eval(/* keepGoing= */ true, ImmutableList.of(aKey));
    assertThat(result.get(aKey)).isNull();
    ErrorInfo errorInfo = result.getError(aKey);
    CycleInfo cycleInfo = Iterables.getOnlyElement(errorInfo.getCycleInfo());
    assertThat(cycleInfo.getCycle()).containsExactly(bKey, dKey).inOrder();
    assertThat(cycleInfo.getPathToCycle()).containsExactly(aKey).inOrder();
  }

  /** Regression test: "value cannot be ready in a cycle". */
  @Test
  public void cycleAboveIndependentCycle() throws Exception {
    graph = new DeterministicHelper.DeterministicProcessableGraph(new InMemoryGraphImpl());
    SkyKey aKey = skyKey("a");
    SkyKey bKey = skyKey("b");
    SkyKey cKey = skyKey("c");
    tester.getOrCreate(aKey).addDependency(bKey);
    tester.getOrCreate(bKey).addDependency(cKey);
    tester.getOrCreate(cKey).addDependency(aKey).addDependency(bKey);
    EvaluationResult<StringValue> result = eval(/* keepGoing= */ true, ImmutableList.of(aKey));
    assertThat(result.get(aKey)).isNull();
    assertThat(result.getError(aKey).getCycleInfo())
        .containsExactly(
            new CycleInfo(ImmutableList.of(aKey, bKey, cKey)),
            new CycleInfo(ImmutableList.of(aKey), ImmutableList.of(bKey, cKey)));
  }

  @Test
  public void valueAboveCycleAndExceptionReportsException() throws Exception {
    graph = new InMemoryGraphImpl();
    SkyKey aKey = skyKey("a");
    SkyKey errorKey = skyKey("error");
    SkyKey bKey = skyKey("b");
    tester.getOrCreate(aKey).addDependency(bKey).addDependency(errorKey);
    tester.getOrCreate(bKey).addDependency(bKey);
    tester.getOrCreate(errorKey).setHasError(true);
    EvaluationResult<StringValue> result = eval(/* keepGoing= */ true, ImmutableList.of(aKey));
    assertThat(result.get(aKey)).isNull();
    assertThat(result.getError(aKey).getException()).isNotNull();
    CycleInfo cycleInfo = Iterables.getOnlyElement(result.getError(aKey).getCycleInfo());
    assertThat(cycleInfo.getCycle()).containsExactly(bKey).inOrder();
    assertThat(cycleInfo.getPathToCycle()).containsExactly(aKey).inOrder();
  }

  @Test
  public void errorValueStored() throws Exception {
    graph = new InMemoryGraphImpl();
    SkyKey errorKey = skyKey("my_error_value");
    tester.getOrCreate(errorKey).setHasError(true);
    EvaluationResult<StringValue> result = eval(false, ImmutableList.of(errorKey));
    assertThatEvaluationResult(result).hasSingletonErrorThat(errorKey);
    // Update value. But builder won't rebuild it.
    tester.getOrCreate(errorKey).setHasError(false);
    tester.set(errorKey, new StringValue("no error?"));
    result = eval(false, ImmutableList.of(errorKey));
    assertThatEvaluationResult(result).hasSingletonErrorThat(errorKey);
  }

  /**
   * Regression test: "OOM in Skyframe cycle detection". We only store the first 20 cycles found
   * below any given root value.
   */
  @Test
  public void manyCycles() throws Exception {
    graph = new InMemoryGraphImpl();
    SkyKey topKey = skyKey("top");
    for (int i = 0; i < 100; i++) {
      SkyKey dep = skyKey(Integer.toString(i));
      tester.getOrCreate(topKey).addDependency(dep);
      tester.getOrCreate(dep).addDependency(dep);
    }
    EvaluationResult<StringValue> result = eval(/* keepGoing= */ true, ImmutableList.of(topKey));
    assertThat(result.get(topKey)).isNull();
    assertManyCycles(result.getError(topKey), topKey, /* selfEdge= */ false);
  }

  /**
   * Regression test: "OOM in Skyframe cycle detection". We filter out multiple paths to a cycle
   * that go through the same child value.
   */
  @Test
  public void manyPathsToCycle() throws Exception {
    graph = new InMemoryGraphImpl();
    SkyKey topKey = skyKey("top");
    SkyKey midKey = skyKey("mid");
    SkyKey cycleKey = skyKey("cycle");
    tester.getOrCreate(topKey).addDependency(midKey);
    tester.getOrCreate(cycleKey).addDependency(cycleKey);
    for (int i = 0; i < 100; i++) {
      SkyKey dep = skyKey(Integer.toString(i));
      tester.getOrCreate(midKey).addDependency(dep);
      tester.getOrCreate(dep).addDependency(cycleKey);
    }
    EvaluationResult<StringValue> result = eval(/* keepGoing= */ true, ImmutableList.of(topKey));
    assertThat(result.get(topKey)).isNull();
    CycleInfo cycleInfo = Iterables.getOnlyElement(result.getError(topKey).getCycleInfo());
    assertThat(cycleInfo.getCycle()).hasSize(1);
    assertThat(cycleInfo.getPathToCycle()).hasSize(3);
    assertThat(cycleInfo.getPathToCycle().subList(0, 2)).containsExactly(topKey, midKey).inOrder();
  }

  /**
   * Checks that errorInfo has many self-edge cycles, and that one of them is a self-edge of topKey,
   * if {@code selfEdge} is true.
   */
  private static void assertManyCycles(ErrorInfo errorInfo, SkyKey topKey, boolean selfEdge) {
    assertThat(Iterables.size(errorInfo.getCycleInfo())).isGreaterThan(1);
    assertThat(Iterables.size(errorInfo.getCycleInfo())).isLessThan(50);
    boolean foundSelfEdge = false;
    for (CycleInfo cycle : errorInfo.getCycleInfo()) {
      assertThat(cycle.getCycle()).hasSize(1); // Self-edge.
      if (!Iterables.isEmpty(cycle.getPathToCycle())) {
        assertThat(cycle.getPathToCycle()).containsExactly(topKey).inOrder();
      } else {
        assertThat(cycle.getCycle()).containsExactly(topKey).inOrder();
        foundSelfEdge = true;
      }
    }
    assertWithMessage(errorInfo + ", " + topKey).that(foundSelfEdge).isEqualTo(selfEdge);
  }

  @Test
  public void manyUnprocessedValuesInCycle() throws Exception {
    graph = new DeterministicHelper.DeterministicProcessableGraph(new InMemoryGraphImpl());
    SkyKey lastSelfKey = skyKey("zlastSelf");
    SkyKey firstSelfKey = skyKey("afirstSelf");
    SkyKey midSelfKey = skyKey("midSelf9");
    // We add firstSelf first so that it is processed last in cycle detection (LIFO), meaning that
    // none of the dep values have to be cleared from firstSelf.
    tester.getOrCreate(firstSelfKey).addDependency(firstSelfKey);
    for (int i = 0; i < 100; i++) {
      SkyKey firstDep = skyKey("first" + i);
      SkyKey midDep = skyKey("midSelf" + i + "dep");
      SkyKey lastDep = skyKey("last" + i);
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
    EvaluationResult<StringValue> result =
        eval(/* keepGoing= */ true, ImmutableList.of(lastSelfKey, firstSelfKey, midSelfKey));
    assertWithMessage(result.toString()).that(result.keyNames()).isEmpty();
    assertThat(result.errorMap().keySet()).containsExactly(lastSelfKey, firstSelfKey, midSelfKey);

    // Check lastSelfKey.
    ErrorInfo errorInfo = result.getError(lastSelfKey);
    assertWithMessage(errorInfo.toString())
        .that(Iterables.size(errorInfo.getCycleInfo()))
        .isEqualTo(1);
    CycleInfo cycleInfo = Iterables.getOnlyElement(errorInfo.getCycleInfo());
    assertThat(cycleInfo.getCycle()).containsExactly(lastSelfKey);
    assertThat(cycleInfo.getPathToCycle()).isEmpty();

    // Check firstSelfKey. It should not have discovered its own self-edge, because there were too
    // many other values before it in the queue.
    assertManyCycles(result.getError(firstSelfKey), firstSelfKey, /* selfEdge= */ false);

    // Check midSelfKey. It should have discovered its own self-edge.
    assertManyCycles(result.getError(midSelfKey), midSelfKey, /* selfEdge= */ true);
  }

  @Test
  public void errorValueStoredWithKeepGoing() throws Exception {
    graph = new InMemoryGraphImpl();
    SkyKey errorKey = skyKey("my_error_value");
    tester.getOrCreate(errorKey).setHasError(true);
    EvaluationResult<StringValue> result = eval(true, ImmutableList.of(errorKey));
    assertThatEvaluationResult(result).hasSingletonErrorThat(errorKey);
    // Update value. But builder won't rebuild it.
    tester.getOrCreate(errorKey).setHasError(false);
    tester.set(errorKey, new StringValue("no error?"));
    result = eval(true, ImmutableList.of(errorKey));
    assertThatEvaluationResult(result).hasSingletonErrorThat(errorKey);
  }

  @Test
  public void continueWithErrorDep() throws Exception {
    graph = new InMemoryGraphImpl();
    SkyKey errorKey = skyKey("my_error_value");
    tester.getOrCreate(errorKey).setHasError(true);
    tester.set("after", new StringValue("after"));
    SkyKey parentKey = skyKey("parent");
    tester
        .getOrCreate(parentKey)
        .addErrorDependency(errorKey, new StringValue("recovered"))
        .setComputedValue(CONCATENATE)
        .addDependency("after");
    EvaluationResult<StringValue> result = eval(/* keepGoing= */ true, ImmutableList.of(parentKey));
    assertThat(result.errorMap()).isEmpty();
    assertThat(result.get(parentKey).getValue()).isEqualTo("recoveredafter");
    result = eval(/* keepGoing= */ false, ImmutableList.of(parentKey));
    assertThatEvaluationResult(result).hasSingletonErrorThat(parentKey);
  }

  @Test
  public void transformErrorDep(@TestParameter boolean keepGoing) throws Exception {
    graph = new InMemoryGraphImpl();
    SkyKey errorKey = skyKey("my_error_value");
    tester.getOrCreate(errorKey).setHasError(true);
    SkyKey parentErrorKey = skyKey("parent");
    tester
        .getOrCreate(parentErrorKey)
        .addErrorDependency(errorKey, new StringValue("recovered"))
        .setHasError(true);
    EvaluationResult<StringValue> result = eval(keepGoing, ImmutableList.of(parentErrorKey));
    assertThatEvaluationResult(result).hasSingletonErrorThat(parentErrorKey);
  }

  @Test
  public void transformErrorDepOneLevelDownKeepGoing() throws Exception {
    graph = new InMemoryGraphImpl();
    SkyKey errorKey = skyKey("my_error_value");
    tester.getOrCreate(errorKey).setHasError(true);
    tester.set("after", new StringValue("after"));
    SkyKey parentErrorKey = skyKey("parent");
    tester.getOrCreate(parentErrorKey).addErrorDependency(errorKey, new StringValue("recovered"));
    tester.set(parentErrorKey, new StringValue("parent value"));
    SkyKey topKey = skyKey("top");
    tester
        .getOrCreate(topKey)
        .addDependency(parentErrorKey)
        .addDependency("after")
        .setComputedValue(CONCATENATE);
    EvaluationResult<StringValue> result = eval(/* keepGoing= */ true, ImmutableList.of(topKey));
    assertThat(ImmutableList.<String>copyOf(result.keyNames())).containsExactly("top");
    assertThat(result.get(topKey).getValue()).isEqualTo("parent valueafter");
    assertThat(result.errorMap()).isEmpty();
  }

  @Test
  public void transformErrorDepOneLevelDownNoKeepGoing() throws Exception {
    graph = new InMemoryGraphImpl();
    SkyKey errorKey = skyKey("my_error_value");
    tester.getOrCreate(errorKey).setHasError(true);
    tester.set("after", new StringValue("after"));
    SkyKey parentErrorKey = skyKey("parent");
    tester.getOrCreate(parentErrorKey).addErrorDependency(errorKey, new StringValue("recovered"));
    tester.set(parentErrorKey, new StringValue("parent value"));
    SkyKey topKey = skyKey("top");
    tester
        .getOrCreate(topKey)
        .addDependency(parentErrorKey)
        .addDependency("after")
        .setComputedValue(CONCATENATE);
    EvaluationResult<StringValue> result = eval(/* keepGoing= */ false, ImmutableList.of(topKey));
    assertThatEvaluationResult(result).hasSingletonErrorThat(topKey);
  }

  @Test
  public void errorDepDoesntStopOtherDep() throws Exception {
    graph = new InMemoryGraphImpl();
    SkyKey errorKey = skyKey("error");
    tester.getOrCreate(errorKey).setHasError(true);
    EvaluationResult<StringValue> result1 = eval(/* keepGoing= */ true, ImmutableList.of(errorKey));
    assertThatEvaluationResult(result1).hasError();
    assertThatEvaluationResult(result1)
        .hasErrorEntryForKeyThat(errorKey)
        .hasExceptionThat()
        .isNotNull();
    SkyKey otherKey = skyKey("other");
    tester.getOrCreate(otherKey).setConstantValue(new StringValue("other"));
    SkyKey topKey = skyKey("top");
    Exception topException = new SomeErrorException("top exception");
    AtomicInteger numComputes = new AtomicInteger(0);
    tester
        .getOrCreate(topKey)
        .setBuilder(
            new SkyFunction() {
              @Nullable
              @Override
              public SkyValue compute(SkyKey skyKey, Environment env)
                  throws SkyFunctionException, InterruptedException {
                SkyframeLookupResult values =
                    env.getValuesAndExceptions(ImmutableList.of(errorKey, otherKey));
                if (numComputes.incrementAndGet() == 1) {
                  assertThat(env.valuesMissing()).isTrue();
                } else {
                  assertThat(numComputes.get()).isEqualTo(2);
                  assertThat(env.valuesMissing()).isFalse();
                }
                try {
                  values.getOrThrow(errorKey, SomeErrorException.class);
                  throw new AssertionError("Should have thrown");
                } catch (SomeErrorException e) {
                  throw new SkyFunctionException(topException, Transience.PERSISTENT) {};
                }
              }
            });
    EvaluationResult<StringValue> result2 = eval(/* keepGoing= */ true, ImmutableList.of(topKey));
    assertThatEvaluationResult(result2).hasError();
    assertThatEvaluationResult(result2)
        .hasErrorEntryForKeyThat(topKey)
        .hasExceptionThat()
        .isSameInstanceAs(topException);
    assertThat(numComputes.get()).isEqualTo(2);
  }

  /** Make sure that multiple unfinished children can be cleared from a cycle value. */
  @Test
  public void cycleWithMultipleUnfinishedChildren() throws Exception {
    graph = new DeterministicHelper.DeterministicProcessableGraph(new InMemoryGraphImpl());
    SkyKey cycleKey = skyKey("zcycle");
    SkyKey midKey = skyKey("mid");
    SkyKey topKey = skyKey("top");
    SkyKey selfEdge1 = skyKey("selfEdge1");
    SkyKey selfEdge2 = skyKey("selfEdge2");
    tester.getOrCreate(topKey).addDependency(midKey).setComputedValue(CONCATENATE);
    // selfEdge* come before cycleKey, so cycleKey's path will be checked first (LIFO), and the
    // cycle with mid will be detected before the selfEdge* cycles are.
    tester
        .getOrCreate(midKey)
        .addDependency(selfEdge1)
        .addDependency(selfEdge2)
        .addDependency(cycleKey)
        .setComputedValue(CONCATENATE);
    tester.getOrCreate(cycleKey).addDependency(midKey);
    tester.getOrCreate(selfEdge1).addDependency(selfEdge1);
    tester.getOrCreate(selfEdge2).addDependency(selfEdge2);
    EvaluationResult<StringValue> result = eval(/* keepGoing= */ true, ImmutableSet.of(topKey));
    assertThat(result.errorMap().keySet()).containsExactly(topKey);
    Iterable<CycleInfo> cycleInfos = result.getError(topKey).getCycleInfo();
    CycleInfo cycleInfo = Iterables.getOnlyElement(cycleInfos);
    assertThat(cycleInfo.getPathToCycle()).containsExactly(topKey);
    assertThat(cycleInfo.getCycle()).containsExactly(midKey, cycleKey);
  }

  /**
   * Regression test: "value in cycle depends on error". The mid value will have two parents -- top
   * and cycle. Error bubbles up from mid to cycle, and we should detect cycle.
   */
  @Test
  public void cycleAndErrorInBubbleUp(@TestParameter boolean keepGoing) throws Exception {
    graph = new DeterministicHelper.DeterministicProcessableGraph(new InMemoryGraphImpl());
    tester = new GraphTester();
    SkyKey errorKey = skyKey("error");
    SkyKey cycleKey = skyKey("cycle");
    SkyKey midKey = skyKey("mid");
    SkyKey topKey = skyKey("top");
    tester.getOrCreate(topKey).addDependency(midKey).setComputedValue(CONCATENATE);
    tester
        .getOrCreate(midKey)
        .addDependency(errorKey)
        .addDependency(cycleKey)
        .setComputedValue(CONCATENATE);

    // We need to ensure that cycle value has finished its work, and we have recorded dependencies
    CountDownLatch cycleFinish = new CountDownLatch(1);
    tester
        .getOrCreate(cycleKey)
        .setBuilder(
            new ChainedFunction(
                null, null, cycleFinish, false, new StringValue(""), ImmutableSet.of(midKey)));
    tester
        .getOrCreate(errorKey)
        .setBuilder(
            new ChainedFunction(
                null, cycleFinish, null, /* waitForException= */ false, null, ImmutableSet.of()));

    EvaluationResult<StringValue> result = eval(keepGoing, ImmutableSet.of(topKey));
    assertThatEvaluationResult(result)
        .hasSingletonErrorThat(topKey)
        .hasCycleInfoThat()
        .containsExactly(
            new CycleInfo(ImmutableList.of(topKey), ImmutableList.of(midKey, cycleKey)));
  }

  /**
   * Regression test: "value in cycle depends on error". We add another value that won't finish
   * building before the threadpool shuts down, to check that the cycle detection can handle
   * unfinished values.
   */
  @Test
  public void cycleAndErrorAndOtherInBubbleUp() throws Exception {
    graph = new DeterministicHelper.DeterministicProcessableGraph(new InMemoryGraphImpl());
    tester = new GraphTester();
    SkyKey errorKey = skyKey("error");
    SkyKey cycleKey = skyKey("cycle");
    SkyKey midKey = skyKey("mid");
    SkyKey topKey = skyKey("top");
    tester.getOrCreate(topKey).addDependency(midKey).setComputedValue(CONCATENATE);
    // We should add cycleKey first and errorKey afterwards. Otherwise there is a chance that
    // during error propagation cycleKey will not be processed, and we will not detect the cycle.
    tester
        .getOrCreate(midKey)
        .addDependency(errorKey)
        .addDependency(cycleKey)
        .setComputedValue(CONCATENATE);
    SkyKey otherTop = skyKey("otherTop");
    CountDownLatch topStartAndCycleFinish = new CountDownLatch(2);
    // In nokeep_going mode, otherTop will wait until the threadpool has received an exception,
    // then request its own dep. This guarantees that there is a value that is not finished when
    // cycle detection happens.
    tester
        .getOrCreate(otherTop)
        .setBuilder(
            new ChainedFunction(
                topStartAndCycleFinish,
                new CountDownLatch(0),
                null,
                /* waitForException= */ true,
                new StringValue("never returned"),
                ImmutableSet.of(skyKey("dep that never builds"))));

    tester
        .getOrCreate(cycleKey)
        .setBuilder(
            new ChainedFunction(
                null,
                null,
                topStartAndCycleFinish,
                /* waitForException= */ false,
                new StringValue(""),
                ImmutableSet.of(midKey)));
    // error waits until otherTop starts and cycle finishes, to make sure otherTop will request
    // its dep before the threadpool shuts down.
    tester
        .getOrCreate(errorKey)
        .setBuilder(
            new ChainedFunction(
                null,
                topStartAndCycleFinish,
                null,
                /* waitForException= */ false,
                null,
                ImmutableSet.of()));
    EvaluationResult<StringValue> result =
        eval(/* keepGoing= */ false, ImmutableSet.of(topKey, otherTop));
    assertThat(result.errorMap().keySet()).containsExactly(topKey);
    Iterable<CycleInfo> cycleInfos = result.getError(topKey).getCycleInfo();
    assertThat(cycleInfos).isNotEmpty();
    CycleInfo cycleInfo = Iterables.getOnlyElement(cycleInfos);
    assertThat(cycleInfo.getPathToCycle()).containsExactly(topKey);
    assertThat(cycleInfo.getCycle()).containsExactly(midKey, cycleKey);
  }

  /**
   * Regression test: "value in cycle depends on error". Here, we add an additional top-level key in
   * error, just to mix it up.
   */
  @Test
  public void cycleAndErrorAndError(@TestParameter boolean keepGoing) throws Exception {
    graph = new DeterministicHelper.DeterministicProcessableGraph(new InMemoryGraphImpl());
    tester = new GraphTester();
    SkyKey errorKey = skyKey("error");
    SkyKey cycleKey = skyKey("cycle");
    SkyKey midKey = skyKey("mid");
    SkyKey topKey = skyKey("top");
    tester.getOrCreate(topKey).addDependency(midKey).setComputedValue(CONCATENATE);
    tester
        .getOrCreate(midKey)
        .addDependency(errorKey)
        .addDependency(cycleKey)
        .setComputedValue(CONCATENATE);
    SkyKey otherTop = skyKey("otherTop");
    CountDownLatch topStartAndCycleFinish = new CountDownLatch(2);
    // In nokeep_going mode, otherTop will wait until the threadpool has received an exception,
    // then throw its own exception. This guarantees that its exception will not be the one
    // bubbling up, but that there is a top-level value with an exception by the time the bubbling
    // up starts.
    tester
        .getOrCreate(otherTop)
        .setBuilder(
            new ChainedFunction(
                topStartAndCycleFinish,
                new CountDownLatch(0),
                null,
                /* waitForException= */ !keepGoing,
                null,
                ImmutableSet.of()));
    // error waits until otherTop starts and cycle finishes, to make sure otherTop will request
    // its dep before the threadpool shuts down.
    tester
        .getOrCreate(errorKey)
        .setBuilder(
            new ChainedFunction(
                null,
                topStartAndCycleFinish,
                null,
                /* waitForException= */ false,
                null,
                ImmutableSet.of()));
    tester
        .getOrCreate(cycleKey)
        .setBuilder(
            new ChainedFunction(
                null,
                null,
                topStartAndCycleFinish,
                /* waitForException= */ false,
                new StringValue(""),
                ImmutableSet.of(midKey)));
    EvaluationResult<StringValue> result = eval(keepGoing, ImmutableSet.of(topKey, otherTop));
    if (keepGoing) {
      assertThatEvaluationResult(result).hasErrorMapThat().hasSize(2);
    }
    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(topKey)
        .hasCycleInfoThat()
        .containsExactly(
            new CycleInfo(ImmutableList.of(topKey), ImmutableList.of(midKey, cycleKey)));
  }

  @Test
  public void testFunctionCrashTrace() {

    class ChildFunction implements SkyFunction {
      @Override
      public SkyValue compute(SkyKey skyKey, Environment env) {
        throw new IllegalStateException("I WANT A PONY!!!");
      }
    }

    class ParentFunction implements SkyFunction {
      @Override
      public SkyValue compute(SkyKey skyKey, Environment env) throws InterruptedException {
        SkyValue dep = env.getValue(ChildKey.create("billy the kid"));
        if (dep == null) {
          return null;
        }
        throw new IllegalStateException(); // Should never get here.
      }
    }

    ImmutableMap<SkyFunctionName, SkyFunction> skyFunctions =
        ImmutableMap.of(
            CHILD_TYPE, new ChildFunction(),
            PARENT_TYPE, new ParentFunction());
    ParallelEvaluator evaluator = makeEvaluator(new InMemoryGraphImpl(), skyFunctions, false);

    RuntimeException e =
        assertThrows(
            RuntimeException.class,
            () -> evaluator.eval(ImmutableList.of(ParentKey.create("octodad"))));
    assertThat(e).hasCauseThat().hasMessageThat().isEqualTo("I WANT A PONY!!!");
    assertThat(e)
        .hasMessageThat()
        .isEqualTo(
            "Unrecoverable error while evaluating node 'child:billy the kid' "
                + "(requested by nodes 'parent:octodad')");
  }

  private static final class SomeOtherErrorException extends Exception {
    SomeOtherErrorException(String msg) {
      super(msg);
    }
  }

  /**
   * This and the following tests are in response to a bug: "Skyframe error propagation model is
   * problematic". They ensure that exceptions a child throws that a value does not specify it can
   * handle in getValueOrThrow do not cause a crash.
   */
  @Test
  public void unexpectedErrorDep(@TestParameter boolean keepGoing) throws Exception {
    graph = new InMemoryGraphImpl();
    SkyKey errorKey = skyKey("my_error_value");
    SomeOtherErrorException exception = new SomeOtherErrorException("error exception");
    tester
        .getOrCreate(errorKey)
        .setBuilder(
            (skyKey, env) -> {
              throw new SkyFunctionException(exception, Transience.PERSISTENT) {};
            });
    SkyKey topKey = skyKey("top");
    tester
        .getOrCreate(topKey)
        .addErrorDependency(errorKey, new StringValue("recovered"))
        .setComputedValue(CONCATENATE);
    EvaluationResult<StringValue> result = eval(keepGoing, ImmutableList.of(topKey));
    assertThatEvaluationResult(result).hasSingletonErrorThat(topKey);
  }

  @Test
  public void unexpectedErrorDepOneLevelDown(@TestParameter boolean keepGoing) throws Exception {
    graph = new InMemoryGraphImpl();
    SkyKey errorKey = skyKey("my_error_value");
    SomeErrorException exception = new SomeErrorException("error exception");
    SomeErrorException topException = new SomeErrorException("top exception");
    StringValue topValue = new StringValue("top");
    tester
        .getOrCreate(errorKey)
        .setBuilder(
            (skyKey, env) -> {
              throw new GenericFunctionException(exception, Transience.PERSISTENT);
            });
    SkyKey topKey = skyKey("top");
    SkyKey parentKey = skyKey("parent");
    tester.getOrCreate(parentKey).addDependency(errorKey).setComputedValue(CONCATENATE);
    tester
        .getOrCreate(topKey)
        .setBuilder(
            (skyKey, env) -> {
              try {
                if (env.getValueOrThrow(parentKey, SomeErrorException.class) == null) {
                  return null;
                }
              } catch (SomeErrorException e) {
                assertWithMessage(e.toString()).that(e).isEqualTo(exception);
              }
              if (keepGoing) {
                return topValue;
              } else {
                throw new GenericFunctionException(topException, Transience.PERSISTENT);
              }
            });
    tester
        .getOrCreate(topKey)
        .addErrorDependency(errorKey, new StringValue("recovered"))
        .setComputedValue(CONCATENATE);
    EvaluationResult<StringValue> result = eval(keepGoing, ImmutableList.of(topKey));
    if (!keepGoing) {
      assertThatEvaluationResult(result).hasSingletonErrorThat(topKey);
    } else {
      assertThatEvaluationResult(result).hasNoError();
      assertThatEvaluationResult(result).hasEntryThat(topKey).isSameInstanceAs(topValue);
    }
  }

  /**
   * Exercises various situations involving groups of deps that overlap -- request one group, then
   * request another group that has a dep in common with the first group.
   *
   * @param sameFirst whether the dep in common in the two groups should be the first dep.
   * @param twoCalls whether the two groups should be requested in two different builder calls.
   */
  @Test
  public void sameDepInTwoGroups(@TestParameter boolean sameFirst, @TestParameter boolean twoCalls)
      throws Exception {
    graph = new InMemoryGraphImpl();
    SkyKey topKey = skyKey("top");
    List<SkyKey> leaves = new ArrayList<>();
    for (int i = 1; i <= 3; i++) {
      SkyKey leaf = skyKey("leaf" + i);
      leaves.add(leaf);
      tester.set(leaf, new StringValue("leaf" + i));
    }
    SkyKey leaf4 = skyKey("leaf4");
    tester.set(leaf4, new StringValue("leaf" + 4));
    tester
        .getOrCreate(topKey)
        .setBuilder(
            (skyKey, env) -> {
              env.getValuesAndExceptions(leaves);
              if (twoCalls && env.valuesMissing()) {
                return null;
              }
              SkyKey first = sameFirst ? leaves.get(0) : leaf4;
              SkyKey second = sameFirst ? leaf4 : leaves.get(2);
              ImmutableList<SkyKey> secondRequest = ImmutableList.of(first, second);
              env.getValuesAndExceptions(secondRequest);
              if (env.valuesMissing()) {
                return null;
              }
              return new StringValue("top");
            });
    eval(/* keepGoing= */ false, topKey);
    assertThat(eval(/* keepGoing= */ false, topKey)).isEqualTo(new StringValue("top"));
  }

  @Test
  public void getValueOrThrowWithErrors(@TestParameter boolean keepGoing) throws Exception {
    graph = new InMemoryGraphImpl();
    SkyKey parentKey = skyKey("parent");
    SkyKey errorDep = skyKey("errorChild");
    SomeErrorException childExn = new SomeErrorException("child error");
    tester
        .getOrCreate(errorDep)
        .setBuilder(
            (skyKey, env) -> {
              throw new GenericFunctionException(childExn, Transience.PERSISTENT);
            });
    List<SkyKey> deps = new ArrayList<>();
    for (int i = 1; i <= 3; i++) {
      SkyKey dep = skyKey("child" + i);
      deps.add(dep);
      tester.set(dep, new StringValue("child" + i));
    }
    SomeErrorException parentExn = new SomeErrorException("parent error");
    tester
        .getOrCreate(parentKey)
        .setBuilder(
            (skyKey, env) -> {
              try {
                SkyValue value = env.getValueOrThrow(errorDep, SomeErrorException.class);
                if (value == null) {
                  return null;
                }
              } catch (SomeErrorException e) {
                // Recover from the child error.
              }
              env.getValuesAndExceptions(deps);
              if (env.valuesMissing()) {
                return null;
              }
              throw new GenericFunctionException(parentExn, Transience.PERSISTENT);
            });
    EvaluationResult<StringValue> evaluationResult = eval(keepGoing, ImmutableList.of(parentKey));
    assertThat(evaluationResult.hasError()).isTrue();
    assertThat(evaluationResult.getError().getException())
        .isEqualTo(keepGoing ? parentExn : childExn);
  }

  @Test
  public void getValuesAndExceptions() throws Exception {
    graph = new InMemoryGraphImpl();
    SkyKey otherKey = skyKey("other");
    SkyKey anotherKey = skyKey("another");
    SkyKey errorExpectedKey = skyKey("errorExpected");
    SkyKey topKey = skyKey("top");
    Exception topException = new SomeErrorException("top exception");
    AtomicInteger numComputes = new AtomicInteger(0);

    tester.set(otherKey, new StringValue("other"));
    tester.set(anotherKey, new StringValue("another"));
    tester.getOrCreate(errorExpectedKey).setHasError(true);
    tester
        .getOrCreate(topKey)
        .setBuilder(
            new SkyFunction() {
              @Nullable
              @Override
              public SkyValue compute(SkyKey skyKey, Environment env)
                  throws SkyFunctionException, InterruptedException {
                ImmutableList<SkyKey> depKeys =
                    ImmutableList.of(otherKey, anotherKey, errorExpectedKey);
                SkyframeLookupResult skyframeLookupResult = env.getValuesAndExceptions(depKeys);
                if (numComputes.incrementAndGet() == 1) {
                  assertThat(env.valuesMissing()).isTrue();
                  for (SkyKey depKey : depKeys.reverse()) {
                    try {
                      assertThat(skyframeLookupResult.getOrThrow(depKey, SomeErrorException.class))
                          .isNull();
                    } catch (SomeErrorException e) {
                      throw new AssertionError("should not have thrown", e);
                    }
                  }
                  return null;
                } else {
                  assertThat(numComputes.get()).isEqualTo(2);
                  SkyValue value1 = skyframeLookupResult.get(otherKey);
                  assertThat(value1).isNotNull();
                  assertThat(env.valuesMissing()).isFalse();
                  try {
                    SkyValue value2 =
                        skyframeLookupResult.getOrThrow(anotherKey, SomeErrorException.class);
                    assertThat(value2).isNotNull();
                    assertThat(env.valuesMissing()).isFalse();
                  } catch (SomeErrorException e) {
                    throw new AssertionError("Should not have thrown", e);
                  }
                  try {
                    skyframeLookupResult.getOrThrow(errorExpectedKey, SomeErrorException.class);
                    throw new AssertionError("Should throw");
                  } catch (SomeErrorException e) {
                    assertThat(env.valuesMissing()).isFalse();
                  }
                  throw new SkyFunctionException(topException, Transience.PERSISTENT) {};
                }
              }
            });
    EvaluationResult<StringValue> result = eval(/* keepGoing= */ true, ImmutableList.of(topKey));

    assertThatEvaluationResult(result).hasError();
    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(topKey)
        .hasExceptionThat()
        .isSameInstanceAs(topException);
    assertThat(numComputes.get()).isEqualTo(2);
  }

  @Test
  public void getValuesAndExceptionsWithErrors() throws Exception {
    graph = new InMemoryGraphImpl();
    SkyKey childKey = skyKey("error");
    SomeErrorException childExn = new SomeErrorException("child error");
    tester
        .getOrCreate(childKey)
        .setBuilder(
            (skyKey, env) -> {
              throw new GenericFunctionException(childExn, Transience.PERSISTENT);
            });
    SkyKey parentKey = skyKey("parent");
    AtomicInteger numComputes = new AtomicInteger(0);
    tester
        .getOrCreate(parentKey)
        .setBuilder(
            (skyKey, env) -> {
              try {
                SkyValue value =
                    env.getValuesAndExceptions(ImmutableList.of(childKey))
                        .getOrThrow(childKey, SomeOtherErrorException.class);
                assertThat(value).isNull();
              } catch (SomeOtherErrorException e) {
                throw new AssertionError("Should not have thrown", e);
              }
              numComputes.incrementAndGet();
              assertThat(env.valuesMissing()).isTrue();
              return null;
            });
    EvaluationResult<StringValue> result = eval(/* keepGoing= */ true, ImmutableList.of(parentKey));
    assertThatEvaluationResult(result).hasError();
    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(parentKey)
        .hasExceptionThat()
        .isSameInstanceAs(childExn);
    assertThat(numComputes.get()).isEqualTo(2);
  }

  @Test
  public void declareDependenciesAndCheckIfValuesMissing() throws Exception {
    graph = new InMemoryGraphImpl();
    SkyKey childKey = skyKey("error");
    SomeErrorException childExn = new SomeErrorException("child error");
    tester
        .getOrCreate(childKey)
        .setBuilder(
            (skyKey, env) -> {
              throw new GenericFunctionException(childExn, Transience.PERSISTENT);
            });
    SkyKey parentKey = skyKey("parent");
    AtomicInteger numComputes = new AtomicInteger(0);
    BugReporter mockReporter = mock(BugReporter.class);
    tester
        .getOrCreate(parentKey)
        .setBuilder(
            (skyKey, env) -> {
              boolean valuesMissing =
                  GraphTraversingHelper.declareDependenciesAndCheckIfValuesMissing(
                      env,
                      ImmutableList.of(childKey),
                      SomeOtherErrorException.class,
                      /* exceptionClass2= */ null,
                      mockReporter);
              numComputes.incrementAndGet();
              assertThat(valuesMissing).isTrue();
              return null;
            });
    EvaluationResult<StringValue> result = eval(/* keepGoing= */ true, ImmutableList.of(parentKey));
    verify(mockReporter)
        .logUnexpected("Value for: '%s' was missing, this should never happen", childKey);
    verifyNoMoreInteractions(mockReporter);
    assertThatEvaluationResult(result).hasError();
    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(parentKey)
        .hasExceptionThat()
        .isSameInstanceAs(childExn);
    assertThat(numComputes.get()).isEqualTo(2);
  }

  @Test
  public void declareDependenciesAndCheckIfNotValuesMissing() throws Exception {
    graph = new InMemoryGraphImpl();
    SkyKey otherKey = skyKey("other");
    SkyKey childKey = skyKey("error");
    SomeErrorException childExn = new SomeErrorException("child error");
    tester.set(otherKey, new StringValue("other"));
    tester
        .getOrCreate(childKey)
        .setBuilder(
            (skyKey, env) -> {
              throw new GenericFunctionException(childExn, Transience.PERSISTENT);
            });
    SkyKey parentKey = skyKey("parent");
    AtomicInteger numComputes = new AtomicInteger(0);
    tester
        .getOrCreate(parentKey)
        .setBuilder(
            (skyKey, env) -> {
              if (numComputes.incrementAndGet() == 1) {
                boolean valuesMissing =
                    GraphTraversingHelper.declareDependenciesAndCheckIfValuesMissing(
                        env, ImmutableList.of(otherKey, childKey), SomeErrorException.class);
                assertThat(valuesMissing).isTrue();
              } else {
                boolean valuesMissing =
                    GraphTraversingHelper.declareDependenciesAndCheckIfValuesMissing(
                        env, ImmutableList.of(otherKey, childKey), SomeErrorException.class);
                assertThat(valuesMissing).isFalse();
              }
              return null;
            });
    EvaluationResult<StringValue> result = eval(/* keepGoing= */ true, ImmutableList.of(parentKey));
    assertThatEvaluationResult(result).hasError();
    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(parentKey)
        .hasExceptionThat()
        .isSameInstanceAs(childExn);
    assertThat(numComputes.get()).isEqualTo(2);
  }

  @Test
  public void validateExceptionTypeInDifferentPosition(
      @TestParameter({"0", "1", "2", "3"}) int exceptionIndex) throws Exception {
    ImmutableList<Class<? extends Exception>> exceptions =
        ImmutableList.of(
            Exception.class,
            SomeOtherErrorException.class,
            IOException.class,
            SomeErrorException.class);
    graph = new InMemoryGraphImpl();
    SkyKey otherKey = skyKey("other");
    tester.set(otherKey, new StringValue("other"));
    SkyKey parentKey = skyKey("parent");
    SomeErrorException parentExn = new SomeErrorException("parent error");
    tester
        .getOrCreate(parentKey)
        .setBuilder(
            (skyKey, env) -> {
              IllegalStateException illegalStateException =
                  assertThrows(
                      IllegalStateException.class,
                      () ->
                          env.getValueOrThrow(
                              otherKey,
                              exceptions.get(exceptionIndex % 4),
                              exceptions.get((exceptionIndex + 1) % 4),
                              exceptions.get((exceptionIndex + 2) % 4),
                              exceptions.get((exceptionIndex + 3) % 4)));
              assertThat(illegalStateException)
                  .hasMessageThat()
                  .contains("is a supertype of RuntimeException");
              assertThat(env.valuesMissing()).isFalse();
              throw new GenericFunctionException(parentExn, Transience.PERSISTENT);
            });
    EvaluationResult<StringValue> result = eval(/* keepGoing= */ true, ImmutableList.of(parentKey));
    assertThat(result.hasError()).isTrue();
    assertThat(result.getError().getException()).isEqualTo(parentExn);
  }

  @Test
  public void validateExceptionTypeWithDifferentException(
      @TestParameter ExceptionOption exceptionOption) throws Exception {
    graph = new InMemoryGraphImpl();
    SkyKey otherKey = skyKey("other");
    tester.set(otherKey, new StringValue("other"));
    SkyKey parentKey = skyKey("parent");
    SomeErrorException parentExn = new SomeErrorException("parent error");
    tester
        .getOrCreate(parentKey)
        .setBuilder(
            (skyKey, env) -> {
              IllegalStateException illegalStateException =
                  assertThrows(
                      IllegalStateException.class,
                      () -> env.getValueOrThrow(otherKey, exceptionOption.exceptionClass));
              assertThat(illegalStateException)
                  .hasMessageThat()
                  .contains(exceptionOption.errorMessage);
              assertThat(env.valuesMissing()).isFalse();
              throw new GenericFunctionException(parentExn, Transience.PERSISTENT);
            });
    EvaluationResult<StringValue> result = eval(/* keepGoing= */ true, ImmutableList.of(parentKey));
    assertThat(result.hasError()).isTrue();
    assertThat(result.getError().getException()).isEqualTo(parentExn);
  }

  private enum ExceptionOption {
    EXCEPTION(Exception.class, "is a supertype of RuntimeException"),
    NULL_POINTER_EXCEPTION(NullPointerException.class, "is a subtype of RuntimeException"),
    INTERRUPTED_EXCEPTION(InterruptedException.class, "is a subtype of InterruptedException");

    final Class<? extends Exception> exceptionClass;
    final String errorMessage;

    ExceptionOption(Class<? extends Exception> exceptionClass, String errorMessage) {
      this.exceptionClass = exceptionClass;
      this.errorMessage = errorMessage;
    }
  }

  @Test
  public void duplicateCycles() throws Exception {
    graph = new InMemoryGraphImpl();
    SkyKey grandparentKey = skyKey("grandparent");
    SkyKey parentKey1 = skyKey("parent1");
    SkyKey parentKey2 = skyKey("parent2");
    SkyKey loopKey1 = skyKey("loop1");
    SkyKey loopKey2 = skyKey("loop2");
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
    assertThat(cycles).hasSize(2);
    int numUniqueCycles = 0;
    CycleDeduper<SkyKey> cycleDeduper = new CycleDeduper<>();
    for (ImmutableList<SkyKey> cycle : cycles) {
      if (!cycleDeduper.alreadySeen(cycle)) {
        numUniqueCycles++;
      }
    }
    assertThat(numUniqueCycles).isEqualTo(1);
  }

  @Test
  public void signalValueEnqueuedAndEvaluated() throws Exception {
    Set<SkyKey> enqueuedValues = Sets.newConcurrentHashSet();
    Set<SkyKey> evaluatedValues = Sets.newConcurrentHashSet();
    EvaluationProgressReceiver progressReceiver =
        new EvaluationProgressReceiver() {
          @Override
          public void enqueueing(SkyKey skyKey) {
            enqueuedValues.add(skyKey);
          }

          @Override
          public void evaluated(
              SkyKey skyKey,
              EvaluationState state,
              @Nullable SkyValue newValue,
              @Nullable ErrorInfo newError,
              @Nullable GroupedDeps directDeps) {
            evaluatedValues.add(skyKey);
          }
        };

    ExtendedEventHandler reporter =
        new Reporter(
            new EventBus(),
            e -> {
              throw new IllegalStateException();
            });

    MemoizingEvaluator evaluator =
        new InMemoryMemoizingEvaluator(
            ImmutableMap.of(GraphTester.NODE_TYPE, tester.getFunction()),
            new SequencedRecordingDifferencer(),
            progressReceiver);

    tester
        .getOrCreate("top1")
        .setComputedValue(CONCATENATE)
        .addDependency("d1")
        .addDependency("d2");
    tester.getOrCreate("top2").setComputedValue(CONCATENATE).addDependency("d3");
    tester.getOrCreate("top3");
    assertThat(enqueuedValues).isEmpty();
    assertThat(evaluatedValues).isEmpty();

    tester.set("d1", new StringValue("1"));
    tester.set("d2", new StringValue("2"));
    tester.set("d3", new StringValue("3"));

    EvaluationContext evaluationContext =
        EvaluationContext.newBuilder()
            .setKeepGoing(false)
            .setParallelism(200)
            .setEventHandler(reporter)
            .build();
    evaluator.evaluate(ImmutableList.of(skyKey("top1")), evaluationContext);
    assertThat(enqueuedValues).containsExactlyElementsIn(GraphTester.toSkyKeys("top1", "d1", "d2"));
    assertThat(evaluatedValues)
        .containsExactlyElementsIn(GraphTester.toSkyKeys("top1", "d1", "d2"));
    enqueuedValues.clear();
    evaluatedValues.clear();

    evaluator.evaluate(ImmutableList.of(skyKey("top2")), evaluationContext);
    assertThat(enqueuedValues).containsExactlyElementsIn(GraphTester.toSkyKeys("top2", "d3"));
    assertThat(evaluatedValues).containsExactlyElementsIn(GraphTester.toSkyKeys("top2", "d3"));
    enqueuedValues.clear();
    evaluatedValues.clear();

    evaluator.evaluate(ImmutableList.of(skyKey("top1")), evaluationContext);
    assertThat(enqueuedValues).isEmpty();
    assertThat(evaluatedValues).containsExactlyElementsIn(GraphTester.toSkyKeys("top1"));
  }

  @Test
  public void runDepOnErrorHaltsNoKeepGoingBuildEagerly(
      @TestParameter boolean childErrorCached, @TestParameter boolean handleChildError)
      throws Exception {
    graph = new InMemoryGraphImpl();
    SkyKey parentKey = skyKey("parent");
    SkyKey childKey = skyKey("child");
    tester.getOrCreate(childKey).setHasError(/* hasError= */ true);
    // The parent should be built exactly twice: once during normal evaluation and once
    // during error bubbling.
    AtomicInteger numParentInvocations = new AtomicInteger(0);
    tester
        .getOrCreate(parentKey)
        .setBuilder(
            (skyKey, env) -> {
              int invocations = numParentInvocations.incrementAndGet();
              if (handleChildError) {
                try {
                  SkyValue value = env.getValueOrThrow(childKey, SomeErrorException.class);
                  // On the first invocation, either the child error should already be cached and
                  // not propagated, or it should be computed freshly and not propagated. On the
                  // second build (error bubbling), the child error should be propagated.
                  assertWithMessage("bogus non-null value " + value).that(value == null).isTrue();
                  assertWithMessage("parent incorrectly re-computed during normal evaluation")
                      .that(invocations)
                      .isEqualTo(1);
                  assertWithMessage("child error not propagated during error bubbling")
                      .that(env.inErrorBubblingForSkyFunctionsThatCanFullyRecoverFromErrors())
                      .isFalse();
                  return value;
                } catch (SomeErrorException e) {
                  assertWithMessage("child error propagated during normal evaluation")
                      .that(env.inErrorBubblingForSkyFunctionsThatCanFullyRecoverFromErrors())
                      .isTrue();
                  assertThat(invocations).isEqualTo(2);
                  return null;
                }
              } else {
                if (invocations == 1) {
                  assertWithMessage("parent's first computation should be during normal evaluation")
                      .that(env.inErrorBubblingForSkyFunctionsThatCanFullyRecoverFromErrors())
                      .isFalse();
                  return env.getValue(childKey);
                } else {
                  assertThat(invocations).isEqualTo(2);
                  assertWithMessage("parent incorrectly re-computed during normal evaluation")
                      .that(env.inErrorBubblingForSkyFunctionsThatCanFullyRecoverFromErrors())
                      .isTrue();
                  return env.getValue(childKey);
                }
              }
            });
    if (childErrorCached) {
      // Ensure that the child is already in the graph.
      evalValueInError(childKey);
    }
    EvaluationResult<StringValue> result =
        eval(/* keepGoing= */ false, ImmutableList.of(parentKey));
    assertThat(numParentInvocations.get()).isEqualTo(2);
    assertThatEvaluationResult(result).hasErrorEntryForKeyThat(parentKey);
  }

  @Test
  public void raceConditionWithNoKeepGoingErrors_FutureError() throws Exception {
    CountDownLatch errorCommitted = new CountDownLatch(1);
    CountDownLatch otherStarted = new CountDownLatch(1);
    CountDownLatch otherParentSignaled = new CountDownLatch(1);
    SkyKey errorParentKey = skyKey("errorParentKey");
    SkyKey errorKey = skyKey("errorKey");
    SkyKey otherParentKey = skyKey("otherParentKey");
    SkyKey otherKey = skyKey("otherKey");
    AtomicInteger numOtherParentInvocations = new AtomicInteger(0);
    AtomicInteger numErrorParentInvocations = new AtomicInteger(0);
    tester
        .getOrCreate(otherParentKey)
        .setBuilder(
            (skyKey, env) -> {
              int invocations = numOtherParentInvocations.incrementAndGet();
              assertWithMessage("otherParentKey should not be restarted")
                  .that(invocations)
                  .isEqualTo(1);
              return env.getValue(otherKey);
            });
    tester
        .getOrCreate(otherKey)
        .setBuilder(
            (skyKey, env) -> {
              otherStarted.countDown();
              TrackingAwaiter.INSTANCE.awaitLatchAndTrackExceptions(
                  errorCommitted, "error didn't get committed to the graph in time");
              return new StringValue("other");
            });
    tester
        .getOrCreate(errorKey)
        .setBuilder(
            (skyKey, env) -> {
              TrackingAwaiter.INSTANCE.awaitLatchAndTrackExceptions(
                  otherStarted, "other didn't start in time");
              throw new GenericFunctionException(
                  new SomeErrorException("error"), Transience.PERSISTENT);
            });
    tester
        .getOrCreate(errorParentKey)
        .setBuilder(
            (skyKey, env) -> {
              int invocations = numErrorParentInvocations.incrementAndGet();
              try {
                SkyValue value = env.getValueOrThrow(errorKey, SomeErrorException.class);
                assertWithMessage("bogus non-null value " + value).that(value == null).isTrue();
                if (invocations == 1) {
                  return null;
                } else {
                  assertThat(env.inErrorBubblingForSkyFunctionsThatCanFullyRecoverFromErrors())
                      .isFalse();
                  fail("RACE CONDITION: errorParentKey was restarted!");
                  return null;
                }
              } catch (SomeErrorException e) {
                assertWithMessage("child error propagated during normal evaluation")
                    .that(env.inErrorBubblingForSkyFunctionsThatCanFullyRecoverFromErrors())
                    .isTrue();
                assertThat(invocations).isEqualTo(2);
                return null;
              }
            });
    graph =
        new NotifyingHelper.NotifyingProcessableGraph(
            new InMemoryGraphImpl(),
            (key, type, order, context) -> {
              if (key.equals(errorKey) && type == EventType.SET_VALUE && order == Order.AFTER) {
                errorCommitted.countDown();
                TrackingAwaiter.INSTANCE.awaitLatchAndTrackExceptions(
                    otherParentSignaled, "otherParent didn't get signaled in time");
                // We try to give some time for ParallelEvaluator to incorrectly re-evaluate
                // 'otherParentKey'. This test case is testing for a real race condition and the
                // 10ms time was chosen experimentally to give a true positive rate of 99.8%
                // (without a sleep it has a 1% true positive rate). There's no good way to do
                // this without sleeping. We *could* introspect ParallelEvaluator's
                // AbstractQueueVisitor to see if the re-evaluation has been enqueued, but that's
                // relying on pretty low-level implementation details.
                Uninterruptibles.sleepUninterruptibly(10, TimeUnit.MILLISECONDS);
              }
              if (key.equals(otherParentKey) && type == EventType.SIGNAL && order == Order.AFTER) {
                otherParentSignaled.countDown();
              }
            });
    EvaluationResult<StringValue> result =
        eval(/* keepGoing= */ false, ImmutableList.of(otherParentKey, errorParentKey));
    assertThat(result.hasError()).isTrue();
    assertThatEvaluationResult(result).hasErrorEntryForKeyThat(errorParentKey);
  }

  @Test
  public void cachedErrorsFromKeepGoingUsedOnNoKeepGoing() throws Exception {
    graph = new DeterministicHelper.DeterministicProcessableGraph(new InMemoryGraphImpl());
    tester = new GraphTester();
    SkyKey errorKey = skyKey("error");
    SkyKey parent1Key = skyKey("parent1");
    SkyKey parent2Key = skyKey("parent2");
    tester
        .getOrCreate(parent1Key)
        .addDependency(errorKey)
        .setConstantValue(new StringValue("parent1"));
    tester
        .getOrCreate(parent2Key)
        .addDependency(errorKey)
        .setConstantValue(new StringValue("parent2"));
    tester.getOrCreate(errorKey).setHasError(true);
    EvaluationResult<StringValue> result =
        eval(/* keepGoing= */ true, ImmutableList.of(parent1Key));
    assertThatEvaluationResult(result).hasSingletonErrorThat(parent1Key);
    result = eval(/* keepGoing= */ false, ImmutableList.of(parent2Key));
    assertThatEvaluationResult(result).hasSingletonErrorThat(parent2Key);
  }

  @Test
  public void cachedTopLevelErrorsShouldHaltNoKeepGoingBuildEarly() throws Exception {
    graph = new DeterministicHelper.DeterministicProcessableGraph(new InMemoryGraphImpl());
    tester = new GraphTester();
    SkyKey errorKey = skyKey("error");
    tester.getOrCreate(errorKey).setHasError(true);
    EvaluationResult<StringValue> result = eval(/* keepGoing= */ true, ImmutableList.of(errorKey));
    assertThatEvaluationResult(result).hasSingletonErrorThat(errorKey);
    SkyKey rogueKey = skyKey("rogue");
    tester
        .getOrCreate(rogueKey)
        .setBuilder(
            (skyKey, env) -> {
              // This SkyFunction could do an arbitrarily bad computation, e.g. loop-forever. So we
              // want to make sure that it is never run when we want to fail-fast anyway.
              fail("eval call should have already terminated");
              return null;
            });
    result = eval(/* keepGoing= */ false, ImmutableList.of(errorKey, rogueKey));
    assertThatEvaluationResult(result).hasErrorMapThat().hasSize(1);
    assertThatEvaluationResult(result).hasErrorEntryForKeyThat(errorKey);
    assertThat(result.errorMap()).doesNotContainKey(rogueKey);
  }

  // Explicit test that we tolerate a SkyFunction that declares different [sequences of] deps each
  // restart. Such behavior from a SkyFunction isn't desired, but Bazel-on-Skyframe does indeed do
  // this.
  @Test
  public void declaresDifferentDepsAfterRestart() throws Exception {
    graph = new DeterministicHelper.DeterministicProcessableGraph(new InMemoryGraphImpl());
    tester = new GraphTester();
    SkyKey grandChild1Key = skyKey("grandChild1");
    tester.getOrCreate(grandChild1Key).setConstantValue(new StringValue("grandChild1"));
    SkyKey child1Key = skyKey("child1");
    tester
        .getOrCreate(child1Key)
        .addDependency(grandChild1Key)
        .setConstantValue(new StringValue("child1"));
    SkyKey grandChild2Key = skyKey("grandChild2");
    tester.getOrCreate(grandChild2Key).setConstantValue(new StringValue("grandChild2"));
    SkyKey child2Key = skyKey("child2");
    tester.getOrCreate(child2Key).setConstantValue(new StringValue("child2"));
    SkyKey parentKey = skyKey("parent");
    AtomicInteger numComputes = new AtomicInteger(0);
    tester
        .getOrCreate(parentKey)
        .setBuilder(
            (skyKey, env) -> {
              switch (numComputes.incrementAndGet()) {
                case 1:
                  env.getValue(child1Key);
                  Preconditions.checkState(env.valuesMissing());
                  return null;
                case 2:
                  env.getValue(child2Key);
                  Preconditions.checkState(env.valuesMissing());
                  return null;
                case 3:
                  return new StringValue("the third time's the charm!");
                default:
                  throw new IllegalStateException();
              }
            });
    EvaluationResult<StringValue> result =
        eval(/* keepGoing= */ false, ImmutableList.of(parentKey));
    assertThatEvaluationResult(result).hasNoError();
    assertThatEvaluationResult(result)
        .hasEntryThat(parentKey)
        .isEqualTo(new StringValue("the third time's the charm!"));
  }

  @Test
  public void runUnhandledTransitiveErrors(
      @TestParameter boolean keepGoing, @TestParameter boolean explicitlyPropagateError)
      throws Exception {
    graph = new DeterministicHelper.DeterministicProcessableGraph(new InMemoryGraphImpl());
    tester = new GraphTester();
    SkyKey grandparentKey = skyKey("grandparent");
    SkyKey parentKey = skyKey("parent");
    SkyKey childKey = skyKey("child");
    AtomicBoolean errorPropagated = new AtomicBoolean(false);
    tester
        .getOrCreate(grandparentKey)
        .setBuilder(
            (skyKey, env) -> {
              try {
                return env.getValueOrThrow(parentKey, SomeErrorException.class);
              } catch (SomeErrorException e) {
                errorPropagated.set(true);
                throw new GenericFunctionException(e, Transience.PERSISTENT);
              }
            });
    tester
        .getOrCreate(parentKey)
        .setBuilder(
            (skyKey, env) -> {
              if (explicitlyPropagateError) {
                try {
                  return env.getValueOrThrow(childKey, SomeErrorException.class);
                } catch (SomeErrorException e) {
                  throw new GenericFunctionException(e);
                }
              } else {
                return env.getValue(childKey);
              }
            });
    tester.getOrCreate(childKey).setHasError(/* hasError= */ true);
    EvaluationResult<StringValue> result = eval(keepGoing, ImmutableList.of(grandparentKey));
    assertThat(errorPropagated.get()).isTrue();
    assertThatEvaluationResult(result).hasSingletonErrorThat(grandparentKey);
  }

  private static class ChildKey extends AbstractSkyKey<String> {
    private static final Interner<ChildKey> interner = BlazeInterners.newWeakInterner();

    private ChildKey(String arg) {
      super(arg);
    }

    static ChildKey create(String arg) {
      return interner.intern(new ChildKey(arg));
    }

    @Override
    public SkyFunctionName functionName() {
      return CHILD_TYPE;
    }
  }

  private static class ParentKey extends AbstractSkyKey<String> {
    private static final Interner<ParentKey> interner = BlazeInterners.newWeakInterner();

    private ParentKey(String arg) {
      super(arg);
    }

    private static ParentKey create(String arg) {
      return interner.intern(new ParentKey(arg));
    }

    @Override
    public SkyFunctionName functionName() {
      return PARENT_TYPE;
    }
  }

  private static class SkyKeyForSkyKeyComputeStateTests extends AbstractSkyKey<String> {
    private static final SkyFunctionName FUNCTION_NAME =
        SkyFunctionName.createHermetic("SKY_KEY_COMPUTE_STATE_TESTS");

    private SkyKeyForSkyKeyComputeStateTests(String arg) {
      super(arg);
    }

    @Override
    public SkyFunctionName functionName() {
      return FUNCTION_NAME;
    }
  }

  // Test for the basic functionality of SkyKeyComputeState.
  @Test
  public void skyKeyComputeState() throws InterruptedException {
    // When we have 3 nodes: key1, key2, key3.
    // (with dependency graph key1 -> key2; key2 -> key3, to be declared later in this test)
    // (and we'll be evaluating key1 later in this test)
    SkyKey key1 = new SkyKeyForSkyKeyComputeStateTests("key1");
    SkyKey key2 = new SkyKeyForSkyKeyComputeStateTests("key2");
    SkyKey key3 = new SkyKeyForSkyKeyComputeStateTests("key3");

    // And an SkyKeyComputeState implementation that tracks global instance counts and per-instance
    // usage counts,
    AtomicInteger globalStateInstanceCounter = new AtomicInteger();
    class State implements SkyKeyComputeState {
      final int instanceCount = globalStateInstanceCounter.incrementAndGet();
      int usageCount = 0;
    }

    // And a SkyFunction for these nodes,
    AtomicLongMap<SkyKey> numCalls = AtomicLongMap.create();
    AtomicReference<WeakReference<State>> stateForKey2Ref = new AtomicReference<>();
    AtomicReference<WeakReference<State>> stateForKey3Ref = new AtomicReference<>();
    SkyFunction skyFunctionForTest =
        // Whose #compute is such that
        (skyKey, env) -> {
          State state = env.getState(State::new);
          state.usageCount++;
          int numCallsForKey = (int) numCalls.incrementAndGet(skyKey);
          // The number of calls to #compute is expected to be equal to the number of usages of
          // the state for that key,
          assertThat(state.usageCount).isEqualTo(numCallsForKey);
          if (skyKey.equals(key1)) {
            // And the semantics for key1 are:

            // The state for key1 is expected to be the first one created (since key1 is expected
            // to be the first node we attempt to compute).
            assertThat(state.instanceCount).isEqualTo(1);
            // And key1 declares a dep on key2,
            if (env.getValue(key2) == null) {
              // (And that dep is expected to be missing on the initial #compute call for key1)
              assertThat(numCallsForKey).isEqualTo(1);
              return null;
            }
            // And if that dep is not missing, then we expect:
            //   - We're on the second #compute call for key1
            assertThat(numCallsForKey).isEqualTo(2);
            //   - The state for key2 should have been eligible for GC. This is because the node
            //     for key2 must have been fully computed, meaning its compute state is no longer
            //     needed, and so ParallelEvaluator ought to have made it eligible for GC.
            GcFinalization.awaitClear(stateForKey2Ref.get());
            return new StringValue("value1");
          } else if (skyKey.equals(key2)) {
            // And the semantics for key2 are:

            // The state for key2 is expected to be the second one created.
            assertThat(state.instanceCount).isEqualTo(2);
            stateForKey2Ref.set(new WeakReference<>(state));
            // And key2 declares a dep on key3,
            if (env.getValue(key3) == null) {
              // (And that dep is expected to be missing on the initial #compute call for key2)
              assertThat(numCallsForKey).isEqualTo(1);
              return null;
            }
            // And if that dep is not missing, then we expect the same sort of things we expected
            // for key1 in this situation.
            assertThat(numCallsForKey).isEqualTo(2);
            GcFinalization.awaitClear(stateForKey3Ref.get());
            return new StringValue("value2");
          } else if (skyKey.equals(key3)) {
            // And the semantics for key3 are:

            // The state for key3 is expected to be the third one created.
            assertThat(state.instanceCount).isEqualTo(3);
            stateForKey3Ref.set(new WeakReference<>(state));
            // And key3 declares no deps.
            return new StringValue("value3");
          }
          throw new IllegalStateException();
        };

    tester.putSkyFunction(SkyKeyForSkyKeyComputeStateTests.FUNCTION_NAME, skyFunctionForTest);
    graph = new InMemoryGraphImpl();
    // Then, when we evaluate key1,
    SkyValue resultValue = eval(/* keepGoing= */ true, key1);
    // It successfully produces the value we expect, confirming all our other expectations about
    // the compute states were correct.
    assertThat(resultValue).isEqualTo(new StringValue("value1"));
  }

  private static class SkyFunctionExceptionForTest extends SkyFunctionException {
    SkyFunctionExceptionForTest(String message) {
      super(new SomeErrorException(message), Transience.PERSISTENT);
    }
  }

  // Test for SkyKeyComputeState in the situation of an error for one node causing normal evaluation
  // to fail-fast, but when there are SkyKeyComputeState instances for other inflight nodes.
  @Test
  public void skyKeyComputeState_noKeepGoingWithAnError() throws InterruptedException {
    // When we have 3 nodes: key1, key2, key3.
    // (with dependency graph key1 -> key2; key3, to be declared later in this test)
    // (and we'll be evaluating key1 & key3 in parallel later in this test)
    SkyKey key1 = new SkyKeyForSkyKeyComputeStateTests("key1");
    SkyKey key2 = new SkyKeyForSkyKeyComputeStateTests("key2");
    SkyKey key3 = new SkyKeyForSkyKeyComputeStateTests("key3");

    class State implements SkyKeyComputeState {}

    // And a SkyFunction for these nodes,
    AtomicReference<WeakReference<State>> stateForKey1Ref = new AtomicReference<>();
    AtomicReference<WeakReference<State>> stateForKey3Ref = new AtomicReference<>();
    CountDownLatch key3SleepingLatch = new CountDownLatch(1);
    AtomicBoolean onNormalEvaluation = new AtomicBoolean(true);
    SkyFunction skyFunctionForTest =
        // Whose #compute is such that
        (skyKey, env) -> {
          if (onNormalEvaluation.get()) {
            // When we're on the normal evaluation:

            State state = env.getState(State::new);
            if (skyKey.equals(key1)) {
              // For key1:

              stateForKey1Ref.set(new WeakReference<>(state));
              // We declare a dep on key.
              return env.getValue(key2);
            } else if (skyKey.equals(key2)) {
              // For key2:

              // We wait for the thread computing key3 to be sleeping
              key3SleepingLatch.await();
              // And then we throw an error, which will fail the normal evaluation and trigger
              // error bubbling.
              onNormalEvaluation.set(false);
              throw new SkyFunctionExceptionForTest("normal evaluation");
            } else if (skyKey.equals(key3)) {
              // For key3:

              stateForKey3Ref.set(new WeakReference<>(state));
              key3SleepingLatch.countDown();
              // We sleep forever. (To be interrupted by ParallelEvaluator when the normal
              // evaluation fails).
              Thread.sleep(Long.MAX_VALUE);
            }
            throw new IllegalStateException();
          } else {
            // When we're in error bubbling:

            // The states for the nodes from normal evaluation should have been eligible for GC.
            // This is because ParallelEvaluator ought to have them eligible for GC before
            // starting error bubbling.
            GcFinalization.awaitClear(stateForKey1Ref.get());
            GcFinalization.awaitClear(stateForKey3Ref.get());

            // We bubble up a unique error message.
            throw new SkyFunctionExceptionForTest("error bubbling for " + skyKey.argument());
          }
        };

    tester.putSkyFunction(SkyKeyForSkyKeyComputeStateTests.FUNCTION_NAME, skyFunctionForTest);
    graph = new InMemoryGraphImpl();
    // Then, when we do a nokeep_going evaluation of key1 and key3 in parallel,
    assertThatEvaluationResult(eval(/* keepGoing= */ false, key1, key3))
        // The evaluation fails (as expected),
        .hasErrorEntryForKeyThat(key1)
        .hasExceptionThat()
        .hasMessageThat()
        // And the error message for key1 is from error bubbling,
        .isEqualTo("error bubbling for key1");
    // Confirming that all our other expectations about the compute states were correct.
  }

  // Demonstrates we're able to drop SkyKeyCompute state intra-evaluation and post-evaluation.
  @Test
  public void skyKeyComputeState_unnecessaryTemporaryStateDropperReceiver()
      throws InterruptedException {
    // When we have 2 nodes: key1, key2
    // (with dependency graph key1 -> key2, to be declared later in this test)
    // (and we'll be evaluating key1 later in this test)
    SkyKey key1 = new SkyKeyForSkyKeyComputeStateTests("key1");
    SkyKey key2 = new SkyKeyForSkyKeyComputeStateTests("key2");

    // And an SkyKeyComputeState implementation that tracks global instance counts and cleanup call
    // counts,
    AtomicInteger globalStateInstanceCounter = new AtomicInteger();
    AtomicInteger globalStateCleanupCounter = new AtomicInteger();
    class State implements SkyKeyComputeState {
      final int instanceCount = globalStateInstanceCounter.incrementAndGet();

      @Override
      public void close() {
        globalStateCleanupCounter.incrementAndGet();
      }
    }

    // And an UnnecessaryTemporaryStateDropperReceiver that,
    AtomicReference<UnnecessaryTemporaryStateDropper> dropperRef = new AtomicReference<>();
    UnnecessaryTemporaryStateDropperReceiver dropperReceiver =
        new UnnecessaryTemporaryStateDropperReceiver() {
          @Override
          public void onEvaluationStarted(UnnecessaryTemporaryStateDropper dropper) {
            // Captures the UnnecessaryTemporaryStateDropper (for our use intra-evaluation)
            dropperRef.set(dropper);
          }

          @Override
          public void onEvaluationFinished() {
            // And then drops everything when the evaluation is done.
            dropperRef.get().drop();
          }
        };

    AtomicReference<WeakReference<State>> stateForKey1Ref = new AtomicReference<>();

    // And a SkyFunction for these nodes,
    SkyFunction skyFunctionForTest =
        // Whose #compute is such that
        (skyKey, env) -> {
          State state = env.getState(State::new);
          if (skyKey.equals(key1)) {
            // The semantics for key1 are:

            // We declare a dep on key2.
            if (env.getValue(key2) == null) {
              // If key2 is missing, that means we're on the initial #compute call for key1,
              // And so we expect the compute state to be the first instance ever.
              assertThat(state.instanceCount).isEqualTo(1);
              stateForKey1Ref.set(new WeakReference<>(state));

              return null;
            } else {
              // But if key2 is not missing, that means we're on the subsequent #compute call for
              // key1. That means we expect the compute state to be the third instance ever,
              // because...
              assertThat(state.instanceCount).isEqualTo(3);

              return new StringValue("value1");
            }
          } else if (skyKey.equals(key2)) {
            // ... The semantics for key2 are:

            // Drop all compute states.
            dropperRef.get().drop();
            // Confirm the old compute state for key1 was GC'd.
            GcFinalization.awaitClear(stateForKey1Ref.get());
            // At this point, both state objects have been cleaned up.
            assertThat(globalStateCleanupCounter.get()).isEqualTo(2);
            // Also confirm key2's compute state is the second instance ever.
            assertThat(state.instanceCount).isEqualTo(2);

            return new StringValue("value2");
          }
          throw new IllegalStateException();
        };

    tester.putSkyFunction(SkyKeyForSkyKeyComputeStateTests.FUNCTION_NAME, skyFunctionForTest);
    graph = new InMemoryGraphImpl();

    ParallelEvaluator parallelEvaluator =
        new ParallelEvaluator(
            graph,
            graphVersion,
            Version.minimal(),
            tester.getSkyFunctionMap(),
            reportedEvents,
            new EmittedEventState(),
            EventFilter.FULL_STORAGE,
            ErrorInfoManager.UseChildErrorInfoIfNecessary.INSTANCE,
            // Doesn't matter for this test case.
            /* keepGoing= */ false,
            revalidationReceiver,
            GraphInconsistencyReceiver.THROWING,
            // We ought not need more than 1 thread for this test case.
            AbstractQueueVisitor.create(
                "test-pool", 1, ParallelEvaluatorErrorClassifier.instance()),
            new SimpleCycleDetector(),
            dropperReceiver);
    // Then, when we evaluate key1,
    SkyValue resultValue = parallelEvaluator.eval(ImmutableList.of(key1)).get(key1);
    // It successfully produces the value we expect, confirming all our other expectations about
    // the compute states were correct.
    assertThat(resultValue).isEqualTo(new StringValue("value1"));
    // And all state objects have been dropped, confirming the #onEvaluationFinished method was
    // called.
    assertThat(globalStateCleanupCounter.get()).isEqualTo(3);
  }

  // Test for the basic functionality of ClassToInstanceMapSkyKeyComputeState, demonstrating
  // that it can hold state associated with different classes and those states are independent.
  @Test
  public void classToInstanceMapSkyKeyComputeState(
      @TestParameter boolean touchStateA, @TestParameter boolean touchStateB)
      throws InterruptedException {
    class StateA implements SkyKeyComputeState {
      boolean touched;
    }
    class StateB implements SkyKeyComputeState {
      boolean touched;
    }
    SkyKey key1 = new SkyKeyForSkyKeyComputeStateTests("key1");
    SkyKey key2 = new SkyKeyForSkyKeyComputeStateTests("key2");
    AtomicLongMap<SkyKey> numCalls = AtomicLongMap.create();
    SkyFunction skyFunctionForTest =
        (skyKey, env) -> {
          int numCallsForKey = (int) numCalls.incrementAndGet(skyKey);
          if (skyKey.equals(key1)) {
            if (numCallsForKey == 1) {
              if (touchStateA) {
                env.getState(ClassToInstanceMapSkyKeyComputeState::new)
                        .getInstance(StateA.class, StateA::new)
                        .touched =
                    true;
              }
              if (touchStateB) {
                env.getState(ClassToInstanceMapSkyKeyComputeState::new)
                        .getInstance(StateB.class, StateB::new)
                        .touched =
                    true;
              }
              assertThat(env.getValue(key2)).isNull();
              return null;
            }
            assertThat(
                    env.getState(ClassToInstanceMapSkyKeyComputeState::new)
                        .getInstance(StateA.class, StateA::new)
                        .touched)
                .isEqualTo(touchStateA);
            assertThat(
                    env.getState(ClassToInstanceMapSkyKeyComputeState::new)
                        .getInstance(StateB.class, StateB::new)
                        .touched)
                .isEqualTo(touchStateB);
            SkyValue value = env.getValue(key2);
            assertThat(value).isEqualTo(new StringValue("value"));
            return value;
          }
          if (skyKey.equals(key2)) {
            return new StringValue("value");
          }
          throw new IllegalStateException();
        };
    tester.putSkyFunction(SkyKeyForSkyKeyComputeStateTests.FUNCTION_NAME, skyFunctionForTest);
    graph = new InMemoryGraphImpl();
    SkyValue resultValue = eval(/* keepGoing= */ true, key1);
    assertThat(resultValue).isEqualTo(new StringValue("value"));
  }

  private static class PartialReevaluationKey extends AbstractSkyKey<String> {
    private static final SkyFunctionName FUNCTION_NAME =
        SkyFunctionName.createHermetic("PARTIAL_REEVALUATION");

    private PartialReevaluationKey(String arg) {
      super(arg);
    }

    @Override
    public SkyFunctionName functionName() {
      return FUNCTION_NAME;
    }

    @Override
    public boolean supportsPartialReevaluation() {
      return true;
    }
  }

  /**
   * A bundle of {@link SkyframeLookupResult}s that can be consumed from as if it was a single one.
   */
  static class DelegatingSkyframeLookupResult implements SkyframeLookupResult {
    private static final Splitter BATCH_SPLITTER = Splitter.on(';');
    private static final Splitter KEY_SPLITTER = Splitter.on(',');
    private final ImmutableMap<SkyKey, SkyframeLookupResult> resultsByKey;

    /**
     * Makes {@link SkyFunction.Environment#getValuesAndExceptions} calls in batches as described by
     * {@code requestBatches} and returns them bundled up.
     *
     * <p>The {@code requestBatches} string is split first by ';' to define an order-sensitive
     * sequence of batches, then by ',' to define an order-insensitive collection of keys.
     */
    static DelegatingSkyframeLookupResult fromRequestBatches(
        String requestBatches,
        SkyFunction.Environment env,
        ImmutableList<? extends AbstractSkyKey<String>> keys)
        throws InterruptedException {
      ImmutableMap<String, ? extends AbstractSkyKey<String>> keysByArg =
          keys.stream().collect(ImmutableMap.toImmutableMap(AbstractSkyKey::argument, k -> k));

      ImmutableMap.Builder<SkyKey, SkyframeLookupResult> builder = new ImmutableMap.Builder<>();
      Iterable<String> batches = BATCH_SPLITTER.split(requestBatches);
      for (String batch : batches) {
        Iterable<String> batchKeys = KEY_SPLITTER.split(batch);
        SkyframeLookupResult result =
            env.getValuesAndExceptions(
                stream(batchKeys).map(keysByArg::get).collect(toImmutableList()));
        for (String batchKey : batchKeys) {
          builder.put(checkNotNull(keysByArg.get(batchKey)), result);
        }
      }
      ImmutableMap<SkyKey, SkyframeLookupResult> resultsByKey = builder.buildOrThrow();

      assertThat(resultsByKey).hasSize(keys.size());
      return new DelegatingSkyframeLookupResult(resultsByKey);
    }

    private DelegatingSkyframeLookupResult(
        ImmutableMap<SkyKey, SkyframeLookupResult> resultsByKey) {
      this.resultsByKey = resultsByKey;
    }

    @Nullable
    @Override
    public <E1 extends Exception, E2 extends Exception, E3 extends Exception> SkyValue getOrThrow(
        SkyKey skyKey,
        @Nullable Class<E1> exceptionClass1,
        @Nullable Class<E2> exceptionClass2,
        @Nullable Class<E3> exceptionClass3)
        throws E1, E2, E3 {
      return checkNotNull(resultsByKey.get(skyKey))
          .getOrThrow(skyKey, exceptionClass1, exceptionClass2, exceptionClass3);
    }

    @Override
    public boolean queryDep(SkyKey key, QueryDepCallback resultCallback) {
      return checkNotNull(resultsByKey.get(key)).queryDep(key, resultCallback);
    }
  }

  @Test
  public void partialReevaluationOneButNotAllDeps(
      @TestParameter({"key2,key3", "key2;key3", "key3;key2"}) String requestBatches)
      throws InterruptedException {
    // The parameterization of this test ensures that the partial reevaluation implementation is
    // insensitive to dep grouping.

    // This test illustrates the basic functionality of partial reevaluation: that a
    // function opting into partial reevaluation can be resumed when one, but not all, of its
    // previously requested-and-not-already-evaluated dependencies finishes evaluation.

    // Graph structure:
    // * key1 depends on key2 and key3

    // Evaluation behavior:
    // * key3 will not finish evaluating until key1 has been restarted with the result of key2

    PartialReevaluationKey key1 = new PartialReevaluationKey("key1");
    PartialReevaluationKey key2 = new PartialReevaluationKey("key2");
    PartialReevaluationKey key3 = new PartialReevaluationKey("key3");
    AtomicInteger key1EvaluationCount = new AtomicInteger();
    CountDownLatch key1ObservesTheValueOfKey2 = new CountDownLatch(1);
    SkyFunction f =
        (skyKey, env) -> {
          PartialReevaluationMailbox mailbox =
              PartialReevaluationMailbox.from(
                  env.getState(ClassToInstanceMapSkyKeyComputeState::new));
          Mail mail = mailbox.getMail();
          assertThat(mailbox.getMail().kind()).isEqualTo(Kind.EMPTY);

          if (skyKey.equals(key1)) {
            int c = key1EvaluationCount.incrementAndGet();
            SkyframeLookupResult result =
                DelegatingSkyframeLookupResult.fromRequestBatches(
                    requestBatches, env, ImmutableList.of(key2, key3));
            if (c == 1) {
              assertThat(mail.kind()).isEqualTo(Kind.FRESHLY_INITIALIZED);
              assertThat(result.get(key2)).isNull();
              assertThat(result.get(key3)).isNull();
              return null;
            }
            if (c == 2) {
              assertThat(mail.kind()).isEqualTo(Kind.CAUSES);
              assertThat(mail.causes().signaledDeps()).containsExactly(key2);
              assertThat(result.get(key2)).isEqualTo(StringValue.of("val2"));
              assertThat(result.get(key3)).isNull();
              key1ObservesTheValueOfKey2.countDown();
              return null;
            }
            assertThat(c).isEqualTo(3);
            assertThat(mail.kind()).isEqualTo(Kind.CAUSES);
            assertThat(mail.causes().signaledDeps()).containsExactly(key3);
            assertThat(result.get(key2)).isEqualTo(StringValue.of("val2"));
            assertThat(result.get(key3)).isEqualTo(StringValue.of("val3"));
            return StringValue.of("val1");
          }
          if (skyKey.equals(key2)) {
            assertThat(mail.kind()).isEqualTo(Kind.FRESHLY_INITIALIZED);
            return StringValue.of("val2");
          }
          assertThat(skyKey).isEqualTo(key3);
          assertThat(mail.kind()).isEqualTo(Kind.FRESHLY_INITIALIZED);
          assertThat(
                  key1ObservesTheValueOfKey2.await(
                      TestUtils.WAIT_TIMEOUT_SECONDS, TimeUnit.SECONDS))
              .isTrue();
          return StringValue.of("val3");
        };
    tester.putSkyFunction(PartialReevaluationKey.FUNCTION_NAME, f);
    graph = new InMemoryGraphImpl();
    SkyValue resultValue = eval(/* keepGoing= */ true, key1);
    assertThat(resultValue).isEqualTo(StringValue.of("val1"));
    assertThat(key1EvaluationCount.get()).isEqualTo(3);
  }

  @Test
  public void partialReevaluationOneDuringAReevaluation(
      @TestParameter({
            "key2,key3,key4",
            "key2,key3;key4",
            "key2;key3,key4",
            "key2;key3;key4",
            // permute (2,3):
            "key3;key2,key4",
            "key3;key2;key4",
            // permute (2,4):
            "key4,key3;key2",
            "key4;key3,key2",
            "key4;key3;key2",
            // permute (3,4):
            "key2,key4;key3",
            "key2;key4;key3",
            // permute (2,3,4):
            "key4;key2;key3"
          })
          String requestBatches)
      throws InterruptedException {
    // The parameterization of this test ensures that the partial reevaluation implementation is
    // insensitive to dep grouping.

    // This test illustrates another bit of partial reevaluation functionality: when a partial
    // reevaluation is currently underway, and when another one of its previously
    // requested-and-not-already-evaluated dependencies finishes evaluation (but not all of them),
    // another partial reevaluation will run when the current one finishes.

    // Graph structure:
    // * key1 depends on key2, key3, and key4
    // * key5 depends on key3 (it's here only to detect when key3 signals its parents)

    // Evaluation behavior:
    // * key4 will not finish evaluating until key1 has been restarted with the result of key3
    // * key1's partial reevaluation observing key2 does not finish until key3 has signaled its
    //   parents (i.e. key1 and key5)
    // * key3 will not finish evaluating until key1 has been restarted with the result of key2

    PartialReevaluationKey key1 = new PartialReevaluationKey("key1");
    PartialReevaluationKey key2 = new PartialReevaluationKey("key2");
    PartialReevaluationKey key3 = new PartialReevaluationKey("key3");
    PartialReevaluationKey key4 = new PartialReevaluationKey("key4");
    PartialReevaluationKey key5 = new PartialReevaluationKey("key5");
    AtomicInteger key1EvaluationCount = new AtomicInteger();
    AtomicInteger key5EvaluationCount = new AtomicInteger();
    AtomicInteger key1EvaluationsInflight = new AtomicInteger();
    CountDownLatch key1ObservesTheValueOfKey2 = new CountDownLatch(1);
    CountDownLatch key5ObservesTheAbsenceOfKey3 = new CountDownLatch(1);
    CountDownLatch key1ObservesTheValueOfKey3 = new CountDownLatch(1);
    CountDownLatch key3SignaledItsParents = new CountDownLatch(1);
    SkyFunction f =
        (skyKey, env) -> {
          Mail mail =
              PartialReevaluationMailbox.from(
                      env.getState(ClassToInstanceMapSkyKeyComputeState::new))
                  .getMail();
          if (skyKey.equals(key1)) {
            assertThat(key1EvaluationsInflight.incrementAndGet()).isEqualTo(1);
            try {
              int c = key1EvaluationCount.incrementAndGet();
              SkyframeLookupResult result =
                  DelegatingSkyframeLookupResult.fromRequestBatches(
                      requestBatches, env, ImmutableList.of(key2, key3, key4));
              if (c == 1) {
                assertThat(mail.kind()).isEqualTo(Kind.FRESHLY_INITIALIZED);
                assertThat(result.get(key2)).isNull();
                assertThat(result.get(key3)).isNull();
                assertThat(result.get(key4)).isNull();
                return null;
              }
              if (c == 2) {
                assertThat(mail.kind()).isEqualTo(Kind.CAUSES);
                assertThat(mail.causes().signaledDeps()).containsExactly(key2);
                assertThat(result.get(key2)).isEqualTo(StringValue.of("val2"));
                assertThat(result.get(key3)).isNull();
                assertThat(result.get(key4)).isNull();
                key1ObservesTheValueOfKey2.countDown();
                assertThat(
                        key3SignaledItsParents.await(
                            TestUtils.WAIT_TIMEOUT_SECONDS, TimeUnit.SECONDS))
                    .isTrue();
                return null;
              }
              if (c == 3) {
                assertThat(mail.kind()).isEqualTo(Kind.CAUSES);
                assertThat(mail.causes().signaledDeps()).containsExactly(key3);
                assertThat(result.get(key2)).isEqualTo(StringValue.of("val2"));
                assertThat(result.get(key3)).isEqualTo(StringValue.of("val3"));
                assertThat(result.get(key4)).isNull();
                key1ObservesTheValueOfKey3.countDown();
                return null;
              }
              assertThat(c).isEqualTo(4);
              assertThat(mail.kind()).isEqualTo(Kind.CAUSES);
              assertThat(mail.causes().signaledDeps()).containsExactly(key4);
              assertThat(result.get(key2)).isEqualTo(StringValue.of("val2"));
              assertThat(result.get(key3)).isEqualTo(StringValue.of("val3"));
              assertThat(result.get(key4)).isEqualTo(StringValue.of("val4"));
              return StringValue.of("val1");
            } finally {
              assertThat(key1EvaluationsInflight.decrementAndGet()).isEqualTo(0);
            }
          }

          if (skyKey.equals(key2)) {
            assertThat(mail.kind()).isEqualTo(Kind.FRESHLY_INITIALIZED);
            return StringValue.of("val2");
          }

          if (skyKey.equals(key3)) {
            assertThat(mail.kind()).isEqualTo(Kind.FRESHLY_INITIALIZED);
            assertThat(
                    key1ObservesTheValueOfKey2.await(
                        TestUtils.WAIT_TIMEOUT_SECONDS, TimeUnit.SECONDS))
                .isTrue();
            assertThat(
                    key5ObservesTheAbsenceOfKey3.await(
                        TestUtils.WAIT_TIMEOUT_SECONDS, TimeUnit.SECONDS))
                .isTrue();
            return StringValue.of("val3");
          }

          if (skyKey.equals(key4)) {
            assertThat(mail.kind()).isEqualTo(Kind.FRESHLY_INITIALIZED);
            assertThat(
                    key1ObservesTheValueOfKey3.await(
                        TestUtils.WAIT_TIMEOUT_SECONDS, TimeUnit.SECONDS))
                .isTrue();
            return StringValue.of("val4");
          }

          assertThat(skyKey).isEqualTo(key5);
          int c = key5EvaluationCount.incrementAndGet();
          if (c == 1) {
            assertThat(mail.kind()).isEqualTo(Kind.FRESHLY_INITIALIZED);
            SkyValue value = env.getValue(key3);
            assertThat(value).isNull();
            key5ObservesTheAbsenceOfKey3.countDown();
            return null;
          }
          assertThat(c).isEqualTo(2);
          assertThat(mail.kind()).isEqualTo(Kind.CAUSES);
          assertThat(mail.causes().signaledDeps()).containsExactly(key3);
          SkyValue value = env.getValue(key3);
          assertThat(value).isEqualTo(StringValue.of("val3"));
          key3SignaledItsParents.countDown();
          return StringValue.of("val5");
        };
    tester.putSkyFunction(PartialReevaluationKey.FUNCTION_NAME, f);
    graph = new InMemoryGraphImpl();
    EvaluationResult<SkyValue> resultValues = eval(/* keepGoing= */ true, key1, key5);
    assertThat(resultValues.get(key1)).isEqualTo(StringValue.of("val1"));
    assertThat(resultValues.get(key5)).isEqualTo(StringValue.of("val5"));
    assertThat(key1EvaluationsInflight.get()).isEqualTo(0);
    assertThat(key1EvaluationCount.get()).isEqualTo(4);
  }

  @Test
  public void partialReevaluationErrorDuringReevaluation(@TestParameter boolean keepGoing)
      throws InterruptedException {
    // This test illustrates how the partial reevaluation implementation handles the case in which a
    // SkyFunction throws an error during a partial reevaluation: the error is ignored if the
    // partially-reevaluated node has not yet been fully signaled. This maintains the invariant that
    // nodes are not committed with a value or error while they have deps which may signal them.

    // Note that Skyframe policy encourages SkyFunction behavior such as this: SkyFunction
    // implementations should eagerly throw errors even when not all their deps are done, to better
    // support error contextualization during no-keep_going builds.

    // Graph structure:
    // * key1 depends on key2, key3, and key4
    // * key5 depends on key3 (it's here only to detect when key3 signals its parents)

    // Evaluation behavior:
    // * key4 will not finish evaluating until key1 has been restarted with the result of key3
    // * key1's partial reevaluation observing key2 does not finish until key3 has signaled its
    //   parents (i.e. key1 and key5)
    // * key3 will not finish evaluating until key1 has been restarted with the result of key2

    PartialReevaluationKey key1 = new PartialReevaluationKey("key1");
    PartialReevaluationKey key2 = new PartialReevaluationKey("key2");
    PartialReevaluationKey key3 = new PartialReevaluationKey("key3");
    PartialReevaluationKey key4 = new PartialReevaluationKey("key4");
    PartialReevaluationKey key5 = new PartialReevaluationKey("key5");
    AtomicInteger key1EvaluationCount = new AtomicInteger();
    AtomicInteger key5EvaluationCount = new AtomicInteger();
    AtomicInteger key1EvaluationsInflight = new AtomicInteger();
    CountDownLatch key1ObservesTheValueOfKey2 = new CountDownLatch(1);
    CountDownLatch key5ObservesTheAbsenceOfKey3 = new CountDownLatch(1);
    CountDownLatch key1ObservesTheValueOfKey3 = new CountDownLatch(1);
    CountDownLatch key3SignaledItsParents = new CountDownLatch(1);
    SkyFunction f =
        (skyKey, env) -> {
          Mail mail =
              PartialReevaluationMailbox.from(
                      env.getState(ClassToInstanceMapSkyKeyComputeState::new))
                  .getMail();
          if (skyKey.equals(key1)) {
            assertThat(key1EvaluationsInflight.incrementAndGet()).isEqualTo(1);
            try {
              int c = key1EvaluationCount.incrementAndGet();
              SkyframeLookupResult result =
                  env.getValuesAndExceptions(ImmutableList.of(key2, key3, key4));
              if (c == 1) {
                assertThat(mail.kind()).isEqualTo(Kind.FRESHLY_INITIALIZED);
                assertThat(result.get(key2)).isNull();
                assertThat(result.get(key3)).isNull();
                assertThat(result.get(key4)).isNull();
                return null;
              }
              if (c == 2) {
                assertThat(mail.kind()).isEqualTo(Kind.CAUSES);
                assertThat(mail.causes().signaledDeps()).containsExactly(key2);
                assertThat(result.get(key2)).isEqualTo(StringValue.of("val2"));
                assertThat(result.get(key3)).isNull();
                assertThat(result.get(key4)).isNull();
                key1ObservesTheValueOfKey2.countDown();
                assertThat(
                        key3SignaledItsParents.await(
                            TestUtils.WAIT_TIMEOUT_SECONDS, TimeUnit.SECONDS))
                    .isTrue();
                throw new SkyFunctionExceptionForTest(
                    "Error thrown during partial reevaluation (1)");
              }
              if (c == 3) {
                // The Skyframe stateCache invalidates its entry for a node when it throws an error:
                assertThat(mail.kind()).isEqualTo(Kind.FRESHLY_INITIALIZED);
                assertThat(result.get(key2)).isEqualTo(StringValue.of("val2"));
                assertThat(result.get(key3)).isEqualTo(StringValue.of("val3"));
                assertThat(result.get(key4)).isNull();
                key1ObservesTheValueOfKey3.countDown();
                throw new SkyFunctionExceptionForTest(
                    "Error thrown during partial reevaluation (2)");
              }
              assertThat(c).isEqualTo(4);
              assertThat(mail.kind()).isEqualTo(Kind.FRESHLY_INITIALIZED);
              assertThat(result.get(key2)).isEqualTo(StringValue.of("val2"));
              assertThat(result.get(key3)).isEqualTo(StringValue.of("val3"));
              assertThat(result.get(key4)).isEqualTo(StringValue.of("val4"));
              throw new SkyFunctionExceptionForTest("Error thrown during final full evaluation");
            } finally {
              assertThat(key1EvaluationsInflight.decrementAndGet()).isEqualTo(0);
            }
          }

          if (skyKey.equals(key2)) {
            assertThat(mail.kind()).isEqualTo(Kind.FRESHLY_INITIALIZED);
            return StringValue.of("val2");
          }

          if (skyKey.equals(key3)) {
            assertThat(mail.kind()).isEqualTo(Kind.FRESHLY_INITIALIZED);
            assertThat(
                    key1ObservesTheValueOfKey2.await(
                        TestUtils.WAIT_TIMEOUT_SECONDS, TimeUnit.SECONDS))
                .isTrue();
            assertThat(
                    key5ObservesTheAbsenceOfKey3.await(
                        TestUtils.WAIT_TIMEOUT_SECONDS, TimeUnit.SECONDS))
                .isTrue();
            return StringValue.of("val3");
          }

          if (skyKey.equals(key4)) {
            assertThat(mail.kind()).isEqualTo(Kind.FRESHLY_INITIALIZED);
            assertThat(
                    key1ObservesTheValueOfKey3.await(
                        TestUtils.WAIT_TIMEOUT_SECONDS, TimeUnit.SECONDS))
                .isTrue();
            return StringValue.of("val4");
          }

          assertThat(skyKey).isEqualTo(key5);
          int c = key5EvaluationCount.incrementAndGet();
          if (c == 1) {
            assertThat(mail.kind()).isEqualTo(Kind.FRESHLY_INITIALIZED);
            SkyValue value = env.getValue(key3);
            assertThat(value).isNull();
            key5ObservesTheAbsenceOfKey3.countDown();
            return null;
          }
          assertThat(c).isEqualTo(2);
          assertThat(mail.kind()).isEqualTo(Kind.CAUSES);
          assertThat(mail.causes().signaledDeps()).containsExactly(key3);
          SkyValue value = env.getValue(key3);
          assertThat(value).isEqualTo(StringValue.of("val3"));
          key3SignaledItsParents.countDown();
          return StringValue.of("val5");
        };
    tester.putSkyFunction(PartialReevaluationKey.FUNCTION_NAME, f);
    graph = new InMemoryGraphImpl();
    EvaluationResult<SkyValue> resultValues = eval(keepGoing, key1, key5);
    assertThat(resultValues.getError(key1).getException()).isInstanceOf(SomeErrorException.class);

    // key4's signal to key1 races with key1's readiness check after completing with an error during
    // partial reevaluation 2. If the signal wins, then key1 should be allowed to complete, because
    // it has no outstanding signals. If the signal loses, then key1 will reevaluate one last time.
    assertThat(resultValues.getError(key1).getException())
        .hasMessageThat()
        .isIn(
            ImmutableList.of(
                "Error thrown during partial reevaluation (2)",
                "Error thrown during final full evaluation"));
    if (keepGoing) {
      assertThat(resultValues.get(key5)).isEqualTo(StringValue.of("val5"));
    }
    assertThat(key1EvaluationsInflight.get()).isEqualTo(0);
    assertThat(key1EvaluationCount.get()).isIn(ImmutableList.of(3, 4));
  }

  @Test
  public void partialReevaluationOneErrorButNotAllDeps(
      @TestParameter boolean keepGoing, @TestParameter boolean enrichError)
      throws InterruptedException {
    // The parameterization of this test ensures that the partial reevaluation implementation is
    // insensitive to dep grouping.

    // This test illustrates partial reevaluation's handling of errors in multiple contexts, with
    // and without keepGoing, and with the error-observing function enriching or recovering from the
    // error.

    // Graph structure:
    // * key1 depends on key2 and key3

    // Evaluation behavior:
    // * key3 will not finish evaluating until key1 has been restarted with the error result of key2
    //   (which notably will never happen in no-keepGoing because by then key2 should have
    //   interrupted evaluations)

    PartialReevaluationKey key1 = new PartialReevaluationKey("key1");
    PartialReevaluationKey key2 = new PartialReevaluationKey("key2");
    PartialReevaluationKey key3 = new PartialReevaluationKey("key3");
    AtomicInteger key1EvaluationCount = new AtomicInteger();
    CountDownLatch key1ObservesTheErrorOfKey2 = new CountDownLatch(1);
    SkyFunction f =
        (skyKey, env) -> {
          Mail mail =
              PartialReevaluationMailbox.from(
                      env.getState(ClassToInstanceMapSkyKeyComputeState::new))
                  .getMail();
          if (skyKey.equals(key1)) {
            int c = key1EvaluationCount.incrementAndGet();
            SkyframeLookupResult result = env.getValuesAndExceptions(ImmutableList.of(key2, key3));
            if (c == 1) {
              assertThat(mail.kind()).isEqualTo(Kind.FRESHLY_INITIALIZED);
              assertThat(result.get(key2)).isNull();
              assertThat(result.get(key3)).isNull();
              return null;
            }
            if (c == 2) {
              if (keepGoing) {
                assertThat(mail.kind()).isEqualTo(Kind.CAUSES);
                assertThat(mail.causes().signaledDeps()).containsExactly(key2);
              } else {
                // The Skyframe stateCache invalidates everything when starting error bubbling:
                assertThat(mail.kind()).isEqualTo(Kind.FRESHLY_INITIALIZED);
                assertThat(env.inErrorBubblingForSkyFunctionsThatCanFullyRecoverFromErrors())
                    .isTrue();
              }
              assertThrows(
                  "key2",
                  SomeErrorException.class,
                  () -> result.getOrThrow(key2, SomeErrorException.class));
              assertThat(result.get(key3)).isNull();
              key1ObservesTheErrorOfKey2.countDown();
              if (enrichError) {
                throw new SkyFunctionExceptionForTest("key1 observed key2 exception (w/o key3)");
              } else {
                return null;
              }
            }
            assertThat(c).isEqualTo(3);
            if (enrichError) {
              // The Skyframe stateCache invalidates its entry for a node when it throws an error:
              assertThat(mail.kind()).isEqualTo(Kind.FRESHLY_INITIALIZED);
            } else {
              assertThat(mail.kind()).isEqualTo(Kind.CAUSES);
              assertThat(mail.causes().signaledDeps()).containsExactly(key3);
            }
            assertThat(keepGoing).isTrue();
            assertThrows(
                "key2",
                SomeErrorException.class,
                () -> result.getOrThrow(key2, SomeErrorException.class));
            assertThat(result.get(key3)).isEqualTo(StringValue.of("val3"));
            if (enrichError) {
              throw new SkyFunctionExceptionForTest("key1 observed key2 exception (w/ key3)");
            } else {
              return StringValue.of("val1");
            }
          }
          if (skyKey.equals(key2)) {
            assertThat(mail.kind()).isEqualTo(Kind.FRESHLY_INITIALIZED);
            throw new SkyFunctionExceptionForTest("key2");
          }
          assertThat(skyKey).isEqualTo(key3);
          assertThat(mail.kind()).isEqualTo(Kind.FRESHLY_INITIALIZED);
          assertThat(
                  key1ObservesTheErrorOfKey2.await(
                      TestUtils.WAIT_TIMEOUT_SECONDS, TimeUnit.SECONDS))
              .isTrue();
          assertThat(keepGoing).isTrue();
          return StringValue.of("val3");
        };
    tester.putSkyFunction(PartialReevaluationKey.FUNCTION_NAME, f);
    graph = new InMemoryGraphImpl();
    EvaluationResult<SkyValue> result = eval(/* keepGoing= */ keepGoing, ImmutableList.of(key1));

    if (keepGoing && enrichError) {
      assertThat(result.getError(key1).getException()).isInstanceOf(SomeErrorException.class);
      assertThat(result.getError(key1).getException())
          .hasMessageThat()
          .isEqualTo("key1 observed key2 exception (w/ key3)");
    } else if (keepGoing) {
      // !enrichError -- expect key1 to have recovered
      assertThat(result.get(key1)).isEqualTo(StringValue.of("val1"));
    } else if (enrichError) {
      // !keepGoing -- expect key3 to have not completed
      assertThat(result.getError(key1).getException()).isInstanceOf(SomeErrorException.class);
      assertThat(result.getError(key1).getException())
          .hasMessageThat()
          .isEqualTo("key1 observed key2 exception (w/o key3)");
    } else {
      // !keepGoing && !enrichError -- expect key2's error
      assertThat(result.getError(key1).getException()).isInstanceOf(SomeErrorException.class);
      assertThat(result.getError(key1).getException()).hasMessageThat().isEqualTo("key2");
    }

    assertThat(key1EvaluationCount.get()).isEqualTo(keepGoing ? 3 : 2);
  }

  @Test
  public void partialReevaluationErrorObservedDuringReevaluation() throws InterruptedException {
    // This test illustrates how the partial reevaluation implementation handles the case in which a
    // SkyFunction doing a partial reevaluation has a dep which completes with an error during a
    // no-keep_going build: that partial reevaluation ends, because
    // ParallelEvaluatorContext.signalParentsOnAbort completely ignores the value returned by
    // entry.signalDep. The node undergoing partial evaluation may or may not be chosen when
    // bubbling up the dep's error; in this test it's chosen because it's the only parent.

    // Note that the partial reevaluation implementation is *not* responsible for this behavior;
    // rather, it is due to "core" Skyframe evaluation policy. However, this test documents the
    // expected behavior of partial reevaluation implementations, in case of future changes to
    // Skyframe.

    // Graph structure:
    // * key1 depends on key2, key3, and key4

    // Evaluation behavior:
    // * key4 will not finish evaluating -- it gets interrupted by key2's error and is never resumed
    // * key3 will not finish evaluating until key1 has been restarted with the result of key2
    // * key1 only observes key2's error during error bubbling

    PartialReevaluationKey key1 = new PartialReevaluationKey("key1");
    PartialReevaluationKey key2 = new PartialReevaluationKey("key2");
    PartialReevaluationKey key3 = new PartialReevaluationKey("key3");
    PartialReevaluationKey key4 = new PartialReevaluationKey("key4");
    AtomicInteger key1EvaluationCount = new AtomicInteger();
    AtomicInteger key4EvaluationCount = new AtomicInteger();
    AtomicInteger key1EvaluationsInflight = new AtomicInteger();
    CountDownLatch key1ObservesTheValueOfKey2 = new CountDownLatch(1);
    CountDownLatch key4WaitsUntilInterruptedByNoKeepGoingEvaluationShutdown = new CountDownLatch(1);
    SkyFunction f =
        (skyKey, env) -> {
          Mail mail =
              PartialReevaluationMailbox.from(
                      env.getState(ClassToInstanceMapSkyKeyComputeState::new))
                  .getMail();
          if (skyKey.equals(key1)) {
            assertThat(key1EvaluationsInflight.incrementAndGet()).isEqualTo(1);
            try {
              int c = key1EvaluationCount.incrementAndGet();
              SkyframeLookupResult result =
                  env.getValuesAndExceptions(ImmutableList.of(key2, key3, key4));
              if (c == 1) {
                assertThat(mail.kind()).isEqualTo(Kind.FRESHLY_INITIALIZED);
                assertThat(result.get(key2)).isNull();
                assertThat(result.get(key3)).isNull();
                assertThat(result.get(key4)).isNull();
                return null;
              }
              if (c == 2) {
                assertThat(mail.kind()).isEqualTo(Kind.CAUSES);
                assertThat(mail.causes().signaledDeps()).containsExactly(key2);
                assertThat(result.get(key2)).isEqualTo(StringValue.of("val2"));
                assertThat(result.get(key3)).isNull();
                assertThat(result.get(key4)).isNull();
                key1ObservesTheValueOfKey2.countDown();
                return null;
              }
              assertThat(c).isEqualTo(3);
              // The Skyframe stateCache invalidates everything when starting error bubbling:
              assertThat(mail.kind()).isEqualTo(Kind.FRESHLY_INITIALIZED);
              assertThat(env.inErrorBubblingForSkyFunctionsThatCanFullyRecoverFromErrors())
                  .isTrue();
              assertThat(result.get(key2)).isEqualTo(StringValue.of("val2"));
              assertThrows(
                  "key3",
                  SomeErrorException.class,
                  () -> result.getOrThrow(key3, SomeErrorException.class));
              assertThat(result.get(key4)).isNull();
              throw new SkyFunctionExceptionForTest("Error thrown after partial reevaluation");
            } finally {
              assertThat(key1EvaluationsInflight.decrementAndGet()).isEqualTo(0);
            }
          }

          if (skyKey.equals(key2)) {
            assertThat(mail.kind()).isEqualTo(Kind.FRESHLY_INITIALIZED);
            return StringValue.of("val2");
          }

          if (skyKey.equals(key3)) {
            assertThat(mail.kind()).isEqualTo(Kind.FRESHLY_INITIALIZED);
            assertThat(
                    key1ObservesTheValueOfKey2.await(
                        TestUtils.WAIT_TIMEOUT_SECONDS, TimeUnit.SECONDS))
                .isTrue();
            throw new SkyFunctionExceptionForTest("key3");
          }

          assertThat(skyKey).isEqualTo(key4);
          assertThat(mail.kind()).isEqualTo(Kind.FRESHLY_INITIALIZED);
          assertThat(key4EvaluationCount.incrementAndGet()).isEqualTo(1);
          throw assertThrows(
              InterruptedException.class,
              () ->
                  key4WaitsUntilInterruptedByNoKeepGoingEvaluationShutdown.await(
                      TestUtils.WAIT_TIMEOUT_SECONDS, TimeUnit.SECONDS));
        };
    tester.putSkyFunction(PartialReevaluationKey.FUNCTION_NAME, f);
    graph = new InMemoryGraphImpl();
    EvaluationResult<SkyValue> resultValues = eval(/* keepGoing= */ false, ImmutableList.of(key1));
    assertThat(resultValues.getError(key1).getException()).isInstanceOf(SomeErrorException.class);
    assertThat(resultValues.getError(key1).getException())
        .hasMessageThat()
        .isEqualTo("Error thrown after partial reevaluation");
    assertThat(key1EvaluationsInflight.get()).isEqualTo(0);
    assertThat(key1EvaluationCount.get()).isEqualTo(3);
  }

  // Regression test for b/225877591 ("Unexpected missing value in PrepareDepsOfPatternsFunction
  // when there's both a dep with a cached cycle and another dep with an error").
  @Test
  public void testDepOnCachedNodeThatItselfDependsOnBothCycleAndError() throws Exception {
    graph = new InMemoryGraphImpl();
    SkyKey parentKey = skyKey("parent");
    SkyKey childKey = skyKey("child");
    SkyKey cycleKey = skyKey("cycle");
    SkyKey errorKey = skyKey("error");
    tester.getOrCreate(cycleKey).addDependency(cycleKey);
    tester.getOrCreate(errorKey).setHasError(true);
    tester.getOrCreate(childKey).addDependency(cycleKey).addDependency(errorKey);

    EvaluationResult<StringValue> result = eval(/* keepGoing= */ true, ImmutableList.of(childKey));
    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(childKey)
        .hasCycleInfoThat()
        .containsExactly(new CycleInfo(ImmutableList.of(childKey), ImmutableList.of(cycleKey)));
    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(childKey)
        .hasExceptionThat()
        .hasMessageThat()
        .contains(errorKey.toString());

    tester
        .getOrCreate(parentKey)
        .setBuilder(
            (key, env) -> {
              SkyframeLookupResult unusedLookupResult =
                  env.getValuesAndExceptions(ImmutableList.of(childKey));
              // env.valuesMissing() should always return true given the graph structure staged
              // above, regardless of the cached state. (This test case specifically stages a fully
              // cached graph state and picks on the getValuesAndExceptions method, because that was
              // the situation of the bug for which this is a regression test.)
              //
              // If childKey is legit not yet in the graph, then of course env.valuesMissing()
              // should return true.
              //
              // If childKey is in the graph, then env.valuesMissing() should still return true
              // because childKey transitively depends on a cycle.
              assertThat(env.valuesMissing()).isTrue();
              return null;
            });
    result = eval(/* keepGoing= */ true, ImmutableList.of(parentKey));
    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(parentKey)
        .hasCycleInfoThat()
        .containsExactly(
            new CycleInfo(ImmutableList.of(parentKey, childKey), ImmutableList.of(cycleKey)));
    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(parentKey)
        .hasExceptionThat()
        .hasMessageThat()
        .contains(errorKey.toString());
  }

  private void evalParallelSkyFunctionAndVerifyResults(
      SkyFunction testFunction,
      QuiescingExecutor testExecutor,
      AtomicInteger actualRunnableCount,
      int expectRunnableCount)
      throws InterruptedException {
    SkyKey parentKey = skyKey("parentKey");
    tester.getOrCreate(parentKey).setBuilder(testFunction);

    graph = new InMemoryGraphImpl();
    ParallelEvaluator parallelEvaluator =
        new ParallelEvaluator(
            graph,
            graphVersion,
            Version.minimal(),
            tester.getSkyFunctionMap(),
            reportedEvents,
            new EmittedEventState(),
            EventFilter.FULL_STORAGE,
            ErrorInfoManager.UseChildErrorInfoIfNecessary.INSTANCE,
            // Doesn't matter for this test case.
            /* keepGoing= */ false,
            revalidationReceiver,
            GraphInconsistencyReceiver.THROWING,
            // We ought not need more than 1 thread for this test case.
            testExecutor,
            new SimpleCycleDetector(),
            UnnecessaryTemporaryStateDropperReceiver.NULL);

    EvaluationResult<StringValue> result = parallelEvaluator.eval(ImmutableList.of(parentKey));
    assertThat(result.hasError()).isFalse();
    assertThat(result.get(parentKey)).isEqualTo(new StringValue("parentValue"));
    assertThat(actualRunnableCount.get()).isEqualTo(expectRunnableCount);
  }

  @Test
  public void testParallelSkyFunctionComputation_runnablesOnBothCurrentAndExternalThreads()
      throws InterruptedException {
    QuiescingExecutor testExecutor =
        AbstractQueueVisitor.create("test-pool", 10, ParallelEvaluatorErrorClassifier.instance());
    AtomicInteger actualRunnableCount = new AtomicInteger(0);

    // Let's arbitrarily set the expected size of Runnables as a random number between 10 and 30.
    int expectRunnableCount = 10 + new Random().nextInt(20);

    SkyFunction testFunction =
        (key, env) -> {
          CountDownLatch countDownLatch = new CountDownLatch(expectRunnableCount);
          BlockingQueue<Runnable> runnablesQueue = new LinkedBlockingQueue<>();
          for (int i = 0; i < expectRunnableCount; ++i) {
            runnablesQueue.put(
                () -> {
                  actualRunnableCount.incrementAndGet();
                  countDownLatch.countDown();
                });
          }

          Runnable drainQueue =
              () -> {
                Runnable next;
                while ((next = runnablesQueue.poll()) != null) {
                  next.run();
                }
              };

          QuiescingExecutor executor = env.getParallelEvaluationExecutor();
          assertThat(executor).isSameInstanceAs(testExecutor);
          for (int i = 0; i < expectRunnableCount; ++i) {
            executor.execute(drainQueue);
          }

          // After dispatching Runnables to threads on the executor, it is preferred that
          // current thread also participates in executing some (or even all) runnables. It is
          // possible that other threads on the executor are all busy and will not be available
          // for a fairly long time. So having the current thread participate will prevent
          // current thread from being completely idle while waiting for the runnableQueue to be
          // fully drained.
          drainQueue.run();

          // It is possible that the last runnable executed on the current thread ends earlier
          // than what are executed on the other threads. So we need to wait for all necessary
          // computations to complete before returning.
          countDownLatch.await();
          return new StringValue("parentValue");
        };

    evalParallelSkyFunctionAndVerifyResults(
        testFunction, testExecutor, actualRunnableCount, expectRunnableCount);
  }

  @Test
  public void testParallelSkyFunctionComputation_runnablesOnExternalThreadsOnly()
      throws InterruptedException {
    QuiescingExecutor testExecutor =
        AbstractQueueVisitor.create("test-pool", 10, ParallelEvaluatorErrorClassifier.instance());
    AtomicInteger actualRunnableCount = new AtomicInteger(0);

    // Let's arbitrarily set the expected size of Runnables as a random number between 10 and 30.
    int expectRunnableCount = 10 + new Random().nextInt(20);

    SkyFunction testFunction =
        (key, env) -> {
          CountDownLatch countDownLatch = new CountDownLatch(expectRunnableCount);
          QuiescingExecutor executor = env.getParallelEvaluationExecutor();
          assertThat(executor).isSameInstanceAs(testExecutor);
          for (int i = 0; i < expectRunnableCount; ++i) {
            executor.execute(
                () -> {
                  actualRunnableCount.incrementAndGet();
                  countDownLatch.countDown();
                });
          }

          // We have to wait for all execution dispatched to external threads to complete before
          // returning.
          countDownLatch.await();
          return new StringValue("parentValue");
        };

    evalParallelSkyFunctionAndVerifyResults(
        testFunction, testExecutor, actualRunnableCount, expectRunnableCount);
  }
}
