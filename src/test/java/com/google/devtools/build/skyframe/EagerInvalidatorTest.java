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
import static com.google.devtools.build.skyframe.GraphTester.CONCATENATE;
import static com.google.devtools.build.skyframe.GraphTester.NODE_TYPE;
import static com.google.devtools.build.skyframe.GraphTester.skyKey;
import static org.junit.Assert.assertThrows;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.collect.Sets;
import com.google.common.eventbus.EventBus;
import com.google.common.testing.GcFinalization;
import com.google.devtools.build.lib.concurrent.AbstractQueueVisitor;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.skyframe.EvaluationContext.UnnecessaryTemporaryStateDropperReceiver;
import com.google.devtools.build.skyframe.GraphTester.StringValue;
import com.google.devtools.build.skyframe.InvalidatingNodeVisitor.DeletingNodeVisitor;
import com.google.devtools.build.skyframe.InvalidatingNodeVisitor.DirtyingInvalidationState;
import com.google.devtools.build.skyframe.InvalidatingNodeVisitor.DirtyingNodeVisitor;
import com.google.devtools.build.skyframe.InvalidatingNodeVisitor.InvalidationType;
import com.google.devtools.build.skyframe.QueryableGraph.Reason;
import java.lang.ref.WeakReference;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import javax.annotation.Nullable;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.experimental.runners.Enclosed;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link InvalidatingNodeVisitor}.
 */
@RunWith(Enclosed.class)
public class EagerInvalidatorTest {
  protected InMemoryGraphImpl graph;
  protected GraphTester tester = new GraphTester();
  protected InvalidatingNodeVisitor.InvalidationState state = newInvalidationState();
  protected AtomicReference<InvalidatingNodeVisitor<?>> visitor = new AtomicReference<>();
  protected DirtyAndInflightTrackingProgressReceiver progressReceiver;
  private IntVersion graphVersion = IntVersion.of(0);

  @After
  public void assertNoTrackedErrors() {
    TrackingAwaiter.INSTANCE.assertNoErrors();
  }

  // The following three methods should be abstract, but junit4 does not allow us to run inner
  // classes in an abstract outer class. Thus, we provide implementations. These methods will never
  // be run because only the inner classes, annotated with @RunWith, will actually be executed.
  InvalidationProgressReceiver.InvalidationState expectedState() {
    throw new UnsupportedOperationException();
  }

  @SuppressWarnings("unused") // Overridden by subclasses.
  void invalidate(
      InMemoryGraph graph,
      DirtyAndInflightTrackingProgressReceiver progressReceiver,
      SkyKey... keys)
      throws InterruptedException {
    throw new UnsupportedOperationException();
  }

  boolean gcExpected() {
    throw new UnsupportedOperationException();
  }

  private boolean isInvalidated(SkyKey key) {
    NodeEntry entry = graph.get(null, Reason.OTHER, key);
    if (gcExpected()) {
      return entry == null;
    } else {
      return entry == null || entry.isDirty();
    }
  }

  private void assertChanged(SkyKey key) {
    NodeEntry entry = graph.get(null, Reason.OTHER, key);
    if (gcExpected()) {
      assertThat(entry).isNull();
    } else {
      assertThat(entry.isChanged()).isTrue();
    }
  }

  private void assertDirtyAndNotChanged(SkyKey key) {
    NodeEntry entry = graph.get(null, Reason.OTHER, key);
    if (gcExpected()) {
      assertThat(entry).isNull();
    } else {
      assertThat(entry.isDirty()).isTrue();
      assertThat(entry.isChanged()).isFalse();
    }

  }

  protected InvalidatingNodeVisitor.InvalidationState newInvalidationState() {
    throw new UnsupportedOperationException("Subclasses must override");
  }

  protected InvalidationType defaultInvalidationType() {
    throw new UnsupportedOperationException("Subclasses must override");
  }

  protected boolean reverseDepsPresent() {
    throw new UnsupportedOperationException("Subclasses must override");
  }

  // Convenience method for eval-ing a single value.
  protected SkyValue eval(boolean keepGoing, SkyKey key) throws InterruptedException {
    SkyKey[] keys = {key};
    return eval(keepGoing, keys).get(key);
  }

  protected <T extends SkyValue> EvaluationResult<T> eval(boolean keepGoing, SkyKey... keys)
      throws InterruptedException {
    return eval(keepGoing, GraphInconsistencyReceiver.THROWING, keys);
  }

  protected <T extends SkyValue> EvaluationResult<T> eval(
      boolean keepGoing, GraphInconsistencyReceiver inconsistencyReceiver, SkyKey... keys)
      throws InterruptedException {
    Reporter reporter = new Reporter(new EventBus());
    ParallelEvaluator evaluator =
        new ParallelEvaluator(
            graph,
            graphVersion,
            Version.minimal(),
            tester.getSkyFunctionMap(),
            reporter,
            new EmittedEventState(),
            EventFilter.FULL_STORAGE,
            ErrorInfoManager.UseChildErrorInfoIfNecessary.INSTANCE,
            keepGoing,
            new DirtyAndInflightTrackingProgressReceiver(EvaluationProgressReceiver.NULL),
            inconsistencyReceiver,
            AbstractQueueVisitor.create(
                "test-pool", 200, ParallelEvaluatorErrorClassifier.instance()),
            new SimpleCycleDetector(),
            UnnecessaryTemporaryStateDropperReceiver.NULL);
    graphVersion = graphVersion.next();
    return evaluator.eval(ImmutableList.copyOf(keys));
  }

  void invalidateWithoutError(
      DirtyAndInflightTrackingProgressReceiver progressReceiver, SkyKey... keys)
      throws InterruptedException {
    invalidate(graph, progressReceiver, keys);
    assertThat(state.isEmpty()).isTrue();
  }

  protected void set(String name, String value) {
    tester.set(name, new StringValue(value));
  }

  private void assertValueValue(String name, String expectedValue) throws InterruptedException {
    StringValue value = (StringValue) eval(false, skyKey(name));
    assertThat(value.getValue()).isEqualTo(expectedValue);
  }

  @Before
  public void setUp() throws Exception {
    progressReceiver =
        new DirtyAndInflightTrackingProgressReceiver(EvaluationProgressReceiver.NULL);
  }

  @Test
  public void receiverWorks() throws Exception {
    Set<SkyKey> invalidated = Sets.newConcurrentHashSet();
    DirtyAndInflightTrackingProgressReceiver receiver =
        new DirtyAndInflightTrackingProgressReceiver(
            new InvalidationProgressReceiver() {
              @Override
              protected void invalidated(SkyKey skyKey, InvalidationState state) {
                Preconditions.checkState(state == expectedState());
                invalidated.add(skyKey);
              }
            });
    graph = new InMemoryGraphImpl();
    SkyKey aKey = GraphTester.nonHermeticKey("a");
    SkyKey bKey = GraphTester.nonHermeticKey("b");
    tester.set(aKey, new StringValue("a"));
    tester.set(bKey, new StringValue("b"));
    tester.getOrCreate("ab").addDependency(aKey).addDependency(bKey).setComputedValue(CONCATENATE);
    assertValueValue("ab", "ab");

    tester.set(aKey, new StringValue("c"));
    invalidateWithoutError(receiver, aKey);
    assertThat(invalidated).containsExactly(aKey, skyKey("ab"));
    assertValueValue("ab", "cb");
    tester.set(bKey, new StringValue("d"));
    invalidateWithoutError(receiver, bKey);
    assertThat(invalidated).containsExactly(aKey, skyKey("ab"), bKey);
  }

  @Test
  public void receiverIsNotifiedAboutNodesInError() throws Exception {
    Set<SkyKey> invalidated = Sets.newConcurrentHashSet();
    DirtyAndInflightTrackingProgressReceiver receiver =
        new DirtyAndInflightTrackingProgressReceiver(
            new InvalidationProgressReceiver() {
              @Override
              protected void invalidated(SkyKey skyKey, InvalidationState state) {
                Preconditions.checkState(state == expectedState());
                invalidated.add(skyKey);
              }
            });

    // Given a graph consisting of two nodes, "a" and "ab" such that "ab" depends on "a",
    // And given "ab" is in error,
    graph = new InMemoryGraphImpl();
    SkyKey aKey = GraphTester.nonHermeticKey("a");
    tester.set(aKey, new StringValue("a"));
    tester.getOrCreate("ab").addDependency(aKey).setHasError(true);
    eval(false, skyKey("ab"));

    // When "a" is invalidated,
    invalidateWithoutError(receiver, aKey);

    // Then the invalidation receiver is notified of both "a" and "ab"'s invalidations.
    assertThat(invalidated).containsExactly(aKey, skyKey("ab"));

    // Note that this behavior isn't strictly required for correctness. This test is
    // meant to document current behavior and protect against programming error.
  }

  @Test
  public void invalidateValuesNotInGraph() throws Exception {
    Set<SkyKey> invalidated = Sets.newConcurrentHashSet();
    DirtyAndInflightTrackingProgressReceiver receiver =
        new DirtyAndInflightTrackingProgressReceiver(
            new InvalidationProgressReceiver() {
              @Override
              protected void invalidated(SkyKey skyKey, InvalidationState state) {
                Preconditions.checkState(state == InvalidationState.DIRTY);
                invalidated.add(skyKey);
              }
            });
    graph = new InMemoryGraphImpl();
    SkyKey aKey = GraphTester.nonHermeticKey("a");
    invalidateWithoutError(receiver, aKey);
    assertThat(invalidated).isEmpty();
    tester.set(aKey, new StringValue("a"));
    StringValue value = (StringValue) eval(false, aKey);
    assertThat(value.getValue()).isEqualTo("a");
    invalidateWithoutError(receiver, GraphTester.nonHermeticKey("b"));
    assertThat(invalidated).isEmpty();
  }

  @Test
  public void invalidatedValuesAreGCedAsExpected() throws Exception {
    SkyKey key = GraphTester.nonHermeticKey("a");
    HeavyValue heavyValue = new HeavyValue();
    WeakReference<HeavyValue> weakRef = new WeakReference<>(heavyValue);
    tester.set(key, heavyValue);

    graph = new InMemoryGraphImpl();
    eval(false, key);
    invalidate(
        graph, new DirtyAndInflightTrackingProgressReceiver(EvaluationProgressReceiver.NULL), key);

    tester = null;
    heavyValue = null;
    if (gcExpected()) {
      GcFinalization.awaitClear(weakRef);
    } else {
      // Not a reliable check, but better than nothing.
      System.gc();
      Thread.sleep(300);
      assertThat(weakRef.get()).isNotNull();
    }
  }

  @Test
  public void reverseDepsConsistent() throws Exception {
    graph = new InMemoryGraphImpl();
    set("a", "a");
    set("b", "b");
    set("c", "c");
    SkyKey abKey = GraphTester.nonHermeticKey("ab");
    tester.getOrCreate(abKey).addDependency("a").addDependency("b").setComputedValue(CONCATENATE);
    tester.getOrCreate("bc").addDependency("b").addDependency("c").setComputedValue(CONCATENATE);
    tester
        .getOrCreate("ab_c")
        .addDependency(abKey)
        .addDependency("c")
        .setComputedValue(CONCATENATE);
    eval(false, skyKey("ab_c"), skyKey("bc"));

    assertThat(graph.get(null, Reason.OTHER, skyKey("a")).getReverseDepsForDoneEntry())
        .containsExactly(abKey);
    assertThat(graph.get(null, Reason.OTHER, skyKey("b")).getReverseDepsForDoneEntry())
        .containsExactly(abKey, skyKey("bc"));
    assertThat(graph.get(null, Reason.OTHER, skyKey("c")).getReverseDepsForDoneEntry())
        .containsExactly(skyKey("ab_c"), skyKey("bc"));

    invalidateWithoutError(
        new DirtyAndInflightTrackingProgressReceiver(EvaluationProgressReceiver.NULL), abKey);
    eval(false);

    // The graph values should be gone.
    assertThat(isInvalidated(abKey)).isTrue();
    assertThat(isInvalidated(skyKey("abc"))).isTrue();

    // The reverse deps to ab and ab_c should have been removed if reverse deps are cleared.
    Set<SkyKey> reverseDeps = new HashSet<>();
    if (reverseDepsPresent()) {
      reverseDeps.add(abKey);
    }
    assertThat(graph.get(null, Reason.OTHER, skyKey("a")).getReverseDepsForDoneEntry())
        .containsExactlyElementsIn(reverseDeps);
    reverseDeps.add(skyKey("bc"));
    assertThat(graph.get(null, Reason.OTHER, skyKey("b")).getReverseDepsForDoneEntry())
        .containsExactlyElementsIn(reverseDeps);
    reverseDeps.clear();
    if (reverseDepsPresent()) {
      reverseDeps.add(skyKey("ab_c"));
    }
    reverseDeps.add(skyKey("bc"));
    assertThat(graph.get(null, Reason.OTHER, skyKey("c")).getReverseDepsForDoneEntry())
        .containsExactlyElementsIn(reverseDeps);
  }

  @Test
  public void interruptChild() throws Exception {
    graph = new InMemoryGraphImpl();
    int numValues = 50; // More values than the invalidator has threads.
    SkyKey[] family = new SkyKey[numValues];
    SkyKey child = GraphTester.nonHermeticKey("child");
    StringValue childValue = new StringValue("child");
    tester.set(child, childValue);
    family[0] = child;
    for (int i = 1; i < numValues; i++) {
      SkyKey member = skyKey(Integer.toString(i));
      tester.getOrCreate(member).addDependency(family[i - 1]).setComputedValue(CONCATENATE);
      family[i] = member;
    }
    SkyKey parent = skyKey("parent");
    tester.getOrCreate(parent).addDependency(family[numValues - 1]).setComputedValue(CONCATENATE);
    eval(/* keepGoing= */ false, parent);
    Thread mainThread = Thread.currentThread();
    AtomicReference<SkyKey> badKey = new AtomicReference<>();
    DirtyAndInflightTrackingProgressReceiver receiver =
        new DirtyAndInflightTrackingProgressReceiver(
            new InvalidationProgressReceiver() {
              @Override
              protected void invalidated(SkyKey skyKey, InvalidationState state) {
                if (skyKey.equals(child)) {
                  // Interrupt on the very first invalidate
                  mainThread.interrupt();
                } else if (!skyKey.functionName().equals(NODE_TYPE)) {
                  // All other invalidations should have the GraphTester's key type.
                  // Exceptions thrown here may be silently dropped, so keep track of errors
                  // ourselves.
                  badKey.set(skyKey);
                }
                try {
                  assertThat(
                          visitor
                              .get()
                              .getInterruptionLatchForTestingOnly()
                              .await(2, TimeUnit.HOURS))
                      .isTrue();
                } catch (InterruptedException e) {
                  // We may well have thrown here because by the time we try to await, the main
                  // thread is already interrupted.
                  Thread.currentThread().interrupt();
                }
              }
            });
    assertThrows(InterruptedException.class, () -> invalidateWithoutError(receiver, child));
    assertThat(badKey.get()).isNull();
    assertThat(state.isEmpty()).isFalse();
    Set<SkyKey> invalidated = Sets.newConcurrentHashSet();
    assertThat(isInvalidated(parent)).isFalse();
    assertThat(graph.get(null, Reason.OTHER, parent).getValue()).isNotNull();
    DirtyAndInflightTrackingProgressReceiver receiver2 =
        new DirtyAndInflightTrackingProgressReceiver(
            new InvalidationProgressReceiver() {
              @Override
              protected void invalidated(SkyKey skyKey, InvalidationState state) {
                invalidated.add(skyKey);
              }
            });
    invalidateWithoutError(receiver2);
    assertThat(invalidated).contains(parent);
    assertThat(state.getInvalidationsForTesting()).isEmpty();

    // Regression test coverage:
    // "all pending values are marked changed on interrupt".
    assertThat(isInvalidated(child)).isTrue();
    assertChanged(child);
    for (int i = 1; i < numValues; i++) {
      assertDirtyAndNotChanged(family[i]);
    }
    assertDirtyAndNotChanged(parent);
  }

  @Test
  public void deepGraph() throws Exception {
    graph = new InMemoryGraphImpl();
    int depth = 1 << 15;
    SkyKey leafKey = GraphTester.nonHermeticKey(Integer.toString(depth));
    for (int i = 0; i < depth; i++) {
      tester
          .getOrCreate(Integer.toString(i))
          .addDependency(i + 1 == depth ? leafKey : skyKey(Integer.toString(i + 1)))
          .setComputedValue(CONCATENATE);
    }
    tester.set(leafKey, new StringValue("leaf"));
    eval(/* keepGoing= */ false, skyKey(Integer.toString(0)));
    invalidateWithoutError(
        new DirtyAndInflightTrackingProgressReceiver(EvaluationProgressReceiver.NULL), leafKey);
  }

  private SkyKey[] constructLargeGraph(int size) {
    Random random = new Random(TestUtils.getRandomSeed());
    SkyKey[] values = new SkyKey[size];
    for (int i = 0; i < size; i++) {
      String iString = Integer.toString(i);
      SkyKey iKey = GraphTester.nonHermeticKey(iString);
      tester.set(iKey, new StringValue(iString));
      set(iString, iString);
      for (int j = 0; j < i; j++) {
        if (random.nextInt(3) == 0) {
          tester.getOrCreate(iKey).addDependency(GraphTester.nonHermeticKey(Integer.toString(j)));
        }
      }
      values[i] = iKey;
    }
    return values;
  }

  /** Returns a subset of {@code nodes} that are still valid and so can be invalidated. */
  private Set<Pair<SkyKey, InvalidationType>> getValuesToInvalidate(SkyKey[] nodes) {
    Set<Pair<SkyKey, InvalidationType>> result = new HashSet<>();
    Random random = new Random(TestUtils.getRandomSeed());
    for (SkyKey node : nodes) {
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
  public void allNodesProcessed() throws Exception {
    graph = new InMemoryGraphImpl();
    ImmutableList.Builder<SkyKey> keysToDelete =
        ImmutableList.builderWithExpectedSize(InvalidatingNodeVisitor.DEFAULT_THREAD_COUNT - 1);
    for (int i = 0; i < InvalidatingNodeVisitor.DEFAULT_THREAD_COUNT - 1; i++) {
      keysToDelete.add(GraphTester.nonHermeticKey("key" + i));
    }
    invalidate(graph, progressReceiver, keysToDelete.build().toArray(new SkyKey[0]));
    assertThat(state.isEmpty()).isTrue();
  }

  @Test
  public void deletingInsideForkJoinPoolWorks() throws Exception {
    graph = new InMemoryGraphImpl();
    ForkJoinPool outerPool = new ForkJoinPool(1);
    outerPool
        .submit(
            () -> {
              try {
                invalidate(graph, progressReceiver, GraphTester.nonHermeticKey("a"));
              } catch (InterruptedException e) {
                throw new IllegalStateException(e);
              }
            })
        .get();
  }

  @Test
  public void interruptRecoversNextTime() throws InterruptedException {
    graph = new InMemoryGraphImpl();
    SkyKey dep = GraphTester.nonHermeticKey("dep");
    SkyKey toDelete = GraphTester.nonHermeticKey("top");
    tester.getOrCreate(toDelete).addDependency(dep).setConstantValue(new StringValue("top"));
    tester.set(dep, new StringValue("dep"));
    eval(/*keepGoing=*/ false, toDelete);
    Thread mainThread = Thread.currentThread();
    assertThrows(
        InterruptedException.class,
        () ->
            invalidateWithoutError(
                new DirtyAndInflightTrackingProgressReceiver(
                    new InvalidationProgressReceiver() {
                      @Override
                      protected void invalidated(SkyKey skyKey, InvalidationState state) {
                        mainThread.interrupt();
                        // Wait for the main thread to be interrupted uninterruptibly, because the
                        // main thread is going to interrupt us, and we don't want to get into an
                        // interrupt fight. Only if we get interrupted without the main thread also
                        // being interrupted will this throw an InterruptedException.
                        TrackingAwaiter.INSTANCE.awaitLatchAndTrackExceptions(
                            visitor.get().getInterruptionLatchForTestingOnly(),
                            "Main thread was not interrupted");
                      }
                    }),
                toDelete));
    invalidateWithoutError(
        new DirtyAndInflightTrackingProgressReceiver(EvaluationProgressReceiver.NULL));
    eval(/* keepGoing= */ false, toDelete);
    invalidateWithoutError(
        new DirtyAndInflightTrackingProgressReceiver(EvaluationProgressReceiver.NULL), toDelete);
    eval(/* keepGoing= */ false, toDelete);
  }

  @Test
  public void interruptThreadInReceiver() throws Exception {
    Random random = new Random(TestUtils.getRandomSeed());
    int graphSize = 1000;
    int tries = 5;
    graph = new InMemoryGraphImpl();
    SkyKey[] values = constructLargeGraph(graphSize);
    eval(/*keepGoing=*/false, values);
    Thread mainThread = Thread.currentThread();
    for (int run = 0; run < tries + 1; run++) {
      Set<Pair<SkyKey, InvalidationType>> valuesToInvalidate = getValuesToInvalidate(values);
      // Find how many invalidations will actually be enqueued for invalidation in the first round,
      // so that we can interrupt before all of them are done.
      int validValuesToDo =
          Sets.difference(valuesToInvalidate, state.getInvalidationsForTesting()).size();
      for (Pair<SkyKey, InvalidationType> pair : state.getInvalidationsForTesting()) {
        if (!isInvalidated(pair.first)) {
          validValuesToDo++;
        }
      }
      int countDownStart = validValuesToDo > 0 ? random.nextInt(validValuesToDo) : 0;
      CountDownLatch countDownToInterrupt = new CountDownLatch(countDownStart);
      // Make sure final invalidation finishes.
      DirtyAndInflightTrackingProgressReceiver receiver =
          run == tries
              ? new DirtyAndInflightTrackingProgressReceiver(EvaluationProgressReceiver.NULL)
              : new DirtyAndInflightTrackingProgressReceiver(
                  new InvalidationProgressReceiver() {
                    @Override
                    protected void invalidated(SkyKey skyKey, InvalidationState state) {
                      countDownToInterrupt.countDown();
                      if (countDownToInterrupt.getCount() == 0) {
                        mainThread.interrupt();
                        // Wait for the main thread to be interrupted uninterruptibly, because the
                        // main thread is going to interrupt us, and we don't want to get into an
                        // interrupt fight. Only if we get interrupted without the main thread also
                        // being interrupted will this throw an InterruptedException.
                        TrackingAwaiter.INSTANCE.awaitLatchAndTrackExceptions(
                            visitor.get().getInterruptionLatchForTestingOnly(),
                            "Main thread was not interrupted");
                      }
                    }
                  });
      try {
        invalidate(
            graph,
            receiver,
            Sets.newHashSet(Iterables.transform(valuesToInvalidate, pair -> pair.first))
                .toArray(new SkyKey[0]));
        assertThat(state.getInvalidationsForTesting()).isEmpty();
      } catch (InterruptedException e) {
        assertThat(run).isLessThan(tries);
      }
      if (state.isEmpty()) {
        // Ran out of values to invalidate.
        break;
      }
    }

    eval(/*keepGoing=*/false, values);
  }

  void setupInvalidatableGraph() throws Exception {
    graph = new InMemoryGraphImpl();
    SkyKey aKey = GraphTester.nonHermeticKey("a");
    tester.set(aKey, new StringValue("a"));
    set("b", "b");
    tester.getOrCreate("ab").addDependency(aKey).addDependency("b").setComputedValue(CONCATENATE);
    assertValueValue("ab", "ab");
    tester.set(aKey, new StringValue("c"));
  }

  private static class HeavyValue implements SkyValue {}

  /** Test suite for the deleting invalidator. */
  @RunWith(JUnit4.class)
  public static class DeletingInvalidatorTest extends EagerInvalidatorTest {
    @Override
    protected void invalidate(
        InMemoryGraph graph,
        DirtyAndInflightTrackingProgressReceiver progressReceiver,
        SkyKey... keys)
        throws InterruptedException {
      Iterable<SkyKey> diff = ImmutableList.copyOf(keys);
      DeletingNodeVisitor deletingNodeVisitor =
          EagerInvalidator.createDeletingVisitorIfNeeded(
              graph,
              diff,
              new DirtyAndInflightTrackingProgressReceiver(progressReceiver),
              (InvalidatingNodeVisitor.DeletingInvalidationState) state,
              true);
      if (deletingNodeVisitor != null) {
        visitor.set(deletingNodeVisitor);
        deletingNodeVisitor.run();
      }
    }

    @Override
    InvalidationProgressReceiver.InvalidationState expectedState() {
      return InvalidationProgressReceiver.InvalidationState.DELETED;
    }

    @Override
    boolean gcExpected() {
      return true;
    }

    @Override
    protected InvalidatingNodeVisitor.DeletingInvalidationState newInvalidationState() {
      return new InvalidatingNodeVisitor.DeletingInvalidationState();
    }

    @Override
    protected InvalidationType defaultInvalidationType() {
      return InvalidationType.DELETED;
    }

    @Override
    protected boolean reverseDepsPresent() {
      return false;
    }

    /**
     * Regression test for b/316606228.
     *
     * <p>Evaluation of {@code top1} is interrupted after it was already signaled by {@code dep},
     * but meanwhile {@code dep} was rewound by {@code top2}, and never gets a chance to complete.
     */
    @Test
    public void interruptDuringRewind_parentAlreadySignaled() throws Exception {
      graph = new InMemoryGraphImpl();
      SkyKey top1 = skyKey("top1");
      SkyKey top2 = skyKey("top2");
      SkyKey dep = skyKey("dep");
      CountDownLatch depDoneObservedByTop1 = new CountDownLatch(1);
      CountDownLatch top2Reset = new CountDownLatch(1);
      tester
          .getOrCreate(top1)
          .setBuilder(
              (skyKey, env) -> {
                if (env.getValue(dep) == null) {
                  return null;
                }
                depDoneObservedByTop1.countDown();
                top2Reset.await();
                throw new InterruptedException();
              });
      tester
          .getOrCreate(top2)
          .setBuilder(
              new SkyFunction() {
                private boolean alreadyReset = false;

                @Nullable
                @Override
                public SkyValue compute(SkyKey skyKey, Environment env)
                    throws InterruptedException {
                  if (alreadyReset) {
                    top2Reset.countDown();
                    throw new InterruptedException();
                  }
                  if (env.getValue(dep) == null) {
                    return null;
                  }
                  depDoneObservedByTop1.await();
                  alreadyReset = true;
                  var rewindGraph = Reset.newRewindGraphFor(top2);
                  rewindGraph.putEdge(top2, dep);
                  return Reset.of(rewindGraph);
                }
              });
      tester.getOrCreate(dep).setConstantValue(new StringValue("depVal"));

      assertThrows(
          InterruptedException.class,
          () ->
              eval(
                  /* keepGoing= */ false,
                  // Tolerate inconsistencies.
                  (key, otherKeys, inconsistency) -> {},
                  top1,
                  top2));

      // Split invalidation into two calls to guarantee that top1 is visited before dep. In
      // practice, deletion is done in a single parallel traversal, but the crash from b/316606228
      // only reproduces when visiting in this order.
      invalidateWithoutError(
          new DirtyAndInflightTrackingProgressReceiver(EvaluationProgressReceiver.NULL), top1);
      invalidateWithoutError(
          new DirtyAndInflightTrackingProgressReceiver(EvaluationProgressReceiver.NULL), dep);
    }

    @Test
    public void dirtyTrackingProgressReceiverWorksWithDeletingInvalidator() throws Exception {
      setupInvalidatableGraph();
      DirtyAndInflightTrackingProgressReceiver receiver =
          new DirtyAndInflightTrackingProgressReceiver(EvaluationProgressReceiver.NULL);

      // Dirty the node, and ensure that the tracker is aware of it:
      ImmutableList<SkyKey> diff = ImmutableList.of(GraphTester.nonHermeticKey("a"));
      InvalidatingNodeVisitor.InvalidationState state1 = new DirtyingInvalidationState();
      Preconditions.checkNotNull(
              EagerInvalidator.createInvalidatingVisitorIfNeeded(graph, diff, receiver, state1))
          .run();
      assertThat(receiver.getUnenqueuedDirtyKeys()).containsExactly(diff.get(0), skyKey("ab"));

      // Delete the node, and ensure that the tracker is no longer tracking it:
      Preconditions.checkNotNull(
              EagerInvalidator.createDeletingVisitorIfNeeded(
                  graph,
                  diff,
                  receiver,
                  (InvalidatingNodeVisitor.DeletingInvalidationState) state,
                  true))
          .run();
      assertThat(receiver.getUnenqueuedDirtyKeys()).isEmpty();
    }
  }

  /**
   * Test suite for the dirtying invalidator.
   */
  @RunWith(JUnit4.class)
  public static class DirtyingInvalidatorTest extends EagerInvalidatorTest {
    @Override
    protected void invalidate(
        InMemoryGraph graph,
        DirtyAndInflightTrackingProgressReceiver progressReceiver,
        SkyKey... keys)
        throws InterruptedException {
      Iterable<SkyKey> diff = ImmutableList.copyOf(keys);
      DirtyingNodeVisitor dirtyingNodeVisitor =
          EagerInvalidator.createInvalidatingVisitorIfNeeded(graph, diff, progressReceiver, state);
      if (dirtyingNodeVisitor != null) {
        visitor.set(dirtyingNodeVisitor);
        dirtyingNodeVisitor.run();
      }
    }

    @Override
    InvalidationProgressReceiver.InvalidationState expectedState() {
      return InvalidationProgressReceiver.InvalidationState.DIRTY;
    }

    @Override
    boolean gcExpected() {
      return false;
    }

    @Override
    protected InvalidatingNodeVisitor.InvalidationState newInvalidationState() {
      return new DirtyingInvalidationState();
    }

    @Override
    protected InvalidationType defaultInvalidationType() {
      return InvalidationType.CHANGED;
    }

    @Override
    protected boolean reverseDepsPresent() {
      return true;
    }

    @Test
    public void dirtyTrackingProgressReceiverWorksWithDirtyingInvalidator() throws Exception {
      setupInvalidatableGraph();
      DirtyAndInflightTrackingProgressReceiver receiver =
          new DirtyAndInflightTrackingProgressReceiver(EvaluationProgressReceiver.NULL);

      // Dirty the node, and ensure that the tracker is aware of it:
      invalidate(graph, receiver, GraphTester.nonHermeticKey("a"));
      assertThat(receiver.getUnenqueuedDirtyKeys()).hasSize(2);
    }
  }
}
