// Copyright 2023 The Bazel Authors. All rights reserved.
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

import static com.google.common.base.MoreObjects.toStringHelper;
import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.skyframe.DirtyBuildingState.DEPTH_SATURATION_BOUND;
import static com.google.devtools.build.skyframe.DirtyBuildingState.EVALUATION_COUNT_SATURATION_BOUND;
import static com.google.devtools.build.skyframe.NotifyingHelper.makeNotifyingTransformer;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Interner;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.concurrent.TieredPriorityExecutor;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.skyframe.EvaluationContext.UnnecessaryTemporaryStateDropperReceiver;
import com.google.devtools.build.skyframe.GraphTester.StringValue;
import com.google.devtools.build.skyframe.NotifyingHelper.Listener;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.util.ArrayList;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicReference;
import javax.annotation.Nullable;
import org.junit.Test;
import org.junit.runner.RunWith;

@RunWith(TestParameterInjector.class)
public final class PriorityTest {
  private static final long POLL_MS = 100;

  /** Low fanout occupies the highest bit that can be set without creating a negative number. */
  private static final int LOW_FANOUT_PRIORITY = 0x4000_0000;

  private final ProcessableGraph graph = new InMemoryGraphImpl();
  private final GraphTester tester = new GraphTester();

  private final StoredEventHandler reportedEvents = new StoredEventHandler();
  private final DirtyTrackingProgressReceiver revalidationReceiver =
      new DirtyTrackingProgressReceiver(null);

  private static final Version VERSION = IntVersion.of(0);

  // TODO(shahan): consider factoring this boilerplate out to a common location.
  private <T extends SkyValue> EvaluationResult<T> eval(SkyKey root, Listener listener)
      throws InterruptedException {
    return new ParallelEvaluator(
            makeNotifyingTransformer(listener).transform(graph),
            VERSION,
            Version.minimal(),
            tester.getSkyFunctionMap(),
            reportedEvents,
            new EmittedEventState(),
            EventFilter.FULL_STORAGE,
            ErrorInfoManager.UseChildErrorInfoIfNecessary.INSTANCE,
            /* keepGoing= */ false,
            revalidationReceiver,
            GraphInconsistencyReceiver.THROWING,
            new TieredPriorityExecutor(
                "test-pool", 6, 3, ParallelEvaluatorErrorClassifier.instance()),
            new SimpleCycleDetector(),
            /* mergingSkyframeAnalysisExecutionPhases= */ false,
            UnnecessaryTemporaryStateDropperReceiver.NULL)
        .eval(ImmutableList.of(root));
  }

  // Some predefined values for use in the tests.
  private static final SkyValue VALUE_A1 = new StringValue("A1");
  private static final SkyValue VALUE_A2 = new StringValue("A2");
  private static final SkyValue VALUE_B1 = new StringValue("B1");
  private static final SkyValue DONE_VALUE = new StringValue("DONE");

  @AutoValue
  abstract static class PriorityInfo {
    private static PriorityInfo fromNode(NodeEntry node) {
      return new AutoValue_PriorityTest_PriorityInfo(node.getPriority(), node.depth());
    }

    private static PriorityInfo of(boolean hasLowFanout, int depth, int evaluationCount) {
      int priority = depth + evaluationCount * evaluationCount;
      if (hasLowFanout) {
        priority += LOW_FANOUT_PRIORITY;
      }
      return new AutoValue_PriorityTest_PriorityInfo(priority, depth);
    }

    abstract int priority();

    abstract int depth();
  }

  @Test
  public void evaluate_incrementsChildDepthAndParentEvaluationCount() throws InterruptedException {
    CPUHeavySkyKey rootKey = Key.create("root");
    CPUHeavySkyKey depKey = Key.create("a");

    tester.getOrCreate(depKey).setConstantValue(VALUE_A1);

    tester
        .getOrCreate(rootKey)
        .setBuilder(
            (unusedKey, env) -> {
              var value = env.getValue(depKey);
              if (value == null) {
                return null;
              }
              assertThat(value).isEqualTo(VALUE_A1);
              return DONE_VALUE;
            });

    var listener = new EvaluationPriorityListener();
    assertThat(eval(rootKey, listener).get(rootKey)).isEqualTo(DONE_VALUE);

    assertThat(listener.priorities())
        .containsExactly(
            rootKey,
                ImmutableList.of(
                    PriorityInfo.of(
                        /* hasLowFanout= */ false, /* depth= */ 0, /* evaluationCount= */ 0),
                    PriorityInfo.of(
                        /* hasLowFanout= */ false, /* depth= */ 0, /* evaluationCount= */ 1)),
            depKey,
                ImmutableList.of(
                    PriorityInfo.of(
                        /* hasLowFanout= */ false, /* depth= */ 1, /* evaluationCount= */ 0)));
  }

  private static class EvaluationPriorityListener implements Listener {
    private final ConcurrentHashMap<SkyKey, ArrayList<PriorityInfo>> priorities =
        new ConcurrentHashMap<>();

    @Override
    public void accept(
        SkyKey key,
        NotifyingHelper.EventType type,
        NotifyingHelper.Order order,
        @Nullable Object context) {
      if (type != NotifyingHelper.EventType.EVALUATE) {
        return;
      }
      priorities
          .computeIfAbsent(key, unusedKey -> new ArrayList<>())
          .add(PriorityInfo.fromNode((NodeEntry) context));
    }

    ConcurrentHashMap<SkyKey, ArrayList<PriorityInfo>> priorities() {
      return priorities;
    }
  }

  @Test
  public void enqueuingChildFromDeeperParent_incrementsDepth() throws InterruptedException {
    CPUHeavySkyKey rootKey = Key.create("root");
    // Both a1 and a2 are children of root. a2 requests a1 again to increase its depth.
    CPUHeavySkyKey a1 = Key.create("a1");
    CPUHeavySkyKey a2 = Key.create("a2");

    // b1 is a child of a1 that causes a1 to restart. b1's SkyFunction holds until a2 has had a
    // chance to re-request a1, so a1's increased depth at re-evaluation can be observed.
    CPUHeavySkyKey b1 = Key.createWithLowFanout("b1");

    var nodeA1 = new AtomicReference<NodeEntry>();

    var listener =
        new EvaluationPriorityListener() {
          @Override
          public void accept(
              SkyKey key,
              NotifyingHelper.EventType type,
              NotifyingHelper.Order order,
              @Nullable Object context) {
            super.accept(key, type, order, context);
            // Captures a1's node in addition to priority information.
            if (key == a1 && type == NotifyingHelper.EventType.EVALUATE) {
              nodeA1.compareAndSet(/* expectedValue= */ null, (NodeEntry) context);
            }
          }
        };

    tester
        .getOrCreate(b1)
        .setBuilder(
            (key, env) -> {
              // To ensure that a1 only re-enqueues after a2 has requested it, waits until a1's
              // depth becomes 2.
              while (nodeA1.get().depth() < 2) {
                try {
                  Thread.sleep(POLL_MS);
                } catch (InterruptedException e) {
                  throw new AssertionError("Unexpected interruption");
                }
              }
              return VALUE_B1;
            });

    tester
        .getOrCreate(a1)
        .setBuilder(
            (key, env) -> {
              var value = env.getValue(b1);
              if (value == null) {
                return null;
              }
              assertThat(value).isEqualTo(VALUE_B1);
              return VALUE_A1;
            });

    tester
        .getOrCreate(a2)
        .setBuilder(
            (key, env) -> {
              var value = env.getValue(a1);
              if (value == null) {
                return null;
              }
              assertThat(value).isEqualTo(VALUE_A1);
              return VALUE_A2;
            });

    tester
        .getOrCreate(rootKey)
        .setBuilder(
            (key, env) -> {
              var value1 = env.getValue(a1);
              var value2 = env.getValue(a2);
              if (value1 == null || value2 == null) {
                return null;
              }
              assertThat(value1).isEqualTo(VALUE_A1);
              assertThat(value2).isEqualTo(VALUE_A2);
              return DONE_VALUE;
            });

    assertThat(eval(rootKey, listener).get(rootKey)).isEqualTo(DONE_VALUE);

    var priorities = listener.priorities();
    assertThat(priorities.keySet()).containsExactly(rootKey, a1, a2, b1);

    assertThat(priorities.get(rootKey))
        .containsExactly(
            PriorityInfo.of(/* hasLowFanout= */ false, /* depth= */ 0, /* evaluationCount= */ 0),
            PriorityInfo.of(/* hasLowFanout= */ false, /* depth= */ 0, /* evaluationCount= */ 1))
        .inOrder();
    assertThat(priorities.get(a1)).hasSize(2);
    assertThat(priorities.get(a1).get(0))
        .isAnyOf(
            // The common case where a1 starts evaluating before a2 requests it.
            PriorityInfo.of(/* hasLowFanout= */ false, /* depth= */ 1, /* evaluationCount= */ 0),
            // Sometimes a2 requests a1 before a1 starts evaluating.
            PriorityInfo.of(/* hasLowFanout= */ false, /* depth= */ 2, /* evaluationCount= */ 0));
    assertThat(priorities.get(a1).get(1))
        .isEqualTo(
            PriorityInfo.of(/* hasLowFanout= */ false, /* depth= */ 2, /* evaluationCount= */ 1));
    assertThat(priorities.get(a2))
        .containsExactly(
            PriorityInfo.of(/* hasLowFanout= */ false, /* depth= */ 1, /* evaluationCount= */ 0),
            PriorityInfo.of(/* hasLowFanout= */ false, /* depth= */ 1, /* evaluationCount= */ 1))
        .inOrder();
    assertThat(priorities.get(b1))
        .containsAnyOf(
            // If a1 requests b1 first.
            PriorityInfo.of(/* hasLowFanout= */ true, /* depth= */ 2, /* evaluationCount= */ 0),
            // If a2 requests a1 first.
            PriorityInfo.of(/* hasLowFanout= */ true, /* depth= */ 3, /* evaluationCount= */ 0));
  }

  @Test
  public void basicPriorityCalculation(@TestParameter boolean hasLowFanout)
      throws InterruptedException {
    var state = new InitialBuildingState(hasLowFanout);
    state.updateDepthIfGreater(5);
    state.incrementEvaluationCount();
    state.incrementEvaluationCount();

    assertThat(state.depth()).isEqualTo(5);
    assertThat(state.getPriority())
        .isEqualTo(
            PriorityInfo.of(hasLowFanout, /* depth= */ 5, /* evaluationCount= */ 2).priority());
  }

  private static final int MAX_DEPTH = 0x7FFF;

  private enum DepthTestCases {
    NEGATIVE_NUMBER(-1, MAX_DEPTH, DEPTH_SATURATION_BOUND),
    OVERFLOWING_VALUE(0xFFFF, MAX_DEPTH, DEPTH_SATURATION_BOUND),
    NON_INTERSECTING_VALUE(0x1_0000, 0, 0);

    private final int proposedDepth;
    private final int resultingDepth;
    private final int priority;

    private DepthTestCases(int proposedDepth, int resultingDepth, int priority) {
      this.proposedDepth = proposedDepth;
      this.resultingDepth = resultingDepth;
      this.priority = priority;
    }

    private int proposedDepth() {
      return proposedDepth;
    }

    private int resultingDepth() {
      return resultingDepth;
    }

    private int priority() {
      return priority;
    }
  }

  @Test
  public void updateDepth(@TestParameter DepthTestCases testCase) {
    var state = new InitialBuildingState(/* hasLowFanout= */ false);

    state.updateDepthIfGreater(testCase.proposedDepth());
    assertThat(state.depth()).isEqualTo(testCase.resultingDepth());
    assertThat(state.getPriority()).isEqualTo(testCase.priority());
  }

  private static final int MAX_EVALUATION_COUNT = 0xFFFF;

  private enum EvaluationCountTestCases {
    SIMPLE(4, 16),
    SATURATING(64, EVALUATION_COUNT_SATURATION_BOUND * EVALUATION_COUNT_SATURATION_BOUND),
    MAXIMUM(
        MAX_EVALUATION_COUNT,
        EVALUATION_COUNT_SATURATION_BOUND * EVALUATION_COUNT_SATURATION_BOUND),
    // If there are somehow over MAX_EVALUATION_COUNT evaluations, it overflows into depth. The
    // only known scenario where this could happen is with partial re-evaluation. There it would
    // make more sense to not mark the corresponding key CPUHeavy to avoid going through the
    // priority queue thousands of times. In that case, its children may receive a small increase
    // in depth per MAX_EVALUATION_COUNT iterations as an unintended side effect.
    OVERFLOW(MAX_EVALUATION_COUNT + 1, 1);

    private final int evaluationCount;
    private final int priority;

    private EvaluationCountTestCases(int evaluationCount, int priority) {
      this.evaluationCount = evaluationCount;
      this.priority = priority;
    }

    private int evaluationCount() {
      return evaluationCount;
    }

    private int priority() {
      return priority;
    }
  }

  @Test
  public void updateEvaluationCount(@TestParameter EvaluationCountTestCases testCase) {
    var state = new InitialBuildingState(/* hasLowFanout= */ false);

    for (int i = 0; i < testCase.evaluationCount(); ++i) {
      state.incrementEvaluationCount();
    }
    assertThat(state.getPriority()).isEqualTo(testCase.priority());
  }

  private static class Key implements CPUHeavySkyKey {
    private static final Interner<Key> interner = BlazeInterners.newWeakInterner();

    private final String arg;
    private final boolean hasLowFanout;

    static Key create(String arg) {
      return interner.intern(new Key(arg, /* hasLowFanout= */ false));
    }

    static Key createWithLowFanout(String arg) {
      return interner.intern(new Key(arg, /* hasLowFanout= */ true));
    }

    private Key(String arg, boolean hasLowFanout) {
      this.arg = arg;
      this.hasLowFanout = hasLowFanout;
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctionName.FOR_TESTING;
    }

    @Override
    public String argument() {
      return arg;
    }

    @Override
    public boolean hasLowFanout() {
      return hasLowFanout;
    }

    @Override
    public int hashCode() {
      return 31 * functionName().hashCode() + arg.hashCode();
    }

    @Override
    public boolean equals(Object obj) {
      if (!(obj instanceof Key)) {
        return false;
      }
      Key that = (Key) obj;
      return arg.equals(that.arg) && hasLowFanout == that.hasLowFanout;
    }

    @Override
    public String toString() {
      return toStringHelper(this).add("arg", arg).add("hasLowFanout", hasLowFanout).toString();
    }
  }
}
