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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.collect.nestedset.NestedSetVisitor;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadHostile;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.events.ExtendedEventHandler.Postable;
import java.io.PrintStream;
import java.util.Map;
import java.util.function.Predicate;
import javax.annotation.Nullable;

/**
 * A graph, defined by a set of functions that can construct values from value keys.
 *
 * <p>The value constructor functions ({@link SkyFunction}s) can declare dependencies on
 * prerequisite {@link SkyValue}s. The {@link MemoizingEvaluator} implementation makes sure that
 * those are created beforehand.
 *
 * <p>The graph caches previously computed value values. Arbitrary values can be invalidated between
 * calls to {@link #evaluate}; they will be recreated the next time they are requested.
 */
public interface MemoizingEvaluator {

  /**
   * Computes the transitive closure of a given set of values at the given {@link Version}. See
   * {@link EagerInvalidator#invalidate}.
   *
   * <p>The returned EvaluationResult is guaranteed to contain a result for at least one root if
   * keepGoing is false. It will contain a result for every root if keepGoing is true, <i>unless</i>
   * the evaluation failed with a "catastrophic" error. In that case, some or all results may be
   * missing.
   */
  <T extends SkyValue> EvaluationResult<T> evaluate(
      Iterable<? extends SkyKey> roots, Version version, EvaluationContext evaluationContext)
      throws InterruptedException;

  /**
   * Ensures that after the next completed {@link #evaluate} call the current values of any value
   * matching this predicate (and all values that transitively depend on them) will be removed from
   * the value cache. All values that were already marked dirty in the graph will also be deleted,
   * regardless of whether or not they match the predicate.
   *
   * <p>If a later call to {@link #evaluate} requests some of the deleted values, those values will
   * be recomputed and the new values stored in the cache again.
   *
   * <p>To delete all dirty values, you can specify a predicate that's always false.
   */
  void delete(Predicate<SkyKey> pred);

  /**
   * Marks dirty values for deletion if they have been dirty for at least as many graph versions
   * as the specified limit.
   *
   * <p>This ensures that after the next completed {@link #evaluate} call, all such values, along
   * with all values that transitively depend on them, will be removed from the value cache. Values
   * that were marked dirty after the threshold version will not be affected by this call.
   *
   * <p>If a later call to {@link #evaluate} requests some of the deleted values, those values will
   * be recomputed and the new values stored in the cache again.
   *
   * <p>To delete all dirty values, you can specify 0 for the limit.
   */
  void deleteDirty(long versionAgeLimit);

  /**
   * Returns the values in the graph.
   *
   * <p>The returned map may be a live view of the graph.
   */
  // TODO(bazel-team): Replace all usages of getValues, getDoneValues, getExistingValue,
  // and getExistingErrorForTesting with usages of WalkableGraph. Changing the getValues usages
  // require some care because getValues gives access to the previous value for changed/dirty nodes.
  Map<SkyKey, SkyValue> getValues();

  /**
   * Returns the node entries in the graph. Should only be called between evaluations. The returned
   * iterable is mutable, but do not mutate it unless you know what you are doing! Naively deleting
   * an entry will break graph invariants and cause a crash.
   */
  Iterable<? extends Map.Entry<SkyKey, ? extends NodeEntry>> getGraphEntries();

  /**
   * Informs the evaluator that a sequence of evaluations at the same version has finished.
   * Evaluators may make optimizations under the assumption that successive evaluations are all at
   * the same version. A call of this method tells the evaluator that the next evaluation is not
   * guaranteed to be at the same version.
   */
  default void noteEvaluationsAtSameVersionMayBeFinished(ExtendedEventHandler eventHandler)
      throws InterruptedException {
    postLoggingStats(eventHandler);
  }

  /**
   * Tells the evaluator to post any logging statistics that it may have accumulated over the last
   * sequence of evaluations. Normally called internally by {@link
   * #noteEvaluationsAtSameVersionMayBeFinished}.
   */
  default void postLoggingStats(ExtendedEventHandler eventHandler) {}

  /**
   * Returns the done (without error) values in the graph.
   *
   * <p>The returned map may be a live view of the graph.
   */
  Map<SkyKey, SkyValue> getDoneValues();

  /**
   * Returns a value if and only if an earlier call to {@link #evaluate} created it; null otherwise.
   *
   * <p>This method should mainly be used by tests that need to verify the presence of a value in
   * the graph after an {@link #evaluate} call.
   */
  @Nullable
  SkyValue getExistingValue(SkyKey key) throws InterruptedException;

  @Nullable
  NodeEntry getExistingEntryAtCurrentlyEvaluatingVersion(SkyKey key) throws InterruptedException;

  /**
   * Returns an error if and only if an earlier call to {@link #evaluate} created it; null
   * otherwise.
   *
   * <p>This method should only be used by tests that need to verify the presence of an error in the
   * graph after an {@link #evaluate} call.
   */
  @SuppressWarnings("VisibleForTestingMisused") // Only exists for testing.
  @VisibleForTesting
  @Nullable
  ErrorInfo getExistingErrorForTesting(SkyKey key) throws InterruptedException;

  /**
   * Tests that want finer control over the graph being used may provide a {@code transformer} here.
   * This {@code transformer} will be applied to the graph for each invalidation/evaluation.
   */
  void injectGraphTransformerForTesting(GraphTransformerForTesting transformer);

  /** Transforms a graph, possibly injecting other functionality. */
  interface GraphTransformerForTesting {
    InMemoryGraph transform(InMemoryGraph graph);

    QueryableGraph transform(QueryableGraph graph);

    ProcessableGraph transform(ProcessableGraph graph);

    GraphTransformerForTesting NO_OP =
        new GraphTransformerForTesting() {
          @Override
          public InMemoryGraph transform(InMemoryGraph graph) {
            return graph;
          }

          @Override
          public QueryableGraph transform(QueryableGraph graph) {
            return graph;
          }

          @Override
          public ProcessableGraph transform(ProcessableGraph graph) {
            return graph;
          }
        };
  }

  /**
   * Write the graph to the output stream. Not necessarily thread-safe. Use only for debugging
   * purposes.
   */
  @ThreadHostile
  void dump(boolean summarize, PrintStream out);

  /** A supplier for creating instances of a particular evaluator implementation. */
  interface EvaluatorSupplier {
    MemoizingEvaluator create(
        ImmutableMap<SkyFunctionName, SkyFunction> skyFunctions,
        Differencer differencer,
        EvaluationProgressReceiver progressReceiver,
        GraphInconsistencyReceiver graphInconsistencyReceiver,
        EventFilter eventFilter,
        EmittedEventState emittedEventState,
        boolean keepEdges);
  }

  /**
   * Keeps track of already-emitted events. Users of the graph should instantiate an
   * {@code EmittedEventState} first and pass it to the graph during creation. This allows them to
   * determine whether or not to replay events.
   */
  class EmittedEventState {
    final NestedSetVisitor.VisitedState<TaggedEvents> eventState =
        new NestedSetVisitor.VisitedState<>();
    final NestedSetVisitor.VisitedState<Postable> postableState =
        new NestedSetVisitor.VisitedState<>();

    public void clear() {
      eventState.clear();
      postableState.clear();
    }
  }
}
