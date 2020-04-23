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
import com.google.common.base.Preconditions;
import com.google.common.graph.GraphBuilder;
import com.google.common.graph.ImmutableGraph;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.util.GroupedList;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * Machinery to evaluate a single value.
 *
 * <p>The SkyFunction {@link #compute} implementation is supposed to access only direct dependencies
 * of the value. However, the direct dependencies need not be known in advance. The implementation
 * can request arbitrary values using {@link Environment#getValue}. If the values are not ready, the
 * call will return {@code null}; in that case the implementation should just return {@code null},
 * in which case the missing dependencies will be computed and the {@link #compute} method will be
 * started again.
 */
public interface SkyFunction {

  /**
   * When a value is requested, this method is called with the name of the value and a
   * dependency-tracking environment.
   *
   * <p>This method should return a non-{@code null} value, or {@code null} if any dependencies were
   * missing ({@link Environment#valuesMissing} was true before returning). In that case the missing
   * dependencies will be computed and the {@code compute} method called again.
   *
   * <p>This method should throw if it fails, or if one of its dependencies fails with an exception
   * and this method cannot recover. If one of its dependencies fails and this method can enrich the
   * exception with additional context, then this method should catch that exception and throw
   * another containing that additional context. If it has no such additional context, then it
   * should allow its dependency's exception to be thrown through it.
   *
   * <p>This method may return {@link Restart} in rare circumstances. See its docs. Do not return
   * values of this type unless you know exactly what you are doing.
   *
   * <p>If version information is discovered for the given {@code skyKey}, {@link
   * Environment#injectVersionForNonHermeticFunction(Version)} may be called on {@code env}.
   *
   * @throws SkyFunctionException on failure
   * @throws InterruptedException if interrupted
   */
  @ThreadSafe
  @Nullable
  SkyValue compute(SkyKey skyKey, Environment env)
      throws SkyFunctionException, InterruptedException;

  /**
   * Extracts a tag (target label) from a SkyKey if it has one. Otherwise return {@code null}.
   *
   * <p>The tag is used for filtering out non-error event messages that do not match --output_filter
   * flag. If a SkyFunction returns {@code null} in this method it means that all the info/warning
   * messages associated with this value will be shown, no matter what --output_filter says.
   */
  @Nullable
  String extractTag(SkyKey skyKey);

  /**
   * Sentinel {@link SkyValue} type for {@link #compute} to return, indicating that something went
   * wrong, and that the evaluation returning this value must be restarted, and the nodes associated
   * with other keys in {@link #rewindGraph()} (whose directed edges should correspond to the nodes'
   * direct dependencies) must also be restarted.
   *
   * <p>An intended cause for returning this is external data loss; e.g., if a dependency's
   * "done-ness" is intended to mean that certain data is available in an external system, but
   * during evaluation of a node that depends on that external data, that data has gone missing, and
   * reevaluation of the dependency is expected to repair the discrepancy.
   *
   * <p>Values of this type will <em>never</em> be returned by {@link Environment}'s getValue
   * methods or from {@link NodeEntry#getValue()}.
   *
   * <p>All {@link ListenableFuture}s used in calls to {@link Environment#dependOnFuture} which were
   * not already complete will be cancelled.
   */
  interface Restart extends SkyValue {
    ImmutableGraph<SkyKey> EMPTY_SKYKEY_GRAPH =
        ImmutableGraph.copyOf(GraphBuilder.directed().allowsSelfLoops(false).build());

    Restart SELF = () -> EMPTY_SKYKEY_GRAPH;

    static Restart selfAnd(ImmutableGraph<SkyKey> rewindGraph) {
      Preconditions.checkArgument(
          rewindGraph.isDirected(), "rewindGraph undirected: %s", rewindGraph);
      Preconditions.checkArgument(
          !rewindGraph.allowsSelfLoops(), "rewindGraph allows self loops: %s", rewindGraph);
      return () -> rewindGraph;
    }

    ImmutableGraph<SkyKey> rewindGraph();
  }

  /**
   * The services provided to the {@link SkyFunction#compute} implementation by the Skyframe
   * evaluation framework.
   */
  interface Environment {
    /**
     * Returns a direct dependency. If the specified value is not in the set of already evaluated
     * direct dependencies, returns {@code null}. Also returns {@code null} if the specified value
     * has already been evaluated and found to be in error.
     *
     * <p>On a subsequent evaluation, if any of this value's dependencies have changed they will be
     * re-evaluated in the same order as originally requested by the {@code SkyFunction} using this
     * {@code getValue} call (see {@link #getValues} for when preserving the order is not
     * important).
     *
     * <p>This method and the ones below may throw {@link InterruptedException}. Such exceptions
     * must not be caught by the {@link SkyFunction#compute} implementation. Instead, they should be
     * propagated up to the caller of {@link SkyFunction#compute}.
     */
    @Nullable
    SkyValue getValue(SkyKey valueName) throws InterruptedException;

    /**
     * Returns a direct dependency. If the specified value is not in the set of already evaluated
     * direct dependencies, returns {@code null}. If the specified value has already been evaluated
     * and found to be in error, throws the exception coming from the error, so long as the
     * exception is of one of the specified types. SkyFunction implementations may use this method
     * to continue evaluation even if one of their dependencies is in error by catching the thrown
     * exception and proceeding. The caller must specify the exception type(s) that might be thrown
     * using the {@code exceptionClass} argument(s). If the dependency's exception is not an
     * instance of {@code exceptionClass}, {@code null} is returned.
     *
     * <p>The exception class given cannot be a supertype or a subtype of {@link RuntimeException},
     * or a subtype of {@link InterruptedException}. See {@link
     * SkyFunctionException#validateExceptionType} for details.
     */
    @Nullable
    <E extends Exception> SkyValue getValueOrThrow(SkyKey depKey, Class<E> exceptionClass)
        throws E, InterruptedException;

    @Nullable
    <E1 extends Exception, E2 extends Exception> SkyValue getValueOrThrow(
        SkyKey depKey, Class<E1> exceptionClass1, Class<E2> exceptionClass2)
        throws E1, E2, InterruptedException;

    @Nullable
    <E1 extends Exception, E2 extends Exception, E3 extends Exception> SkyValue getValueOrThrow(
        SkyKey depKey,
        Class<E1> exceptionClass1,
        Class<E2> exceptionClass2,
        Class<E3> exceptionClass3)
        throws E1, E2, E3, InterruptedException;

    @Nullable
    <E1 extends Exception, E2 extends Exception, E3 extends Exception, E4 extends Exception>
        SkyValue getValueOrThrow(
            SkyKey depKey,
            Class<E1> exceptionClass1,
            Class<E2> exceptionClass2,
            Class<E3> exceptionClass3,
            Class<E4> exceptionClass4)
            throws E1, E2, E3, E4, InterruptedException;

    @Nullable
    <
            E1 extends Exception,
            E2 extends Exception,
            E3 extends Exception,
            E4 extends Exception,
            E5 extends Exception>
        SkyValue getValueOrThrow(
            SkyKey depKey,
            Class<E1> exceptionClass1,
            Class<E2> exceptionClass2,
            Class<E3> exceptionClass3,
            Class<E4> exceptionClass4,
            Class<E5> exceptionClass5)
            throws E1, E2, E3, E4, E5, InterruptedException;

    /**
     * Requests {@code depKeys} "in parallel", independent of each others' values. These keys may be
     * thought of as a "dependency group" -- they are requested together by this value.
     *
     * <p>In general, if the result of one getValue call can affect the argument of a later getValue
     * call, the two calls cannot be merged into a single getValues call, since the result of the
     * first call might change on a later evaluation. Inversely, if the result of one getValue call
     * cannot affect the parameters of the next getValue call, the two keys can form a dependency
     * group and the two getValue calls should be merged into one getValues call. In the latter
     * case, if we fail to combine the _multiple_ getValue (or getValues) calls into one _single_
     * getValues call, it would result in multiple dependency groups with an implicit ordering
     * between them. This would unnecessarily cause sequential evaluations of these groups and could
     * impact overall performance.
     *
     * <p>On subsequent evaluations, when checking to see if dependencies require re-evaluation, all
     * the values within one group may be simultaneously checked. A SkyFunction should request a
     * dependency group if checking the deps serially on a subsequent evaluation would take too
     * long, and if the {@link #compute} method would request all deps anyway as long as no earlier
     * deps had changed. SkyFunction.Environment implementations may also choose to request these
     * deps in parallel on the first evaluation, potentially speeding it up.
     *
     * <p>While re-evaluating every value in the group may take longer than re-evaluating just the
     * first one and finding that it has changed, no extra work is done: the contract of the
     * dependency group means that the {@link #compute} method, when called to re-evaluate this
     * value, will request all values in the group again anyway, so they would have to have been
     * built in any case.
     *
     * <p>Example of when to use getValues: A ListProcessor value is built with key inputListRef.
     * The {@link #compute} method first calls getValue(InputList.key(inputListRef)), and retrieves
     * inputList. It then iterates through inputList, calling getValue on each input. Finally, it
     * processes the whole list and returns. Say inputList is (a, b, c). Since the {@link #compute}
     * method will unconditionally call getValue(a), getValue(b), and getValue (c), the {@link
     * #compute} method can instead just call getValues({a, b, c}). If the value is later dirtied
     * the evaluator will evaluate a, b, and c in parallel (assuming the inputList value was
     * unchanged), and re-evaluate the ListProcessor value only if at least one of them was changed.
     * On the other hand, if the InputList changes to be (a, b, d), then the evaluator will see that
     * the first dep has changed, and call the {@link #compute} method to re-evaluate from scratch,
     * without considering the dep group of {a, b, c}.
     *
     * <p>Example of when not to use getValues: A BestMatch value is built with key
     * &lt;potentialMatchesRef, matchCriterion&gt;. The {@link #compute} method first calls
     * getValue(PotentialMatches.key(potentialMatchesRef) and retrieves potentialMatches. It then
     * iterates through potentialMatches, calling getValue on each potential match until it finds
     * one that satisfies matchCriterion. In this case, if potentialMatches is (a, b, c), it would
     * be <i>incorrect</i> to call getValues({a, b, c}), because it is not known yet whether
     * requesting b or c will be necessary -- if a matches, then we will never call b or c.
     *
     * <p>Returns a map, {@code m}. For all {@code k} in {@code depKeys}, {@code m.containsKey(k)}
     * is {@code true}, and, {@code m.get(k) != null} iff the dependency was already evaluated and
     * was not in error.
     */
    Map<SkyKey, SkyValue> getValues(Iterable<? extends SkyKey> depKeys) throws InterruptedException;

    /**
     * Similar to {@link #getValues} but allows the caller to specify a set of types that are proper
     * subtypes of Exception (see {@link SkyFunctionException} for more details) to find out whether
     * any of the dependencies' evaluations resulted in exceptions of those types. The returned
     * objects may throw when attempting to retrieve their value.
     *
     * <p>Callers should prioritize their responsibility to detect and handle errors in the returned
     * map over their responsibility to return {@code null} if values are missing. This is because
     * in nokeep_going evaluations, an error from a low level dependency is given a chance to be
     * enriched by its reverse-dependencies, if possible.
     *
     * <p>Returns a map, {@code m}. For all {@code k} in {@code depKeys}, {@code m.get(k) != null}.
     * For all {@code v} such that there is some {@code k} such that {@code m.get(k) == v}, the
     * following is true: {@code v.get() != null} iff the dependency {@code k} was already evaluated
     * and was not in error. {@code v.get()} throws {@code E} iff the dependency {@code k} was
     * already evaluated with an error in the specified set of {@link Exception} types.
     */
    <E extends Exception> Map<SkyKey, ValueOrException<E>> getValuesOrThrow(
        Iterable<? extends SkyKey> depKeys, Class<E> exceptionClass) throws InterruptedException;

    <E1 extends Exception, E2 extends Exception>
        Map<SkyKey, ValueOrException2<E1, E2>> getValuesOrThrow(
            Iterable<? extends SkyKey> depKeys,
            Class<E1> exceptionClass1,
            Class<E2> exceptionClass2)
            throws InterruptedException;

    <E1 extends Exception, E2 extends Exception, E3 extends Exception>
        Map<SkyKey, ValueOrException3<E1, E2, E3>> getValuesOrThrow(
            Iterable<? extends SkyKey> depKeys,
            Class<E1> exceptionClass1,
            Class<E2> exceptionClass2,
            Class<E3> exceptionClass3)
            throws InterruptedException;

    <E1 extends Exception, E2 extends Exception, E3 extends Exception, E4 extends Exception>
        Map<SkyKey, ValueOrException4<E1, E2, E3, E4>> getValuesOrThrow(
            Iterable<? extends SkyKey> depKeys,
            Class<E1> exceptionClass1,
            Class<E2> exceptionClass2,
            Class<E3> exceptionClass3,
            Class<E4> exceptionClass4)
            throws InterruptedException;

    <
            E1 extends Exception,
            E2 extends Exception,
            E3 extends Exception,
            E4 extends Exception,
            E5 extends Exception>
        Map<SkyKey, ValueOrException5<E1, E2, E3, E4, E5>> getValuesOrThrow(
            Iterable<? extends SkyKey> depKeys,
            Class<E1> exceptionClass1,
            Class<E2> exceptionClass2,
            Class<E3> exceptionClass3,
            Class<E4> exceptionClass4,
            Class<E5> exceptionClass5)
            throws InterruptedException;

    /**
     * Returns whether there was a previous getValue[s][OrThrow] that indicated a missing
     * dependency. Formally, returns true iff at least one of the following occurred:
     *
     * <ul>
     *   <li>getValue[OrThrow](k[, c]) returned {@code null} for some k
     *   <li>getValues(ks).get(k) == {@code null} for some ks and k such that ks.contains(k)
     *   <li>getValuesOrThrow(ks, c).get(k).get() == {@code null} for some ks and k such that
     *       ks.contains(k)
     * </ul>
     *
     * <p>If this returns true, the {@link SkyFunction} must return {@code null}.
     */
    boolean valuesMissing();

    /**
     * Returns the {@link ExtendedEventHandler} that a SkyFunction should use to print any errors,
     * warnings, or progress messages during execution of {@link SkyFunction#compute}.
     */
    ExtendedEventHandler getListener();

    /**
     * A live view of deps known to have already been requested either through an earlier call to
     * {@link SkyFunction#compute} or inferred during change pruning. Should return {@code null} if
     * unknown.
     */
    @Nullable
    default GroupedList<SkyKey> getTemporaryDirectDeps() {
      return null;
    }

    /**
     * Injects non-hermetic {@link Version} information for this environment.
     *
     * <p>This may be called during the course of {@link SkyFunction#compute(SkyKey, Environment)}
     * if the function discovers version information for the {@link SkyKey}.
     *
     * <p>Environments that either do not need or wish to ignore non-hermetic version information
     * may keep the default no-op implementation.
     */
    default void injectVersionForNonHermeticFunction(Version version) {}

    /**
     * Register dependencies on keys without necessarily requiring their values.
     *
     * <p>WARNING: Dependencies here MUST be done! Only use this function if you know what you're
     * doing.
     *
     * <p>If the {@link EvaluationVersionBehavior} is {@link
     * EvaluationVersionBehavior#MAX_CHILD_VERSIONS} then this method may fall back to just doing a
     * {@link #getValues} call internally. Thus, any graph evaluations that require this method to
     * be performant <i>must</i> run with {@link EvaluationVersionBehavior#GRAPH_VERSION}.
     */
    default void registerDependencies(Iterable<SkyKey> keys) throws InterruptedException {
      getValues(keys);
    }

    /** Returns whether we are currently in error bubbling. */
    @VisibleForTesting
    boolean inErrorBubblingForTesting();

    /**
     * Adds a dependency on a Skyframe-external event. If the given future is already complete, this
     * method silently returns without doing anything (to avoid unnecessary function restarts).
     * Otherwise, Skyframe adds a listener to the passed-in future, and only re-enqueues the current
     * node after the future completes and all requested deps are done. The added listener will
     * perform the minimum amount of work on the thread completing the future necessary for Skyframe
     * bookkeeping.
     *
     * <p>Callers of this method must check {@link #valuesMissing} before returning {@code null}
     * from a {@link SkyFunction}.
     *
     * <p>This API is intended for performing async computations (e.g., remote execution) in another
     * thread pool without blocking the current Skyframe thread.
     */
    void dependOnFuture(ListenableFuture<?> future);
  }
}
