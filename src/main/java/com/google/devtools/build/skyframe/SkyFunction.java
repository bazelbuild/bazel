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

import static com.google.common.base.Preconditions.checkState;

import com.google.common.base.Preconditions;
import com.google.common.graph.GraphBuilder;
import com.google.common.graph.ImmutableGraph;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.events.Reportable;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.io.IOException;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Supplier;
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
   * <p>Be aware that during error bubbling Skyframe will interpret a thrown {@link
   * InterruptedException} to mean that this method has no additional context to contribute to a
   * dependency's exception. Also note that Skyframe interrupts the evaluating thread when, during
   * error bubbling, this method requests a dependency which failed with an exception. Prefer (if
   * possible) exception enrichment logic simple enough to be insensitive to the evaluating thread's
   * interrupt state.
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
  default String extractTag(SkyKey skyKey) {
    return null;
  }

  /**
   * Returns the max transitive source version that would be injected via {@link
   * SkyFunctionEnvironment#injectVersionForNonHermeticFunction} if {@link #compute(SkyKey,
   * Environment)} were invoked for the given {@link SkyKey}/{@link SkyValue} pair, or null if no
   * call for version injection would be made.
   */
  @Nullable
  default Version getMaxTransitiveSourceVersionToInjectForNonHermeticFunction(
      SkyKey skyKey, SkyValue skyValue) throws IOException {
    checkState(skyKey.functionName().getHermeticity() == FunctionHermeticity.HERMETIC);
    return null;
  }

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
   *
   * <p>This may only be returned by {@link #compute} if {@link Environment#restartPermitted} is
   * true. If restarting is not permitted, {@link #compute} should throw an appropriate {@link
   * SkyFunctionException}.
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
   * Value lookup subset of services provided to {@link SkyFunction} implementations.
   *
   * <p>See {@link Environment} for the full set of services.
   */
  interface LookupEnvironment {
    /**
     * Returns a direct dependency. If the specified value is not in the set of already evaluated
     * direct dependencies, returns {@code null}. Also returns {@code null} if the specified value
     * has already been evaluated and found to be in error.
     *
     * <p>On a subsequent evaluation, if any of this value's dependencies have changed they will be
     * re-evaluated in the same order as originally requested by the {@code SkyFunction} using this
     * {@code getValue} call (see {@link #getValuesAndExceptions} for when preserving the order is
     * not important).
     *
     * <p>This method and the ones below may throw {@link InterruptedException}. Such exceptions
     * must not be caught by the {@link SkyFunction#compute} implementation. Instead, they should be
     * propagated up to the caller of {@link SkyFunction#compute}.
     */
    @CanIgnoreReturnValue
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
    @CanIgnoreReturnValue
    @Nullable
    <E extends Exception> SkyValue getValueOrThrow(SkyKey depKey, Class<E> exceptionClass)
        throws E, InterruptedException;

    @CanIgnoreReturnValue
    @Nullable
    <E1 extends Exception, E2 extends Exception> SkyValue getValueOrThrow(
        SkyKey depKey, Class<E1> exceptionClass1, Class<E2> exceptionClass2)
        throws E1, E2, InterruptedException;

    @CanIgnoreReturnValue
    @Nullable
    <E1 extends Exception, E2 extends Exception, E3 extends Exception> SkyValue getValueOrThrow(
        SkyKey depKey,
        Class<E1> exceptionClass1,
        Class<E2> exceptionClass2,
        Class<E3> exceptionClass3)
        throws E1, E2, E3, InterruptedException;

    @CanIgnoreReturnValue
    @Nullable
    <E1 extends Exception, E2 extends Exception, E3 extends Exception, E4 extends Exception>
        SkyValue getValueOrThrow(
            SkyKey depKey,
            Class<E1> exceptionClass1,
            Class<E2> exceptionClass2,
            Class<E3> exceptionClass3,
            Class<E4> exceptionClass4)
            throws E1, E2, E3, E4, InterruptedException;

    /**
     * Requests {@code depKeys} "in parallel", independent of each others' results. These keys may
     * be thought of as a "dependency group" -- they are requested together by this value.
     *
     * <p>In general, if the result of one getValue call can affect the argument of a later getValue
     * call, the two calls cannot be merged into a single getValuesAndExceptions call, since the
     * result of the first call might change on a later evaluation. Inversely, if the result of one
     * getValue call cannot affect the parameters of the next getValue call, the two keys can form a
     * dependency group and the two getValue calls should be merged into one getValuesAndExceptions
     * call. In the latter case, if we fail to combine the _multiple_ getValue (or
     * getValuesAndExceptions) calls into one _single_ getValuesAndExceptions call, it would result
     * in multiple dependency groups with an implicit ordering between them. This would
     * unnecessarily cause sequential evaluations of these groups and could impact overall
     * performance.
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
     * <p>Example of when to use getValuesAndExceptions: A ListProcessor value is built with key
     * inputListRef. The {@link #compute} method first calls getValue(InputList.key(inputListRef)),
     * and retrieves inputList. It then iterates through inputList, calling getValue on each input.
     * Finally, it processes the whole list and returns. Say inputList is (a, b, c). Since the
     * {@link #compute} method will unconditionally call getValue(a), getValue(b), and getValue(c),
     * the {@link #compute} method can instead just call getValuesAndExceptions({a, b, c}). If the
     * value is later dirtied the evaluator will evaluate a, b, and c in parallel (assuming the
     * inputList value was unchanged), and re-evaluate the ListProcessor value only if at least one
     * of them was changed. On the other hand, if the InputList changes to be (a, b, d), then the
     * evaluator will see that the first dep has changed, and call the {@link #compute} method to
     * re-evaluate from scratch, without considering the dep group of {a, b, c}.
     *
     * <p>Example of when not to use getValuesAndExceptions: A BestMatch value is built with key
     * &lt;potentialMatchesRef, matchCriterion&gt;. The {@link #compute} method first calls
     * getValue(PotentialMatches.key(potentialMatchesRef) and retrieves potentialMatches. It then
     * iterates through potentialMatches, calling getValue on each potential match until it finds
     * one that satisfies matchCriterion. In this case, if potentialMatches is (a, b, c), it would
     * be <i>incorrect</i> to call getValuesAndExceptions({a, b, c}), because it is not known yet
     * whether requesting b or c will be necessary -- if a matches, then we will never call b or c.
     *
     * <p>Returns a {@link SkyframeLookupResult}, which allows the calling {@code SkyFunction} to
     * get a value or throw an exception per SkyKey.
     */
    @CanIgnoreReturnValue
    SkyframeLookupResult getValuesAndExceptions(Iterable<? extends SkyKey> depKeys)
        throws InterruptedException;

    /**
     * Returns a lookup result containing previously requested dependencies.
     *
     * <p>NB: this may contain fewer dependencies than expected if the node is restarted before all
     * its dependencies have signaled. The two known cases are error bubbling and partial
     * re-evaluation. In error bubbling, an error should be present.
     */
    SkyframeLookupResult getLookupHandleForPreviouslyRequestedDeps();
  }

  /**
   * The services provided to the {@link SkyFunction#compute} implementation by the Skyframe
   * evaluation framework.
   */
  interface Environment extends LookupEnvironment {
    /**
     * Returns whether there was a previous getValue[s][OrThrow] that indicated a missing
     * dependency. Formally, returns true iff at least one of the following occurred:
     *
     * <ul>
     *   <li>getValue[OrThrow](k[, c]) returned {@code null} for some k
     *   <li>A call to {@code result#get[OrThrow](k[, c])} returned {@code null} where result =
     *       getValuesAndExceptions(ks) for some ks
     *   <li>A call to {@code result#queryDep(k, cb)} returned {@code false} where result =
     *       getValuesAndExceptions(ks) for some ks
     * </ul>
     *
     * <p>If this returns true, the {@link SkyFunction} must return {@code null} or throw a {@link
     * SkyFunctionException} if it detected an error even with values missing.
     */
    boolean valuesMissing();

    /**
     * Returns the {@link ExtendedEventHandler} that a {@link SkyFunction} should use to print any
     * errors, warnings, or progress messages during execution of {@link SkyFunction#compute}.
     *
     * <p>{@link Reportable#storeForReplay} is used to determine when to actually {@linkplain
     * Reportable#reportTo report} events passed to the listener. A return of {@code false}
     * indicates that the event's relevance is tied to the time at which it is created, so it is
     * reported immediately. All other events are temporarily stored in the environment and only
     * reported after the function completes. If the function returns {@code null} due to a missing
     * dependency, these events are discarded. It is the responsibility of the function to emit the
     * events again after it is restarted. Note that if using {@link #getState} to prune work, the
     * function may need to store events in the {@link SkyKeyComputeState} so that they can be
     * replayed on a subsequent invocation.
     */
    ExtendedEventHandler getListener();

    /**
     * A live view of deps known to have already been requested either through an earlier call to
     * {@link SkyFunction#compute} or inferred during change pruning. Should return {@code null} if
     * unknown. Only for special use cases: do not use in general unless you know exactly what
     * you're doing!
     */
    @Nullable
    default GroupedDeps getTemporaryDirectDeps() {
      return null;
    }

    /**
     * Injects non-hermetic {@link Version} information for the currently evaluating {@link SkyKey}.
     *
     * <p>This may be called during the course of {@link SkyFunction#compute} if the function
     * determines that the currently evaluating key's source dependencies have not changed since the
     * given {@code version}.
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
     * <p>If {@linkplain NodeEntry#getMaxTransitiveSourceVersion max transitive source versions} are
     * being tracked, then this method must not be called.
     */
    void registerDependencies(Iterable<SkyKey> keys);

    /**
     * Returns whether we are currently in error bubbling. Should only be used by SkyFunctions that
     * can fully recover from a dependency's throwing an exception in --keep_going mode, returning a
     * value instead of transforming the exception. {@link
     * com.google.devtools.build.lib.skyframe.TargetPatternFunction} is the classic example of such
     * a SkyFunction, since it can encounter errors while processing target patterns like
     * '//foo/...' but still return the list of all found targets.
     *
     * <p>Such a SkyFunction cannot unconditionally return a value, since in --nokeep_going mode it
     * may be called upon to transform a lower-level exception. This method can tell it whether to
     * transform a dependency's exception or ignore it and return a value as usual.
     */
    boolean inErrorBubblingForSkyFunctionsThatCanFullyRecoverFromErrors();

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

    /**
     * A {@link SkyFunction#compute} call may return {@link Restart} only if this returns {@code
     * true}.
     */
    boolean restartPermitted();

    /**
     * Container for data stored in between calls to {@link #compute} for the same {@link SkyKey}.
     *
     * <p>See the javadoc of {@link #getState} for motivation and an example.
     */
    interface SkyKeyComputeState extends AutoCloseable {
      /**
       * {@inheritDoc}
       *
       * <p>Can be overridden to make sure {@link SkyKeyComputeState} objects are cleaned up. Note
       * that, while this ostensibly opens up the possibility for {@link SkyKeyComputeState} to hold
       * on to any kind of external resource, doing so might still be dangerous as we only actively
       * drop {@link SkyKeyComputeState} objects on high memory pressure. If the external resource
       * being held on to is approaching starvation, we currently don't do anything to alleviate
       * that pressure. So think *hard* before you start doing that!
       *
       * <p>Implementations <strong>MUST</strong> be idempotent.
       *
       * <p>Note also that this method should not perform any heavy work (especially blocking
       * operations).
       */
      @Override
      default void close() {}
    }

    /**
     * Canonical type-safe heterogeneous container for use with {@link #getState} in SkyFunction
     * implementations that employ complex or abstract compositional strategies.
     */
    // Must be threadsafe: used by PartialReevaluationMailbox#from on multiple threads, to save
    // signals from deps.
    @ThreadSafe
    class ClassToInstanceMapSkyKeyComputeState implements SkyKeyComputeState {

      private final ConcurrentHashMap<Class<? extends SkyKeyComputeState>, SkyKeyComputeState> map =
          new ConcurrentHashMap<>();

      public <T extends SkyKeyComputeState> T getInstance(
          Class<T> type, Supplier<T> stateSupplier) {
        return type.cast(map.computeIfAbsent(type, ignored -> stateSupplier.get()));
      }
    }

    /**
     * Returns (or creates and returns) a "state" object to assist with temporary computations for
     * the {@link SkyKey} associated with this {@link Environment}.
     *
     * <p>The {@link SkyKeyComputeState} will either be freshly created via the given {@link
     * Supplier}, or will be the same exact instance used on the previous call to this method for
     * the same {@link SkyKey}. This allows {@link SkyFunction} implementations to avoid redoing the
     * same intermediate work over-and-over again on each {@link #compute} call for the same {@link
     * SkyKey}, due to missing Skyframe dependencies. For example,
     *
     * <pre>
     *   class MyFunction implements SkyFunction {
     *     public SkyValue compute(SkyKey skyKey, Environment env) throws InterruptedException {
     *       int x = (Integer) skyKey.argument();
     *       SkyKey myDependencyKey = getSkyKeyForValue(someExpensiveComputation(x));
     *       SkyValue myDependencyValue = env.getValue(myDependencyKey);
     *       if (env.valuesMissing()) {
     *         return null;
     *       }
     *       return createMyValue(myDependencyValue);
     *     }
     *   }
     * </pre>
     *
     * <p>If the dependency was missing, then we'll end up evaluating {@code
     * someExpensiveComputation(x)} twice, once on the initial call to {@link #compute} and then
     * again on the subsequent call after the dependency was computed.
     *
     * <p>To fix this, we can use a mutable {@link SkyKeyComputeState} implementation and store the
     * result of {@code someExpensiveComputation(x)} in there:
     *
     * <pre>
     *   class MyFunction implements SkyFunction {
     *     private static class State implements SkyKeyComputeState {
     *       private Integer result;
     *     }
     *
     *     public SkyValue compute(SkyKey skyKey, Environment env) throws InterruptedException {
     *       int x = (Integer) skyKey.argument();
     *       State state = env.getState(State::new);
     *       if (state.result == null) {
     *         state.result = someExpensiveComputation(x);
     *       }
     *       SkyKey myDependencyKey = getSkyKeyForValue(state.result);
     *       SkyValue myDependencyValue = env.getValue(myDependencyKey);
     *       if (env.valuesMissing()) {
     *         return null;
     *       }
     *       return createMyValue(myDependencyValue);
     *     }
     *   }
     * </pre>
     *
     * <p>Now {@code someExpensiveComputation(x)} gets called exactly once for each {@code x}!
     *
     * <p>Important: There's no guarantee the {@link SkyKeyComputeState} instance will be the same
     * exact instance used on the previous call to this method for the same {@link SkyKey}. The
     * above example was just illustrating the best-case outcome. Therefore, {@link SkyFunction}
     * implementations should make use of this feature only as a performance optimization.
     *
     * <p>Note that {@link SkyKeyComputeState#close()} allows us to hold on to other kinds of
     * external resources and clean them up when necessary, but see the Javadoc there for caveats.
     *
     * <p>A notable example of the above note is that if {@link #compute} returns a {@link Restart}
     * then a call to {@link #getState} on the subsequent call to {@link #compute} will definitely
     * use the {@code stateSupplier}. It's important that Skyframe do this because {@link Restart}
     * indicates that work should be redone, and so it'd be wrong to reuse work from the previous
     * {@link #compute} call.
     */
    <T extends SkyKeyComputeState> T getState(Supplier<T> stateSupplier);

    /**
     * Returns the max transitive source version of a {@link NodeEntry}.
     *
     * <p>This value might not consider all deps' source versions if called before all deps have
     * been requested or if {@link #valuesMissing} returns {@code true}.
     *
     * <p>Rules for calculation of the max transitive source version:
     *
     * <ul>
     *   <li>Returns {@code null} during cycle detection and error bubbling, or for transient
     *       errors.
     *   <li>If the node is {@link FunctionHermeticity#NONHERMETIC}, returns the version passed to
     *       {@link #injectVersionForNonHermeticFunction} if it was called, or else {@code null}.
     *   <li>For all other nodes, queries {@link NodeEntry#getMaxTransitiveSourceVersion} of direct
     *       dependency nodes and chooses the maximal version seen (according to {@link
     *       Version#atMost}). If there are no direct dependencies, returns {@link
     *       ParallelEvaluatorContext#getMinimalVersion}. If any direct dependency node has a {@code
     *       null} MTSV, returns {@code null}.
     * </ul>
     */
    @Nullable
    Version getMaxTransitiveSourceVersionSoFar();
  }
}
