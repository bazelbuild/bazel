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

import com.google.common.annotations.VisibleForTesting;
import com.google.devtools.build.lib.events.EventHandler;

import java.util.Map;

import javax.annotation.Nullable;

/**
 * Machinery to evaluate a single value.
 *
 * <p>The builder is supposed to access only direct dependencies of the value. However, the direct
 * dependencies need not be known in advance. The builder can request arbitrary values using
 * {@link Environment#getValue}. If the values are not ready, the call will return null; in that
 * case the builder can either try to proceed (and potentially indicate more dependencies by
 * additional {@code getValue} calls), or just return null, in which case the missing dependencies
 * will be computed and the builder will be started again.
 */
public interface SkyFunction {

  /**
   * When a value is requested, this method is called with the name of the value and a value
   * building environment.
   *
   * <p>This method should return a constructed value, or null if any dependencies were missing
   * ({@link Environment#valuesMissing} was true before returning). In that case the missing
   * dependencies will be computed and the value builder restarted.
   *
   * <p>Implementations must be threadsafe and reentrant.
   *
   * @throws SkyFunctionException on failure
   * @throws InterruptedException when the user interrupts the build
   */
  @Nullable SkyValue compute(SkyKey skyKey, Environment env) throws SkyFunctionException,
      InterruptedException;

  /**
   * Extracts a tag (target label) from a SkyKey if it has one. Otherwise return null.
   *
   * <p>The tag is used for filtering out non-error event messages that do not match --output_filter
   * flag. If a SkyFunction returns null in this method it means that all the info/warning messages
   * associated with this value will be shown, no matter what --output_filter says.
   */
  @Nullable
  String extractTag(SkyKey skyKey);

  /**
   * The services provided to the value builder by the graph implementation.
   */
  interface Environment {
    /**
     * Returns a direct dependency. If the specified value is not in the set of already evaluated
     * direct dependencies, returns null. Also returns null if the specified value has already been
     * evaluated and found to be in error.
     *
     * <p>On a subsequent build, if any of this value's dependencies have changed they will be
     * re-evaluated in the same order as originally requested by the {@code SkyFunction} using
     * this {@code getValue} call (see {@link #getValues} for when preserving the order is not
     * important).
     */
    @Nullable
    SkyValue getValue(SkyKey valueName);

    /**
     * Returns a direct dependency. If the specified value is not in the set of already evaluated
     * direct dependencies, returns null. If the specified value has already been evaluated and
     * found to be in error, throws the exception coming from the error. Value builders may
     * use this method to continue evaluation even if one of their children is in error by catching
     * the thrown exception and proceeding. The caller must specify the exception that might be
     * thrown using the {@code exceptionClass} argument. If the child's exception is not an instance
     * of {@code exceptionClass}, returns null without throwing.
     *
     * <p>The exception class given cannot be a supertype or a subtype of {@link RuntimeException},
     * or a subtype of {@link InterruptedException}. See
     * {@link SkyFunctionException#validateExceptionType} for details.
     */
    @Nullable
    <E extends Exception> SkyValue getValueOrThrow(SkyKey depKey, Class<E> exceptionClass) throws E;
    @Nullable
    <E1 extends Exception, E2 extends Exception> SkyValue getValueOrThrow(SkyKey depKey,
        Class<E1> exceptionClass1, Class<E2> exceptionClass2) throws E1, E2;
    @Nullable
    <E1 extends Exception, E2 extends Exception, E3 extends Exception> SkyValue getValueOrThrow(
        SkyKey depKey, Class<E1> exceptionClass1, Class<E2> exceptionClass2,
        Class<E3> exceptionClass3) throws E1, E2, E3;
    @Nullable
    <E1 extends Exception, E2 extends Exception, E3 extends Exception, E4 extends Exception>
        SkyValue getValueOrThrow(SkyKey depKey, Class<E1> exceptionClass1,
        Class<E2> exceptionClass2, Class<E3> exceptionClass3, Class<E4> exceptionClass4)
            throws E1, E2, E3, E4;

    /**
     * Returns true iff any of the past {@link #getValue}(s) or {@link #getValueOrThrow} method
     * calls for this instance returned null (because the value was not yet present and done in the
     * graph).
     *
     * <p>If this returns true, the {@link SkyFunction} must return {@code null}.
     */
    boolean valuesMissing();

    /**
     * Requests {@code depKeys} "in parallel", independent of each others' values. These keys may be
     * thought of as a "dependency group" -- they are requested together by this value.
     *
     * <p>In general, if the result of one getValue call can affect the argument of a later getValue
     * call, the two calls cannot be merged into a single getValues call, since the result of the
     * first call might change on a later build. Inversely, if the result of one getValue call
     * cannot affect the parameters of the next getValue call, the two keys can form a dependency
     * group and the two getValue calls merged into one getValues call.
     *
     * <p>This means that on subsequent builds, when checking to see if a value requires rebuilding,
     * all the values in this group may be simultaneously checked. A SkyFunction should request a
     * dependency group if checking the deps serially on a subsequent build would take too long, and
     * if the builder would request all deps anyway as long as no earlier deps had changed.
     * SkyFunction.Environment implementations may also choose to request these deps in
     * parallel on the first build, potentially speeding up the build.
     *
     * <p>While re-evaluating every value in the group may take longer than re-evaluating just the
     * first one and finding that it has changed, no extra work is done: the contract of the
     * dependency group means that the builder, when called to rebuild this value, will request all
     * values in the group again anyway, so they would have to have been built in any case.
     *
     * <p>Example of when to use getValues: A ListProcessor value is built with key inputListRef.
     * The builder first calls getValue(InputList.key(inputListRef)), and retrieves inputList. It
     * then iterates through inputList, calling getValue on each input. Finally, it processes the
     * whole list and returns. Say inputList is (a, b, c). Since the builder will unconditionally
     * call getValue(a), getValue(b), and getValue(c), the builder can instead just call
     * getValues({a, b, c}). If the value is later dirtied the evaluator will build a, b, and c in
     * parallel (assuming the inputList value was unchanged), and re-evaluate the ListProcessor
     * value only if at least one of them was changed. On the other hand, if the InputList changes
     * to be (a, b, d), then the evaluator will see that the first dep has changed, and call the
     * builder to rebuild from scratch, without considering the dep group of {a, b, c}.
     *
     * <p>Example of when not to use getValues: A BestMatch value is built with key
     * &lt;potentialMatchesRef, matchCriterion&gt;. The builder first calls
     * getValue(PotentialMatches.key(potentialMatchesRef) and retrieves potentialMatches. It then
     * iterates through potentialMatches, calling getValue on each potential match until it finds
     * one that satisfies matchCriterion. In this case, if potentialMatches is (a, b, c), it would
     * be <i>incorrect</i> to call getValues({a, b, c}), because it is not known yet whether
     * requesting b or c will be necessary -- if a matches, then we will never call b or c.
     */
    Map<SkyKey, SkyValue> getValues(Iterable<SkyKey> depKeys);

    /**
     * The same as {@link #getValues} but the returned objects may throw when attempting to retrieve
     * their value. Note that even if the requested values can throw different kinds of exceptions,
     * only exceptions of type {@code E} will be preserved in the returned objects. All others will
     * be null.
     */
    <E extends Exception> Map<SkyKey, ValueOrException<E>> getValuesOrThrow(
        Iterable<SkyKey> depKeys, Class<E> exceptionClass);
    <E1 extends Exception, E2 extends Exception> Map<SkyKey, ValueOrException2<E1, E2>>
    getValuesOrThrow(Iterable<SkyKey> depKeys, Class<E1> exceptionClass1,
        Class<E2> exceptionClass2);
    <E1 extends Exception, E2 extends Exception, E3 extends Exception>
    Map<SkyKey, ValueOrException3<E1, E2, E3>> getValuesOrThrow(Iterable<SkyKey> depKeys,
        Class<E1> exceptionClass1, Class<E2> exceptionClass2, Class<E3> exceptionClass3);
    <E1 extends Exception, E2 extends Exception, E3 extends Exception, E4 extends Exception>
    Map<SkyKey, ValueOrException4<E1, E2, E3, E4>> getValuesOrThrow(Iterable<SkyKey> depKeys,
        Class<E1> exceptionClass1, Class<E2> exceptionClass2, Class<E3> exceptionClass3,
        Class<E4> exceptionClass4);

    /**
     * Returns the {@link EventHandler} that a SkyFunction should use to print any errors,
     * warnings, or progress messages while building.
     */
    EventHandler getListener();

    /** Returns whether we are currently in error bubbling. */
    @VisibleForTesting
    boolean inErrorBubblingForTesting();
  }
}
