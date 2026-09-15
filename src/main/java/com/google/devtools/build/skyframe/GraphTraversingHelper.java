// Copyright 2022 The Bazel Authors. All rights reserved.
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
import com.google.common.base.Throwables;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.bugreport.BugReporter;
import javax.annotation.Nullable;

/**
 * Helper to check if requested nodes are done in the graph. The only method {@link
 * #declareDependenciesAndCheckIfValuesMissing} calls {@link
 * SkyFunction.Environment#getValuesAndExceptions} and checks that all nodes are done, either with
 * values or expected exceptions.
 *
 * <p>Only use this helper if you know what you are doing! Most Skyframe users should not need to
 * call it.
 */
public final class GraphTraversingHelper {

  /**
   * Returns false iff for each key in {@code skyKeys}, the corresponding node is done with values
   * in the Skyframe graph.
   */
  public static boolean declareDependenciesAndCheckIfValuesMissing(
      SkyFunction.Environment env, Iterable<? extends SkyKey> skyKeys) throws InterruptedException {
    return declareDependenciesAndCheckIfValuesMissing(
        env, skyKeys, /*exceptionClass1=*/ null, /*exceptionClass2=*/ null);
  }

  /**
   * Returns false iff for each key in {@code skyKeys}, corresponding node is done with values or
   * specified {@code exceptionClass} errors in the Skyframe graph.
   *
   * <p>The exception class given cannot be a supertype or a subtype of {@link RuntimeException}, or
   * a subtype of {@link InterruptedException}. See {@link
   * SkyFunctionException#validateExceptionType} for details.
   */
  public static <E extends Exception> boolean declareDependenciesAndCheckIfValuesMissing(
      SkyFunction.Environment env, Iterable<? extends SkyKey> skyKeys, Class<E> exceptionClass)
      throws InterruptedException {
    return declareDependenciesAndCheckIfValuesMissing(
        env, skyKeys, exceptionClass, /*exceptionClass2=*/ null);
  }

  public static <E1 extends Exception, E2 extends Exception>
      boolean declareDependenciesAndCheckIfValuesMissing(
          SkyFunction.Environment env,
          Iterable<? extends SkyKey> skyKeys,
          @Nullable Class<E1> exceptionClass1,
          @Nullable Class<E2> exceptionClass2)
          throws InterruptedException {
    return declareDependenciesAndCheckIfValuesMissing(
        env, skyKeys, exceptionClass1, exceptionClass2, BugReporter.defaultInstance());
  }

  @VisibleForTesting
  static <E1 extends Exception, E2 extends Exception>
      boolean declareDependenciesAndCheckIfValuesMissing(
          SkyFunction.Environment env,
          Iterable<? extends SkyKey> skyKeys,
          @Nullable Class<E1> exceptionClass1,
          @Nullable Class<E2> exceptionClass2,
          BugReporter bugReporter)
          throws InterruptedException {
    SkyframeLookupResult result = env.getValuesAndExceptions(skyKeys);
    if (env.valuesMissing()) {
      return true;
    }
    for (SkyKey key : skyKeys) {
      try {
        SkyValue value = result.getOrThrow(key, exceptionClass1, exceptionClass2);
        if (value == null) {
          bugReporter.logUnexpected("Value for: '%s' was missing, this should never happen", key);
          return true;
        }
      } catch (Exception e) {
        if ((exceptionClass1 != null && exceptionClass1.isInstance(e))
            || (exceptionClass2 != null && exceptionClass2.isInstance(e))) {
          continue;
        }
        Throwables.throwIfUnchecked(e);
        throw new IllegalStateException(
            "unexpected exception from " + Iterables.toString(skyKeys), e);
      }
    }
    return false;
  }

  /**
   * Returns false iff for each key in {@code skyKeys}, the corresponding node is done with values
   * in the Skyframe graph, and every node evaluated successfully without an exception.
   *
   * <p>Prefer {@link #declareDependenciesAndCheckIfValuesMissing} when possible. This method is for
   * {@link SkyFunction} callers that don't handle child exceptions themselves, and just want to
   * propagate child exceptions upwards via Skyframe.
   */
  public static boolean declareDependenciesAndCheckIfValuesMissingMaybeWithExceptions(
      SkyFunction.Environment env, Iterable<? extends SkyKey> skyKeys) throws InterruptedException {
    SkyframeLookupResult result = env.getValuesAndExceptions(skyKeys);
    if (env.valuesMissing()) {
      return true;
    }
    for (SkyKey key : skyKeys) {
      if (result.get(key) == null) {
        return true;
      }
    }
    return false;
  }

  private GraphTraversingHelper() {}
}
