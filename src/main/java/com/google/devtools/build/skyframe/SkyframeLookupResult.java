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

import com.google.errorprone.annotations.CanIgnoreReturnValue;
import javax.annotation.Nullable;

/**
 * A map-like result of getting Skyframe dependencies via {@link
 * SkyFunction.Environment#getValuesAndExceptions}. Callers can use the {@link #get}, {@link
 * #getOrThrow} and {@link #queryDep} methods to obtain elements by key.
 *
 * <p>Note that a {@link SkyFunction} cannot guarantee that {@link
 * SkyFunction.Environment#valuesMissing} will be true upon receipt of a {@code
 * SkyframeLookupResult}. The elements must all be fetched. If {@link #get} or {@link #getOrThrow}
 * returns {@code null}, or {@link #queryDep} returns false, only then will {@link
 * SkyFunction.Environment#valuesMissing} be guaranteed to return true.
 */
public interface SkyframeLookupResult {

  /**
   * Returns a direct dependency. If the dependency is not in the set of already evaluated direct
   * dependencies, returns {@code null}. Also returns {@code null} if the dependency has already
   * been evaluated and found to be in error. In either of these cases, {@link
   * SkyFunction.Environment#valuesMissing} will subsequently return true.
   */
  @Nullable
  default SkyValue get(SkyKey skyKey) {
    return getOrThrow(skyKey, null, null, null);
  }

  /**
   * Returns a direct dependency. If the dependency is not in the set of already evaluated direct
   * dependencies, returns {@code null}. If the dependency has already been evaluated and found to
   * be in error, throws the exception coming from the error, so long as the exception is of one of
   * the specified types. SkyFunction implementations may use this method to catch and rethrow a
   * more informative exception or to continue evaluation. The caller must specify the exception
   * type(s) that might be thrown using the {@code exceptionClass} argument(s). If the dependency's
   * exception is not an instance of {@code exceptionClass}, {@code null} is returned. In this case,
   * {@link SkyFunction.Environment#valuesMissing} will subsequently return true.
   *
   * <p>The exception class given cannot be a supertype or a subtype of {@link RuntimeException}, or
   * a subtype of {@link InterruptedException}. See {@link
   * SkyFunctionException#validateExceptionType} for details.
   */
  @CanIgnoreReturnValue
  @Nullable
  default <E extends Exception> SkyValue getOrThrow(SkyKey skyKey, Class<E> exceptionClass)
      throws E {
    return getOrThrow(skyKey, exceptionClass, null, null);
  }

  /** Similar to {@link #getOrThrow(SkyKey, Class)}, but takes two exception class parameters. */
  @CanIgnoreReturnValue
  @Nullable
  default <E1 extends Exception, E2 extends Exception> SkyValue getOrThrow(
      SkyKey skyKey, Class<E1> exceptionClass1, Class<E2> exceptionClass2) throws E1, E2 {
    return getOrThrow(skyKey, exceptionClass1, exceptionClass2, null);
  }

  /** Similar to {@link #getOrThrow(SkyKey, Class)}, but takes three exception class parameters. */
  @CanIgnoreReturnValue
  @Nullable
  <E1 extends Exception, E2 extends Exception, E3 extends Exception> SkyValue getOrThrow(
      SkyKey skyKey,
      @Nullable Class<E1> exceptionClass1,
      @Nullable Class<E2> exceptionClass2,
      @Nullable Class<E3> exceptionClass3)
      throws E1, E2, E3;

  /**
   * Similar to {@link #getOrThrow}, but supports more flexible control flows.
   *
   * <p>This method provides a more generic exception handling interface than {@link #getOrThrow},
   * for cases where the caller cannot pass specific exception types.
   *
   * @return true if either a value was passed to {@link QueryDepCallback#acceptValue} or an
   *     exception was successfully handled by {@link QueryDepCallback#tryHandleException}. False
   *     indicates that either the key is unavailable or an exception was unhandled by {@link
   *     QueryDepCallback#tryHandleException}.
   */
  boolean queryDep(SkyKey key, QueryDepCallback resultCallback);

  /**
   * An interface specifying a dependency query.
   *
   * <p>{@link #queryDep} calls one of {@link QueryDepCallback#acceptValue} or {@link
   * QueryDepCallback#tryHandleException} if a result is available.
   */
  @FunctionalInterface
  interface QueryDepCallback {
    /**
     * Accepts a value.
     *
     * @param key the key associated with the value.
     */
    void acceptValue(SkyKey key, SkyValue value);

    /**
     * Offers an exception to this query.
     *
     * @param key the key associated with the exception.
     * @return true if the exception was handled.
     */
    default boolean tryHandleException(SkyKey key, Exception e) {
      return false; // Default implementation that handles no exceptions.
    }
  }
}
