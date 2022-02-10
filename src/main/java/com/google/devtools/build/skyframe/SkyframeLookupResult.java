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

import com.google.common.base.Function;
import javax.annotation.Nullable;

/**
 * An Map-like result of getting Skyframe dependencies via {@link
 * SkyFunction.Environment#getValuesAndExceptions}. Callers can use the {@link #get} and {@link
 * #getOrThrow} methods to obtain elements by key.
 *
 * <p>Note that a {@link SkyFunction} cannot guarantee that {@link
 * SkyFunction.Environment#valuesMissing} will be true upon receipt of a {@code
 * SkyframeLookupResult}. The elements must all be fetched. If {@link #get} or {@link #getOrThrow}
 * returns {@code null}, only then will {@link SkyFunction.Environment#valuesMissing} be guaranteed
 * to return true.
 */
public final class SkyframeLookupResult {
  private final Runnable valuesMissingCallback;
  private final Function<SkyKey, ValueOrUntypedException> valuesOrExceptions;

  SkyframeLookupResult(
      Runnable valuesMissingCallback,
      Function<SkyKey, ValueOrUntypedException> valuesOrExceptions) {
    this.valuesMissingCallback = checkNotNull(valuesMissingCallback);
    this.valuesOrExceptions = checkNotNull(valuesOrExceptions);
  }

  /**
   * Returns a direct dependency. If the dependency is not in the set of already evaluated direct
   * dependencies, returns {@code null}. Also returns {@code null} if the dependency has already
   * been evaluated and found to be in error. In either of these cases, {@link
   * SkyFunction.Environment#valuesMissing} will subsequently return true.
   */
  @Nullable
  public SkyValue get(SkyKey skyKey) {
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
  @Nullable
  public <E1 extends Exception> SkyValue getOrThrow(SkyKey skyKey, Class<E1> exceptionClass)
      throws E1 {
    return getOrThrow(skyKey, exceptionClass, null, null);
  }

  /**
   * Similar to {@link getOrThrow} one exceptionClass version, but take two exceptionClass
   * parameters.
   */
  @Nullable
  public <E1 extends Exception, E2 extends Exception> SkyValue getOrThrow(
      SkyKey skyKey, Class<E1> exceptionClass1, Class<E2> exceptionClass2) throws E1, E2 {
    return getOrThrow(skyKey, exceptionClass1, exceptionClass2, null);
  }

  /**
   * Similar to {@link getOrThrow} one exceptionClass version, but take three exceptionClass
   * parameters.
   */
  @Nullable
  public <E1 extends Exception, E2 extends Exception, E3 extends Exception> SkyValue getOrThrow(
      SkyKey skyKey,
      @Nullable Class<E1> exceptionClass1,
      @Nullable Class<E2> exceptionClass2,
      @Nullable Class<E3> exceptionClass3)
      throws E1, E2, E3 {
    ValueOrUntypedException voe = valuesOrExceptions.apply(skyKey);
    SkyValue value = voe.getValue();
    if (value != null) {
      return value;
    }
    Exception e = voe.getException();
    if (e != null) {
      if (exceptionClass1 != null && exceptionClass1.isInstance(e)) {
        throw exceptionClass1.cast(e);
      }
      if (exceptionClass2 != null && exceptionClass2.isInstance(e)) {
        throw exceptionClass2.cast(e);
      }
      if (exceptionClass3 != null && exceptionClass3.isInstance(e)) {
        throw exceptionClass3.cast(e);
      }
    }
    valuesMissingCallback.run();
    return null;
  }
}
