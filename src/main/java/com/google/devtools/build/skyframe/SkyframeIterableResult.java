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

import static com.google.common.base.Preconditions.checkNotNull;

import java.util.Iterator;
import java.util.NoSuchElementException;
import javax.annotation.Nullable;

/**
 * An iterable-like result of getting Skyframe dependencies via {@link
 * SkyFunction.Environment#getOrderedValuesAndExceptions}. Callers can use the {@link #next} and
 * {@link #nextOrThrow} methods to obtain successive elements.
 *
 * <p>Can only be iterated through once: if an iterable of size k was passed to {@link
 * SkyFunction.Environment#getOrderedValuesAndExceptions}, then the returned {@link
 * SkyframeIterableResult} can have {@link #next} and {@link #nextOrThrow} called a total of k
 * times.
 *
 * <p>Note that a {@link SkyFunction} cannot guarantee that {@link
 * SkyFunction.Environment#valuesMissing} will be true upon receipt of a {@code
 * SkyframeIterableResult}. The elements must be iterated over. If {@link #next} or {@link
 * #nextOrThrow} returns {@code null}, only then will {@link SkyFunction.Environment#valuesMissing}
 * be guaranteed to return true.
 */
public final class SkyframeIterableResult {
  private final Iterator<ValueOrUntypedException> valuesOrExceptions;
  private final Runnable valuesMissingCallback;

  SkyframeIterableResult(
      Runnable valuesMissingCallback, Iterator<ValueOrUntypedException> valuesOrExceptions) {
    this.valuesMissingCallback = checkNotNull(valuesMissingCallback);
    this.valuesOrExceptions = checkNotNull(valuesOrExceptions);
  }

  /**
   * Returns true if {@link SkyframeIterableResult} has more elements. In other words, returns true
   * if {@link #next} or {@link #nextOrThrow} would return an element rather than throwing a {@link
   * NoSuchElementException}.
   */
  public boolean hasNext() {
    return valuesOrExceptions.hasNext();
  }

  /**
   * Returns the next direct dependency in {@link SkyframeIterableResult}. If the dependency is not
   * in the set of already evaluated direct dependencies, returns {@code null}. Also returns {@code
   * null} if the dependency has already been evaluated and found to be in error. In either of these
   * cases, {@link SkyFunction.Environment#valuesMissing} will subsequently return true.
   */
  public SkyValue next() {
    return nextOrThrow(null, null, null);
  }

  /**
   * Returns the next direct dependency in {@link SkyframeIterableResult}. If the dependency is not
   * in the set of already evaluated direct dependencies, returns {@code null}. If the dependency
   * has already been evaluated and found to be in error, throws the exception coming from the
   * error, so long as the exception is of one of the specified types. SkyFunction implementations
   * may use this method to catch and rethrow a more informative exception or to continue
   * evaluation. The caller must specify the exception type(s) that might be thrown using the {@code
   * exceptionClass} argument(s). If the dependency's exception is not an instance of {@code
   * exceptionClass}, {@code null} is returned. In this case, {@link
   * SkyFunction.Environment#valuesMissing} will subsequently return true.
   *
   * <p>The exception class given cannot be a supertype or a subtype of {@link RuntimeException}, or
   * a subtype of {@link InterruptedException}. See {@link
   * SkyFunctionException#validateExceptionType} for details.
   */
  public <E1 extends Exception> SkyValue nextOrThrow(Class<E1> exceptionClass) throws E1 {
    return nextOrThrow(exceptionClass, null, null, null);
  }

  public <E1 extends Exception, E2 extends Exception> SkyValue nextOrThrow(
      Class<E1> exceptionClass1, Class<E2> exceptionClass2) throws E1, E2 {
    return nextOrThrow(exceptionClass1, exceptionClass2, null, null);
  }

  public <E1 extends Exception, E2 extends Exception, E3 extends Exception> SkyValue nextOrThrow(
      Class<E1> exceptionClass1, Class<E2> exceptionClass2, Class<E3> exceptionClass3)
      throws E1, E2, E3 {
    return nextOrThrow(exceptionClass1, exceptionClass2, exceptionClass3, null);
  }

  @Nullable
  public <E1 extends Exception, E2 extends Exception, E3 extends Exception, E4 extends Exception>
      SkyValue nextOrThrow(
          @Nullable Class<E1> exceptionClass1,
          @Nullable Class<E2> exceptionClass2,
          @Nullable Class<E3> exceptionClass3,
          @Nullable Class<E4> exceptionClass4)
          throws E1, E2, E3, E4 {
    ValueOrUntypedException voe = valuesOrExceptions.next();
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
      if (exceptionClass4 != null && exceptionClass4.isInstance(e)) {
        throw exceptionClass4.cast(e);
      }
    }
    valuesMissingCallback.run();
    return null;
  }
}
