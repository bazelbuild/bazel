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

import com.google.common.base.Preconditions;

import javax.annotation.Nullable;

/**
 * Wrapper for a value or the exception thrown when trying to build it.
 *
 * <p>This is intended only for use in alternative {@code MemoizingEvaluator} implementations.
 *
 * @param <E> Exception class that may have been thrown when building requested value.
 */
public final class ValueOrException<E extends Throwable> {
  @Nullable private final SkyValue value;
  @Nullable private final E exception;

  /** Gets the stored value. Throws an exception if one was thrown when building this value. */
  @Nullable public SkyValue get() throws E {
    if (exception != null) {
      throw exception;
    }
    return value;
  }

  private ValueOrException(@Nullable SkyValue value) {
    this.value = value;
    this.exception = null;
  }

  private ValueOrException(E exception) {
    this.exception = Preconditions.checkNotNull(exception);
    this.value = null;
  }

  /**
   * Returns a {@code ValueOrException} representing success.
   *
   * <p>This is intended only for use in alternative {@code MemoizingEvaluator} implementations.
   */
  public static <F extends Throwable> ValueOrException<F> ofValue(SkyValue value) {
    return new ValueOrException<>(value);
  }

  /**
   * Returns a {@code ValueOrException} representing failure.
   *
   * <p>This is intended only for use in alternative {@code MemoizingEvaluator} implementations.
   */
  public static <F extends Throwable> ValueOrException<F> ofException(F exception) {
    return new ValueOrException<>(exception);
  }

  @SuppressWarnings("unchecked") // Cast to ValueOrException<F>.
  static <F extends Throwable> ValueOrException<F> ofNull() {
    return (ValueOrException<F>) NULL;
  }

  private static final ValueOrException<Throwable> NULL =
      new ValueOrException<>((SkyValue) null);
}
