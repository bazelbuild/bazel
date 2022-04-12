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

import com.google.common.base.Throwables;
import javax.annotation.Nullable;

/**
 * Wrapper for a value or the untyped exception thrown when trying to compute it.
 *
 * <p>Sky functions should generally prefer using exception-typed subclasses such as {@link
 * ValueOrException}, {@link ValueOrException2}, etc. Occasionally, it may be useful to work with
 * {@link ValueOrUntypedException} instead, for example to share logic for processing Skyframe
 * requests that use different exception types or arity.
 */
public abstract class ValueOrUntypedException {

  /**
   * Returns the stored value, or {@code null} if it has not yet been computed. If an exception is
   * stored, it is wrapped in {@link IllegalStateException} and thrown.
   *
   * <p>This method is useful when a value is not expected to store an exception but was requested
   * in the same batch as values which may store an exception.
   */
  @Nullable
  public final SkyValue getUnchecked() {
    Exception e = getException();
    if (e != null) {
      throw new IllegalStateException(e);
    }
    return getValue();
  }

  /**
   * Returns the stored value, or {@code null} if it has not yet been computed. If an exception is
   * stored, it is thrown as-is if it is an instance of {@code exceptionClass} and wrapped in {@link
   * IllegalStateException} otherwise.
   *
   * <p>This method is useful when a value is only expected to store {@code exceptionClass} but was
   * requested in the same batch as values which may store other types of exceptions.
   */
  @Nullable
  public final <E extends Exception> SkyValue getOrThrow(Class<E> exceptionClass) throws E {
    Exception e = getException();
    if (e != null) {
      Throwables.throwIfInstanceOf(e, exceptionClass);
      throw new IllegalStateException(e);
    }
    return getValue();
  }

  /** Returns the stored value, if there was one. */
  @Nullable
  abstract SkyValue getValue();

  /** Returns the stored exception, if there was one. */
  @Nullable
  abstract Exception getException();

  public static ValueOrUntypedException ofValueUntyped(SkyValue value) {
    return new ValueOrUntypedExceptionImpl(value);
  }

  public static ValueOrUntypedException ofNull() {
    return ValueOrUntypedExceptionImpl.NULL;
  }

  public static ValueOrUntypedException ofExn(Exception e) {
    return new ValueOrUntypedExceptionExnImpl(e);
  }

  private static final class ValueOrUntypedExceptionImpl extends ValueOrUntypedException {
    static final ValueOrUntypedExceptionImpl NULL = new ValueOrUntypedExceptionImpl(null);
    @Nullable private final SkyValue value;

    ValueOrUntypedExceptionImpl(@Nullable SkyValue value) {
      this.value = value;
    }

    @Override
    @Nullable
    public SkyValue getValue() {
      return value;
    }

    @Override
    public Exception getException() {
      return null;
    }

    @Override
    public String toString() {
      return "ValueOrUntypedExceptionValueImpl:" + value;
    }
  }

  private static final class ValueOrUntypedExceptionExnImpl extends ValueOrUntypedException {
    private final Exception exception;

    ValueOrUntypedExceptionExnImpl(Exception exception) {
      this.exception = exception;
    }

    @Override
    @Nullable
    public SkyValue getValue() {
      return null;
    }

    @Override
    public Exception getException() {
      return exception;
    }
  }
}
