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

import javax.annotation.Nullable;

/** Wrapper for a value or the typed exception thrown when trying to compute it. */
public abstract class ValueOrException<E extends Exception> extends ValueOrUntypedException {

  /** Gets the stored value. Throws an exception if one was thrown when computing this value. */
  @Nullable
  public abstract SkyValue get() throws E;

  static <E extends Exception> ValueOrException<E> fromUntypedException(
      ValueOrUntypedException voe, Class<E> exceptionClass) {
    SkyValue value = voe.getValue();
    if (value != null) {
      return ofValue(value);
    }
    Exception e = voe.getException();
    if (e != null) {
      if (exceptionClass.isInstance(e)) {
        return ofExn1(exceptionClass.cast(e));
      }
    }
    return ofNullValue();
  }

  private static <E extends Exception> ValueOrException<E> ofExn1(E e) {
    return new ValueOrExceptionExnImpl<>(e);
  }

  private static <E extends Exception> ValueOrException<E> ofNullValue() {
    return ValueOrExceptionValueImpl.ofNullValue();
  }

  private static <E extends Exception> ValueOrException<E> ofValue(SkyValue value) {
    return new ValueOrExceptionValueImpl<>(value);
  }

  private static class ValueOrExceptionValueImpl<E extends Exception> extends ValueOrException<E> {
    private static final ValueOrExceptionValueImpl<Exception> NULL =
        new ValueOrExceptionValueImpl<>(null);

    @Nullable private final SkyValue value;

    private ValueOrExceptionValueImpl(@Nullable SkyValue value) {
      this.value = value;
    }

    @Override
    @Nullable
    public SkyValue get() {
      return value;
    }

    @Override
    @Nullable
    public SkyValue getValue() {
      return value;
    }

    @Override
    @Nullable
    public Exception getException() {
      return null;
    }

    @Override
    public String toString() {
      return "ValueOrExceptionValueImpl:" + value;
    }

    @SuppressWarnings("unchecked")
    static <E extends Exception> ValueOrExceptionValueImpl<E> ofNullValue() {
      return (ValueOrExceptionValueImpl<E>) NULL;
    }
  }

  private static class ValueOrExceptionExnImpl<E extends Exception> extends ValueOrException<E> {
    private final E e;

    private ValueOrExceptionExnImpl(E e) {
      this.e = e;
    }

    @Override
    public SkyValue get() throws E {
      throw e;
    }

    @Override
    @Nullable
    public SkyValue getValue() {
      return null;
    }

    @Override
    public Exception getException() {
      return e;
    }
  }
}
