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
public abstract class ValueOrException3<E1 extends Exception, E2 extends Exception,
    E3 extends Exception> extends ValueOrUntypedException {

  /** Gets the stored value. Throws an exception if one was thrown when computing this value. */
  @Nullable
  public abstract SkyValue get() throws E1, E2, E3;

  static <E1 extends Exception, E2 extends Exception, E3 extends Exception>
      ValueOrException3<E1, E2, E3> fromUntypedException(
          ValueOrUntypedException voe,
          Class<E1> exceptionClass1,
          Class<E2> exceptionClass2,
          Class<E3> exceptionClass3) {
    SkyValue value = voe.getValue();
    if (value != null) {
      return ofValue(value);
    }
    Exception e = voe.getException();
    if (e != null) {
      if (exceptionClass1.isInstance(e)) {
        return ofExn1(exceptionClass1.cast(e));
      }
      if (exceptionClass2.isInstance(e)) {
        return ofExn2(exceptionClass2.cast(e));
      }
      if (exceptionClass3.isInstance(e)) {
        return ofExn3(exceptionClass3.cast(e));
      }
    }
    return ofNullValue();
  }

  private static <E1 extends Exception, E2 extends Exception, E3 extends Exception>
      ValueOrException3<E1, E2, E3> ofNullValue() {
    return ValueOrException3ValueImpl.ofNullValue();
  }

  private static <E1 extends Exception, E2 extends Exception, E3 extends Exception>
      ValueOrException3<E1, E2, E3> ofValue(SkyValue value) {
    return new ValueOrException3ValueImpl<>(value);
  }

  private static <E1 extends Exception, E2 extends Exception, E3 extends Exception>
      ValueOrException3<E1, E2, E3> ofExn1(E1 e) {
    return new ValueOrException3Exn1Impl<>(e);
  }

  private static <E1 extends Exception, E2 extends Exception, E3 extends Exception>
      ValueOrException3<E1, E2, E3> ofExn2(E2 e) {
    return new ValueOrException3Exn2Impl<>(e);
  }

  private static <E1 extends Exception, E2 extends Exception, E3 extends Exception>
      ValueOrException3<E1, E2, E3> ofExn3(E3 e) {
    return new ValueOrException3Exn3Impl<>(e);
  }

  private static class ValueOrException3ValueImpl<
          E1 extends Exception, E2 extends Exception, E3 extends Exception>
      extends ValueOrException3<E1, E2, E3> {
    private static final ValueOrException3ValueImpl<Exception, Exception, Exception> NULL =
        new ValueOrException3ValueImpl<>(null);

    @Nullable private final SkyValue value;

    ValueOrException3ValueImpl(@Nullable SkyValue value) {
      this.value = value;
    }

    @Override
    @Nullable
    public SkyValue get() {
      return value;
    }

    @Override
    @Nullable
    public Exception getException() {
      return null;
    }

    @Override
    @Nullable
    public SkyValue getValue() {
      return value;
    }

    @SuppressWarnings("unchecked")
    static <E1 extends Exception, E2 extends Exception, E3 extends Exception>
        ValueOrException3ValueImpl<E1, E2, E3> ofNullValue() {
      return (ValueOrException3ValueImpl<E1, E2, E3>) NULL;
    }
  }

  private static class ValueOrException3Exn1Impl<
          E1 extends Exception, E2 extends Exception, E3 extends Exception>
      extends ValueOrException3<E1, E2, E3> {
    private final E1 e;

    private ValueOrException3Exn1Impl(E1 e) {
      this.e = e;
    }

    @Override
    public SkyValue get() throws E1 {
      throw e;
    }

    @Override
    public Exception getException() {
      return e;
    }

    @Override
    @Nullable
    public SkyValue getValue() {
      return null;
    }
  }

  private static class ValueOrException3Exn2Impl<
          E1 extends Exception, E2 extends Exception, E3 extends Exception>
      extends ValueOrException3<E1, E2, E3> {
    private final E2 e;

    private ValueOrException3Exn2Impl(E2 e) {
      this.e = e;
    }

    @Override
    public SkyValue get() throws E2 {
      throw e;
    }

    @Override
    public Exception getException() {
      return e;
    }

    @Override
    @Nullable
    public SkyValue getValue() {
      return null;
    }
  }

  private static class ValueOrException3Exn3Impl<
          E1 extends Exception, E2 extends Exception, E3 extends Exception>
      extends ValueOrException3<E1, E2, E3> {
    private final E3 e;

    private ValueOrException3Exn3Impl(E3 e) {
      this.e = e;
    }

    @Override
    public SkyValue get() throws E3 {
      throw e;
    }

    @Override
    public Exception getException() {
      return e;
    }

    @Override
    @Nullable
    public SkyValue getValue() {
      return null;
    }
  }
}
