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
public abstract class ValueOrException5<E1 extends Exception, E2 extends Exception,
    E3 extends Exception, E4 extends Exception, E5 extends Exception>
    extends ValueOrUntypedException {

  /** Gets the stored value. Throws an exception if one was thrown when computing this value. */
  @Nullable
  public abstract SkyValue get() throws E1, E2, E3, E4, E5;

  static <
          E1 extends Exception,
          E2 extends Exception,
          E3 extends Exception,
          E4 extends Exception,
          E5 extends Exception>
      ValueOrException5<E1, E2, E3, E4, E5> fromUntypedException(
          ValueOrUntypedException voe,
          Class<E1> exceptionClass1,
          Class<E2> exceptionClass2,
          Class<E3> exceptionClass3,
          Class<E4> exceptionClass4,
          Class<E5> exceptionClass5) {
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
      if (exceptionClass4.isInstance(e)) {
        return ofExn4(exceptionClass4.cast(e));
      }
      if (exceptionClass5.isInstance(e)) {
        return ofExn5(exceptionClass5.cast(e));
      }
    }
    return ofNullValue();
  }

  private static <
          E1 extends Exception,
          E2 extends Exception,
          E3 extends Exception,
          E4 extends Exception,
          E5 extends Exception>
      ValueOrException5<E1, E2, E3, E4, E5> ofNullValue() {
    return ValueOrException5ValueImpl.ofNullValue();
  }

  private static <
          E1 extends Exception,
          E2 extends Exception,
          E3 extends Exception,
          E4 extends Exception,
          E5 extends Exception>
      ValueOrException5<E1, E2, E3, E4, E5> ofValue(SkyValue value) {
    return new ValueOrException5ValueImpl<>(value);
  }

  private static <
          E1 extends Exception,
          E2 extends Exception,
          E3 extends Exception,
          E4 extends Exception,
          E5 extends Exception>
      ValueOrException5<E1, E2, E3, E4, E5> ofExn1(E1 e) {
    return new ValueOrException5Exn1Impl<>(e);
  }

  private static <
          E1 extends Exception,
          E2 extends Exception,
          E3 extends Exception,
          E4 extends Exception,
          E5 extends Exception>
      ValueOrException5<E1, E2, E3, E4, E5> ofExn2(E2 e) {
    return new ValueOrException5Exn2Impl<>(e);
  }

  private static <
          E1 extends Exception,
          E2 extends Exception,
          E3 extends Exception,
          E4 extends Exception,
          E5 extends Exception>
      ValueOrException5<E1, E2, E3, E4, E5> ofExn3(E3 e) {
    return new ValueOrException5Exn3Impl<>(e);
  }

  private static <
          E1 extends Exception,
          E2 extends Exception,
          E3 extends Exception,
          E4 extends Exception,
          E5 extends Exception>
      ValueOrException5<E1, E2, E3, E4, E5> ofExn4(E4 e) {
    return new ValueOrException5Exn4Impl<>(e);
  }

  private static <
          E1 extends Exception,
          E2 extends Exception,
          E3 extends Exception,
          E4 extends Exception,
          E5 extends Exception>
      ValueOrException5<E1, E2, E3, E4, E5> ofExn5(E5 e) {
    return new ValueOrException5Exn5Impl<>(e);
  }

  private static class ValueOrException5ValueImpl<
          E1 extends Exception,
          E2 extends Exception,
          E3 extends Exception,
          E4 extends Exception,
          E5 extends Exception>
      extends ValueOrException5<E1, E2, E3, E4, E5> {
    private static final ValueOrException5ValueImpl<
            Exception, Exception, Exception, Exception, Exception>
        NULL = new ValueOrException5ValueImpl<>(null);

    @Nullable private final SkyValue value;

    ValueOrException5ValueImpl(@Nullable SkyValue value) {
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
    static <
            E1 extends Exception,
            E2 extends Exception,
            E3 extends Exception,
            E4 extends Exception,
            E5 extends Exception>
        ValueOrException5ValueImpl<E1, E2, E3, E4, E5> ofNullValue() {
      return (ValueOrException5ValueImpl<E1, E2, E3, E4, E5>) NULL;
    }
  }

  private static class ValueOrException5Exn1Impl<
          E1 extends Exception,
          E2 extends Exception,
          E3 extends Exception,
          E4 extends Exception,
          E5 extends Exception>
      extends ValueOrException5<E1, E2, E3, E4, E5> {
    private final E1 e;

    private ValueOrException5Exn1Impl(E1 e) {
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

  private static class ValueOrException5Exn2Impl<
          E1 extends Exception,
          E2 extends Exception,
          E3 extends Exception,
          E4 extends Exception,
          E5 extends Exception>
      extends ValueOrException5<E1, E2, E3, E4, E5> {
    private final E2 e;

    private ValueOrException5Exn2Impl(E2 e) {
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

  private static class ValueOrException5Exn3Impl<
          E1 extends Exception,
          E2 extends Exception,
          E3 extends Exception,
          E4 extends Exception,
          E5 extends Exception>
      extends ValueOrException5<E1, E2, E3, E4, E5> {
    private final E3 e;

    private ValueOrException5Exn3Impl(E3 e) {
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

  private static class ValueOrException5Exn4Impl<
          E1 extends Exception,
          E2 extends Exception,
          E3 extends Exception,
          E4 extends Exception,
          E5 extends Exception>
      extends ValueOrException5<E1, E2, E3, E4, E5> {
    private final E4 e;

    private ValueOrException5Exn4Impl(E4 e) {
      this.e = e;
    }

    @Override
    public SkyValue get() throws E4 {
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

  private static class ValueOrException5Exn5Impl<
          E1 extends Exception,
          E2 extends Exception,
          E3 extends Exception,
          E4 extends Exception,
          E5 extends Exception>
      extends ValueOrException5<E1, E2, E3, E4, E5> {
    private final E5 e;

    private ValueOrException5Exn5Impl(E5 e) {
      this.e = e;
    }

    @Override
    public SkyValue get() throws E5 {
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
