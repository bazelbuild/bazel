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

import com.google.common.annotations.VisibleForTesting;

import javax.annotation.Nullable;

/** Utilities for producing and consuming ValueOrException(2|3|4)? instances. */
@VisibleForTesting
public class ValueOrExceptionUtils {

  /** The bottom exception type. */
  class BottomException extends Exception {
  }

  @Nullable
  public static SkyValue downconvert(ValueOrException<BottomException> voe) {
    return voe.getValue();
  }

  public static <E1 extends Exception> ValueOrException<E1> downconvert(
      ValueOrException2<E1, BottomException> voe, Class<E1> exceptionClass1) {
    Exception e = voe.getException();
    if (e == null) {
      return new ValueOrExceptionValueImpl<>(voe.getValue());
    }
    // Here and below, we use type-safe casts for performance reasons. Another approach would be
    // cascading try-catch-rethrow blocks, but that has a higher performance penalty.
    if (exceptionClass1.isInstance(e)) {
      return new ValueOrExceptionExnImpl<>(exceptionClass1.cast(e));
    }
    throw new IllegalStateException("shouldn't reach here " + e.getClass() + " " + exceptionClass1,
        e);
  }

  public static <E1 extends Exception, E2 extends Exception> ValueOrException2<E1, E2> downconvert(
      ValueOrException3<E1, E2, BottomException> voe,
      Class<E1> exceptionClass1,
      Class<E2> exceptionClass2) {
    Exception e = voe.getException();
    if (e == null) {
      return new ValueOrException2ValueImpl<>(voe.getValue());
    }
    if (exceptionClass1.isInstance(e)) {
      return new ValueOrException2Exn1Impl<>(exceptionClass1.cast(e));
    }
    if (exceptionClass2.isInstance(e)) {
      return new ValueOrException2Exn2Impl<>(exceptionClass2.cast(e));
    }
    throw new IllegalStateException("shouldn't reach here " + e.getClass() + " " + exceptionClass1
        + " " + exceptionClass2, e);
  }

  public static <E1 extends Exception, E2 extends Exception, E3 extends Exception>
      ValueOrException3<E1, E2, E3> downconvert(
          ValueOrException4<E1, E2, E3, BottomException> voe,
          Class<E1> exceptionClass1,
          Class<E2> exceptionClass2,
          Class<E3> exceptionClass3) {
    Exception e = voe.getException();
    if (e == null) {
      return new ValueOrException3ValueImpl<>(voe.getValue());
    }
    if (exceptionClass1.isInstance(e)) {
      return new ValueOrException3Exn1Impl<>(exceptionClass1.cast(e));
    }
    if (exceptionClass2.isInstance(e)) {
      return new ValueOrException3Exn2Impl<>(exceptionClass2.cast(e));
    }
    if (exceptionClass3.isInstance(e)) {
      return new ValueOrException3Exn3Impl<>(exceptionClass3.cast(e));
    }
    throw new IllegalStateException("shouldn't reach here " + e.getClass() + " " + exceptionClass1
        + " " + exceptionClass2 + " " + exceptionClass3, e);
  }

  public static <E1 extends Exception, E2 extends Exception, E3 extends Exception,
                 E4 extends Exception>
      ValueOrException4<E1, E2, E3, E4> downconvert(
          ValueOrException5<E1, E2, E3, E4, BottomException> voe,
          Class<E1> exceptionClass1,
          Class<E2> exceptionClass2,
          Class<E3> exceptionClass3,
          Class<E4> exceptionClass4) {
    Exception e = voe.getException();
    if (e == null) {
      return new ValueOrException4ValueImpl<>(voe.getValue());
    }
    if (exceptionClass1.isInstance(e)) {
      return new ValueOrException4Exn1Impl<>(exceptionClass1.cast(e));
    }
    if (exceptionClass2.isInstance(e)) {
      return new ValueOrException4Exn2Impl<>(exceptionClass2.cast(e));
    }
    if (exceptionClass3.isInstance(e)) {
      return new ValueOrException4Exn3Impl<>(exceptionClass3.cast(e));
    }
    if (exceptionClass4.isInstance(e)) {
      return new ValueOrException4Exn4Impl<>(exceptionClass4.cast(e));
    }
    throw new IllegalStateException("shouldn't reach here " + e.getClass() + " " + exceptionClass1
        + " " + exceptionClass2 + " " + exceptionClass3 + " " + exceptionClass4, e);
  }


  public static <E extends Exception> ValueOrException<E> ofNull() {
    return ValueOrExceptionValueImpl.ofNull();
  }

  public static ValueOrUntypedException ofValueUntyped(SkyValue value) {
    return new ValueOrUntypedExceptionImpl(value);
  }

  public static <E extends Exception> ValueOrException<E> ofExn(E e) {
    return new ValueOrExceptionExnImpl<>(e);
  }

  public static <E1 extends Exception, E2 extends Exception, E3 extends Exception,
                 E4 extends Exception, E5 extends Exception>
      ValueOrException5<E1, E2, E3, E4, E5> ofNullValue() {
    return ValueOrException5ValueImpl.ofNullValue();
  }

  public static <E1 extends Exception, E2 extends Exception, E3 extends Exception,
                 E4 extends Exception, E5 extends Exception>
      ValueOrException5<E1, E2, E3, E4, E5> ofValue(SkyValue value) {
    return new ValueOrException5ValueImpl<>(value);
  }

  public static <E1 extends Exception, E2 extends Exception, E3 extends Exception,
                 E4 extends Exception, E5 extends Exception>
      ValueOrException5<E1, E2, E3, E4, E5> ofExn1(E1 e) {
    return new ValueOrException5Exn1Impl<>(e);
  }

  public static <E1 extends Exception, E2 extends Exception, E3 extends Exception,
                 E4 extends Exception, E5 extends Exception>
      ValueOrException5<E1, E2, E3, E4, E5> ofExn2(E2 e) {
    return new ValueOrException5Exn2Impl<>(e);
  }

  public static <E1 extends Exception, E2 extends Exception, E3 extends Exception,
                 E4 extends Exception, E5 extends Exception>
      ValueOrException5<E1, E2, E3, E4, E5> ofExn3(E3 e) {
    return new ValueOrException5Exn3Impl<>(e);
  }

  public static <E1 extends Exception, E2 extends Exception, E3 extends Exception,
                 E4 extends Exception, E5 extends Exception>
      ValueOrException5<E1, E2, E3, E4, E5> ofExn4(E4 e) {
    return new ValueOrException5Exn4Impl<>(e);
  }

  public static <E1 extends Exception, E2 extends Exception, E3 extends Exception,
                 E4 extends Exception, E5 extends Exception>
      ValueOrException5<E1, E2, E3, E4, E5> ofExn5(E5 e) {
    return new ValueOrException5Exn5Impl<>(e);
  }

  private static class ValueOrUntypedExceptionImpl extends ValueOrUntypedException {
    @Nullable
    private final SkyValue value;

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
  }

  private static class ValueOrExceptionValueImpl<E extends Exception> extends ValueOrException<E> {
    private static final ValueOrExceptionValueImpl<Exception> NULL =
        new ValueOrExceptionValueImpl<Exception>((SkyValue) null);

    @Nullable
    private final SkyValue value;

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

    @SuppressWarnings("unchecked")
    public static <E extends Exception> ValueOrExceptionValueImpl<E> ofNull() {
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

  private static class ValueOrException2ValueImpl<E1 extends Exception, E2 extends Exception>
      extends ValueOrException2<E1, E2> {
    @Nullable
    private final SkyValue value;

    ValueOrException2ValueImpl(@Nullable SkyValue value) {
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
  }

  private static class ValueOrException2Exn1Impl<E1 extends Exception, E2 extends Exception>
      extends ValueOrException2<E1, E2> {
    private final E1 e;

    private ValueOrException2Exn1Impl(E1 e) {
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

  private static class ValueOrException2Exn2Impl<E1 extends Exception, E2 extends Exception>
      extends ValueOrException2<E1, E2> {
    private final E2 e;

    private ValueOrException2Exn2Impl(E2 e) {
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

  private static class ValueOrException3ValueImpl<E1 extends Exception, E2 extends Exception,
      E3 extends Exception> extends ValueOrException3<E1, E2, E3> {
    @Nullable
    private final SkyValue value;

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
  }

  private static class ValueOrException3Exn1Impl<E1 extends Exception, E2 extends Exception,
      E3 extends Exception> extends ValueOrException3<E1, E2, E3> {
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

  private static class ValueOrException3Exn2Impl<E1 extends Exception, E2 extends Exception,
      E3 extends Exception> extends ValueOrException3<E1, E2, E3> {
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

  private static class ValueOrException3Exn3Impl<E1 extends Exception, E2 extends Exception,
      E3 extends Exception> extends ValueOrException3<E1, E2, E3> {
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

  private static class ValueOrException4ValueImpl<E1 extends Exception, E2 extends Exception,
      E3 extends Exception, E4 extends Exception> extends ValueOrException4<E1, E2, E3, E4> {
 
    @Nullable
    private final SkyValue value;

    ValueOrException4ValueImpl(@Nullable SkyValue value) {
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
  }

  private static class ValueOrException4Exn1Impl<E1 extends Exception, E2 extends Exception,
      E3 extends Exception, E4 extends Exception> extends ValueOrException4<E1, E2, E3, E4> {
    private final E1 e;

    private ValueOrException4Exn1Impl(E1 e) {
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

  private static class ValueOrException4Exn2Impl<E1 extends Exception, E2 extends Exception,
      E3 extends Exception, E4 extends Exception> extends ValueOrException4<E1, E2, E3, E4> {
    private final E2 e;

    private ValueOrException4Exn2Impl(E2 e) {
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

  private static class ValueOrException4Exn3Impl<E1 extends Exception, E2 extends Exception,
      E3 extends Exception, E4 extends Exception> extends ValueOrException4<E1, E2, E3, E4> {
    private final E3 e;

    private ValueOrException4Exn3Impl(E3 e) {
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

  private static class ValueOrException4Exn4Impl<E1 extends Exception, E2 extends Exception,
      E3 extends Exception, E4 extends Exception> extends ValueOrException4<E1, E2, E3, E4> {
    private final E4 e;

    private ValueOrException4Exn4Impl(E4 e) {
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

  private static class ValueOrException5ValueImpl<E1 extends Exception, E2 extends Exception,
      E3 extends Exception, E4 extends Exception, E5 extends Exception>
      extends ValueOrException5<E1, E2, E3, E4, E5> {
    private static final ValueOrException5ValueImpl<Exception, Exception, Exception,
        Exception, Exception> NULL = new ValueOrException5ValueImpl<>((SkyValue) null);

    @Nullable
    private final SkyValue value;

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
    private static <E1 extends Exception, E2 extends Exception, E3 extends Exception,
                    E4 extends Exception, E5 extends Exception>
        ValueOrException5ValueImpl<E1, E2, E3, E4, E5> ofNullValue() {
      return (ValueOrException5ValueImpl<E1, E2, E3, E4, E5>) NULL;
    }
  }

  private static class ValueOrException5Exn1Impl<E1 extends Exception, E2 extends Exception,
      E3 extends Exception, E4 extends Exception, E5 extends Exception>
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

  private static class ValueOrException5Exn2Impl<E1 extends Exception, E2 extends Exception,
      E3 extends Exception, E4 extends Exception, E5 extends Exception>
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

  private static class ValueOrException5Exn3Impl<E1 extends Exception, E2 extends Exception,
      E3 extends Exception, E4 extends Exception, E5 extends Exception>
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

  private static class ValueOrException5Exn4Impl<E1 extends Exception, E2 extends Exception,
      E3 extends Exception, E4 extends Exception, E5 extends Exception>
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

    private static class ValueOrException5Exn5Impl<E1 extends Exception, E2 extends Exception,
        E3 extends Exception, E4 extends Exception, E5 extends Exception>
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
