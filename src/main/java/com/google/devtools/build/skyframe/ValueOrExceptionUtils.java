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

/** Utilities for producing and consuming ValueOrException(2|3|4)? instances. */
class ValueOrExceptionUtils {

  /** The bottom exception type. */
  class BottomException extends Exception {
  }

  @Nullable
  public static SkyValue downcovert(ValueOrException<BottomException> voe) {
    return voe.getValue();
  }

  public static <E1 extends Exception> ValueOrException<E1> downcovert(
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
      ValueOrException3<E1, E2, BottomException> voe, Class<E1> exceptionClass1,
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
      ValueOrException3<E1, E2, E3> downconvert(ValueOrException4<E1, E2, E3, BottomException> voe,
          Class<E1> exceptionClass1, Class<E2> exceptionClass2, Class<E3> exceptionClass3) {
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

  public static <E extends Exception> ValueOrException<E> ofNull() {
    return ValueOrExceptionValueImpl.ofNull();
  }

  public static ValueOrUntypedException ofValueUntyped(SkyValue value) {
    return new ValueImplBase(value);
  }

  public static <E extends Exception> ValueOrException<E> ofExn(E e) {
    return new ValueOrExceptionExnImpl<>(e);
  }

  public static <E1 extends Exception, E2 extends Exception, E3 extends Exception,
      E4 extends Exception> ValueOrException4<E1, E2, E3, E4> ofNullValue() {
    return ValueOrException4ValueImpl.ofNullValue();
  }

  public static <E1 extends Exception, E2 extends Exception, E3 extends Exception,
      E4 extends Exception> ValueOrException4<E1, E2, E3, E4> ofValue(SkyValue value) {
    return new ValueOrException4ValueImpl<>(value);
  }

  public static <E1 extends Exception, E2 extends Exception, E3 extends Exception,
      E4 extends Exception> ValueOrException4<E1, E2, E3, E4> ofExn1(E1 e) {
    return new ValueOrException4Exn1Impl<>(e);
  }

  public static <E1 extends Exception, E2 extends Exception, E3 extends Exception,
      E4 extends Exception> ValueOrException4<E1, E2, E3, E4> ofExn2(E2 e) {
    return new ValueOrException4Exn2Impl<>(e);
  }

  public static <E1 extends Exception, E2 extends Exception, E3 extends Exception,
      E4 extends Exception> ValueOrException4<E1, E2, E3, E4> ofExn3(E3 e) {
    return new ValueOrException4Exn3Impl<>(e);
  }

  public static <E1 extends Exception, E2 extends Exception, E3 extends Exception,
      E4 extends Exception> ValueOrException4<E1, E2, E3, E4> ofExn4(E4 e) {
    return new ValueOrException4Exn4Impl<>(e);
  }

  private static class ValueImplBase implements ValueOrUntypedException {

    @Nullable
    private final SkyValue value;

    ValueImplBase(@Nullable SkyValue value) {
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

  private static class ExnImplBase<E extends Exception> implements ValueOrUntypedException {

    private final E e;

    ExnImplBase(E e) {
      this.e = Preconditions.checkNotNull(e);
    }

    @Override
    public SkyValue getValue() {
      return null;
    }

    @Override
    public E getException() {
      return e;
    }
  }

  private static class ValueOrExceptionValueImpl<E extends Exception>
      extends ValueImplBase implements ValueOrException<E> {

    private static final ValueOrExceptionValueImpl<Exception> NULL =
        new ValueOrExceptionValueImpl<Exception>((SkyValue) null);

    private ValueOrExceptionValueImpl(@Nullable SkyValue value) {
      super(value);
    }

    @Override
    @Nullable
    public SkyValue get() {
      return getValue();
    }

    @SuppressWarnings("unchecked")
    public static <E extends Exception> ValueOrExceptionValueImpl<E> ofNull() {
      return (ValueOrExceptionValueImpl<E>) NULL;
    }
  }

  private static class ValueOrExceptionExnImpl<E extends Exception>
      extends ExnImplBase<E> implements ValueOrException<E> {

    private ValueOrExceptionExnImpl(E e) {
      super(e);
    }

    @Override
    public SkyValue get() throws E {
      throw getException();
    }
  }

  private static class ValueOrException2ValueImpl<E1 extends Exception, E2 extends Exception>
      extends ValueImplBase implements ValueOrException2<E1, E2> {

    private ValueOrException2ValueImpl(@Nullable SkyValue value) {
      super(value);
    }

    @Override
    @Nullable
    public SkyValue get() {
      return getValue();
    }
  }

  private static class ValueOrException2Exn1Impl<E1 extends Exception, E2 extends Exception>
      extends ExnImplBase<E1> implements ValueOrException2<E1, E2> {

    private ValueOrException2Exn1Impl(E1 e) {
      super(e);
    }

    @Override
    public SkyValue get() throws E1 {
      throw getException();
    }
  }

  private static class ValueOrException2Exn2Impl<E1 extends Exception, E2 extends Exception>
      extends ExnImplBase<E2> implements ValueOrException2<E1, E2> {

    private ValueOrException2Exn2Impl(E2 e) {
      super(e);
    }

    @Override
    public SkyValue get() throws E2 {
      throw getException();
    }
  }

  private static class ValueOrException3ValueImpl<E1 extends Exception, E2 extends Exception,
      E3 extends Exception> extends ValueImplBase implements ValueOrException3<E1, E2, E3> {

    private ValueOrException3ValueImpl(@Nullable SkyValue value) {
      super(value);
    }

    @Override
    @Nullable
    public SkyValue get() {
      return getValue();
    }
  }

  private static class ValueOrException3Exn1Impl<E1 extends Exception, E2 extends Exception,
      E3 extends Exception> extends ExnImplBase<E1> implements ValueOrException3<E1, E2, E3> {

    private ValueOrException3Exn1Impl(E1 e) {
      super(e);
    }

    @Override
    public SkyValue get() throws E1 {
      throw getException();
    }
  }

  private static class ValueOrException3Exn2Impl<E1 extends Exception, E2 extends Exception,
      E3 extends Exception> extends ExnImplBase<E2> implements ValueOrException3<E1, E2, E3> {

    private ValueOrException3Exn2Impl(E2 e) {
      super(e);
    }

    @Override
    public SkyValue get() throws E2 {
      throw getException();
    }
  }

  private static class ValueOrException3Exn3Impl<E1 extends Exception, E2 extends Exception,
      E3 extends Exception> extends ExnImplBase<E3> implements ValueOrException3<E1, E2, E3> {

    private ValueOrException3Exn3Impl(E3 e) {
      super(e);
    }

    @Override
    public SkyValue get() throws E3 {
      throw getException();
    }
  }

  private static class ValueOrException4ValueImpl<E1 extends Exception, E2 extends Exception,
      E3 extends Exception, E4 extends Exception> extends ValueImplBase
      implements ValueOrException4<E1, E2, E3, E4> {

    private static final ValueOrException4ValueImpl<Exception, Exception, Exception,
        Exception> NULL = new ValueOrException4ValueImpl<>((SkyValue) null);

    private ValueOrException4ValueImpl(@Nullable SkyValue value) {
      super(value);
    }

    @Override
    @Nullable
    public SkyValue get() {
      return getValue();
    }

    @SuppressWarnings("unchecked")
    private static <E1 extends Exception, E2 extends Exception, E3 extends Exception,
        E4 extends Exception>ValueOrException4ValueImpl<E1, E2, E3, E4> ofNullValue() {
      return (ValueOrException4ValueImpl<E1, E2, E3, E4>) NULL;
    }
  }

  private static class ValueOrException4Exn1Impl<E1 extends Exception, E2 extends Exception,
      E3 extends Exception, E4 extends Exception> extends ExnImplBase<E1>
      implements ValueOrException4<E1, E2, E3, E4> {

    private ValueOrException4Exn1Impl(E1 e) {
      super(e);
    }

    @Override
    public SkyValue get() throws E1 {
      throw getException();
    }
  }

  private static class ValueOrException4Exn2Impl<E1 extends Exception, E2 extends Exception,
      E3 extends Exception, E4 extends Exception> extends ExnImplBase<E2>
      implements ValueOrException4<E1, E2, E3, E4> {

    private ValueOrException4Exn2Impl(E2 e) {
      super(e);
    }

    @Override
    public SkyValue get() throws E2 {
      throw getException();
    }
  }

  private static class ValueOrException4Exn3Impl<E1 extends Exception, E2 extends Exception,
      E3 extends Exception, E4 extends Exception> extends ExnImplBase<E3>
      implements ValueOrException4<E1, E2, E3, E4> {

    private ValueOrException4Exn3Impl(E3 e) {
      super(e);
    }

    @Override
    public SkyValue get() throws E3 {
      throw getException();
    }
  }

  private static class ValueOrException4Exn4Impl<E1 extends Exception, E2 extends Exception,
      E3 extends Exception, E4 extends Exception> extends ExnImplBase<E4>
      implements ValueOrException4<E1, E2, E3, E4> {

    private ValueOrException4Exn4Impl(E4 e) {
      super(e);
    }

    @Override
    public SkyValue get() throws E4 {
      throw getException();
    }
  }
}
