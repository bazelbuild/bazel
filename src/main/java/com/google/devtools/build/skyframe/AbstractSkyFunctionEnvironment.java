// Copyright 2015 The Bazel Authors. All rights reserved.
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
import com.google.common.base.Function;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Maps;
import com.google.devtools.build.skyframe.ValueOrExceptionUtils.BottomException;

import java.util.Collections;
import java.util.Map;
import java.util.Set;

import javax.annotation.Nullable;

/**
 * Basic implementation of {@link SkyFunction.Environment} in which all convenience methods delegate
 * to a single abstract method.
 */
@VisibleForTesting
public abstract class AbstractSkyFunctionEnvironment implements SkyFunction.Environment {
  protected boolean valuesMissing = false;
  private <E extends Exception> ValueOrException<E> getValueOrException(SkyKey depKey,
      Class<E> exceptionClass) {
    return ValueOrExceptionUtils.downconvert(
        getValueOrException(depKey, exceptionClass, BottomException.class), exceptionClass);
  }

  private <E1 extends Exception, E2 extends Exception> ValueOrException2<E1, E2>
  getValueOrException(SkyKey depKey, Class<E1> exceptionClass1, Class<E2> exceptionClass2) {
    return ValueOrExceptionUtils.downconvert(getValueOrException(depKey, exceptionClass1,
        exceptionClass2, BottomException.class), exceptionClass1, exceptionClass2);
  }

  private <E1 extends Exception, E2 extends Exception, E3 extends Exception>
      ValueOrException3<E1, E2, E3> getValueOrException(
          SkyKey depKey,
          Class<E1> exceptionClass1,
          Class<E2> exceptionClass2,
          Class<E3> exceptionClass3) {
    return ValueOrExceptionUtils.downconvert(
        getValueOrException(depKey, exceptionClass1, exceptionClass2, exceptionClass3,
            BottomException.class),
        exceptionClass1,
        exceptionClass2,
        exceptionClass3);
  }

  private <E1 extends Exception, E2 extends Exception, E3 extends Exception, E4 extends Exception>
      ValueOrException4<E1, E2, E3, E4> getValueOrException(
          SkyKey depKey,
          Class<E1> exceptionClass1,
          Class<E2> exceptionClass2,
          Class<E3> exceptionClass3,
          Class<E4> exceptionClass4) {
    return ValueOrExceptionUtils.downconvert(
        getValueOrException(depKey, exceptionClass1, exceptionClass2, exceptionClass3,
            exceptionClass4, BottomException.class),
        exceptionClass1,
        exceptionClass2,
        exceptionClass3,
        exceptionClass4);
  }

  private <E1 extends Exception, E2 extends Exception, E3 extends Exception,
           E4 extends Exception, E5 extends Exception>
      ValueOrException5<E1, E2, E3, E4, E5> getValueOrException(
          SkyKey depKey,
          Class<E1> exceptionClass1,
          Class<E2> exceptionClass2,
          Class<E3> exceptionClass3,
          Class<E4> exceptionClass4,
          Class<E5> exceptionClass5) {
    return getValueOrExceptions(
        ImmutableSet.of(depKey),
        exceptionClass1,
        exceptionClass2,
        exceptionClass3,
        exceptionClass4,
        exceptionClass5).get(depKey);
  }

  private <E1 extends Exception, E2 extends Exception, E3 extends Exception,
           E4 extends Exception, E5 extends Exception>
      Map<SkyKey, ValueOrException5<E1, E2, E3, E4, E5>> getValueOrExceptions(
          Set<SkyKey> depKeys,
          Class<E1> exceptionClass1,
          Class<E2> exceptionClass2,
          Class<E3> exceptionClass3,
          Class<E4> exceptionClass4,
          Class<E5> exceptionClass5) {
    SkyFunctionException.validateExceptionType(exceptionClass1);
    SkyFunctionException.validateExceptionType(exceptionClass2);
    SkyFunctionException.validateExceptionType(exceptionClass3);
    SkyFunctionException.validateExceptionType(exceptionClass4);
    SkyFunctionException.validateExceptionType(exceptionClass5);
    Map<SkyKey, ValueOrUntypedException> valueOrExceptions =
        getValueOrUntypedExceptions(depKeys);
    ImmutableMap.Builder<SkyKey, ValueOrException5<E1, E2, E3, E4, E5>> builder =
        ImmutableMap.builder();
    for (SkyKey depKey : depKeys) {
      ValueOrUntypedException voe = valueOrExceptions.get(depKey);
      SkyValue value = voe.getValue();
      if (value != null) {
        builder.put(depKey, ValueOrExceptionUtils.<E1, E2, E3, E4, E5>ofValue(value));
        continue;
      }
      Exception e = voe.getException();
      if (e != null) {
        if (exceptionClass1.isInstance(e)) {
          builder.put(depKey, ValueOrExceptionUtils.<E1, E2, E3, E4, E5>ofExn1(
              exceptionClass1.cast(e)));
          continue;
        }
        if (exceptionClass2.isInstance(e)) {
          builder.put(depKey, ValueOrExceptionUtils.<E1, E2, E3, E4, E5>ofExn2(
              exceptionClass2.cast(e)));
          continue;
        }
        if (exceptionClass3.isInstance(e)) {
          builder.put(depKey, ValueOrExceptionUtils.<E1, E2, E3, E4, E5>ofExn3(
              exceptionClass3.cast(e)));
          continue;
        }
        if (exceptionClass4.isInstance(e)) {
          builder.put(depKey, ValueOrExceptionUtils.<E1, E2, E3, E4, E5>ofExn4(
              exceptionClass4.cast(e)));
          continue;
        }
        if (exceptionClass5.isInstance(e)) {
          builder.put(depKey, ValueOrExceptionUtils.<E1, E2, E3, E4, E5>ofExn5(
              exceptionClass5.cast(e)));
          continue;
        }
      }
      valuesMissing = true;
      builder.put(depKey, ValueOrExceptionUtils.<E1, E2, E3, E4, E5>ofNullValue());
    }
    return builder.build();
  }

  /** Implementations should set {@link #valuesMissing} as necessary. */
  protected abstract Map<SkyKey, ValueOrUntypedException> getValueOrUntypedExceptions(
      Set<SkyKey> depKeys);

  @Override
  @Nullable
  public SkyValue getValue(SkyKey depKey) {
    try {
      return getValueOrThrow(depKey, BottomException.class);
    } catch (BottomException e) {
      throw new IllegalStateException("shouldn't reach here");
    }
  }

  @Override
  @Nullable
  public <E extends Exception> SkyValue getValueOrThrow(SkyKey depKey, Class<E> exceptionClass)
      throws E {
    return getValueOrException(depKey, exceptionClass).get();
  }

  @Override
  @Nullable
  public <E1 extends Exception, E2 extends Exception> SkyValue getValueOrThrow(
      SkyKey depKey,
      Class<E1> exceptionClass1,
      Class<E2> exceptionClass2) throws E1, E2 {
    return getValueOrException(depKey, exceptionClass1, exceptionClass2).get();
  }

  @Override
  @Nullable
  public <E1 extends Exception, E2 extends Exception, E3 extends Exception>
      SkyValue getValueOrThrow(
          SkyKey depKey,
          Class<E1> exceptionClass1,
          Class<E2> exceptionClass2,
          Class<E3> exceptionClass3) throws E1, E2, E3 {
    return getValueOrException(depKey, exceptionClass1, exceptionClass2, exceptionClass3).get();
  }

  @Override
  public <E1 extends Exception, E2 extends Exception, E3 extends Exception, E4 extends Exception>
      SkyValue getValueOrThrow(
          SkyKey depKey,
          Class<E1> exceptionClass1,
          Class<E2> exceptionClass2,
          Class<E3> exceptionClass3,
          Class<E4> exceptionClass4) throws E1, E2, E3, E4 {
    return getValueOrException(
        depKey,
        exceptionClass1,
        exceptionClass2,
        exceptionClass3,
        exceptionClass4).get();
  }

  @Override
  public <E1 extends Exception, E2 extends Exception, E3 extends Exception, E4 extends Exception,
          E5 extends Exception>
      SkyValue getValueOrThrow(
          SkyKey depKey,
          Class<E1> exceptionClass1,
          Class<E2> exceptionClass2,
          Class<E3> exceptionClass3,
          Class<E4> exceptionClass4,
          Class<E5> exceptionClass5
  ) throws E1, E2, E3, E4, E5 {
    return getValueOrException(
        depKey,
        exceptionClass1,
        exceptionClass2,
        exceptionClass3,
        exceptionClass4,
        exceptionClass5).get();
  }

  @Override
  public Map<SkyKey, SkyValue> getValues(Iterable<SkyKey> depKeys) {
    return Maps.transformValues(getValuesOrThrow(depKeys, BottomException.class),
        GET_VALUE_FROM_VOE);
  }

  @Override
  public <E extends Exception> Map<SkyKey, ValueOrException<E>> getValuesOrThrow(
      Iterable<SkyKey> depKeys, Class<E> exceptionClass) {
    return Maps.transformValues(
        getValuesOrThrow(depKeys, exceptionClass, BottomException.class),
        makeSafeDowncastToVOEFunction(exceptionClass));
  }

  @Override
  public <E1 extends Exception, E2 extends Exception>
      Map<SkyKey, ValueOrException2<E1, E2>> getValuesOrThrow(
          Iterable<SkyKey> depKeys,
          Class<E1> exceptionClass1,
          Class<E2> exceptionClass2) {
    return Maps.transformValues(
        getValuesOrThrow(depKeys, exceptionClass1, exceptionClass2, BottomException.class),
        makeSafeDowncastToVOE2Function(exceptionClass1, exceptionClass2));
  }

  @Override
  public <E1 extends Exception, E2 extends Exception, E3 extends Exception>
      Map<SkyKey, ValueOrException3<E1, E2, E3>> getValuesOrThrow(
          Iterable<SkyKey> depKeys,
          Class<E1> exceptionClass1,
          Class<E2> exceptionClass2,
          Class<E3> exceptionClass3) {
    return Maps.transformValues(
        getValuesOrThrow(depKeys, exceptionClass1, exceptionClass2, exceptionClass3,
            BottomException.class),
        makeSafeDowncastToVOE3Function(exceptionClass1, exceptionClass2, exceptionClass3));
  }

  @Override
  public <E1 extends Exception, E2 extends Exception, E3 extends Exception, E4 extends Exception>
      Map<SkyKey, ValueOrException4<E1, E2, E3, E4>> getValuesOrThrow(
          Iterable<SkyKey> depKeys,
          Class<E1> exceptionClass1,
          Class<E2> exceptionClass2,
          Class<E3> exceptionClass3,
          Class<E4> exceptionClass4) {
    return Maps.transformValues(
        getValuesOrThrow(depKeys, exceptionClass1, exceptionClass2, exceptionClass3,
            exceptionClass4, BottomException.class),
        makeSafeDowncastToVOE4Function(exceptionClass1, exceptionClass2, exceptionClass3,
            exceptionClass4));
  }

  @Override
  public <E1 extends Exception, E2 extends Exception, E3 extends Exception,
          E4 extends Exception, E5 extends Exception>
      Map<SkyKey, ValueOrException5<E1, E2, E3, E4, E5>> getValuesOrThrow(
          Iterable<SkyKey> depKeys,
          Class<E1> exceptionClass1,
          Class<E2> exceptionClass2,
          Class<E3> exceptionClass3,
          Class<E4> exceptionClass4,
          Class<E5> exceptionClass5) {
    Set<SkyKey> keys = ImmutableSet.copyOf(depKeys);
    Map<SkyKey, ValueOrException5<E1, E2, E3, E4, E5>> result = getValueOrExceptions(keys,
        exceptionClass1, exceptionClass2, exceptionClass3, exceptionClass4, exceptionClass5);
    return Collections.unmodifiableMap(result);
  }


  @Override
  public boolean valuesMissing() {
    return valuesMissing;
  }

  private static final Function<ValueOrException<BottomException>, SkyValue> GET_VALUE_FROM_VOE =
      new Function<ValueOrException<BottomException>, SkyValue>() {
        @Override
        public SkyValue apply(ValueOrException<BottomException> voe) {
          return ValueOrExceptionUtils.downconvert(voe);
        }
      };

  private static <E extends Exception>
      Function<ValueOrException2<E, BottomException>, ValueOrException<E>>
          makeSafeDowncastToVOEFunction(final Class<E> exceptionClass) {
    return new Function<ValueOrException2<E, BottomException>, ValueOrException<E>>() {
      @Override
      public ValueOrException<E> apply(ValueOrException2<E, BottomException> voe) {
        return ValueOrExceptionUtils.downconvert(voe, exceptionClass);
      }
    };
  }

  private static <E1 extends Exception, E2 extends Exception>
      Function<ValueOrException3<E1, E2, BottomException>, ValueOrException2<E1, E2>>
      makeSafeDowncastToVOE2Function(
          final Class<E1> exceptionClass1,
          final Class<E2> exceptionClass2) {
    return new Function<ValueOrException3<E1, E2, BottomException>, ValueOrException2<E1, E2>>() {
      @Override
      public ValueOrException2<E1, E2> apply(ValueOrException3<E1, E2, BottomException> voe) {
        return ValueOrExceptionUtils.downconvert(voe, exceptionClass1, exceptionClass2);
      }
    };
  }

  private static <E1 extends Exception, E2 extends Exception, E3 extends Exception>
      Function<ValueOrException4<E1, E2, E3, BottomException>, ValueOrException3<E1, E2, E3>>
      makeSafeDowncastToVOE3Function(
          final Class<E1> exceptionClass1,
          final Class<E2> exceptionClass2,
          final Class<E3> exceptionClass3) {
    return new Function<ValueOrException4<E1, E2, E3, BottomException>,
        ValueOrException3<E1, E2, E3>>() {
      @Override
      public ValueOrException3<E1, E2, E3> apply(
          ValueOrException4<E1, E2, E3, BottomException> voe) {
        return ValueOrExceptionUtils.downconvert(
            voe, exceptionClass1, exceptionClass2, exceptionClass3);
      }
    };
  }

  private static <E1 extends Exception, E2 extends Exception, E3 extends Exception,
                  E4 extends Exception>
      Function<ValueOrException5<E1, E2, E3, E4, BottomException>,
          ValueOrException4<E1, E2, E3, E4>>
      makeSafeDowncastToVOE4Function(
          final Class<E1> exceptionClass1,
          final Class<E2> exceptionClass2,
          final Class<E3> exceptionClass3,
          final Class<E4> exceptionClass4) {
    return new Function<ValueOrException5<E1, E2, E3, E4, BottomException>,
                        ValueOrException4<E1, E2, E3, E4>>() {
      @Override
      public ValueOrException4<E1, E2, E3, E4> apply(
          ValueOrException5<E1, E2, E3, E4, BottomException> voe) {
        return ValueOrExceptionUtils.downconvert(
            voe, exceptionClass1, exceptionClass2, exceptionClass3, exceptionClass4);
      }
    };
  }
}
