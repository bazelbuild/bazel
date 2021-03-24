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
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.flogger.GoogleLogger;
import com.google.common.flogger.StackSize;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.util.GroupedList;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * Basic implementation of {@link SkyFunction.Environment} in which all convenience methods delegate
 * to a single abstract method.
 */
@VisibleForTesting
public abstract class AbstractSkyFunctionEnvironment implements SkyFunction.Environment {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  protected boolean valuesMissing = false;
  // Hack for the common case that there are no errors in the retrieved values. In that case, we
  // don't have to filter out any impermissible exceptions. Hack because we communicate this in an
  // out-of-band way from #getValueOrUntypedExceptions. It's out-of-band because we don't want to
  // incur the garbage overhead of returning a more complex data structure from
  // #getValueOrUntypedExceptions.
  protected boolean errorMightHaveBeenFound = false;
  @Nullable private final GroupedList<SkyKey> temporaryDirectDeps;
  @Nullable protected List<ListenableFuture<?>> externalDeps;

  public AbstractSkyFunctionEnvironment(@Nullable GroupedList<SkyKey> temporaryDirectDeps) {
    this.temporaryDirectDeps = temporaryDirectDeps;
  }

  public AbstractSkyFunctionEnvironment() {
    this(null);
  }

  @Override
  public GroupedList<SkyKey> getTemporaryDirectDeps() {
    return temporaryDirectDeps;
  }

  /** Implementations should set {@link #valuesMissing} as necessary. */
  protected abstract Map<SkyKey, ValueOrUntypedException> getValueOrUntypedExceptions(
      Iterable<? extends SkyKey> depKeys) throws InterruptedException;

  /** Implementations should set {@link #valuesMissing} as necessary. */
  protected abstract List<ValueOrUntypedException> getOrderedValueOrUntypedExceptions(
      Iterable<? extends SkyKey> depKeys) throws InterruptedException;

  @Override
  @Nullable
  public SkyValue getValue(SkyKey depKey) throws InterruptedException {
    return getValues(ImmutableSet.of(depKey)).get(depKey);
  }

  @Override
  @Nullable
  public <E extends Exception> SkyValue getValueOrThrow(SkyKey depKey, Class<E> exceptionClass)
      throws E, InterruptedException {
    return getValuesOrThrow(ImmutableSet.of(depKey), exceptionClass).get(depKey).get();
  }

  @Override
  @Nullable
  public <E1 extends Exception, E2 extends Exception> SkyValue getValueOrThrow(
      SkyKey depKey, Class<E1> exceptionClass1, Class<E2> exceptionClass2)
      throws E1, E2, InterruptedException {
    return getValuesOrThrow(ImmutableSet.of(depKey), exceptionClass1, exceptionClass2)
        .get(depKey)
        .get();
  }

  @Override
  @Nullable
  public <E1 extends Exception, E2 extends Exception, E3 extends Exception>
      SkyValue getValueOrThrow(
          SkyKey depKey,
          Class<E1> exceptionClass1,
          Class<E2> exceptionClass2,
          Class<E3> exceptionClass3)
          throws E1, E2, E3, InterruptedException {
    return getValuesOrThrow(
            ImmutableSet.of(depKey), exceptionClass1, exceptionClass2, exceptionClass3)
        .get(depKey)
        .get();
  }

  @Override
  public <E1 extends Exception, E2 extends Exception, E3 extends Exception, E4 extends Exception>
      SkyValue getValueOrThrow(
          SkyKey depKey,
          Class<E1> exceptionClass1,
          Class<E2> exceptionClass2,
          Class<E3> exceptionClass3,
          Class<E4> exceptionClass4)
          throws E1, E2, E3, E4, InterruptedException {
    return getValuesOrThrow(
            ImmutableSet.of(depKey),
            exceptionClass1,
            exceptionClass2,
            exceptionClass3,
            exceptionClass4)
        .get(depKey)
        .get();
  }

  @Override
  public <
          E1 extends Exception,
          E2 extends Exception,
          E3 extends Exception,
          E4 extends Exception,
          E5 extends Exception>
      SkyValue getValueOrThrow(
          SkyKey depKey,
          Class<E1> exceptionClass1,
          Class<E2> exceptionClass2,
          Class<E3> exceptionClass3,
          Class<E4> exceptionClass4,
          Class<E5> exceptionClass5)
          throws E1, E2, E3, E4, E5, InterruptedException {
    return getValuesOrThrow(
            ImmutableSet.of(depKey),
            exceptionClass1,
            exceptionClass2,
            exceptionClass3,
            exceptionClass4,
            exceptionClass5)
        .get(depKey)
        .get();
  }

  @Override
  public Map<SkyKey, SkyValue> getValues(Iterable<? extends SkyKey> depKeys)
      throws InterruptedException {
    Map<SkyKey, ValueOrUntypedException> valuesOrExceptions = getValueOrUntypedExceptions(depKeys);
    checkValuesMissingBecauseOfFilteredError(valuesOrExceptions, null, null, null, null, null);
    return Collections.unmodifiableMap(
        Maps.transformValues(valuesOrExceptions, ValueOrUntypedException::getValue));
  }

  @Override
  public <E extends Exception> Map<SkyKey, ValueOrException<E>> getValuesOrThrow(
      Iterable<? extends SkyKey> depKeys, Class<E> exceptionClass) throws InterruptedException {
    SkyFunctionException.validateExceptionType(exceptionClass);
    Map<SkyKey, ValueOrUntypedException> valuesOrExceptions = getValueOrUntypedExceptions(depKeys);
    checkValuesMissingBecauseOfFilteredError(
        valuesOrExceptions, exceptionClass, null, null, null, null);
    return Collections.unmodifiableMap(
        Maps.transformValues(
            valuesOrExceptions, voe -> ValueOrException.fromUntypedException(voe, exceptionClass)));
  }

  @Override
  public <E1 extends Exception, E2 extends Exception>
      Map<SkyKey, ValueOrException2<E1, E2>> getValuesOrThrow(
          Iterable<? extends SkyKey> depKeys, Class<E1> exceptionClass1, Class<E2> exceptionClass2)
              throws InterruptedException {
    SkyFunctionException.validateExceptionType(exceptionClass1);
    SkyFunctionException.validateExceptionType(exceptionClass2);
    Map<SkyKey, ValueOrUntypedException> valuesOrExceptions = getValueOrUntypedExceptions(depKeys);
    checkValuesMissingBecauseOfFilteredError(
        valuesOrExceptions, exceptionClass1, exceptionClass2, null, null, null);
    return Collections.unmodifiableMap(
        Maps.transformValues(
            valuesOrExceptions,
            voe -> ValueOrException2.fromUntypedException(voe, exceptionClass1, exceptionClass2)));
  }

  @Override
  public <E1 extends Exception, E2 extends Exception, E3 extends Exception>
      Map<SkyKey, ValueOrException3<E1, E2, E3>> getValuesOrThrow(
          Iterable<? extends SkyKey> depKeys,
          Class<E1> exceptionClass1,
          Class<E2> exceptionClass2,
          Class<E3> exceptionClass3)
              throws InterruptedException {
    SkyFunctionException.validateExceptionType(exceptionClass1);
    SkyFunctionException.validateExceptionType(exceptionClass2);
    SkyFunctionException.validateExceptionType(exceptionClass3);
    Map<SkyKey, ValueOrUntypedException> valuesOrExceptions = getValueOrUntypedExceptions(depKeys);
    checkValuesMissingBecauseOfFilteredError(
        valuesOrExceptions, exceptionClass1, exceptionClass2, exceptionClass3, null, null);
    return Collections.unmodifiableMap(
        Maps.transformValues(
            valuesOrExceptions,
            voe ->
                ValueOrException3.fromUntypedException(
                    voe, exceptionClass1, exceptionClass2, exceptionClass3)));
  }

  @Override
  public <E1 extends Exception, E2 extends Exception, E3 extends Exception, E4 extends Exception>
      Map<SkyKey, ValueOrException4<E1, E2, E3, E4>> getValuesOrThrow(
          Iterable<? extends SkyKey> depKeys,
          Class<E1> exceptionClass1,
          Class<E2> exceptionClass2,
          Class<E3> exceptionClass3,
          Class<E4> exceptionClass4)
           throws InterruptedException {
    SkyFunctionException.validateExceptionType(exceptionClass1);
    SkyFunctionException.validateExceptionType(exceptionClass2);
    SkyFunctionException.validateExceptionType(exceptionClass3);
    SkyFunctionException.validateExceptionType(exceptionClass4);
    Map<SkyKey, ValueOrUntypedException> valuesOrExceptions = getValueOrUntypedExceptions(depKeys);
    checkValuesMissingBecauseOfFilteredError(
        valuesOrExceptions,
        exceptionClass1,
        exceptionClass2,
        exceptionClass3,
        exceptionClass4,
        null);
    return Collections.unmodifiableMap(
        Maps.transformValues(
            valuesOrExceptions,
            voe ->
                ValueOrException4.fromUntypedException(
                    voe, exceptionClass1, exceptionClass2, exceptionClass3, exceptionClass4)));
  }

  @Override
  public <
          E1 extends Exception,
          E2 extends Exception,
          E3 extends Exception,
          E4 extends Exception,
          E5 extends Exception>
      Map<SkyKey, ValueOrException5<E1, E2, E3, E4, E5>> getValuesOrThrow(
          Iterable<? extends SkyKey> depKeys,
          Class<E1> exceptionClass1,
          Class<E2> exceptionClass2,
          Class<E3> exceptionClass3,
          Class<E4> exceptionClass4,
          Class<E5> exceptionClass5)
              throws InterruptedException {
    SkyFunctionException.validateExceptionType(exceptionClass1);
    SkyFunctionException.validateExceptionType(exceptionClass2);
    SkyFunctionException.validateExceptionType(exceptionClass3);
    SkyFunctionException.validateExceptionType(exceptionClass4);
    SkyFunctionException.validateExceptionType(exceptionClass5);
    Map<SkyKey, ValueOrUntypedException> valuesOrExceptions = getValueOrUntypedExceptions(depKeys);
    checkValuesMissingBecauseOfFilteredError(
        valuesOrExceptions,
        exceptionClass1,
        exceptionClass2,
        exceptionClass3,
        exceptionClass4,
        exceptionClass5);
    return Collections.unmodifiableMap(
        Maps.transformValues(
            valuesOrExceptions,
            voe ->
                ValueOrException5.fromUntypedException(
                    voe,
                    exceptionClass1,
                    exceptionClass2,
                    exceptionClass3,
                    exceptionClass4,
                    exceptionClass5)));
  }

  private <
          E1 extends Exception,
          E2 extends Exception,
          E3 extends Exception,
          E4 extends Exception,
          E5 extends Exception>
      void checkValuesMissingBecauseOfFilteredError(
          Map<SkyKey, ValueOrUntypedException> voes,
          @Nullable Class<E1> exceptionClass1,
          @Nullable Class<E2> exceptionClass2,
          @Nullable Class<E3> exceptionClass3,
          @Nullable Class<E4> exceptionClass4,
          @Nullable Class<E5> exceptionClass5) {
    checkValuesMissingBecauseOfFilteredError(
        voes.values(),
        exceptionClass1,
        exceptionClass2,
        exceptionClass3,
        exceptionClass4,
        exceptionClass5);
  }

  private <
          E1 extends Exception,
          E2 extends Exception,
          E3 extends Exception,
          E4 extends Exception,
          E5 extends Exception>
      void checkValuesMissingBecauseOfFilteredError(
          Collection<ValueOrUntypedException> voes,
          @Nullable Class<E1> exceptionClass1,
          @Nullable Class<E2> exceptionClass2,
          @Nullable Class<E3> exceptionClass3,
          @Nullable Class<E4> exceptionClass4,
          @Nullable Class<E5> exceptionClass5) {
    if (!errorMightHaveBeenFound) {
      // Short-circuit in the common case of no errors.
      return;
    }
    for (ValueOrUntypedException voe : voes) {
      SkyValue value = voe.getValue();
      if (value == null) {
        Exception e = voe.getException();
        if (e == null
            || ((exceptionClass1 == null || !exceptionClass1.isInstance(e))
                && (exceptionClass2 == null || !exceptionClass2.isInstance(e))
                && (exceptionClass3 == null || !exceptionClass3.isInstance(e))
                && (exceptionClass4 == null || !exceptionClass4.isInstance(e))
                && (exceptionClass5 == null || !exceptionClass5.isInstance(e)))) {
          valuesMissing = true;
          // TODO(b/166268889): Remove when debugged.
          if (e instanceof IOException) {
            logger.atInfo().withStackTrace(StackSize.SMALL).withCause(e).log(
                "IOException suppressed by lack of Skyframe declaration (%s %s %s %s %s)",
                exceptionClass1,
                exceptionClass2,
                exceptionClass3,
                exceptionClass4,
                exceptionClass5);
          }
          return;
        }
      }
    }
  }

  @Override
  public List<SkyValue> getOrderedValues(Iterable<? extends SkyKey> depKeys)
      throws InterruptedException {
    List<ValueOrUntypedException> valuesOrExceptions = getOrderedValueOrUntypedExceptions(depKeys);
    checkValuesMissingBecauseOfFilteredError(valuesOrExceptions, null, null, null, null, null);
    return Collections.unmodifiableList(
        Lists.transform(valuesOrExceptions, ValueOrUntypedException::getValue));
  }

  @Override
  public <E extends Exception> List<ValueOrException<E>> getOrderedValuesOrThrow(
      Iterable<? extends SkyKey> depKeys, Class<E> exceptionClass) throws InterruptedException {
    SkyFunctionException.validateExceptionType(exceptionClass);
    List<ValueOrUntypedException> valuesOrExceptions = getOrderedValueOrUntypedExceptions(depKeys);
    checkValuesMissingBecauseOfFilteredError(
        valuesOrExceptions, exceptionClass, null, null, null, null);
    return Collections.unmodifiableList(
        Lists.transform(
            valuesOrExceptions, voe -> ValueOrException.fromUntypedException(voe, exceptionClass)));
  }

  @Override
  public <E1 extends Exception, E2 extends Exception>
      List<ValueOrException2<E1, E2>> getOrderedValuesOrThrow(
          Iterable<? extends SkyKey> depKeys, Class<E1> exceptionClass1, Class<E2> exceptionClass2)
          throws InterruptedException {
    SkyFunctionException.validateExceptionType(exceptionClass1);
    SkyFunctionException.validateExceptionType(exceptionClass2);
    List<ValueOrUntypedException> valuesOrExceptions = getOrderedValueOrUntypedExceptions(depKeys);
    checkValuesMissingBecauseOfFilteredError(
        valuesOrExceptions, exceptionClass1, exceptionClass2, null, null, null);
    return Collections.unmodifiableList(
        Lists.transform(
            valuesOrExceptions,
            voe -> ValueOrException2.fromUntypedException(voe, exceptionClass1, exceptionClass2)));
  }

  @Override
  public <E1 extends Exception, E2 extends Exception, E3 extends Exception>
      List<ValueOrException3<E1, E2, E3>> getOrderedValuesOrThrow(
          Iterable<? extends SkyKey> depKeys,
          Class<E1> exceptionClass1,
          Class<E2> exceptionClass2,
          Class<E3> exceptionClass3)
          throws InterruptedException {
    SkyFunctionException.validateExceptionType(exceptionClass1);
    SkyFunctionException.validateExceptionType(exceptionClass2);
    SkyFunctionException.validateExceptionType(exceptionClass3);
    List<ValueOrUntypedException> valuesOrExceptions = getOrderedValueOrUntypedExceptions(depKeys);
    checkValuesMissingBecauseOfFilteredError(
        valuesOrExceptions, exceptionClass1, exceptionClass2, exceptionClass3, null, null);
    return Collections.unmodifiableList(
        Lists.transform(
            valuesOrExceptions,
            voe ->
                ValueOrException3.fromUntypedException(
                    voe, exceptionClass1, exceptionClass2, exceptionClass3)));
  }

  @Override
  public <E1 extends Exception, E2 extends Exception, E3 extends Exception, E4 extends Exception>
      List<ValueOrException4<E1, E2, E3, E4>> getOrderedValuesOrThrow(
          Iterable<? extends SkyKey> depKeys,
          Class<E1> exceptionClass1,
          Class<E2> exceptionClass2,
          Class<E3> exceptionClass3,
          Class<E4> exceptionClass4)
          throws InterruptedException {
    SkyFunctionException.validateExceptionType(exceptionClass1);
    SkyFunctionException.validateExceptionType(exceptionClass2);
    SkyFunctionException.validateExceptionType(exceptionClass3);
    SkyFunctionException.validateExceptionType(exceptionClass4);
    List<ValueOrUntypedException> valuesOrExceptions = getOrderedValueOrUntypedExceptions(depKeys);
    checkValuesMissingBecauseOfFilteredError(
        valuesOrExceptions,
        exceptionClass1,
        exceptionClass2,
        exceptionClass3,
        exceptionClass4,
        null);
    return Collections.unmodifiableList(
        Lists.transform(
            valuesOrExceptions,
            voe ->
                ValueOrException4.fromUntypedException(
                    voe, exceptionClass1, exceptionClass2, exceptionClass3, exceptionClass4)));
  }

  @Override
  public <
          E1 extends Exception,
          E2 extends Exception,
          E3 extends Exception,
          E4 extends Exception,
          E5 extends Exception>
      List<ValueOrException5<E1, E2, E3, E4, E5>> getOrderedValuesOrThrow(
          Iterable<? extends SkyKey> depKeys,
          Class<E1> exceptionClass1,
          Class<E2> exceptionClass2,
          Class<E3> exceptionClass3,
          Class<E4> exceptionClass4,
          Class<E5> exceptionClass5)
          throws InterruptedException {
    SkyFunctionException.validateExceptionType(exceptionClass1);
    SkyFunctionException.validateExceptionType(exceptionClass2);
    SkyFunctionException.validateExceptionType(exceptionClass3);
    SkyFunctionException.validateExceptionType(exceptionClass4);
    SkyFunctionException.validateExceptionType(exceptionClass5);
    List<ValueOrUntypedException> valuesOrExceptions = getOrderedValueOrUntypedExceptions(depKeys);
    checkValuesMissingBecauseOfFilteredError(
        valuesOrExceptions,
        exceptionClass1,
        exceptionClass2,
        exceptionClass3,
        exceptionClass4,
        exceptionClass5);
    return Collections.unmodifiableList(
        Lists.transform(
            valuesOrExceptions,
            voe ->
                ValueOrException5.fromUntypedException(
                    voe,
                    exceptionClass1,
                    exceptionClass2,
                    exceptionClass3,
                    exceptionClass4,
                    exceptionClass5)));
  }

  @Override
  public boolean valuesMissing() {
    return valuesMissing || (externalDeps != null);
  }

  @Override
  public void dependOnFuture(ListenableFuture<?> future) {
    if (future.isDone()) {
      // No need to track a dependency on something that's already done.
      return;
    }
    if (externalDeps == null) {
      externalDeps = new ArrayList<>();
    }
    externalDeps.add(future);
  }
}
