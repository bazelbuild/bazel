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
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.util.GroupedList;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * Basic implementation of {@link SkyFunction.Environment} in which all convenience methods delegate
 * to a single abstract method.
 */
@VisibleForTesting
public abstract class AbstractSkyFunctionEnvironment implements SkyFunction.Environment {

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
    return getValueOrThrowHelper(depKey, null, null, null, null);
  }

  @Override
  @Nullable
  public <E extends Exception> SkyValue getValueOrThrow(SkyKey depKey, Class<E> exceptionClass)
      throws E, InterruptedException {
    SkyFunctionException.validateExceptionType(exceptionClass);
    return getValueOrThrowHelper(depKey, exceptionClass, null, null, null);
  }

  @Override
  @Nullable
  public <E1 extends Exception, E2 extends Exception> SkyValue getValueOrThrow(
      SkyKey depKey, Class<E1> exceptionClass1, Class<E2> exceptionClass2)
      throws E1, E2, InterruptedException {
    SkyFunctionException.validateExceptionType(exceptionClass1);
    SkyFunctionException.validateExceptionType(exceptionClass2);
    return getValueOrThrowHelper(depKey, exceptionClass1, exceptionClass2, null, null);
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
    SkyFunctionException.validateExceptionType(exceptionClass1);
    SkyFunctionException.validateExceptionType(exceptionClass2);
    SkyFunctionException.validateExceptionType(exceptionClass3);
    return getValueOrThrowHelper(depKey, exceptionClass1, exceptionClass2, exceptionClass3, null);
  }

  @Override
  @Nullable
  public <E1 extends Exception, E2 extends Exception, E3 extends Exception, E4 extends Exception>
      SkyValue getValueOrThrow(
          SkyKey depKey,
          Class<E1> exceptionClass1,
          Class<E2> exceptionClass2,
          Class<E3> exceptionClass3,
          Class<E4> exceptionClass4)
          throws E1, E2, E3, E4, InterruptedException {
    SkyFunctionException.validateExceptionType(exceptionClass1);
    SkyFunctionException.validateExceptionType(exceptionClass2);
    SkyFunctionException.validateExceptionType(exceptionClass3);
    SkyFunctionException.validateExceptionType(exceptionClass4);
    return getValueOrThrowHelper(
        depKey, exceptionClass1, exceptionClass2, exceptionClass3, exceptionClass4);
  }

  @Nullable
  private <E1 extends Exception, E2 extends Exception, E3 extends Exception, E4 extends Exception>
      SkyValue getValueOrThrowHelper(
          SkyKey depKey,
          @Nullable Class<E1> exceptionClass1,
          @Nullable Class<E2> exceptionClass2,
          @Nullable Class<E3> exceptionClass3,
          @Nullable Class<E4> exceptionClass4)
          throws E1, E2, E3, E4, InterruptedException {
    SkyframeIterableResult result = getOrderedValuesAndExceptions(ImmutableSet.of(depKey));
    return result.nextOrThrow(exceptionClass1, exceptionClass2, exceptionClass3, exceptionClass4);
  }

  @Override
  public SkyframeLookupResult getValuesAndExceptions(Iterable<? extends SkyKey> depKeys)
      throws InterruptedException {
    Map<SkyKey, ValueOrUntypedException> valuesOrExceptions = getValueOrUntypedExceptions(depKeys);
    return new SkyframeLookupResult(() -> valuesMissing = true, valuesOrExceptions::get);
  }

  @Override
  public SkyframeIterableResult getOrderedValuesAndExceptions(Iterable<? extends SkyKey> depKeys)
      throws InterruptedException {
    List<ValueOrUntypedException> valuesOrExceptions = getOrderedValueOrUntypedExceptions(depKeys);
    Iterator<ValueOrUntypedException> valuesOrExceptionsi = valuesOrExceptions.iterator();
    return new SkyframeIterableResult(() -> valuesMissing = true, valuesOrExceptionsi);
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
