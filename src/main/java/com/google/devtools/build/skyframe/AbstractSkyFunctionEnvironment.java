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

import com.google.common.util.concurrent.ListenableFuture;
import com.google.errorprone.annotations.ForOverride;
import java.util.ArrayList;
import java.util.List;
import javax.annotation.Nullable;

/**
 * Basic implementation of {@link SkyFunction.Environment} in which all convenience methods delegate
 * to a single abstract method.
 */
abstract class AbstractSkyFunctionEnvironment implements SkyFunction.Environment {

  protected boolean valuesMissing = false;
  @Nullable protected List<ListenableFuture<?>> externalDeps;

  @Override
  @Nullable
  public final SkyValue getValue(SkyKey depKey) throws InterruptedException {
    return getValueOrThrowInternal(depKey, null, null, null, null);
  }

  @Override
  @Nullable
  public final <E extends Exception> SkyValue getValueOrThrow(
      SkyKey depKey, Class<E> exceptionClass) throws E, InterruptedException {
    SkyFunctionException.validateExceptionType(exceptionClass);
    return getValueOrThrowInternal(depKey, exceptionClass, null, null, null);
  }

  @Override
  @Nullable
  public final <E1 extends Exception, E2 extends Exception> SkyValue getValueOrThrow(
      SkyKey depKey, Class<E1> exceptionClass1, Class<E2> exceptionClass2)
      throws E1, E2, InterruptedException {
    SkyFunctionException.validateExceptionType(exceptionClass1);
    SkyFunctionException.validateExceptionType(exceptionClass2);
    return getValueOrThrowInternal(depKey, exceptionClass1, exceptionClass2, null, null);
  }

  @Override
  @Nullable
  public final <E1 extends Exception, E2 extends Exception, E3 extends Exception>
      SkyValue getValueOrThrow(
          SkyKey depKey,
          Class<E1> exceptionClass1,
          Class<E2> exceptionClass2,
          Class<E3> exceptionClass3)
          throws E1, E2, E3, InterruptedException {
    SkyFunctionException.validateExceptionType(exceptionClass1);
    SkyFunctionException.validateExceptionType(exceptionClass2);
    SkyFunctionException.validateExceptionType(exceptionClass3);
    return getValueOrThrowInternal(depKey, exceptionClass1, exceptionClass2, exceptionClass3, null);
  }

  @Override
  @Nullable
  public final <
          E1 extends Exception, E2 extends Exception, E3 extends Exception, E4 extends Exception>
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
    return getValueOrThrowInternal(
        depKey, exceptionClass1, exceptionClass2, exceptionClass3, exceptionClass4);
  }

  @ForOverride
  @Nullable
  abstract <E1 extends Exception, E2 extends Exception, E3 extends Exception, E4 extends Exception>
      SkyValue getValueOrThrowInternal(
          SkyKey depKey,
          @Nullable Class<E1> exceptionClass1,
          @Nullable Class<E2> exceptionClass2,
          @Nullable Class<E3> exceptionClass3,
          @Nullable Class<E4> exceptionClass4)
          throws E1, E2, E3, E4, InterruptedException;

  @Override
  public final boolean valuesMissing() {
    return valuesMissing || externalDeps != null;
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
