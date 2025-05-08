// Copyright 2024 The Bazel Authors. All rights reserved.
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

import static com.google.common.base.Preconditions.checkState;

import com.google.devtools.build.skyframe.SkyFunction.LookupEnvironment;
import javax.annotation.Nullable;

/**
 * Used when multiple bazel State Machines' {@link com.google.devtools.build.skyframe.state.Driver}s
 * need to concurrently access {@link LookupEnvironment} and {@link SkyframeLookupResult} to query
 * and fetch dependency values.
 *
 * <p>In the bazel State Machine logic, we assume that {@link LookupEnvironment} and {@link
 * SkyframeLookupResult} refer to the same instance. So {@link ConcurrentSkyFunctionEnvironment}
 * implements methods from both interfaces and all relevant operations are thread-safe.
 */
public final class ConcurrentSkyFunctionEnvironment
    implements LookupEnvironment, SkyframeLookupResult {
  private final SkyFunctionEnvironment delegate;

  public ConcurrentSkyFunctionEnvironment(SkyFunctionEnvironment delegate) {
    this.delegate = delegate;
  }

  @Nullable
  @Override
  public synchronized SkyValue getValue(SkyKey valueName) throws InterruptedException {
    return delegate.getValue(valueName);
  }

  @Nullable
  @Override
  public synchronized <E extends Exception> SkyValue getValueOrThrow(
      SkyKey depKey, Class<E> exceptionClass) throws E, InterruptedException {
    return delegate.getValueOrThrow(depKey, exceptionClass);
  }

  @Nullable
  @Override
  public synchronized <E1 extends Exception, E2 extends Exception> SkyValue getValueOrThrow(
      SkyKey depKey, Class<E1> exceptionClass1, Class<E2> exceptionClass2)
      throws E1, E2, InterruptedException {
    return delegate.getValueOrThrow(depKey, exceptionClass1, exceptionClass2);
  }

  @Nullable
  @Override
  public synchronized <E1 extends Exception, E2 extends Exception, E3 extends Exception>
      SkyValue getValueOrThrow(
          SkyKey depKey,
          Class<E1> exceptionClass1,
          Class<E2> exceptionClass2,
          Class<E3> exceptionClass3)
          throws E1, E2, E3, InterruptedException {
    return delegate.getValueOrThrow(depKey, exceptionClass1, exceptionClass2, exceptionClass3);
  }

  @Nullable
  @Override
  public synchronized <
          E1 extends Exception, E2 extends Exception, E3 extends Exception, E4 extends Exception>
      SkyValue getValueOrThrow(
          SkyKey depKey,
          Class<E1> exceptionClass1,
          Class<E2> exceptionClass2,
          Class<E3> exceptionClass3,
          Class<E4> exceptionClass4)
          throws E1, E2, E3, E4, InterruptedException {
    return delegate.getValueOrThrow(
        depKey, exceptionClass1, exceptionClass2, exceptionClass3, exceptionClass4);
  }

  @Override
  public synchronized SkyframeLookupResult getValuesAndExceptions(
      Iterable<? extends SkyKey> depKeys) throws InterruptedException {
    // When calling `SkyFunctionEnvironment#getValuesAndExceptions`, `delegate` looks up deps in the
    // dependency graph and returns itself as a `SkyframeLookupResult` instance. The `checkState`
    // verifies the returned instance does not change.
    //
    // `getLookupHandleForPreviouslyRequestedDeps` below follows the same intuition.
    checkState(delegate.getValuesAndExceptions(depKeys) == delegate);
    return this;
  }

  @Override
  public synchronized SkyframeLookupResult getLookupHandleForPreviouslyRequestedDeps() {
    checkState(delegate.getLookupHandleForPreviouslyRequestedDeps() == delegate);
    return this;
  }

  @Nullable
  @Override
  public synchronized <E1 extends Exception, E2 extends Exception, E3 extends Exception>
      SkyValue getOrThrow(
          SkyKey skyKey,
          @Nullable Class<E1> exceptionClass1,
          @Nullable Class<E2> exceptionClass2,
          @Nullable Class<E3> exceptionClass3)
          throws E1, E2, E3 {
    return delegate.getOrThrow(skyKey, exceptionClass1, exceptionClass2, exceptionClass3);
  }

  @Override
  public synchronized boolean queryDep(SkyKey key, QueryDepCallback resultCallback) {
    return delegate.queryDep(key, resultCallback);
  }
}
