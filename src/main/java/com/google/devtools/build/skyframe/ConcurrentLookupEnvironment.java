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

import com.google.devtools.build.skyframe.SkyFunction.LookupEnvironment;
import javax.annotation.Nullable;

/**
 * Used when {@link SkyFunction#compute} needs to concurrently access {@link LookupEnvironment}.
 *
 * <p>A common use case can be {@link SkyFunction#compute} drives several in-parallel {@link
 * com.google.devtools.build.skyframe.state.StateMachine}s, which could concurrently query deps in
 * {@link LookupEnvironment}.
 */
final class ConcurrentLookupEnvironment implements LookupEnvironment {
  private final LookupEnvironment delegate;

  ConcurrentLookupEnvironment(LookupEnvironment delegate) {
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
    return delegate.getValuesAndExceptions(depKeys);
  }

  @Override
  public synchronized SkyframeLookupResult getLookupHandleForPreviouslyRequestedDeps() {
    return delegate.getLookupHandleForPreviouslyRequestedDeps();
  }
}
