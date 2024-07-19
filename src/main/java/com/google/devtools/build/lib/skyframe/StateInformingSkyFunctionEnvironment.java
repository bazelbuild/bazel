// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe;

import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.skyframe.GroupedDeps;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.SkyframeLookupResult;
import com.google.devtools.build.skyframe.Version;
import java.util.function.Supplier;
import javax.annotation.Nullable;

/** An environment that wraps each call to its delegate by informing injected {@link Informee}s. */
final class StateInformingSkyFunctionEnvironment implements SkyFunction.Environment {
  private final SkyFunction.Environment delegate;
  private final Informee preFetch;
  private final Informee postFetch;

  StateInformingSkyFunctionEnvironment(
      SkyFunction.Environment delegate, Informee preFetch, Informee postFetch) {
    this.delegate = delegate;
    this.preFetch = preFetch;
    this.postFetch = postFetch;
  }

  @Nullable
  @Override
  public SkyValue getValue(SkyKey valueName) throws InterruptedException {
    preFetch.inform();
    try {
      return delegate.getValue(valueName);
    } finally {
      postFetch.inform();
    }
  }

  @Nullable
  @Override
  public <E extends Exception> SkyValue getValueOrThrow(SkyKey depKey, Class<E> exceptionClass)
      throws E, InterruptedException {
    preFetch.inform();
    try {
      return delegate.getValueOrThrow(depKey, exceptionClass);
    } finally {
      postFetch.inform();
    }
  }

  @Nullable
  @Override
  public <E1 extends Exception, E2 extends Exception> SkyValue getValueOrThrow(
      SkyKey depKey, Class<E1> exceptionClass1, Class<E2> exceptionClass2)
      throws E1, E2, InterruptedException {
    preFetch.inform();
    try {
      return delegate.getValueOrThrow(depKey, exceptionClass1, exceptionClass2);
    } finally {
      postFetch.inform();
    }
  }

  @Nullable
  @Override
  public <E1 extends Exception, E2 extends Exception, E3 extends Exception>
      SkyValue getValueOrThrow(
          SkyKey depKey,
          Class<E1> exceptionClass1,
          Class<E2> exceptionClass2,
          Class<E3> exceptionClass3)
          throws E1, E2, E3, InterruptedException {
    preFetch.inform();
    try {
      return delegate.getValueOrThrow(depKey, exceptionClass1, exceptionClass2, exceptionClass3);
    } finally {
      postFetch.inform();
    }
  }

  @Nullable
  @Override
  public <E1 extends Exception, E2 extends Exception, E3 extends Exception, E4 extends Exception>
      SkyValue getValueOrThrow(
          SkyKey depKey,
          Class<E1> exceptionClass1,
          Class<E2> exceptionClass2,
          Class<E3> exceptionClass3,
          Class<E4> exceptionClass4)
          throws E1, E2, E3, E4, InterruptedException {
    preFetch.inform();
    try {
      return delegate.getValueOrThrow(
          depKey, exceptionClass1, exceptionClass2, exceptionClass3, exceptionClass4);
    } finally {
      postFetch.inform();
    }
  }

  @Override
  public boolean valuesMissing() {
    return delegate.valuesMissing();
  }

  @Override
  public SkyframeLookupResult getValuesAndExceptions(Iterable<? extends SkyKey> depKeys)
      throws InterruptedException {
    preFetch.inform();
    try {
      return delegate.getValuesAndExceptions(depKeys);
    } finally {
      postFetch.inform();
    }
  }

  @Override
  public ExtendedEventHandler getListener() {
    return delegate.getListener();
  }

  @Override
  public boolean inErrorBubbling() {
    return delegate.inErrorBubbling();
  }

  @Nullable
  @Override
  public GroupedDeps getTemporaryDirectDeps() {
    return delegate.getTemporaryDirectDeps();
  }

  @Override
  public void injectVersionForNonHermeticFunction(Version version) {
    delegate.injectVersionForNonHermeticFunction(version);
  }

  @Override
  public void registerDependencies(Iterable<SkyKey> keys) {
    delegate.registerDependencies(keys);
  }

  @Override
  public void dependOnFuture(ListenableFuture<?> future) {
    delegate.dependOnFuture(future);
  }

  @Override
  public SkyframeLookupResult getLookupHandleForPreviouslyRequestedDeps() {
    return delegate.getLookupHandleForPreviouslyRequestedDeps();
  }

  @Override
  public <T extends SkyKeyComputeState> T getState(Supplier<T> stateSupplier) {
    return delegate.getState(stateSupplier);
  }

  interface Informee {
    void inform() throws InterruptedException;
  }

  @Override
  @Nullable
  public Version getMaxTransitiveSourceVersionSoFar() {
    return delegate.getMaxTransitiveSourceVersionSoFar();
  }
}
