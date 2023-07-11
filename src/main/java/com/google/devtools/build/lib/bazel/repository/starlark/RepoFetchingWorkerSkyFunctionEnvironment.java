// Copyright 2023 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.repository.starlark;

import com.google.common.collect.ImmutableList;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.SkyframeLookupResult;
import com.google.devtools.build.skyframe.Version;
import java.util.function.Supplier;
import javax.annotation.Nullable;

/**
 * A {@link SkyFunction.Environment} implementation designed to be used in a different thread than
 * the corresponding SkyFunction runs in. It relies on a delegate Environment object to do
 * underlying work. Its {@link #getValue} and {@link #getValueOrThrow} methods do not return {@code
 * null} when the {@link SkyValue} in question is not available. Instead, it blocks and waits for
 * the host Skyframe thread to restart, and replaces the delegate Environment with a fresh one from
 * the restarted SkyFunction before continuing. (Note that those methods <em>do</em> return {@code
 * null} if the SkyValue was evaluated but found to be in error.)
 *
 * <p>Crucially, the delegate Environment object must not be used by multiple threads at the same
 * time. In effect, this is guaranteed by only one of the worker thread and host thread being active
 * at any given time.
 */
class RepoFetchingWorkerSkyFunctionEnvironment
    implements SkyFunction.Environment, ExtendedEventHandler, SkyframeLookupResult {
  private final RepoFetchingSkyKeyComputeState state;
  private SkyFunction.Environment delegate;

  RepoFetchingWorkerSkyFunctionEnvironment(
      RepoFetchingSkyKeyComputeState state, SkyFunction.Environment delegate) {
    this.state = state;
    this.delegate = delegate;
  }

  @Override
  public boolean valuesMissing() {
    return delegate.valuesMissing();
  }

  @Override
  public SkyframeLookupResult getValuesAndExceptions(Iterable<? extends SkyKey> depKeys)
      throws InterruptedException {
    delegate.getValuesAndExceptions(depKeys);
    if (!delegate.valuesMissing()) {
      // Do NOT just return the return value of `delegate.getValuesAndExceptions` here! That would
      // cause anyone holding onto the returned // result object to potentially use a stale version
      // of it after a skyfunction restart.
      return this;
    }
    // We null out `delegate` before blocking for the fresh env so that the old one becomes
    // eligible for GC.
    delegate = null;
    delegate = state.signalForFreshEnv();
    delegate.getValuesAndExceptions(depKeys);
    return this;
  }

  @Nullable
  @Override
  public <E1 extends Exception, E2 extends Exception, E3 extends Exception> SkyValue getOrThrow(
      SkyKey skyKey, Class<E1> e1, Class<E2> e2, Class<E3> e3) throws E1, E2, E3 {
    return delegate.getLookupHandleForPreviouslyRequestedDeps().getOrThrow(skyKey, e1, e2, e3);
  }

  @Override
  public boolean queryDep(SkyKey key, QueryDepCallback resultCallback) {
    return delegate.getLookupHandleForPreviouslyRequestedDeps().queryDep(key, resultCallback);
  }

  @Nullable
  @Override
  public SkyValue getValue(SkyKey depKey) throws InterruptedException {
    return getValuesAndExceptions(ImmutableList.of(depKey)).get(depKey);
  }

  @Nullable
  @Override
  public <E1 extends Exception> SkyValue getValueOrThrow(SkyKey depKey, Class<E1> e1)
      throws E1, InterruptedException {
    return getValuesAndExceptions(ImmutableList.of(depKey)).getOrThrow(depKey, e1);
  }

  @Nullable
  @Override
  public <E1 extends Exception, E2 extends Exception> SkyValue getValueOrThrow(
      SkyKey depKey, Class<E1> e1, Class<E2> e2) throws E1, E2, InterruptedException {
    return getValuesAndExceptions(ImmutableList.of(depKey)).getOrThrow(depKey, e1, e2);
  }

  @Nullable
  @Override
  public <E1 extends Exception, E2 extends Exception, E3 extends Exception>
      SkyValue getValueOrThrow(SkyKey depKey, Class<E1> e1, Class<E2> e2, Class<E3> e3)
          throws E1, E2, E3, InterruptedException {
    return getValuesAndExceptions(ImmutableList.of(depKey)).getOrThrow(depKey, e1, e2, e3);
  }

  @Nullable
  @Override
  public <E1 extends Exception, E2 extends Exception, E3 extends Exception, E4 extends Exception>
      SkyValue getValueOrThrow(
          SkyKey depKey, Class<E1> e1, Class<E2> e2, Class<E3> e3, Class<E4> e4)
          throws E1, E2, E3, E4, InterruptedException {
    SkyValue value = delegate.getValueOrThrow(depKey, e1, e2, e3, e4);
    if (value != null) {
      return value;
    }
    // We null out `delegate` before blocking for the fresh env so that the old one becomes
    // eligible for GC.
    delegate = null;
    delegate = state.signalForFreshEnv();
    return delegate.getValueOrThrow(depKey, e1, e2, e3, e4);
  }

  @Override
  public ExtendedEventHandler getListener() {
    // Do NOT just return `delegate.getListener()` here! That would cause anyone holding onto the
    // returned listener to potentially post events to a stale listener.
    return this;
  }

  @Override
  public void post(Postable obj) {
    delegate.getListener().post(obj);
  }

  @Override
  public void handle(Event event) {
    delegate.getListener().handle(event);
  }

  @Override
  public void registerDependencies(Iterable<SkyKey> keys) {
    delegate.registerDependencies(keys);
  }

  @Override
  public boolean inErrorBubblingForSkyFunctionsThatCanFullyRecoverFromErrors() {
    return delegate.inErrorBubblingForSkyFunctionsThatCanFullyRecoverFromErrors();
  }

  @Override
  public void dependOnFuture(ListenableFuture<?> future) {
    delegate.dependOnFuture(future);
  }

  @Override
  public boolean restartPermitted() {
    return delegate.restartPermitted();
  }

  @Override
  public SkyframeLookupResult getLookupHandleForPreviouslyRequestedDeps() {
    return delegate.getLookupHandleForPreviouslyRequestedDeps();
  }

  @Override
  public <T extends SkyKeyComputeState> T getState(Supplier<T> stateSupplier) {
    return delegate.getState(stateSupplier);
  }

  @Nullable
  @Override
  public Version getMaxTransitiveSourceVersionSoFar() {
    return delegate.getMaxTransitiveSourceVersionSoFar();
  }
}
