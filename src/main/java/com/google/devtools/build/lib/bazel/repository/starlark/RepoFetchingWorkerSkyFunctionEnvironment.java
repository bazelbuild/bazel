package com.google.devtools.build.lib.bazel.repository.starlark;

import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.SettableFuture;
import com.google.devtools.build.lib.bazel.repository.starlark.RepoFetchingSkyKeyComputeState.Message;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.events.Reportable;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.SkyframeLookupResult;
import com.google.devtools.build.skyframe.Version;
import java.util.concurrent.ExecutionException;
import java.util.function.Supplier;
import javax.annotation.Nullable;

/**
 * A {@link com.google.devtools.build.skyframe.SkyFunction.Environment} implementation whose {@link
 * #getValue} and {@link #getValueOrThrow} methods do not return {@code null} when the {@link
 * SkyValue} in question is not available. Instead, it blocks on a {@link SettableFuture} until the
 * {@link SkyValue} in question is available.
 */
class RepoFetchingWorkerSkyFunctionEnvironment
    implements SkyFunction.Environment, SkyframeLookupResult, ExtendedEventHandler {
  private final RepoFetchingSkyKeyComputeState state;

  RepoFetchingWorkerSkyFunctionEnvironment(RepoFetchingSkyKeyComputeState state) {
    this.state = state;
  }

  @Override
  public boolean valuesMissing() {
    // Contrary to popular belief, I never miss. *wink*
    return false;
  }

  @Nullable
  @Override
  public SkyValue getValue(SkyKey depKey) throws InterruptedException {
    return getValueOrThrow(depKey, null, null, null, null);
  }

  @Nullable
  @Override
  public <E1 extends Exception> SkyValue getValueOrThrow(SkyKey depKey, Class<E1> e1)
      throws E1, InterruptedException {
    return getValueOrThrow(depKey, e1, null, null, null);
  }

  @Nullable
  @Override
  public <E1 extends Exception, E2 extends Exception> SkyValue getValueOrThrow(
      SkyKey depKey, Class<E1> e1, Class<E2> e2) throws E1, E2, InterruptedException {
    return getValueOrThrow(depKey, e1, e2, null, null);
  }

  @Nullable
  @Override
  public <E1 extends Exception, E2 extends Exception, E3 extends Exception>
      SkyValue getValueOrThrow(SkyKey depKey, Class<E1> e1, Class<E2> e2, Class<E3> e3)
          throws E1, E2, E3, InterruptedException {
    return getValueOrThrow(depKey, e1, e2, e3, null);
  }

  @Nullable
  @Override
  public <E1 extends Exception, E2 extends Exception, E3 extends Exception, E4 extends Exception>
      SkyValue getValueOrThrow(
          SkyKey depKey, Class<E1> e1, Class<E2> e2, Class<E3> e3, Class<E4> e4)
          throws E1, E2, E3, E4, InterruptedException {
    SettableFuture<SkyValue> future = SettableFuture.create();
    state.postMessage(new Message.NewSkyframeDependency<>(depKey, e1, e2, e3, e4, future));
    try {
      return future.get();
    } catch (ExecutionException e) {
      SkyFunctionException.throwIfInstanceOf((Exception) e.getCause(), e1, e2, e3, e4);
      throw new IllegalStateException("somehow we have a stray e: " + e.getMessage(), e.getCause());
    }
  }

  @Override
  public SkyframeLookupResult getValuesAndExceptions(Iterable<? extends SkyKey> depKeys) {
    return this;
  }

  @Nullable
  @Override
  public <E1 extends Exception, E2 extends Exception, E3 extends Exception> SkyValue getOrThrow(
      SkyKey skyKey, @Nullable Class<E1> e1, @Nullable Class<E2> e2, @Nullable Class<E3> e3)
      throws E1, E2, E3 {
    try {
      return getValueOrThrow(skyKey, e1, e2, e3);
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      return null;
    }
  }

  @Override
  public boolean queryDep(SkyKey key, QueryDepCallback resultCallback) {
    throw new UnsupportedOperationException("queryDep");
  }

  @Override
  public ExtendedEventHandler getListener() {
    return this;
  }

  private void report(Reportable reportable) {
    try {
      state.postMessage(new Message.Event(reportable));
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
    }
  }

  @Override
  public void handle(Event event) {
    report(event);
  }

  @Override
  public void post(Postable obj) {
    report(obj);
  }

  @Override
  public void registerDependencies(Iterable<SkyKey> keys) {
    throw new UnsupportedOperationException("registerDependencies");
  }

  @Override
  public boolean inErrorBubblingForSkyFunctionsThatCanFullyRecoverFromErrors() {
    throw new UnsupportedOperationException(
        "inErrorBubblingForSkyFunctionsThatCanFullyRecoverFromErrors");
  }

  @Override
  public void dependOnFuture(ListenableFuture<?> future) {
    throw new UnsupportedOperationException("dependOnFuture");
  }

  @Override
  public boolean restartPermitted() {
    return false;
  }

  @Override
  public SkyframeLookupResult getLookupHandleForPreviouslyRequestedDeps() {
    return this;
  }

  @Override
  public <T extends SkyKeyComputeState> T getState(Supplier<T> stateSupplier) {
    throw new UnsupportedOperationException("getState");
  }

  @Nullable
  @Override
  public Version getMaxTransitiveSourceVersionSoFar() {
    throw new UnsupportedOperationException("getMaxTransitiveSourceVersionSoFar");
  }
}
