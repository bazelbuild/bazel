// Copyright 2022 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.errorprone.annotations.ForOverride;
import java.util.Map;
import java.util.function.Supplier;
import javax.annotation.Nullable;

/**
 * Partial {@link SkyFunction.Environment} implementation that allows tests to deal with simple
 * types like {@link ValueOrUntypedException} and {@link ImmutableMap}.
 */
public abstract class AbstractSkyFunctionEnvironmentForTesting
    extends AbstractSkyFunctionEnvironment {

  /**
   * Gets a map of values or exceptions.
   *
   * <p>Implementations should set {@link #valuesMissing} as necessary.
   */
  @ForOverride
  protected abstract ImmutableMap<SkyKey, ValueOrUntypedException> getValueOrUntypedExceptions(
      Iterable<? extends SkyKey> depKeys) throws InterruptedException;

  @Nullable
  @Override
  final <E1 extends Exception, E2 extends Exception, E3 extends Exception, E4 extends Exception>
      SkyValue getValueOrThrowInternal(
          SkyKey depKey,
          @Nullable Class<E1> exceptionClass1,
          @Nullable Class<E2> exceptionClass2,
          @Nullable Class<E3> exceptionClass3,
          @Nullable Class<E4> exceptionClass4)
          throws E1, E2, E3, E4, InterruptedException {
    ValueOrUntypedException voe = getValueOrUntypedExceptions(ImmutableList.of(depKey)).get(depKey);
    SkyValue value = voe.getValue();
    if (value != null) {
      return value;
    }
    SkyFunctionException.throwIfInstanceOf(
        voe.getException(), exceptionClass1, exceptionClass2, exceptionClass3, exceptionClass4);
    valuesMissing = true;
    return null;
  }

  @Override
  public final SkyframeLookupResult getValuesAndExceptions(Iterable<? extends SkyKey> depKeys)
      throws InterruptedException {
    Map<SkyKey, ValueOrUntypedException> valuesOrExceptions = getValueOrUntypedExceptions(depKeys);
    return new SimpleSkyframeLookupResult(() -> valuesMissing = true, valuesOrExceptions::get);
  }

  @Override
  public SkyframeLookupResult getLookupHandleForPreviouslyRequestedDeps() {
    throw new UnsupportedOperationException();
  }

  @Override
  public void registerDependencies(Iterable<SkyKey> keys) {
    throw new UnsupportedOperationException();
  }

  @Override
  public void dependOnFuture(ListenableFuture<?> future) {
    throw new UnsupportedOperationException();
  }

  @Override
  public Version getMaxTransitiveSourceVersionSoFar() {
    throw new UnsupportedOperationException();
  }

  @Override
  public boolean inErrorBubbling() {
    return false;
  }

  @Override
  public <T extends SkyKeyComputeState> T getState(Supplier<T> stateSupplier) {
    return stateSupplier.get();
  }
}
