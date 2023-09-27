// Copyright 2017 The Bazel Authors. All rights reserved.
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

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.collect.Streams.stream;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.skyframe.AbstractSkyFunctionEnvironmentForTesting;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.ValueOrUntypedException;
import com.google.devtools.build.skyframe.Version;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;
import javax.annotation.Nullable;

/**
 * A {@link SkyFunction.Environment} backed by a {@link SkyframeExecutor} that can be used to
 * evaluate arbitrary {@link SkyKey}s for testing.
 */
public final class SkyFunctionEnvironmentForTesting
    extends AbstractSkyFunctionEnvironmentForTesting {

  private final ExtendedEventHandler eventHandler;
  private final SkyframeExecutor skyframeExecutor;

  public SkyFunctionEnvironmentForTesting(
      ExtendedEventHandler eventHandler, SkyframeExecutor skyframeExecutor) {
    this.eventHandler = eventHandler;
    this.skyframeExecutor = skyframeExecutor;
  }

  @Override
  protected ValueOrUntypedException getSingleValueOrUntypedException(SkyKey depKey) {
    return getOrderedValueOrUntypedExceptions(ImmutableList.of(depKey)).get(0);
  }

  @Override
  protected Map<SkyKey, ValueOrUntypedException> getValueOrUntypedExceptions(
      Iterable<? extends SkyKey> depKeys) {
    ImmutableMap.Builder<SkyKey, ValueOrUntypedException> resultMap = ImmutableMap.builder();
    Iterable<SkyKey> keysToEvaluate = ImmutableList.copyOf(depKeys);
    EvaluationResult<SkyValue> evaluationResult =
        skyframeExecutor.evaluateSkyKeys(eventHandler, keysToEvaluate, true);
    for (SkyKey depKey : ImmutableSet.copyOf(depKeys)) {
      resultMap.put(depKey, ValueOrUntypedException.ofValueUntyped(evaluationResult.get(depKey)));
    }
    return resultMap.buildOrThrow();
  }

  @Override
  public ExtendedEventHandler getListener() {
    return eventHandler;
  }

  @Override
  public void registerDependencies(Iterable<SkyKey> keys) {
    throw new UnsupportedOperationException();
  }

  @Override
  protected List<ValueOrUntypedException> getOrderedValueOrUntypedExceptions(
      Iterable<? extends SkyKey> depKeys) {
    EvaluationResult<SkyValue> evaluationResult =
        skyframeExecutor.evaluateSkyKeys(eventHandler, depKeys, true);
    return stream(depKeys)
        .map(evaluationResult::get)
        .map(ValueOrUntypedException::ofValueUntyped)
        .collect(toImmutableList());
  }

  @Override
  public boolean inErrorBubblingForSkyFunctionsThatCanFullyRecoverFromErrors() {
    return false;
  }

  @Override
  public void dependOnFuture(ListenableFuture<?> future) {
    throw new UnsupportedOperationException();
  }

  @Override
  public <T extends SkyKeyComputeState> T getState(Supplier<T> stateSupplier) {
    return stateSupplier.get();
  }

  @Override
  @Nullable
  public Version getMaxTransitiveSourceVersionSoFar() {
    throw new UnsupportedOperationException();
  }
}
