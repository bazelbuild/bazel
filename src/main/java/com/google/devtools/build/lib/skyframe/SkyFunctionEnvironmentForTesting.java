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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.util.ResourceUsage;
import com.google.devtools.build.skyframe.AbstractSkyFunctionEnvironment;
import com.google.devtools.build.skyframe.BuildDriver;
import com.google.devtools.build.skyframe.EvaluationContext;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.ValueOrUntypedException;
import java.util.Map;

/** A skyframe environment that can be used to evaluate arbitrary skykeys for testing. */
public class SkyFunctionEnvironmentForTesting extends AbstractSkyFunctionEnvironment
    implements SkyFunction.Environment {

  private final BuildDriver buildDriver;
  private final ExtendedEventHandler eventHandler;
  private final SkyframeExecutor skyframeExecutor;

  /** Creates a SkyFunctionEnvironmentForTesting that uses a BuildDriver to evaluate skykeys. */
  public SkyFunctionEnvironmentForTesting(
      BuildDriver buildDriver,
      ExtendedEventHandler eventHandler,
      SkyframeExecutor skyframeExecutor) {
    this.buildDriver = buildDriver;
    this.eventHandler = eventHandler;
    this.skyframeExecutor = skyframeExecutor;
  }

  @Override
  protected Map<SkyKey, ValueOrUntypedException> getValueOrUntypedExceptions(
      Iterable<? extends SkyKey> depKeys) throws InterruptedException {
    ImmutableMap.Builder<SkyKey, ValueOrUntypedException> resultMap = ImmutableMap.builder();
    Iterable<SkyKey> keysToEvaluate = ImmutableList.copyOf(depKeys);
    EvaluationContext evaluationContext =
        EvaluationContext.newBuilder()
            .setKeepGoing(true)
            .setNumThreads(ResourceUsage.getAvailableProcessors())
            .setEventHander(eventHandler)
            .build();
    EvaluationResult<SkyValue> evaluationResult =
        skyframeExecutor.evaluateSkyKeys(eventHandler, keysToEvaluate, true);
    buildDriver.evaluate(depKeys, evaluationContext);
    for (SkyKey depKey : ImmutableSet.copyOf(depKeys)) {
      resultMap.put(depKey, ValueOrUntypedException.ofValueUntyped(evaluationResult.get(depKey)));
    }
    return resultMap.build();
  }

  @Override
  public ExtendedEventHandler getListener() {
    return eventHandler;
  }

  @Override
  public boolean inErrorBubblingForTesting() {
    return false;
  }
}
