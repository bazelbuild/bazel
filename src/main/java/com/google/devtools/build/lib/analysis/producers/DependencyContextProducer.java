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
package com.google.devtools.build.lib.analysis.producers;

import com.google.devtools.build.lib.analysis.TargetAndConfiguration;
import com.google.devtools.build.lib.analysis.ToolchainCollection;
import com.google.devtools.build.lib.analysis.TransitiveDependencyState;
import com.google.devtools.build.lib.analysis.config.ConfigConditions;
import com.google.devtools.build.lib.analysis.platform.PlatformInfo;
import com.google.devtools.build.lib.skyframe.ConfiguredValueCreationException;
import com.google.devtools.build.lib.skyframe.toolchains.ToolchainException;
import com.google.devtools.build.lib.skyframe.toolchains.UnloadedToolchainContext;
import com.google.devtools.build.skyframe.state.StateMachine;
import javax.annotation.Nullable;

/**
 * This class computes the unloaded toolchain context and {@link ConfigConditions}.
 *
 * <p>It uses {@link PlatformInfo} derived from the unloaded toolchain contexts to compute config
 * conditions, creating a sequential dependency between the two.
 *
 * <p>It's possible to use {@link DependencyContextProducerWithCompatibilityCheck} here instead but
 * that necessarily evaluates {@link ConfigConditions} before computing the unloaded toolchain
 * contexts, which in turn requires evaluating {@link PlatformInfo} in advance. This ordering is
 * necessary because the compatibility check must precede the unloaded toolchain contexts
 * computation.
 *
 * <p>This producer optimizes for the case where no compatibility check is needed and saves memory
 * by using the {@link PlatformInfo} computed as a side effect of the unloaded toolchain contexts.
 */
public final class DependencyContextProducer
    implements StateMachine,
        UnloadedToolchainContextsProducer.ResultSink,
        ConfigConditionsProducer.ResultSink {
  /**
   * Accepts results for both {@link DependencyContextProducer} and {@link
   * DependencyContextProducerWithCompatibilityCheck}.
   */
  public interface ResultSink {
    void acceptDependencyContext(DependencyContext value);

    void acceptDependencyContextError(DependencyContextError error);
  }

  // -------------------- Input --------------------
  private final UnloadedToolchainContextsInputs unloadedToolchainContextsInputs;
  private final TargetAndConfiguration targetAndConfiguration;
  private final TransitiveDependencyState transitiveState;

  // -------------------- Output --------------------
  private final ResultSink sink;

  // -------------------- Internal State --------------------
  @Nullable // Will be null if the target doesn't require toolchain resolution.
  private ToolchainCollection<UnloadedToolchainContext> unloadedToolchainContexts;
  private ConfigConditions configConditions;
  boolean hasError = false;

  public DependencyContextProducer(
      UnloadedToolchainContextsInputs unloadedToolchainContextsInputs,
      TargetAndConfiguration targetAndConfiguration,
      TransitiveDependencyState transitiveState,
      ResultSink sink) {
    this.unloadedToolchainContextsInputs = unloadedToolchainContextsInputs;
    this.targetAndConfiguration = targetAndConfiguration;
    this.transitiveState = transitiveState;
    this.sink = sink;
  }

  @Override
  public StateMachine step(Tasks tasks) {
    return new UnloadedToolchainContextsProducer(
        unloadedToolchainContextsInputs,
        (UnloadedToolchainContextsProducer.ResultSink) this,
        /* runAfter= */ this::computeConfigConditions);
  }

  @Override
  public void acceptUnloadedToolchainContexts(
      @Nullable ToolchainCollection<UnloadedToolchainContext> unloadedToolchainContexts) {
    this.unloadedToolchainContexts = unloadedToolchainContexts;
  }

  @Override
  public void acceptUnloadedToolchainContextsError(ToolchainException error) {
    this.hasError = true;
    sink.acceptDependencyContextError(DependencyContextError.of(error));
  }

  private StateMachine computeConfigConditions(Tasks tasks) {
    if (hasError) {
      return DONE;
    }

    return new ConfigConditionsProducer(
        targetAndConfiguration,
        unloadedToolchainContexts == null ? null : unloadedToolchainContexts.getTargetPlatform(),
        transitiveState,
        (ConfigConditionsProducer.ResultSink) this,
        /* runAfter= */ this::constructResult);
  }

  @Override
  public void acceptConfigConditions(ConfigConditions configConditions) {
    this.configConditions = configConditions;
  }

  @Override
  public void acceptConfigConditionsError(ConfiguredValueCreationException error) {
    this.hasError = true;
    sink.acceptDependencyContextError(DependencyContextError.of(error));
  }

  private StateMachine constructResult(Tasks tasks) {
    if (hasError) {
      return DONE;
    }

    sink.acceptDependencyContext(
        DependencyContext.create(unloadedToolchainContexts, configConditions));
    return DONE;
  }
}
