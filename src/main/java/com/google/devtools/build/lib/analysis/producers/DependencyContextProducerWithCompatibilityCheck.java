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

import com.google.devtools.build.lib.analysis.PlatformConfiguration;
import com.google.devtools.build.lib.analysis.TargetAndConfiguration;
import com.google.devtools.build.lib.analysis.ToolchainCollection;
import com.google.devtools.build.lib.analysis.TransitiveDependencyState;
import com.google.devtools.build.lib.analysis.config.ConfigConditions;
import com.google.devtools.build.lib.analysis.constraints.IncompatibleTargetChecker.IncompatibleTargetException;
import com.google.devtools.build.lib.analysis.constraints.IncompatibleTargetChecker.IncompatibleTargetProducer;
import com.google.devtools.build.lib.analysis.platform.PlatformInfo;
import com.google.devtools.build.lib.packages.ConfiguredAttributeMapper.ValidationException;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import com.google.devtools.build.lib.skyframe.ConfiguredValueCreationException;
import com.google.devtools.build.lib.skyframe.RuleConfiguredTargetValue;
import com.google.devtools.build.lib.skyframe.toolchains.PlatformLookupUtil.InvalidPlatformException;
import com.google.devtools.build.lib.skyframe.toolchains.ToolchainException;
import com.google.devtools.build.lib.skyframe.toolchains.UnloadedToolchainContext;
import com.google.devtools.build.skyframe.state.StateMachine;
import java.util.Optional;
import javax.annotation.Nullable;

/**
 * Computes the {@link DependencyContext} while checking for platform compatibility.
 *
 * <p>See <a href="https://bazel.build/extending/platforms#skipping-incompatible-targets">Skipping
 * Incompatible Targets</a> for more details on platform compatibility.
 */
public final class DependencyContextProducerWithCompatibilityCheck
    implements StateMachine,
        PlatformInfoProducer.ResultSink,
        ConfigConditionsProducer.ResultSink,
        IncompatibleTargetProducer.ResultSink,
        UnloadedToolchainContextsProducer.ResultSink {
  // -------------------- Input --------------------
  private final TargetAndConfiguration targetAndConfiguration;
  private final ConfiguredTargetKey configuredTargetKey;
  private final UnloadedToolchainContextsInputs unloadedToolchainContextsInputs;

  private final TransitiveDependencyState transitiveState;

  // -------------------- Output --------------------
  private final DependencyContextProducer.ResultSink sink;

  // -------------------- Internal State --------------------
  private PlatformInfo targetPlatformInfo;
  private ConfigConditions configConditions;
  @Nullable // Will be null if the target doesn't require toolchain resolution.
  private ToolchainCollection<UnloadedToolchainContext> unloadedToolchainContexts;
  private boolean hasError = false;

  public DependencyContextProducerWithCompatibilityCheck(
      TargetAndConfiguration targetAndConfiguration,
      ConfiguredTargetKey configuredTargetKey,
      UnloadedToolchainContextsInputs unloadedToolchainContextsInputs,
      TransitiveDependencyState transitiveState,
      DependencyContextProducer.ResultSink sink) {
    this.targetAndConfiguration = targetAndConfiguration;
    this.configuredTargetKey = configuredTargetKey;
    this.unloadedToolchainContextsInputs = unloadedToolchainContextsInputs;
    this.transitiveState = transitiveState;
    this.sink = sink;
  }

  @Override
  public StateMachine step(Tasks tasks) {
    var defaultToolchainContextKey = unloadedToolchainContextsInputs.targetToolchainContextKey();
    if (defaultToolchainContextKey == null) {
      // If `defaultToolchainContextKey` is null, there's no platform info, incompatibility check
      // or toolchain resolution. Short-circuits and computes only the ConfigConditions.
      return new ConfigConditionsProducer(
          targetAndConfiguration,
          /* targetPlatformInfo= */ null,
          transitiveState,
          (ConfigConditionsProducer.ResultSink) this,
          /* runAfter= */ this::constructResult);
    }

    // Non-null `defaultToolchainContextKey` guarantees that `platformConfiguration` is non-null.
    var platformConfiguration =
        targetAndConfiguration.getConfiguration().getFragment(PlatformConfiguration.class);
    // Checks for incompatibility before toolchain resolution so that known missing
    // toolchains mark the target incompatible instead of failing the build.
    return new PlatformInfoProducer(
        ConfiguredTargetKey.builder()
            .setLabel(platformConfiguration.getTargetPlatform())
            .setConfigurationKey(defaultToolchainContextKey.configurationKey())
            .build(),
        (PlatformInfoProducer.ResultSink) this,
        /* runAfter= */ this::computeConfigConditions);
  }

  @Override
  public void acceptPlatformInfo(PlatformInfo info) {
    this.targetPlatformInfo = info;
  }

  @Override
  public void acceptPlatformInfoError(InvalidPlatformException error) {
    this.hasError = true;
    sink.acceptDependencyContextError(DependencyContextError.of(error));
  }

  private StateMachine computeConfigConditions(Tasks tasks) {
    if (hasError) {
      return DONE;
    }

    return new ConfigConditionsProducer(
        targetAndConfiguration,
        targetPlatformInfo,
        transitiveState,
        (ConfigConditionsProducer.ResultSink) this,
        /* runAfter= */ this::checkCompatibility);
  }

  // -------------------- ConfigConditionsProducer.ResultSink --------------------
  @Override
  public void acceptConfigConditions(ConfigConditions configConditions) {
    this.configConditions = configConditions;
  }

  @Override
  public void acceptConfigConditionsError(ConfiguredValueCreationException error) {
    this.hasError = true;
    sink.acceptDependencyContextError(DependencyContextError.of(error));
  }

  private StateMachine checkCompatibility(Tasks tasks) {
    if (hasError) {
      return DONE;
    }

    return new IncompatibleTargetProducer(
        targetAndConfiguration,
        configuredTargetKey,
        configConditions,
        targetPlatformInfo,
        transitiveState,
        (IncompatibleTargetProducer.ResultSink) this,
        /* runAfter= */ this::computeUnloadedToolchainContexts);
  }

  @Override
  public void acceptIncompatibleTarget(Optional<RuleConfiguredTargetValue> incompatibleTarget) {
    if (!incompatibleTarget.isEmpty()) {
      this.hasError = true;
      sink.acceptDependencyContextError(
          DependencyContextError.of(new IncompatibleTargetException(incompatibleTarget.get())));
    }
  }

  @Override
  public void acceptValidationException(ValidationException e) {
    this.hasError = true;
    sink.acceptDependencyContextError(DependencyContextError.of(e));
  }

  private StateMachine computeUnloadedToolchainContexts(Tasks tasks) {
    if (hasError) {
      return DONE;
    }

    return new UnloadedToolchainContextsProducer(
        unloadedToolchainContextsInputs,
        (UnloadedToolchainContextsProducer.ResultSink) this,
        /* runAfter= */ this::constructResult);
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

  private StateMachine constructResult(Tasks tasks) {
    if (hasError) {
      return DONE;
    }

    sink.acceptDependencyContext(
        DependencyContext.create(unloadedToolchainContexts, configConditions));
    return DONE;
  }
}
