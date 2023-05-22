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

package com.google.devtools.build.lib.skyframe.toolchains;

import static com.google.common.collect.ImmutableList.toImmutableList;
import static java.util.stream.Collectors.joining;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.PlatformConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.analysis.config.ConfigMatchingProvider;
import com.google.devtools.build.lib.analysis.config.ToolchainTypeRequirement;
import com.google.devtools.build.lib.analysis.platform.ConstraintCollection;
import com.google.devtools.build.lib.analysis.platform.ConstraintSettingInfo;
import com.google.devtools.build.lib.analysis.platform.DeclaredToolchainInfo;
import com.google.devtools.build.lib.analysis.platform.PlatformInfo;
import com.google.devtools.build.lib.analysis.platform.ToolchainTypeInfo;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import com.google.devtools.build.lib.skyframe.toolchains.PlatformLookupUtil.InvalidPlatformException;
import com.google.devtools.build.lib.skyframe.toolchains.RegisteredToolchainsFunction.InvalidToolchainLabelException;
import com.google.devtools.build.lib.skyframe.toolchains.SingleToolchainResolutionValue.SingleToolchainResolutionKey;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.errorprone.annotations.FormatMethod;
import com.google.errorprone.annotations.FormatString;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import javax.annotation.Nullable;

/** {@link SkyFunction} which performs toolchain resolution for a single toolchain type. */
public class SingleToolchainResolutionFunction implements SkyFunction {

  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws ToolchainResolutionFunctionException, InterruptedException {
    SingleToolchainResolutionKey key = (SingleToolchainResolutionKey) skyKey.argument();

    // This call could be combined with the call below, but this SkyFunction is evaluated so rarely
    // it's not worth optimizing.
    BuildConfigurationValue configuration =
        (BuildConfigurationValue) env.getValue(key.configurationKey());
    if (env.valuesMissing()) {
      return null;
    }

    // Get all toolchains.
    RegisteredToolchainsValue toolchains;
    try {
      toolchains =
          (RegisteredToolchainsValue)
              env.getValueOrThrow(
                  RegisteredToolchainsValue.key(key.configurationKey()),
                  InvalidToolchainLabelException.class);
      if (toolchains == null) {
        return null;
      }
    } catch (InvalidToolchainLabelException e) {
      throw new ToolchainResolutionFunctionException(e);
    }

    // Check if we are debugging the target or the toolchain type.
    boolean debug =
        key.debugTarget()
            || configuration
                .getFragment(PlatformConfiguration.class)
                .debugToolchainResolution(key.toolchainType().toolchainType());

    // Find the right one.
    return resolveConstraints(
        key.toolchainType(),
        key.toolchainTypeInfo(),
        key.availableExecutionPlatformKeys(),
        key.targetPlatformKey(),
        toolchains.registeredToolchains(),
        env,
        debug ? env.getListener() : null);
  }

  /**
   * Given the available execution platforms and toolchains, find the set of platform, toolchain
   * pairs that are compatible a) with each other, and b) with the toolchain type and target
   * platform.
   */
  @Nullable
  private static SingleToolchainResolutionValue resolveConstraints(
      ToolchainTypeRequirement toolchainType,
      ToolchainTypeInfo toolchainTypeInfo,
      List<ConfiguredTargetKey> availableExecutionPlatformKeys,
      ConfiguredTargetKey targetPlatformKey,
      ImmutableList<DeclaredToolchainInfo> toolchains,
      Environment env,
      @Nullable EventHandler eventHandler)
      throws ToolchainResolutionFunctionException, InterruptedException {

    // Load the PlatformInfo needed to check constraints.
    Map<ConfiguredTargetKey, PlatformInfo> platforms;
    try {
      platforms =
          PlatformLookupUtil.getPlatformInfo(
              ImmutableList.<ConfiguredTargetKey>builderWithExpectedSize(
                      availableExecutionPlatformKeys.size() + 1)
                  .add(targetPlatformKey)
                  .addAll(availableExecutionPlatformKeys)
                  .build(),
              env);
      if (env.valuesMissing()) {
        return null;
      }
    } catch (InvalidPlatformException e) {
      throw new ToolchainResolutionFunctionException(e);
    }

    PlatformInfo targetPlatform = platforms.get(targetPlatformKey);

    // Platforms may exist multiple times in availableExecutionPlatformKeys. The Set lets this code
    // check whether a platform has already been seen during processing.
    Set<ConfiguredTargetKey> platformKeysSeen = new HashSet<>();
    ImmutableMap.Builder<ConfiguredTargetKey, Label> builder = ImmutableMap.builder();

    // Pre-filter for the correct toolchain type. This simplifies the loop and makes debugging
    // toolchain resolution much, much easier.
    ImmutableList<DeclaredToolchainInfo> filteredToolchains =
        toolchains.stream()
            .filter(
                toolchain ->
                    toolchain.toolchainType().typeLabel().equals(toolchainType.toolchainType()))
            .collect(toImmutableList());

    for (DeclaredToolchainInfo toolchain : filteredToolchains) {
      // Make sure the target setting matches.
      if (!toolchain.targetSettings().stream().allMatch(ConfigMatchingProvider::matches)) {
        String mismatchValues =
            toolchain.targetSettings().stream()
                .filter(configProvider -> !configProvider.matches())
                .map(configProvider -> configProvider.label().getName())
                .collect(joining(", "));
        debugMessage(
            eventHandler,
            "    Type %s: %s platform %s: Rejected toolchain %s; mismatching config settings: %s",
            toolchainType.toolchainType(),
            "target",
            targetPlatform.label(),
            toolchain.toolchainLabel(),
            mismatchValues);
        continue;
      }

      // Make sure the target platform matches.
      if (!checkConstraints(
          eventHandler,
          toolchain.targetConstraints(),
          "target",
          targetPlatform,
          toolchainType,
          toolchain.toolchainLabel())) {
        continue;
      }

      // Find the matching execution platforms.
      for (ConfiguredTargetKey executionPlatformKey : availableExecutionPlatformKeys) {
        // Only check the toolchains if this is a new platform.
        if (platformKeysSeen.contains(executionPlatformKey)) {
          debugMessage(
              eventHandler,
              "    Type %s: target platform %s: execution platform %s: Skipping toolchain %s;"
                  + " execution platform already has selected toolchain",
              toolchainType.toolchainType(),
              targetPlatform.label(),
              executionPlatformKey.getLabel(),
              toolchain.toolchainLabel());
          continue;
        }

        PlatformInfo executionPlatform = platforms.get(executionPlatformKey);
        if (!checkConstraints(
            eventHandler,
            toolchain.execConstraints(),
            "execution",
            executionPlatform,
            toolchainType,
            toolchain.toolchainLabel())) {
          continue;
        }

        debugMessage(
            eventHandler,
            "  Type %s: target platform %s: execution %s: Selected toolchain %s",
            toolchainType.toolchainType(),
            targetPlatform.label(),
            executionPlatformKey.getLabel(),
            toolchain.toolchainLabel());
        builder.put(executionPlatformKey, toolchain.toolchainLabel());
        platformKeysSeen.add(executionPlatformKey);
      }
    }

    ImmutableMap<ConfiguredTargetKey, Label> resolvedToolchainLabels = builder.buildOrThrow();
    if (resolvedToolchainLabels.isEmpty()) {
      debugMessage(
          eventHandler,
          "  Type %s: target platform %s: No toolchains found.",
          toolchainType.toolchainType(),
          targetPlatform.label());
    }

    return SingleToolchainResolutionValue.create(toolchainTypeInfo, resolvedToolchainLabels);
  }

  /**
   * Helper method to print a debugging message, if the given {@link EventHandler} is not {@code
   * null}.
   */
  @FormatMethod
  private static void debugMessage(
      @Nullable EventHandler eventHandler, @FormatString String template, Object... args) {
    if (eventHandler == null) {
      return;
    }

    eventHandler.handle(Event.info("ToolchainResolution: " + String.format(template, args)));
  }

  /**
   * Returns {@code true} iff all constraints set by the toolchain and in the {@link PlatformInfo}
   * match.
   */
  private static boolean checkConstraints(
      @Nullable EventHandler eventHandler,
      ConstraintCollection toolchainConstraints,
      String platformType,
      PlatformInfo platform,
      ToolchainTypeRequirement toolchainType,
      Label toolchainLabel) {

    // Check every constraint_setting in either the toolchain or the platform.
    ImmutableSet<ConstraintSettingInfo> mismatchSettings =
        toolchainConstraints.diff(platform.constraints());

    // If a constraint_setting has a default_constraint_value, and the platform
    // sets a non-default constraint value for the same constraint_setting, then
    // even toolchains with no reference to that constraint_setting will detect
    // a mismatch here. This manifests as a toolchain resolution failure (#8778).
    //
    // To allow combining rulesets with their own toolchains in a single top-level
    // workspace, toolchains that do not reference a constraint_setting should not
    // be forced to match with it.
    ImmutableSet<ConstraintSettingInfo> mismatchSettingsWithDefault =
        mismatchSettings.stream()
            .filter(toolchainConstraints::hasWithoutDefault)
            .collect(ImmutableSet.toImmutableSet());

    if (!mismatchSettingsWithDefault.isEmpty()) {
      String mismatchValues =
          mismatchSettingsWithDefault.stream()
              .filter(toolchainConstraints::has)
              .map(s -> toolchainConstraints.get(s).label().getName())
              .collect(joining(", "));
      if (!mismatchValues.isEmpty()) {
        mismatchValues = "; mismatching values: " + mismatchValues;
      }

      String missingSettings =
          mismatchSettingsWithDefault.stream()
              .filter(s -> !toolchainConstraints.has(s))
              .map(s -> s.label().getName())
              .collect(joining(", "));
      if (!missingSettings.isEmpty()) {
        missingSettings = "; missing: " + missingSettings;
      }
      debugMessage(
          eventHandler,
          "    Type %s: %s platform %s: Rejected toolchain %s%s%s",
          toolchainType.toolchainType(),
          platformType,
          platform.label(),
          toolchainLabel,
          mismatchValues,
          missingSettings);
    }

    return mismatchSettingsWithDefault.isEmpty();
  }

  /**
   * Used to indicate errors during the computation of an {@link SingleToolchainResolutionValue}.
   */
  private static final class ToolchainResolutionFunctionException extends SkyFunctionException {
    ToolchainResolutionFunctionException(InvalidToolchainLabelException e) {
      super(e, Transience.PERSISTENT);
    }

    ToolchainResolutionFunctionException(InvalidPlatformException e) {
      super(e, Transience.PERSISTENT);
    }
  }
}
