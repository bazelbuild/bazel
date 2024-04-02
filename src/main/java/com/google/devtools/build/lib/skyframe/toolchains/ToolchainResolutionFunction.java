// Copyright 2019 The Bazel Authors. All rights reserved.
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
import static com.google.common.collect.ImmutableSet.toImmutableSet;
import static java.util.stream.Collectors.joining;

import com.google.auto.value.AutoValue;
import com.google.common.base.Preconditions;
import com.google.common.collect.HashBasedTable;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSetMultimap;
import com.google.common.collect.Table;
import com.google.devtools.build.lib.analysis.PlatformConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.analysis.config.CommonOptions;
import com.google.devtools.build.lib.analysis.config.ToolchainTypeRequirement;
import com.google.devtools.build.lib.analysis.platform.ConstraintValueInfo;
import com.google.devtools.build.lib.analysis.platform.PlatformInfo;
import com.google.devtools.build.lib.analysis.platform.ToolchainTypeInfo;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.server.FailureDetails.Toolchain.Code;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import com.google.devtools.build.lib.skyframe.config.BuildConfigurationKey;
import com.google.devtools.build.lib.skyframe.toolchains.ConstraintValueLookupUtil.InvalidConstraintValueException;
import com.google.devtools.build.lib.skyframe.toolchains.PlatformLookupUtil.InvalidPlatformException;
import com.google.devtools.build.lib.skyframe.toolchains.RegisteredExecutionPlatformsFunction.InvalidExecutionPlatformLabelException;
import com.google.devtools.build.lib.skyframe.toolchains.RegisteredToolchainsFunction.InvalidToolchainLabelException;
import com.google.devtools.build.lib.skyframe.toolchains.SingleToolchainResolutionFunction.InvalidConfigurationDuringToolchainResolutionException;
import com.google.devtools.build.lib.skyframe.toolchains.SingleToolchainResolutionValue.SingleToolchainResolutionKey;
import com.google.devtools.build.lib.skyframe.toolchains.ToolchainTypeLookupUtil.InvalidToolchainTypeException;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyframeLookupResult;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * Sky function which performs toolchain resolution for multiple toolchain types, including
 * selecting the execution platform.
 */
public class ToolchainResolutionFunction implements SkyFunction {

  @Nullable
  @Override
  public UnloadedToolchainContext compute(SkyKey skyKey, Environment env)
      throws ToolchainResolutionFunctionException, InterruptedException {
    ToolchainContextKey key = (ToolchainContextKey) skyKey.argument();

    try {
      UnloadedToolchainContextImpl.Builder builder = UnloadedToolchainContextImpl.builder(key);

      // Determine the configuration being used.
      BuildConfigurationValue configuration =
          (BuildConfigurationValue) env.getValue(key.configurationKey());
      if (configuration == null) {
        throw new ValueMissingException();
      }
      PlatformConfiguration platformConfiguration =
          Preconditions.checkNotNull(configuration.getFragment(PlatformConfiguration.class));

      // Check if debug output should be generated.
      boolean debug =
          key.debugTarget()
              || configuration
                  .getFragment(PlatformConfiguration.class)
                  .debugToolchainResolution(
                      key.toolchainTypes().stream()
                          .map(ToolchainTypeRequirement::toolchainType)
                          .collect(toImmutableSet()));

      // Create keys for all platforms that will be used, and validate them early.
      // Do this early, to catch platform errors early.
      PlatformKeys platformKeys =
          loadPlatformKeys(
              env,
              debug,
              configuration.getKey(),
              platformConfiguration,
              key.execConstraintLabels());
      if (env.valuesMissing()) {
        return null;
      }

      // Load the configured target for the toolchain types to ensure that they are valid and
      // resolve aliases.
      ImmutableMap<Label, ToolchainTypeInfo> resolvedToolchainTypeInfos =
          loadToolchainTypeInfos(
              env,
              configuration,
              key.toolchainTypes().stream()
                  .map(ToolchainTypeRequirement::toolchainType)
                  .collect(toImmutableSet()));
      builder.setRequestedLabelToToolchainType(resolvedToolchainTypeInfos);
      ImmutableSet<ToolchainType> resolvedToolchainTypes =
          loadToolchainTypes(resolvedToolchainTypeInfos, key.toolchainTypes());

      // Determine the actual toolchain implementations to use.
      determineToolchainImplementations(
          env,
          key.configurationKey(),
          resolvedToolchainTypes,
          key.forceExecutionPlatform().map(platformKeys::find),
          builder,
          platformKeys,
          key.debugTarget());

      UnloadedToolchainContext unloadedToolchainContext = builder.build();
      if (debug) {
        String selectedToolchains =
            unloadedToolchainContext.toolchainTypeToResolved().entries().stream()
                .map(
                    e ->
                        String.format(
                            "type %s -> toolchain %s", e.getKey().typeLabel(), e.getValue()))
                .collect(joining(", "));
        env.getListener()
            .handle(
                Event.info(
                    String.format(
                        "ToolchainResolution: Target platform %s: Selected execution platform %s,"
                            + " %s",
                        unloadedToolchainContext.targetPlatform().label(),
                        unloadedToolchainContext.executionPlatform().label(),
                        selectedToolchains)));
      }
      return unloadedToolchainContext;
    } catch (ToolchainException e) {
      throw new ToolchainResolutionFunctionException(e);
    } catch (ValueMissingException e) {
      return null;
    }
  }

  @AutoValue
  abstract static class ToolchainType {
    abstract ToolchainTypeRequirement toolchainTypeRequirement();

    abstract ToolchainTypeInfo toolchainTypeInfo();

    static ToolchainType create(
        ToolchainTypeRequirement toolchainTypeRequirement, ToolchainTypeInfo toolchainTypeInfo) {
      return new AutoValue_ToolchainResolutionFunction_ToolchainType(
          toolchainTypeRequirement, toolchainTypeInfo);
    }

    public boolean mandatory() {
      return toolchainTypeRequirement().mandatory();
    }
  }

  /**
   * Returns a map from the requested toolchain type Label (after any alias chains) to the {@link
   * ToolchainTypeInfo} provider.
   */
  private static ImmutableMap<Label, ToolchainTypeInfo> loadToolchainTypeInfos(
      Environment environment,
      BuildConfigurationValue configuration,
      ImmutableSet<Label> toolchainTypeLabels)
      throws InvalidToolchainTypeException, InterruptedException, ValueMissingException {
    ImmutableSet<ConfiguredTargetKey> toolchainTypeKeys =
        toolchainTypeLabels.stream()
            .map(
                label ->
                    ConfiguredTargetKey.builder()
                        .setLabel(label)
                        .setConfiguration(configuration)
                        .build())
            .collect(toImmutableSet());

    ImmutableMap<Label, ToolchainTypeInfo> resolvedToolchainTypes =
        ToolchainTypeLookupUtil.resolveToolchainTypes(environment, toolchainTypeKeys);
    if (environment.valuesMissing()) {
      throw new ValueMissingException();
    }
    return resolvedToolchainTypes;
  }

  /**
   * Returns a map from the actual post-alias Label to the ToolchainTypeRequirement for that type.
   */
  private ImmutableSet<ToolchainType> loadToolchainTypes(
      ImmutableMap<Label, ToolchainTypeInfo> resolvedToolchainTypeInfos,
      ImmutableSet<ToolchainTypeRequirement> toolchainTypes) {
    ImmutableSet.Builder<ToolchainType> resolved = new ImmutableSet.Builder<>();

    for (ToolchainTypeRequirement toolchainTypeRequirement : toolchainTypes) {
      // Find the actual Label.
      Label toolchainTypeLabel = toolchainTypeRequirement.toolchainType();
      ToolchainTypeInfo toolchainTypeInfo = resolvedToolchainTypeInfos.get(toolchainTypeLabel);
      if (toolchainTypeInfo != null) {
        toolchainTypeLabel = toolchainTypeInfo.typeLabel();
      }

      // If the labels don't match, re-build the TTR.
      if (!toolchainTypeLabel.equals(toolchainTypeRequirement.toolchainType())) {
        toolchainTypeRequirement =
            toolchainTypeRequirement.toBuilder().toolchainType(toolchainTypeLabel).build();
      }

      resolved.add(ToolchainType.create(toolchainTypeRequirement, toolchainTypeInfo));
    }
    return resolved.build();
  }

  @AutoValue
  abstract static class PlatformKeys {
    abstract ConfiguredTargetKey hostPlatformKey();

    abstract ConfiguredTargetKey targetPlatformKey();

    abstract ImmutableList<ConfiguredTargetKey> executionPlatformKeys();

    @Nullable
    public ConfiguredTargetKey find(Label platformLabel) {
      if (platformLabel.equals(hostPlatformKey().getLabel())) {
        return hostPlatformKey();
      }
      if (platformLabel.equals(targetPlatformKey().getLabel())) {
        return targetPlatformKey();
      }

      for (ConfiguredTargetKey configuredTargetKey : executionPlatformKeys()) {
        if (platformLabel.equals(configuredTargetKey.getLabel())) {
          return configuredTargetKey;
        }
      }

      return null;
    }

    static PlatformKeys create(
        ConfiguredTargetKey hostPlatformKey,
        ConfiguredTargetKey targetPlatformKey,
        List<ConfiguredTargetKey> executionPlatformKeys) {
      return new AutoValue_ToolchainResolutionFunction_PlatformKeys(
          hostPlatformKey, targetPlatformKey, ImmutableList.copyOf(executionPlatformKeys));
    }
  }

  private static PlatformKeys loadPlatformKeys(
      SkyFunction.Environment environment,
      boolean debug,
      BuildConfigurationKey configurationKey,
      PlatformConfiguration platformConfiguration,
      ImmutableSet<Label> execConstraintLabels)
      throws InterruptedException,
          ValueMissingException,
          InvalidConstraintValueException,
          InvalidPlatformException,
          InvalidExecutionPlatformLabelException {
    // Determine the target and host platform keys.
    Label hostPlatformLabel = platformConfiguration.getHostPlatform();
    Label targetPlatformLabel = platformConfiguration.getTargetPlatform();

    ConfiguredTargetKey hostPlatformKey =
        ConfiguredTargetKey.builder()
            .setLabel(hostPlatformLabel)
            .setConfigurationKey(BuildConfigurationKey.create(CommonOptions.EMPTY_OPTIONS))
            .build();
    ConfiguredTargetKey targetPlatformKey =
        ConfiguredTargetKey.builder()
            .setLabel(targetPlatformLabel)
            .setConfigurationKey(BuildConfigurationKey.create(CommonOptions.EMPTY_OPTIONS))
            .build();

    // Load the host and target platforms early, to check for errors.
    var unused =
        PlatformLookupUtil.getPlatformInfo(
            ImmutableList.of(hostPlatformKey, targetPlatformKey), environment);
    if (environment.valuesMissing()) {
      throw new ValueMissingException();
    }

    ImmutableList<ConfiguredTargetKey> executionPlatformKeys =
        loadExecutionPlatformKeys(
            environment,
            debug,
            configurationKey,
            hostPlatformKey,
            execConstraintLabels);

    return PlatformKeys.create(hostPlatformKey, targetPlatformKey, executionPlatformKeys);
  }

  private static ImmutableList<ConfiguredTargetKey> loadExecutionPlatformKeys(
      SkyFunction.Environment environment,
      boolean debug,
      BuildConfigurationKey configurationKey,
      ConfiguredTargetKey defaultPlatformKey,
      ImmutableSet<Label> execConstraintLabels)
      throws InterruptedException,
          ValueMissingException,
          InvalidConstraintValueException,
          InvalidPlatformException,
          InvalidExecutionPlatformLabelException {
    RegisteredExecutionPlatformsValue registeredExecutionPlatforms =
        (RegisteredExecutionPlatformsValue)
            environment.getValueOrThrow(
                RegisteredExecutionPlatformsValue.key(configurationKey),
                InvalidPlatformException.class,
                InvalidExecutionPlatformLabelException.class);
    if (registeredExecutionPlatforms == null) {
      throw new ValueMissingException();
    }

    ImmutableList<ConfiguredTargetKey> availableExecutionPlatformKeys =
        new ImmutableList.Builder<ConfiguredTargetKey>()
            .addAll(registeredExecutionPlatforms.registeredExecutionPlatformKeys())
            .add(defaultPlatformKey)
            .build();

    // Filter out execution platforms that don't satisfy the extra constraints.
    ImmutableList<ConfiguredTargetKey> execConstraintKeys =
        execConstraintLabels.stream()
            .map(
                label ->
                    ConfiguredTargetKey.builder()
                        .setLabel(label)
                        .setConfigurationKey(
                            BuildConfigurationKey.create(CommonOptions.EMPTY_OPTIONS))
                        .build())
            .collect(toImmutableList());

    return filterAvailablePlatforms(
        environment, debug, availableExecutionPlatformKeys, execConstraintKeys);
  }

  /** Returns only the platform keys that match the given constraints. */
  private static ImmutableList<ConfiguredTargetKey> filterAvailablePlatforms(
      SkyFunction.Environment environment,
      boolean debug,
      ImmutableList<ConfiguredTargetKey> platformKeys,
      ImmutableList<ConfiguredTargetKey> constraintKeys)
      throws InterruptedException,
          ValueMissingException,
          InvalidConstraintValueException,
          InvalidPlatformException {

    // Short circuit if not needed.
    if (constraintKeys.isEmpty()) {
      return platformKeys;
    }

    // At this point the host and target platforms have been loaded, but not necessarily the chosen
    // execution platform (it might be the same as the host platform, and might not).
    //
    // It's not worth trying to optimize away this call, since in the optimizable case (the exec
    // platform is the host platform), Skyframe will return the correct results immediately without
    // need of a restart.
    Map<ConfiguredTargetKey, PlatformInfo> platformInfoMap =
        PlatformLookupUtil.getPlatformInfo(platformKeys, environment);
    if (platformInfoMap == null) {
      throw new ValueMissingException();
    }
    List<ConstraintValueInfo> constraints =
        ConstraintValueLookupUtil.getConstraintValueInfo(constraintKeys, environment);
    if (constraints == null) {
      throw new ValueMissingException();
    }

    return platformKeys.stream()
        .filter(key -> filterPlatform(environment, debug, platformInfoMap.get(key), constraints))
        .collect(toImmutableList());
  }

  /** Returns {@code true} if the given platform has all of the constraints. */
  private static boolean filterPlatform(
      SkyFunction.Environment environment,
      boolean debug,
      PlatformInfo platformInfo,
      List<ConstraintValueInfo> constraints) {
    ImmutableList<ConstraintValueInfo> missingConstraints =
        platformInfo.constraints().findMissing(constraints);
    if (debug) {
      for (ConstraintValueInfo constraint : missingConstraints) {
        // The value for this setting is not present in the platform, or doesn't match the expected
        // value.
        environment
            .getListener()
            .handle(
                Event.info(
                    String.format(
                        "ToolchainResolution: Removed execution platform %s from"
                            + " available execution platforms, it is missing constraint %s",
                        platformInfo.label(), constraint.label())));
      }
    }

    return missingConstraints.isEmpty();
  }

  private static void determineToolchainImplementations(
      Environment environment,
      BuildConfigurationKey configurationKey,
      ImmutableSet<ToolchainType> toolchainTypes,
      Optional<ConfiguredTargetKey> forcedExecutionPlatform,
      UnloadedToolchainContextImpl.Builder builder,
      PlatformKeys platformKeys,
      boolean debugTarget)
      throws InterruptedException,
          ValueMissingException,
          InvalidPlatformException,
          UnresolvedToolchainsException,
          InvalidToolchainLabelException,
          InvalidConfigurationDuringToolchainResolutionException {

    // Find the toolchains for the requested toolchain types.
    List<SingleToolchainResolutionKey> registeredToolchainKeys = new ArrayList<>();
    for (ToolchainType toolchainType : toolchainTypes) {
      registeredToolchainKeys.add(
          SingleToolchainResolutionValue.key(
              configurationKey,
              toolchainType.toolchainTypeRequirement(),
              toolchainType.toolchainTypeInfo(),
              platformKeys.targetPlatformKey(),
              platformKeys.executionPlatformKeys(),
              debugTarget));
    }

    SkyframeLookupResult results = environment.getValuesAndExceptions(registeredToolchainKeys);
    boolean valuesMissing = false;

    // Determine the potential set of toolchains.
    Table<ConfiguredTargetKey, ToolchainTypeInfo, Label> resolvedToolchains =
        HashBasedTable.create();
    List<Label> missingMandatoryToolchains = new ArrayList<>();
    for (SingleToolchainResolutionKey key : registeredToolchainKeys) {
      SingleToolchainResolutionValue singleToolchainResolutionValue =
          (SingleToolchainResolutionValue)
              results.getOrThrow(
                  key,
                  InvalidToolchainLabelException.class,
                  InvalidConfigurationDuringToolchainResolutionException.class);
      if (singleToolchainResolutionValue == null) {
        valuesMissing = true;
        continue;
      }

      if (!singleToolchainResolutionValue.availableToolchainLabels().isEmpty()) {
        ToolchainTypeInfo requiredToolchainType = singleToolchainResolutionValue.toolchainType();
        resolvedToolchains.putAll(
            findPlatformsAndLabels(requiredToolchainType, singleToolchainResolutionValue));
      } else if (key.toolchainType().mandatory()) {
        // Save the missing type and continue looping to check for more.
        missingMandatoryToolchains.add(key.toolchainType().toolchainType());
      }
      // TODO(katre): track missing optional toolchains?
    }

    // Verify that all mandatory toolchain types have a toolchain.
    if (!missingMandatoryToolchains.isEmpty()) {
      throw new UnresolvedToolchainsException(missingMandatoryToolchains);
    }

    if (valuesMissing) {
      throw new ValueMissingException();
    }

    // Find and return the first execution platform which has all mandatory toolchains.
    Optional<ConfiguredTargetKey> selectedExecutionPlatformKey =
        findExecutionPlatformForToolchains(
            toolchainTypes,
            forcedExecutionPlatform,
            platformKeys.executionPlatformKeys(),
            resolvedToolchains);

    ImmutableSet<ToolchainTypeRequirement> toolchainTypeRequirements =
        toolchainTypes.stream()
            .map(ToolchainType::toolchainTypeRequirement)
            .collect(toImmutableSet());
    if (selectedExecutionPlatformKey.isEmpty()) {
      builder.setToolchainTypes(toolchainTypeRequirements);
      builder.setExecutionPlatform(PlatformInfo.EMPTY_PLATFORM_INFO);
      builder.setTargetPlatform(PlatformInfo.EMPTY_PLATFORM_INFO);
      builder.setErrorData(
          NoMatchingPlatformData.builder()
              .setToolchainTypes(toolchainTypeRequirements)
              .setAvailableExecutionPlatformKeys(platformKeys.executionPlatformKeys())
              .setTargetPlatformKey(platformKeys.targetPlatformKey())
              .build());
      return;
    }

    Map<ConfiguredTargetKey, PlatformInfo> platforms =
        PlatformLookupUtil.getPlatformInfo(
            ImmutableList.of(selectedExecutionPlatformKey.get(), platformKeys.targetPlatformKey()),
            environment);
    if (platforms == null) {
      throw new ValueMissingException();
    }

    builder.setToolchainTypes(toolchainTypeRequirements);
    builder.setExecutionPlatform(platforms.get(selectedExecutionPlatformKey.get()));
    builder.setTargetPlatform(platforms.get(platformKeys.targetPlatformKey()));

    Map<ToolchainTypeInfo, Label> toolchains =
        resolvedToolchains.row(selectedExecutionPlatformKey.get());
    builder.setToolchainTypeToResolved(ImmutableSetMultimap.copyOf(toolchains.entrySet()));
  }

  /**
   * Adds all of toolchain labels from {@code toolchainResolutionValue} to {@code
   * resolvedToolchains}.
   */
  private static Table<ConfiguredTargetKey, ToolchainTypeInfo, Label> findPlatformsAndLabels(
      ToolchainTypeInfo requiredToolchainType,
      SingleToolchainResolutionValue singleToolchainResolutionValue) {

    Table<ConfiguredTargetKey, ToolchainTypeInfo, Label> resolvedToolchains =
        HashBasedTable.create();
    for (Map.Entry<ConfiguredTargetKey, Label> entry :
        singleToolchainResolutionValue.availableToolchainLabels().entrySet()) {
      resolvedToolchains.put(entry.getKey(), requiredToolchainType, entry.getValue());
    }
    return resolvedToolchains;
  }

  /**
   * Finds the first platform from {@code availableExecutionPlatformKeys} that is present in {@code
   * resolvedToolchains} and has all required toolchain types.
   */
  private static Optional<ConfiguredTargetKey> findExecutionPlatformForToolchains(
      ImmutableSet<ToolchainType> toolchainTypes,
      Optional<ConfiguredTargetKey> forcedExecutionPlatform,
      ImmutableList<ConfiguredTargetKey> availableExecutionPlatformKeys,
      Table<ConfiguredTargetKey, ToolchainTypeInfo, Label> resolvedToolchains) {

    if (forcedExecutionPlatform.isPresent()) {
      // Is the forced platform suitable?
      if (isPlatformSuitable(forcedExecutionPlatform.get(), toolchainTypes, resolvedToolchains)) {
        return forcedExecutionPlatform;
      }
    }

    var candidatePlatforms =
        availableExecutionPlatformKeys.stream()
            .filter(epk -> isPlatformSuitable(epk, toolchainTypes, resolvedToolchains));

    var toolchainTypeInfos =
        toolchainTypes.stream().map(ToolchainType::toolchainTypeInfo).collect(toImmutableSet());

    // Sort by the number of toolchains (the sort is stable)
    return candidatePlatforms.max(
        Comparator.comparingLong(
            epk -> countToolchainsOnPlatform(epk, toolchainTypeInfos, resolvedToolchains)));
  }

  private static boolean isPlatformSuitable(
      ConfiguredTargetKey executionPlatformKey,
      ImmutableSet<ToolchainType> toolchainTypes,
      Table<ConfiguredTargetKey, ToolchainTypeInfo, Label> resolvedToolchains) {
    if (toolchainTypes.isEmpty()) {
      // Since there aren't any toolchains, we should be able to use any execution platform that
      // has made it this far.
      return true;
    }

    // Determine whether all mandatory toolchains are present.
    return resolvedToolchains
        .row(executionPlatformKey)
        .keySet()
        .containsAll(
            toolchainTypes.stream()
                .filter(ToolchainType::mandatory)
                .map(ToolchainType::toolchainTypeInfo)
                .collect(toImmutableSet()));
  }

  private static long countToolchainsOnPlatform(
      ConfiguredTargetKey executionPlatformKey,
      ImmutableSet<ToolchainTypeInfo> toolchainTypeInfos,
      Table<ConfiguredTargetKey, ToolchainTypeInfo, Label> resolvedToolchains) {
    if (toolchainTypeInfos.isEmpty()) {
      return 0;
    }

    // Determine the number of optional toolchains.
    Set<ToolchainTypeInfo> platformToolchains =
        resolvedToolchains.row(executionPlatformKey).keySet();
    return toolchainTypeInfos.stream().filter(platformToolchains::contains).count();
  }

  private static final class ValueMissingException extends Exception {
    private ValueMissingException() {
      super();
    }
  }

  /** Exception used when a toolchain type is required but no matching toolchain is found. */
  static final class UnresolvedToolchainsException extends ToolchainException {
    UnresolvedToolchainsException(List<Label> missingToolchainTypes) {
      super(getMessage(missingToolchainTypes));
    }

    @Override
    protected Code getDetailedCode() {
      return Code.NO_MATCHING_TOOLCHAIN;
    }

    private static String getMessage(List<Label> missingToolchainTypes) {
      ImmutableList<String> labelStrings =
          missingToolchainTypes.stream().map(Label::toString).collect(toImmutableList());
      return String.format(
          "No matching toolchains found for types %s."
              + "\nTo debug, rerun with --toolchain_resolution_debug='%s'"
              + "\nIf platforms or toolchains are a new concept for you, we'd encourage reading "
              + "https://bazel.build/concepts/platforms-intro.",
          String.join(", ", labelStrings), String.join("|", labelStrings));
    }
  }

  /** Used to indicate errors during the computation of an {@link UnloadedToolchainContextImpl}. */
  private static final class ToolchainResolutionFunctionException extends SkyFunctionException {
    ToolchainResolutionFunctionException(ToolchainException e) {
      super(e, Transience.PERSISTENT);
    }
  }
}
