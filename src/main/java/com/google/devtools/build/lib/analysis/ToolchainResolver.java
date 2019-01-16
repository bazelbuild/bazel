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

package com.google.devtools.build.lib.analysis;

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.collect.ImmutableList.toImmutableList;
import static java.util.stream.Collectors.joining;

import com.google.auto.value.AutoValue;
import com.google.common.collect.HashBasedTable;
import com.google.common.collect.ImmutableBiMap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Table;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.platform.ConstraintValueInfo;
import com.google.devtools.build.lib.analysis.platform.PlatformInfo;
import com.google.devtools.build.lib.analysis.platform.PlatformProviderUtils;
import com.google.devtools.build.lib.analysis.platform.ToolchainInfo;
import com.google.devtools.build.lib.analysis.platform.ToolchainTypeInfo;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.skyframe.BuildConfigurationValue;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetAndData;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import com.google.devtools.build.lib.skyframe.ConstraintValueLookupUtil;
import com.google.devtools.build.lib.skyframe.ConstraintValueLookupUtil.InvalidConstraintValueException;
import com.google.devtools.build.lib.skyframe.PlatformLookupUtil;
import com.google.devtools.build.lib.skyframe.PlatformLookupUtil.InvalidPlatformException;
import com.google.devtools.build.lib.skyframe.RegisteredExecutionPlatformsValue;
import com.google.devtools.build.lib.skyframe.RegisteredToolchainsFunction.InvalidToolchainLabelException;
import com.google.devtools.build.lib.skyframe.ToolchainException;
import com.google.devtools.build.lib.skyframe.ToolchainResolutionFunction.NoToolchainFoundException;
import com.google.devtools.build.lib.skyframe.ToolchainResolutionValue;
import com.google.devtools.build.lib.util.OrderedSetMultimap;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.ValueOrException2;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;
import javax.annotation.Nullable;

/**
 * Performs the toolchain resolution process to determine the correct toolchain target dependencies
 * for a target being configured, based on the required toolchain types, target platform, and
 * available execution platforms.
 */
public class ToolchainResolver {
  // Required data.
  private final Environment environment;
  private final BuildConfigurationValue.Key configurationKey;

  // Optional data.
  private String targetDescription = "";
  private ImmutableSet<Label> requiredToolchainTypeLabels = ImmutableSet.of();
  private ImmutableSet<Label> execConstraintLabels = ImmutableSet.of();

  // Determined during execution.
  private boolean debug = false;

  /**
   * Creates a new {@link ToolchainResolver} to help find the required toolchains for a configured
   * target.
   *
   * @param env the environment to use to request dependent Skyframe nodes
   * @param configurationKey The build configuration to use for dependent targets
   */
  public ToolchainResolver(Environment env, BuildConfigurationValue.Key configurationKey) {
    this.environment = checkNotNull(env);
    this.configurationKey = checkNotNull(configurationKey);
  }

  /**
   * Sets a description of the target that toolchains will be resolved for. This is primarily useful
   * for printing informative error messages, so that users can tell which targets had difficulty.
   */
  public ToolchainResolver setTargetDescription(String targetDescription) {
    this.targetDescription = targetDescription;
    return this;
  }

  /**
   * Sets the labels of the required toolchain types that this resolver needs to find toolchains
   * for.
   */
  public ToolchainResolver setRequiredToolchainTypes(Set<Label> requiredToolchainTypeLabels) {
    this.requiredToolchainTypeLabels = ImmutableSet.copyOf(requiredToolchainTypeLabels);
    return this;
  }

  /**
   * Sets extra constraints on the execution platform. Targets can use this to ensure that the
   * execution platform has some desired characteristics, such as having enough memory to run tests.
   */
  public ToolchainResolver setExecConstraintLabels(Set<Label> execConstraintLabels) {
    this.execConstraintLabels = ImmutableSet.copyOf(execConstraintLabels);
    return this;
  }

  /**
   * Determines the specific toolchains that are required, given the requested toolchain types,
   * target platform, and configuration.
   *
   * <p>In order to resolve toolchains, first the {@link ToolchainResolver} must be created, and
   * then an {@link UnloadedToolchainContext} generated. The {@link UnloadedToolchainContext} will
   * report the specific toolchain targets to depend on, and those can be found using the typical
   * dependency machinery. Once dependencies, including toolchains, have been loaded, the {@link
   * UnloadedToolchainContext#load} method can be called to generate the final {@link
   * ToolchainContext} to be used by the target.
   *
   * <p>This makes several SkyFrame calls, particularly to {@link
   * com.google.devtools.build.lib.skyframe.ConfiguredTargetFunction} (to load platforms and
   * toolchains), to {@link
   * com.google.devtools.build.lib.skyframe.RegisteredExecutionPlatformsFunction}, and to {@link
   * com.google.devtools.build.lib.skyframe.ToolchainResolutionFunction}. This method return {@code
   * null} to signal a SkyFrame restart is needed to resolve dependencies.
   */
  @Nullable
  public UnloadedToolchainContext resolve() throws InterruptedException, ToolchainException {

    try {
      UnloadedToolchainContext.Builder unloadedToolchainContext =
          UnloadedToolchainContext.builder();
      unloadedToolchainContext.setTargetDescription(targetDescription);

      // Determine the configuration being used.
      BuildConfigurationValue value =
          (BuildConfigurationValue) environment.getValue(configurationKey);
      if (value == null) {
        throw new ValueMissingException();
      }
      BuildConfiguration configuration = value.getConfiguration();
      PlatformConfiguration platformConfiguration =
          configuration.getFragment(PlatformConfiguration.class);
      if (platformConfiguration == null) {
        throw new ValueMissingException();
      }

      // Check if debug output should be generated.
      this.debug = configuration.getOptions().get(PlatformOptions.class).toolchainResolutionDebug;

      // Create keys for all platforms that will be used, and validate them early.
      PlatformKeys platformKeys = loadPlatformKeys(configuration, platformConfiguration);
      if (environment.valuesMissing()) {
        return null;
      }

      // Determine the actual toolchain implementations to use.
      determineToolchainImplementations(unloadedToolchainContext, platformKeys);

      return unloadedToolchainContext.build();
    } catch (ValueMissingException e) {
      return null;
    }
  }

  @AutoValue
  abstract static class PlatformKeys {
    abstract ConfiguredTargetKey hostPlatformKey();

    abstract ConfiguredTargetKey targetPlatformKey();

    abstract ImmutableList<ConfiguredTargetKey> executionPlatformKeys();

    static PlatformKeys create(
        ConfiguredTargetKey hostPlatformKey,
        ConfiguredTargetKey targetPlatformKey,
        List<ConfiguredTargetKey> executionPlatformKeys) {
      return new AutoValue_ToolchainResolver_PlatformKeys(
          hostPlatformKey, targetPlatformKey, ImmutableList.copyOf(executionPlatformKeys));
    }
  }

  private PlatformKeys loadPlatformKeys(
      BuildConfiguration configuration, PlatformConfiguration platformConfiguration)
      throws InterruptedException, InvalidPlatformException, ValueMissingException,
          InvalidConstraintValueException {
    // Determine the target and host platform keys.
    Label hostPlatformLabel = platformConfiguration.getHostPlatform();
    Label targetPlatformLabel = platformConfiguration.getTargetPlatform();

    ConfiguredTargetKey hostPlatformKey = ConfiguredTargetKey.of(hostPlatformLabel, configuration);
    ConfiguredTargetKey targetPlatformKey =
        ConfiguredTargetKey.of(targetPlatformLabel, configuration);

    // Load the host and target platforms early, to check for errors.
    PlatformLookupUtil.getPlatformInfo(
        ImmutableList.of(hostPlatformKey, targetPlatformKey), environment);
    if (environment.valuesMissing()) {
      throw new ValueMissingException();
    }

    ImmutableList<ConfiguredTargetKey> executionPlatformKeys =
        loadExecutionPlatformKeys(configuration, hostPlatformKey);

    return PlatformKeys.create(hostPlatformKey, targetPlatformKey, executionPlatformKeys);
  }

  private ImmutableList<ConfiguredTargetKey> loadExecutionPlatformKeys(
      BuildConfiguration configuration, ConfiguredTargetKey defaultPlatformKey)
      throws InvalidPlatformException, InterruptedException, InvalidConstraintValueException,
          ValueMissingException {
    RegisteredExecutionPlatformsValue registeredExecutionPlatforms =
        (RegisteredExecutionPlatformsValue)
            environment.getValueOrThrow(
                RegisteredExecutionPlatformsValue.key(configurationKey),
                InvalidPlatformException.class);
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
            .map(label -> ConfiguredTargetKey.of(label, configuration))
            .collect(toImmutableList());

    return filterAvailablePlatforms(availableExecutionPlatformKeys, execConstraintKeys);
  }

  /** Returns only the platform keys that match the given constraints. */
  private ImmutableList<ConfiguredTargetKey> filterAvailablePlatforms(
      ImmutableList<ConfiguredTargetKey> platformKeys,
      ImmutableList<ConfiguredTargetKey> constraintKeys)
      throws InterruptedException, InvalidPlatformException, InvalidConstraintValueException,
          ValueMissingException {

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
        .filter(key -> filterPlatform(platformInfoMap.get(key), constraints))
        .collect(toImmutableList());
  }

  /** Returns {@code true} if the given platform has all of the constraints. */
  private boolean filterPlatform(PlatformInfo platformInfo, List<ConstraintValueInfo> constraints) {
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
                        "ToolchainResolver: Removed execution platform %s from"
                            + " available execution platforms, it is missing constraint %s",
                        platformInfo.label(), constraint.label())));
        }
    }

    return missingConstraints.isEmpty();
  }

  private void determineToolchainImplementations(
      UnloadedToolchainContext.Builder unloadedToolchainContext, PlatformKeys platformKeys)
      throws InterruptedException, ToolchainException, ValueMissingException {

    // Find the toolchains for the required toolchain types.
    List<ToolchainResolutionValue.Key> registeredToolchainKeys = new ArrayList<>();
    for (Label toolchainTypeLabel : requiredToolchainTypeLabels) {
      registeredToolchainKeys.add(
          ToolchainResolutionValue.key(
              configurationKey,
              toolchainTypeLabel,
              platformKeys.targetPlatformKey(),
              platformKeys.executionPlatformKeys()));
    }

    Map<SkyKey, ValueOrException2<NoToolchainFoundException, InvalidToolchainLabelException>>
        results =
            environment.getValuesOrThrow(
                registeredToolchainKeys,
                NoToolchainFoundException.class,
                InvalidToolchainLabelException.class);
    boolean valuesMissing = false;

    // Determine the potential set of toolchains.
    Table<ConfiguredTargetKey, ToolchainTypeInfo, Label> resolvedToolchains =
        HashBasedTable.create();
    ImmutableSet.Builder<ToolchainTypeInfo> requiredToolchainTypesBuilder = ImmutableSet.builder();
    List<Label> missingToolchains = new ArrayList<>();
    for (Map.Entry<
            SkyKey, ValueOrException2<NoToolchainFoundException, InvalidToolchainLabelException>>
        entry : results.entrySet()) {
      try {
        ValueOrException2<NoToolchainFoundException, InvalidToolchainLabelException>
            valueOrException = entry.getValue();
        ToolchainResolutionValue toolchainResolutionValue =
            (ToolchainResolutionValue) valueOrException.get();
        if (toolchainResolutionValue == null) {
          valuesMissing = true;
          continue;
        }

        ToolchainTypeInfo requiredToolchainType = toolchainResolutionValue.toolchainType();
        requiredToolchainTypesBuilder.add(requiredToolchainType);
        resolvedToolchains.putAll(
            findPlatformsAndLabels(requiredToolchainType, toolchainResolutionValue));
      } catch (NoToolchainFoundException e) {
        // Save the missing type and continue looping to check for more.
        missingToolchains.add(e.missingToolchainTypeLabel());
      }
    }

    if (!missingToolchains.isEmpty()) {
      throw new UnresolvedToolchainsException(missingToolchains);
    }

    if (valuesMissing) {
      throw new ValueMissingException();
    }

    ImmutableSet<ToolchainTypeInfo> requiredToolchainTypes = requiredToolchainTypesBuilder.build();

    // Find and return the first execution platform which has all required toolchains.
    Optional<ConfiguredTargetKey> selectedExecutionPlatformKey;
    if (requiredToolchainTypeLabels.isEmpty()
        && platformKeys.executionPlatformKeys().contains(platformKeys.hostPlatformKey())) {
      // Fall back to the legacy behavior: use the host platform if it's available, otherwise the
      // first execution platform.
      selectedExecutionPlatformKey = Optional.of(platformKeys.hostPlatformKey());
    } else {
      // If there are no toolchains, this will return the first execution platform.
      selectedExecutionPlatformKey =
          findExecutionPlatformForToolchains(
              requiredToolchainTypes, platformKeys.executionPlatformKeys(), resolvedToolchains);
    }

    if (!selectedExecutionPlatformKey.isPresent()) {
      throw new NoMatchingPlatformException(
          requiredToolchainTypeLabels,
          platformKeys.executionPlatformKeys(),
          platformKeys.targetPlatformKey());
    }

    Map<ConfiguredTargetKey, PlatformInfo> platforms =
        PlatformLookupUtil.getPlatformInfo(
            ImmutableList.of(selectedExecutionPlatformKey.get(), platformKeys.targetPlatformKey()),
            environment);
    if (platforms == null) {
      throw new ValueMissingException();
    }

    unloadedToolchainContext.setRequiredToolchainTypes(requiredToolchainTypes);
    unloadedToolchainContext.setExecutionPlatform(
        platforms.get(selectedExecutionPlatformKey.get()));
    unloadedToolchainContext.setTargetPlatform(platforms.get(platformKeys.targetPlatformKey()));

    Map<ToolchainTypeInfo, Label> toolchains =
        resolvedToolchains.row(selectedExecutionPlatformKey.get());
    unloadedToolchainContext.setToolchainTypeToResolved(ImmutableBiMap.copyOf(toolchains));
  }

  /**
   * Adds all of toolchain labels from{@code toolchainResolutionValue} to {@code
   * resolvedToolchains}.
   */
  private static Table<ConfiguredTargetKey, ToolchainTypeInfo, Label> findPlatformsAndLabels(
      ToolchainTypeInfo requiredToolchainType, ToolchainResolutionValue toolchainResolutionValue) {

    Table<ConfiguredTargetKey, ToolchainTypeInfo, Label> resolvedToolchains =
        HashBasedTable.create();
    for (Map.Entry<ConfiguredTargetKey, Label> entry :
        toolchainResolutionValue.availableToolchainLabels().entrySet()) {
      resolvedToolchains.put(entry.getKey(), requiredToolchainType, entry.getValue());
    }
    return resolvedToolchains;
  }

  /**
   * Finds the first platform from {@code availableExecutionPlatformKeys} that is present in {@code
   * resolvedToolchains} and has all required toolchain types.
   */
  private Optional<ConfiguredTargetKey> findExecutionPlatformForToolchains(
      ImmutableSet<ToolchainTypeInfo> requiredToolchainTypes,
      ImmutableList<ConfiguredTargetKey> availableExecutionPlatformKeys,
      Table<ConfiguredTargetKey, ToolchainTypeInfo, Label> resolvedToolchains) {
    for (ConfiguredTargetKey executionPlatformKey : availableExecutionPlatformKeys) {
      Map<ToolchainTypeInfo, Label> toolchains = resolvedToolchains.row(executionPlatformKey);

      if (!toolchains.keySet().containsAll(requiredToolchainTypes)) {
        // Not all toolchains are present, keep going
        continue;
      }

      if (debug) {
        String selectedToolchains =
            toolchains.entrySet().stream()
                .map(
                    e ->
                        String.format(
                            "type %s -> toolchain %s", e.getKey().typeLabel(), e.getValue()))
                .collect(joining(", "));
        environment
            .getListener()
            .handle(
                Event.info(
                    String.format(
                        "ToolchainResolver: Selected execution platform %s, %s",
                        executionPlatformKey.getLabel(), selectedToolchains)));
      }
      return Optional.of(executionPlatformKey);
    }

    return Optional.empty();
  }

  /**
   * Represents the state of toolchain resolution once the specific required toolchains have been
   * determined, but before the toolchain dependencies have been resolved.
   */
  @AutoValue
  public abstract static class UnloadedToolchainContext {

    static Builder builder() {
      return new AutoValue_ToolchainResolver_UnloadedToolchainContext.Builder();
    }

    /** Builder class to help create the {@link UnloadedToolchainContext}. */
    @AutoValue.Builder
    public interface Builder {
      /** Sets a description of the target being used, for error messaging. */
      Builder setTargetDescription(String targetDescription);

      /** Sets the selected execution platform that these toolchains use. */
      Builder setExecutionPlatform(PlatformInfo executionPlatform);

      /** Sets the target platform that these toolchains generate output for. */
      Builder setTargetPlatform(PlatformInfo targetPlatform);

      /** Sets the toolchain types that were requested. */
      Builder setRequiredToolchainTypes(Set<ToolchainTypeInfo> requiredToolchainTypes);

      Builder setToolchainTypeToResolved(
          ImmutableBiMap<ToolchainTypeInfo, Label> toolchainTypeToResolved);

      UnloadedToolchainContext build();
    }

    /** Returns a description of the target being used, for error messaging. */
    abstract String targetDescription();

    /** Returns the selected execution platform that these toolchains use. */
    abstract PlatformInfo executionPlatform();

    /** Returns the target platform that these toolchains generate output for. */
    abstract PlatformInfo targetPlatform();

    /** Returns the toolchain types that were requested. */
    abstract ImmutableSet<ToolchainTypeInfo> requiredToolchainTypes();

    /** The map of toolchain type to resolved toolchain to be used. */
    abstract ImmutableBiMap<ToolchainTypeInfo, Label> toolchainTypeToResolved();

    /** Returns the labels of the specific toolchains being used. */
    public ImmutableSet<Label> resolvedToolchainLabels() {
      return toolchainTypeToResolved().values();
    }

    /**
     * Finishes preparing the {@link ToolchainContext} by finding the specific toolchain providers
     * to be used for each toolchain type.
     */
    public ToolchainContext load(
        OrderedSetMultimap<Attribute, ConfiguredTargetAndData> prerequisiteMap) {

      ToolchainContext.Builder toolchainContext =
          ToolchainContext.builder()
              .setTargetDescription(targetDescription())
              .setExecutionPlatform(executionPlatform())
              .setTargetPlatform(targetPlatform())
              .setRequiredToolchainTypes(requiredToolchainTypes())
              .setResolvedToolchainLabels(resolvedToolchainLabels());

      // Find the prerequisites associated with PlatformSemantics.RESOLVED_TOOLCHAINS_ATTR.
      Optional<Attribute> toolchainAttribute =
          prerequisiteMap.keys().stream()
              .filter(Objects::nonNull)
              .filter(
                  attribute ->
                      attribute.getName().equals(PlatformSemantics.RESOLVED_TOOLCHAINS_ATTR))
              .findFirst();
      ImmutableMap.Builder<ToolchainTypeInfo, ToolchainInfo> toolchains =
          new ImmutableMap.Builder<>();
      ImmutableList.Builder<TemplateVariableInfo> templateVariableProviders =
          new ImmutableList.Builder<>();
      if (toolchainAttribute.isPresent()) {
        for (ConfiguredTargetAndData target : prerequisiteMap.get(toolchainAttribute.get())) {
          Label discoveredLabel = target.getTarget().getLabel();
          ToolchainTypeInfo toolchainType =
              toolchainTypeToResolved().inverse().get(discoveredLabel);

          // If the toolchainType hadn't been resolved to an actual toolchain, resolution would have
          // failed with an error much earlier. This null check is just for safety.
          if (toolchainType != null) {
            ToolchainInfo toolchainInfo =
                PlatformProviderUtils.toolchain(target.getConfiguredTarget());
            if (toolchainType != null) {
              toolchains.put(toolchainType, toolchainInfo);
            }
          }

          // Find any template variables present for this toolchain.
          TemplateVariableInfo templateVariableInfo =
              target.getConfiguredTarget().get(TemplateVariableInfo.PROVIDER);
          if (templateVariableInfo != null) {
            templateVariableProviders.add(templateVariableInfo);
          }
        }
      }

      return toolchainContext
          .setToolchains(toolchains.build())
          .setTemplateVariableProviders(templateVariableProviders.build())
          .build();
    }
  }

  private static final class ValueMissingException extends Exception {
    private ValueMissingException() {
      super();
    }
  }

  /** Exception used when no execution platform can be found. */
  static final class NoMatchingPlatformException extends ToolchainException {
    NoMatchingPlatformException(
        Set<Label> requiredToolchainTypeLabels,
        ImmutableList<ConfiguredTargetKey> availableExecutionPlatformKeys,
        ConfiguredTargetKey targetPlatformKey) {
      super(
          formatError(
              requiredToolchainTypeLabels, availableExecutionPlatformKeys, targetPlatformKey));
    }

    private static String formatError(
        Set<Label> requiredToolchainTypeLabels,
        ImmutableList<ConfiguredTargetKey> availableExecutionPlatformKeys,
        ConfiguredTargetKey targetPlatformKey) {
      if (requiredToolchainTypeLabels.isEmpty()) {
        return String.format(
            "Unable to find an execution platform for target platform %s"
                + " from available execution platforms [%s]",
            targetPlatformKey.getLabel(),
            availableExecutionPlatformKeys.stream()
                .map(key -> key.getLabel().toString())
                .collect(Collectors.joining(", ")));
      }
      return String.format(
          "Unable to find an execution platform for toolchains [%s] and target platform %s"
              + " from available execution platforms [%s]",
          requiredToolchainTypeLabels.stream().map(Label::toString).collect(joining(", ")),
          targetPlatformKey.getLabel(),
          availableExecutionPlatformKeys.stream()
              .map(key -> key.getLabel().toString())
              .collect(Collectors.joining(", ")));
    }
  }

  /** Exception used when a toolchain type is required but no matching toolchain is found. */
  static final class UnresolvedToolchainsException extends ToolchainException {
    UnresolvedToolchainsException(List<Label> missingToolchainTypes) {
      super(
          String.format(
              "no matching toolchains found for types %s",
              missingToolchainTypes.stream().map(Label::toString).collect(joining(", "))));
    }
  }
}
