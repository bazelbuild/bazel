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
import static java.util.stream.Collectors.joining;

import com.google.auto.value.AutoValue;
import com.google.common.base.Joiner;
import com.google.common.base.Optional;
import com.google.common.collect.HashBasedTable;
import com.google.common.collect.ImmutableBiMap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Table;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.PlatformConfiguration;
import com.google.devtools.build.lib.analysis.PlatformOptions;
import com.google.devtools.build.lib.analysis.ToolchainContext;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.platform.ConstraintValueInfo;
import com.google.devtools.build.lib.analysis.platform.PlatformInfo;
import com.google.devtools.build.lib.analysis.platform.PlatformProviderUtils;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.pkgcache.FilteringPolicy;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetFunction.ConfiguredValueCreationException;
import com.google.devtools.build.lib.skyframe.RegisteredToolchainsFunction.InvalidToolchainLabelException;
import com.google.devtools.build.lib.skyframe.ToolchainResolutionFunction.NoToolchainFoundException;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.ValueOrException;
import com.google.devtools.build.skyframe.ValueOrException4;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import javax.annotation.Nullable;

/**
 * Common code to create a {@link ToolchainContext} given a set of required toolchain type labels.
 */
// TODO(katre): Refactor this and ToolchainContext into something nicer to work with and with
// fewer static methods everywhere.
public class ToolchainUtil {

  /**
   * Returns a new {@link ToolchainContext}, containing:
   *
   * <ul>
   *   <li>If {@code requiredToolchains} was non-empty, the resolved toolchains and execution
   *       platform (as labels), based on the results of the {@link ToolchainResolutionFunction}
   *   <li>If {@code requiredToolchains} was empty:
   *       <ul>
   *         <li>The resolved toolchains will be empty.
   *         <li>The execution platform will be the host platform, if the host platform was in the
   *             set of available execution platforms.
   *         <li>Otherwise, the execution platform will be the first available execution platform.
   *       </ul>
   * </ul>
   *
   * @param env the Skyframe environment to use to acquire dependencies
   * @param targetDescription a description of the target use, for error and debug message context
   * @param requiredToolchains the required toolchain types that must be resolved
   * @param execConstraintLabels extra constraints on the execution platform to select
   * @param configurationKey the build configuration to use for resolving other targets
   */
  @Nullable
  static ToolchainContext createToolchainContext(
      Environment env,
      String targetDescription,
      Set<Label> requiredToolchains,
      Set<Label> execConstraintLabels,
      @Nullable BuildConfigurationValue.Key configurationKey)
      throws ToolchainContextException, InterruptedException {

    // In some cases this is called with a missing configuration, so we skip toolchain context.
    if (configurationKey == null) {
      return null;
    }

    // This call could be combined with the call below, but this SkyFunction is evaluated so rarely
    // it's not worth optimizing.
    BuildConfigurationValue value = (BuildConfigurationValue) env.getValue(configurationKey);
    if (env.valuesMissing()) {
      return null;
    }
    BuildConfiguration configuration = value.getConfiguration();

    // Load the target and host platform keys.
    PlatformConfiguration platformConfiguration =
        configuration.getFragment(PlatformConfiguration.class);
    if (platformConfiguration == null) {
      return null;
    }
    Label hostPlatformLabel = platformConfiguration.getHostPlatform();
    Label targetPlatformLabel = platformConfiguration.getTargetPlatforms().get(0);

    ConfiguredTargetKey hostPlatformKey = ConfiguredTargetKey.of(hostPlatformLabel, configuration);
    ConfiguredTargetKey targetPlatformKey =
        ConfiguredTargetKey.of(targetPlatformLabel, configuration);
    ImmutableList<ConfiguredTargetKey> execConstraintKeys =
        execConstraintLabels
            .stream()
            .map(label -> ConfiguredTargetKey.of(label, configuration))
            .collect(toImmutableList());

    // Load the host and target platforms early, to check for errors.
    getPlatformInfo(ImmutableList.of(hostPlatformKey, targetPlatformKey), env);

    // Load all available execution platform keys. This will find any errors in the execution
    // platform definitions.
    RegisteredExecutionPlatformsValue registeredExecutionPlatforms =
        loadRegisteredExecutionPlatforms(env, configurationKey);
    if (registeredExecutionPlatforms == null) {
      return null;
    }

    ImmutableList<ConfiguredTargetKey> availableExecutionPlatformKeys =
        new ImmutableList.Builder<ConfiguredTargetKey>()
            .addAll(registeredExecutionPlatforms.registeredExecutionPlatformKeys())
            .add(hostPlatformKey)
            .build();

    // Filter out execution platforms that don't satisfy the extra constraints.
    boolean debug = configuration.getOptions().get(PlatformOptions.class).toolchainResolutionDebug;
    availableExecutionPlatformKeys =
        filterPlatforms(availableExecutionPlatformKeys, execConstraintKeys, env, debug);
    if (availableExecutionPlatformKeys == null) {
      return null;
    }

    ResolvedToolchains resolvedToolchains =
        resolveToolchainLabels(
            env,
            requiredToolchains,
            configurationKey,
            hostPlatformKey,
            availableExecutionPlatformKeys,
            targetPlatformKey,
            debug);
    if (resolvedToolchains == null) {
      return null;
    }

    return createContext(
        env,
        targetDescription,
        resolvedToolchains.executionPlatformKey(),
        resolvedToolchains.targetPlatformKey(),
        requiredToolchains,
        resolvedToolchains.toolchains());
  }

  private static RegisteredExecutionPlatformsValue loadRegisteredExecutionPlatforms(
      Environment env, BuildConfigurationValue.Key configurationKey)
      throws InterruptedException, ToolchainContextException {
    try {
      RegisteredExecutionPlatformsValue registeredExecutionPlatforms =
          (RegisteredExecutionPlatformsValue)
              env.getValueOrThrow(
                  RegisteredExecutionPlatformsValue.key(configurationKey),
                  InvalidPlatformException.class);
      if (registeredExecutionPlatforms == null) {
        return null;
      }
      return registeredExecutionPlatforms;
    } catch (InvalidPlatformException e) {
      throw new ToolchainContextException(e);
    }
  }

  @Nullable
  static Map<ConfiguredTargetKey, PlatformInfo> getPlatformInfo(
      Iterable<ConfiguredTargetKey> platformKeys, Environment env)
      throws InterruptedException, ToolchainContextException {

    Map<SkyKey, ValueOrException<ConfiguredValueCreationException>> values =
        env.getValuesOrThrow(platformKeys, ConfiguredValueCreationException.class);
    boolean valuesMissing = env.valuesMissing();
    Map<ConfiguredTargetKey, PlatformInfo> platforms = valuesMissing ? null : new HashMap<>();
    try {
      for (ConfiguredTargetKey key : platformKeys) {
        PlatformInfo platformInfo = findPlatformInfo(values.get(key));
        if (!valuesMissing && platformInfo != null) {
          platforms.put(key, platformInfo);
        }
      }
    } catch (ConfiguredValueCreationException e) {
      throw new ToolchainContextException(e);
    }
    if (valuesMissing) {
      return null;
    }
    return platforms;
  }

  /**
   * Returns the {@link PlatformInfo} provider from the {@link ConfiguredTarget} in the {@link
   * ValueOrException}, or {@code null} if the {@link ConfiguredTarget} is not present. If the
   * {@link ConfiguredTarget} does not have a {@link PlatformInfo} provider, a {@link
   * InvalidPlatformException} is thrown, wrapped in a {@link ToolchainContextException}.
   */
  @Nullable
  private static PlatformInfo findPlatformInfo(
      ValueOrException<ConfiguredValueCreationException> valueOrException)
      throws ConfiguredValueCreationException, ToolchainContextException {

    ConfiguredTargetValue ctv = (ConfiguredTargetValue) valueOrException.get();
    if (ctv == null) {
      return null;
    }

    ConfiguredTarget configuredTarget = ctv.getConfiguredTarget();
    PlatformInfo platformInfo = PlatformProviderUtils.platform(configuredTarget);
    if (platformInfo == null) {
      throw new ToolchainContextException(
          new InvalidPlatformException(configuredTarget.getLabel()));
    }

    return platformInfo;
  }

  /** Data class to hold the result of resolving toolchain labels. */
  @AutoValue
  protected abstract static class ResolvedToolchains {

    abstract ConfiguredTargetKey executionPlatformKey();

    abstract ConfiguredTargetKey targetPlatformKey();

    abstract ImmutableBiMap<Label, Label> toolchains();

    protected static ResolvedToolchains create(
        ConfiguredTargetKey executionPlatformKey,
        ConfiguredTargetKey targetPlatformKey,
        Map<Label, Label> toolchains) {
      return new AutoValue_ToolchainUtil_ResolvedToolchains(
          executionPlatformKey, targetPlatformKey, ImmutableBiMap.copyOf(toolchains));
    }
  }

  @Nullable
  private static ResolvedToolchains resolveToolchainLabels(
      Environment env,
      Set<Label> requiredToolchains,
      BuildConfigurationValue.Key configurationKey,
      ConfiguredTargetKey hostPlatformKey,
      ImmutableList<ConfiguredTargetKey> availableExecutionPlatformKeys,
      ConfiguredTargetKey targetPlatformKey,
      boolean debug)
      throws InterruptedException, ToolchainContextException {

    // Find the toolchains for the required toolchain types.
    List<ToolchainResolutionValue.Key> registeredToolchainKeys = new ArrayList<>();
    for (Label toolchainType : requiredToolchains) {
      registeredToolchainKeys.add(
          ToolchainResolutionValue.key(
              configurationKey, toolchainType, targetPlatformKey, availableExecutionPlatformKeys));
    }

    Map<
            SkyKey,
            ValueOrException4<
                NoToolchainFoundException, ConfiguredValueCreationException,
                InvalidToolchainLabelException, EvalException>>
        results =
            env.getValuesOrThrow(
                registeredToolchainKeys,
                NoToolchainFoundException.class,
                ConfiguredValueCreationException.class,
                InvalidToolchainLabelException.class,
                EvalException.class);
    boolean valuesMissing = false;

    // Determine the potential set of toolchains.
    Table<ConfiguredTargetKey, Label, Label> resolvedToolchains = HashBasedTable.create();
    List<Label> missingToolchains = new ArrayList<>();
    for (Map.Entry<
            SkyKey,
            ValueOrException4<
                NoToolchainFoundException, ConfiguredValueCreationException,
                InvalidToolchainLabelException, EvalException>>
        entry : results.entrySet()) {
      try {
        Label requiredToolchainType =
            ((ToolchainResolutionValue.Key) entry.getKey().argument()).toolchainType();
        ValueOrException4<
                NoToolchainFoundException, ConfiguredValueCreationException,
                InvalidToolchainLabelException, EvalException>
            valueOrException = entry.getValue();
        if (valueOrException.get() == null) {
          valuesMissing = true;
          continue;
        }

        ToolchainResolutionValue toolchainResolutionValue =
            (ToolchainResolutionValue) valueOrException.get();
        addPlatformsAndLabels(resolvedToolchains, requiredToolchainType, toolchainResolutionValue);
      } catch (NoToolchainFoundException e) {
        // Save the missing type and continue looping to check for more.
        missingToolchains.add(e.missingToolchainType());
      } catch (ConfiguredValueCreationException e) {
        throw new ToolchainContextException(e);
      } catch (InvalidToolchainLabelException e) {
        throw new ToolchainContextException(e);
      } catch (EvalException e) {
        throw new ToolchainContextException(e);
      }
    }

    if (!missingToolchains.isEmpty()) {
      throw new ToolchainContextException(new UnresolvedToolchainsException(missingToolchains));
    }

    if (valuesMissing) {
      return null;
    }

    // Find and return the first execution platform which has all required toolchains.
    Optional<ConfiguredTargetKey> selectedExecutionPlatformKey;
    if (requiredToolchains.isEmpty() && availableExecutionPlatformKeys.contains(hostPlatformKey)) {
      // Fall back to the legacy behavior: use the host platform if it's available, otherwise the
      // first execution platform.
      selectedExecutionPlatformKey = Optional.of(hostPlatformKey);
    } else {
      // If there are no toolchains, this will return the first execution platform.
      selectedExecutionPlatformKey =
          findExecutionPlatformForToolchains(
              env, requiredToolchains, availableExecutionPlatformKeys, resolvedToolchains, debug);
    }

    if (!selectedExecutionPlatformKey.isPresent()) {
      throw new ToolchainContextException(
          new NoMatchingPlatformException(
              requiredToolchains, availableExecutionPlatformKeys, targetPlatformKey));
    }

    return ResolvedToolchains.create(
        selectedExecutionPlatformKey.get(),
        targetPlatformKey,
        resolvedToolchains.row(selectedExecutionPlatformKey.get()));
  }

  private static Optional<ConfiguredTargetKey> findExecutionPlatformForToolchains(
      Environment env,
      Set<Label> requiredToolchains,
      ImmutableList<ConfiguredTargetKey> availableExecutionPlatformKeys,
      Table<ConfiguredTargetKey, Label, Label> resolvedToolchains,
      boolean debug) {
    for (ConfiguredTargetKey executionPlatformKey : availableExecutionPlatformKeys) {
      // PlatformInfo executionPlatform = platforms.get(executionPlatformKey);
      Map<Label, Label> toolchains = resolvedToolchains.row(executionPlatformKey);
      if (!toolchains.keySet().containsAll(requiredToolchains)) {
        // Not all toolchains are present, keep going
        continue;
      }

      if (debug) {
        env.getListener()
            .handle(
                Event.info(
                    String.format(
                        "ToolchainUtil: Selected execution platform %s, %s",
                        executionPlatformKey.getLabel(),
                        toolchains
                            .entrySet()
                            .stream()
                            .map(
                                e ->
                                    String.format(
                                        "type %s -> toolchain %s", e.getKey(), e.getValue()))
                            .collect(joining(", ")))));
      }
      return Optional.of(executionPlatformKey);
    }

    return Optional.absent();
  }

  private static void addPlatformsAndLabels(
      Table<ConfiguredTargetKey, Label, Label> resolvedToolchains,
      Label requiredToolchainType,
      ToolchainResolutionValue toolchainResolutionValue) {

    for (Map.Entry<ConfiguredTargetKey, Label> entry :
        toolchainResolutionValue.availableToolchainLabels().entrySet()) {
      resolvedToolchains.put(entry.getKey(), requiredToolchainType, entry.getValue());
    }
  }

  @Nullable
  private static ToolchainContext createContext(
      Environment env,
      String targetDescription,
      ConfiguredTargetKey executionPlatformKey,
      ConfiguredTargetKey targetPlatformKey,
      Set<Label> requiredToolchains,
      ImmutableBiMap<Label, Label> toolchains)
      throws ToolchainContextException, InterruptedException {

    Map<ConfiguredTargetKey, PlatformInfo> platforms =
        getPlatformInfo(ImmutableList.of(executionPlatformKey, targetPlatformKey), env);

    if (platforms == null) {
      return null;
    }

    return ToolchainContext.create(
        targetDescription,
        platforms.get(executionPlatformKey),
        platforms.get(targetPlatformKey),
        requiredToolchains,
        toolchains);
  }

  @Nullable
  static ImmutableList<Label> expandTargetPatterns(
      Environment env, List<String> targetPatterns, FilteringPolicy filteringPolicy)
      throws InvalidTargetPatternException, InterruptedException {

    // First parse the patterns, and throw any errors immediately.
    List<TargetPatternValue.TargetPatternKey> patternKeys = new ArrayList<>();
    for (TargetPatternValue.TargetPatternSkyKeyOrException keyOrException :
        TargetPatternValue.keys(targetPatterns, filteringPolicy, "")) {

      try {
        patternKeys.add(keyOrException.getSkyKey());
      } catch (TargetParsingException e) {
        throw new InvalidTargetPatternException(keyOrException.getOriginalPattern(), e);
      }
    }

    // Then, resolve the patterns.
    Map<SkyKey, ValueOrException<TargetParsingException>> resolvedPatterns =
        env.getValuesOrThrow(patternKeys, TargetParsingException.class);
    boolean valuesMissing = env.valuesMissing();
    ImmutableList.Builder<Label> labels = valuesMissing ? null : new ImmutableList.Builder<>();

    for (TargetPatternValue.TargetPatternKey pattern : patternKeys) {
      TargetPatternValue value;
      try {
        value = (TargetPatternValue) resolvedPatterns.get(pattern).get();
        if (!valuesMissing && value != null) {
          labels.addAll(value.getTargets().getTargets());
        }
      } catch (TargetParsingException e) {
        throw new InvalidTargetPatternException(pattern.getPattern(), e);
      }
    }

    if (valuesMissing) {
      return null;
    }

    return labels.build();
  }

  @Nullable
  private static ImmutableList<ConfiguredTargetKey> filterPlatforms(
      ImmutableList<ConfiguredTargetKey> platformKeys,
      ImmutableList<ConfiguredTargetKey> constraintKeys,
      Environment env,
      boolean debug)
      throws ToolchainContextException, InterruptedException {

    // Short circuit if not needed.
    if (constraintKeys.isEmpty()) {
      return platformKeys;
    }

    Map<ConfiguredTargetKey, PlatformInfo> platformInfoMap = getPlatformInfo(platformKeys, env);
    if (platformInfoMap == null) {
      return null;
    }
    List<ConstraintValueInfo> constraints = getConstraintValueInfo(constraintKeys, env);
    if (constraints == null) {
      return null;
    }

    return platformKeys
        .stream()
        .filter(key -> filterPlatform(platformInfoMap.get(key), constraints, env, debug))
        .collect(toImmutableList());
  }

  @Nullable
  private static List<ConstraintValueInfo> getConstraintValueInfo(
      ImmutableList<ConfiguredTargetKey> constraintKeys, Environment env)
      throws InterruptedException, ToolchainContextException {

    Map<SkyKey, ValueOrException<ConfiguredValueCreationException>> values =
        env.getValuesOrThrow(constraintKeys, ConfiguredValueCreationException.class);
    boolean valuesMissing = env.valuesMissing();
    List<ConstraintValueInfo> constraintValues = valuesMissing ? null : new ArrayList<>();
    try {
      for (ConfiguredTargetKey key : constraintKeys) {
        ConstraintValueInfo constraintValueInfo = findConstraintValueInfo(values.get(key));
        if (!valuesMissing && constraintValueInfo != null) {
          constraintValues.add(constraintValueInfo);
        }
      }
    } catch (ConfiguredValueCreationException e) {
      throw new ToolchainContextException(e);
    }
    if (valuesMissing) {
      return null;
    }
    return constraintValues;
  }

  @Nullable
  private static ConstraintValueInfo findConstraintValueInfo(
      ValueOrException<ConfiguredValueCreationException> valueOrException)
      throws ConfiguredValueCreationException, ToolchainContextException {

    ConfiguredTargetValue configuredTargetValue = (ConfiguredTargetValue) valueOrException.get();
    if (configuredTargetValue == null) {
      return null;
    }

    ConfiguredTarget configuredTarget = configuredTargetValue.getConfiguredTarget();
    ConstraintValueInfo constraintValueInfo =
        PlatformProviderUtils.constraintValue(configuredTarget);
    if (constraintValueInfo == null) {
      throw new ToolchainContextException(
          new InvalidConstraintValueException(configuredTarget.getLabel()));
    }

    return constraintValueInfo;
  }

  private static boolean filterPlatform(
      PlatformInfo platformInfo,
      List<ConstraintValueInfo> constraints,
      Environment env,
      boolean debug) {
    for (ConstraintValueInfo filterConstraint : constraints) {
      ConstraintValueInfo platformInfoConstraint =
          platformInfo.getConstraint(filterConstraint.constraint());
      if (platformInfoConstraint == null || !platformInfoConstraint.equals(filterConstraint)) {
        // The value for this setting is not present in the platform, or doesn't match the expected
        // value.
        if (debug) {
          env.getListener()
              .handle(
                  Event.info(
                      String.format(
                          "ToolchainUtil: Removed execution platform %s from"
                              + " available execution platforms, it is missing constraint %s",
                          platformInfo.label(), filterConstraint.label())));
        }
        return false;
      }
    }

    return true;
  }

  /** Exception used when no execution platform can be found. */
  static final class NoMatchingPlatformException extends Exception {
    NoMatchingPlatformException() {
      super("No available execution platform satisfies all requested toolchain types");
    }

    public NoMatchingPlatformException(
        Set<Label> requiredToolchains,
        ImmutableList<ConfiguredTargetKey> availableExecutionPlatformKeys,
        ConfiguredTargetKey targetPlatformKey) {
      super(formatError(requiredToolchains, availableExecutionPlatformKeys, targetPlatformKey));
    }

    private static String formatError(
        Set<Label> requiredToolchains,
        ImmutableList<ConfiguredTargetKey> availableExecutionPlatformKeys,
        ConfiguredTargetKey targetPlatformKey) {
      if (requiredToolchains.isEmpty()) {
        return String.format(
            "Unable to find an execution platform for target platform %s"
                + " from available execution platforms [%s]",
            targetPlatformKey.getLabel(),
            availableExecutionPlatformKeys
                .stream()
                .map(key -> key.getLabel().toString())
                .collect(Collectors.joining(", ")));
      }
      return String.format(
          "Unable to find an execution platform for toolchains [%s] and target platform %s"
              + " from available execution platforms [%s]",
          Joiner.on(", ").join(requiredToolchains),
          targetPlatformKey.getLabel(),
          availableExecutionPlatformKeys
              .stream()
              .map(key -> key.getLabel().toString())
              .collect(Collectors.joining(", ")));
    }
  }

  /**
   * Exception used when an error occurs in {@link #expandTargetPatterns(Environment, List,
   * FilteringPolicy)}.
   */
  static final class InvalidTargetPatternException extends Exception {
    private String invalidPattern;
    private TargetParsingException tpe;

    public InvalidTargetPatternException(String invalidPattern, TargetParsingException tpe) {
      super(tpe);
      this.invalidPattern = invalidPattern;
      this.tpe = tpe;
    }

    public String getInvalidPattern() {
      return invalidPattern;
    }

    public TargetParsingException getTpe() {
      return tpe;
    }
  }

  /** Exception used when a platform label is not a valid platform. */
  static final class InvalidPlatformException extends Exception {
    InvalidPlatformException(Label label) {
      super(formatError(label));
    }

    InvalidPlatformException(Label label, ConfiguredValueCreationException e) {
      super(formatError(label), e);
    }

    private static String formatError(Label label) {
      return String.format(
          "Target %s was referenced as a platform, but does not provide PlatformInfo", label);
    }
  }

  /** Exception used when a constraint value label is not a valid constraint value. */
  static final class InvalidConstraintValueException extends Exception {
    InvalidConstraintValueException(Label label) {
      super(formatError(label));
    }

    InvalidConstraintValueException(Label label, ConfiguredValueCreationException e) {
      super(formatError(label), e);
    }

    private static String formatError(Label label) {
      return String.format(
          "Target %s was referenced as a constraint_value,"
              + " but does not provide ConstraintValueInfo",
          label);
    }
  }

  /** Exception used when a toolchain type is required but no matching toolchain is found. */
  public static final class UnresolvedToolchainsException extends Exception {
    private final ImmutableList<Label> missingToolchainTypes;

    public UnresolvedToolchainsException(List<Label> missingToolchainTypes) {
      super(
          String.format(
              "no matching toolchains found for types %s",
              Joiner.on(", ").join(missingToolchainTypes)));
      this.missingToolchainTypes = ImmutableList.copyOf(missingToolchainTypes);
    }

    public ImmutableList<Label> missingToolchainTypes() {
      return missingToolchainTypes;
    }
  }

  /** Exception used to wrap exceptions during toolchain resolution. */
  public static class ToolchainContextException extends Exception {
    public ToolchainContextException(NoMatchingPlatformException e) {
      super(e);
    }

    public ToolchainContextException(InvalidPlatformException e) {
      super(e);
    }

    public ToolchainContextException(InvalidConstraintValueException e) {
      super(e);
    }

    public ToolchainContextException(UnresolvedToolchainsException e) {
      super(e);
    }

    public ToolchainContextException(ConfiguredValueCreationException e) {
      super(e);
    }

    public ToolchainContextException(InvalidToolchainLabelException e) {
      super(e);
    }

    public ToolchainContextException(EvalException e) {
      super(e);
    }
  }
}
