package com.google.devtools.build.lib.skyframe;

import static com.google.common.collect.ImmutableList.toImmutableList;
import static java.util.stream.Collectors.joining;

import com.google.auto.value.AutoValue;
import com.google.common.collect.HashBasedTable;
import com.google.common.collect.ImmutableBiMap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Table;
import com.google.devtools.build.lib.analysis.PlatformConfiguration;
import com.google.devtools.build.lib.analysis.PlatformOptions;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.platform.ConstraintValueInfo;
import com.google.devtools.build.lib.analysis.platform.PlatformInfo;
import com.google.devtools.build.lib.analysis.platform.ToolchainTypeInfo;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.skyframe.ConstraintValueLookupUtil.InvalidConstraintValueException;
import com.google.devtools.build.lib.skyframe.PlatformLookupUtil.InvalidPlatformException;
import com.google.devtools.build.lib.skyframe.RegisteredToolchainsFunction.InvalidToolchainLabelException;
import com.google.devtools.build.lib.skyframe.SingleToolchainResolutionFunction.NoToolchainFoundException;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.ValueOrException2;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;
import javax.annotation.Nullable;

/**
 * {@link SkyFunction} which performs toolchain resolution for a multiple toolchain types, including
 * selecting the execution platform.
 */
public class ToolchainResolutionFunction implements SkyFunction {

  @Nullable
  @Override
  public UnloadedToolchainContext compute(SkyKey skyKey, Environment env)
      throws ToolchainResolutionFunctionException, InterruptedException {
    UnloadedToolchainContext.Key key = (UnloadedToolchainContext.Key) skyKey.argument();

    try {
      UnloadedToolchainContext.Builder unloadedToolchainContext =
          UnloadedToolchainContext.builder();

      // Determine the configuration being used.
      BuildConfigurationValue value =
          (BuildConfigurationValue) env.getValue(key.configurationKey());
      if (value == null) {
        throw new ValueMissingException();
      }
      BuildConfiguration configuration = value.getConfiguration();
      PlatformConfiguration platformConfiguration =
          configuration.getFragment(PlatformConfiguration.class);
      if (platformConfiguration == null) {
        // throw new ValueMissingException();
        throw new ToolchainException("missing platform config fragment");
      }

      // Check if debug output should be generated.
      boolean debug =
          configuration.getOptions().get(PlatformOptions.class).toolchainResolutionDebug;

      // Create keys for all platforms that will be used, and validate them early.
      PlatformKeys platformKeys =
          loadPlatformKeys(
              env,
              debug,
              key.configurationKey(),
              configuration,
              platformConfiguration,
              key.execConstraintLabels());
      if (env.valuesMissing()) {
        return null;
      }

      // Determine the actual toolchain implementations to use.
      determineToolchainImplementations(
          env,
          debug,
          key.configurationKey(),
          key.requiredToolchainTypeLabels(),
          unloadedToolchainContext,
          platformKeys);

      return unloadedToolchainContext.build();
    } catch (ToolchainException e) {
      throw new ToolchainResolutionFunctionException(e);
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
      return new AutoValue_ToolchainResolutionFunction_PlatformKeys(
          hostPlatformKey, targetPlatformKey, ImmutableList.copyOf(executionPlatformKeys));
    }
  }

  private PlatformKeys loadPlatformKeys(
      SkyFunction.Environment environment,
      boolean debug,
      BuildConfigurationValue.Key configurationKey,
      BuildConfiguration configuration,
      PlatformConfiguration platformConfiguration,
      ImmutableSet<Label> execConstraintLabels)
      throws InterruptedException, ValueMissingException, InvalidConstraintValueException,
          InvalidPlatformException {
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
        loadExecutionPlatformKeys(
            environment,
            debug,
            configurationKey,
            configuration,
            hostPlatformKey,
            execConstraintLabels);

    return PlatformKeys.create(hostPlatformKey, targetPlatformKey, executionPlatformKeys);
  }

  private ImmutableList<ConfiguredTargetKey> loadExecutionPlatformKeys(
      SkyFunction.Environment environment,
      boolean debug,
      BuildConfigurationValue.Key configurationKey,
      BuildConfiguration configuration,
      ConfiguredTargetKey defaultPlatformKey,
      ImmutableSet<Label> execConstraintLabels)
      throws InterruptedException, ValueMissingException, InvalidConstraintValueException,
          InvalidPlatformException {
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

    return filterAvailablePlatforms(
        environment, debug, availableExecutionPlatformKeys, execConstraintKeys);
  }

  /** Returns only the platform keys that match the given constraints. */
  private ImmutableList<ConfiguredTargetKey> filterAvailablePlatforms(
      SkyFunction.Environment environment,
      boolean debug,
      ImmutableList<ConfiguredTargetKey> platformKeys,
      ImmutableList<ConfiguredTargetKey> constraintKeys)
      throws InterruptedException, ValueMissingException, InvalidConstraintValueException,
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
  private boolean filterPlatform(
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

  private void determineToolchainImplementations(
      SkyFunction.Environment environment,
      boolean debug,
      BuildConfigurationValue.Key configurationKey,
      ImmutableSet<Label> requiredToolchainTypeLabels,
      UnloadedToolchainContext.Builder unloadedToolchainContext,
      PlatformKeys platformKeys)
      throws InterruptedException, ValueMissingException, InvalidPlatformException,
          NoMatchingPlatformException, UnresolvedToolchainsException,
          InvalidToolchainLabelException {

    // Find the toolchains for the required toolchain types.
    List<SingleToolchainResolutionValue.Key> registeredToolchainKeys = new ArrayList<>();
    for (Label toolchainTypeLabel : requiredToolchainTypeLabels) {
      registeredToolchainKeys.add(
          SingleToolchainResolutionValue.key(
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
        SingleToolchainResolutionValue singleToolchainResolutionValue =
            (SingleToolchainResolutionValue) valueOrException.get();
        if (singleToolchainResolutionValue == null) {
          valuesMissing = true;
          continue;
        }

        ToolchainTypeInfo requiredToolchainType = singleToolchainResolutionValue.toolchainType();
        requiredToolchainTypesBuilder.add(requiredToolchainType);
        resolvedToolchains.putAll(
            findPlatformsAndLabels(requiredToolchainType, singleToolchainResolutionValue));
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
    if (requiredToolchainTypeLabels.isEmpty()) {
      if (platformKeys.executionPlatformKeys().contains(platformKeys.hostPlatformKey())) {
        // Fall back to the legacy behavior: use the host platform if it's available, otherwise the
        // first execution platform.
        selectedExecutionPlatformKey = Optional.of(platformKeys.hostPlatformKey());
      } else if (!platformKeys.executionPlatformKeys().isEmpty()) {
        // The host platform is invalid and no toolchains were requested: use the first execution
        // platform, if there is one.
        selectedExecutionPlatformKey = Optional.of(platformKeys.executionPlatformKeys().get(0));
      } else {
        // There's nothing left to try.
        selectedExecutionPlatformKey = Optional.empty();
      }
    } else {
      selectedExecutionPlatformKey =
          findExecutionPlatformForToolchains(
              environment,
              debug,
              requiredToolchainTypes,
              platformKeys.executionPlatformKeys(),
              resolvedToolchains);
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
      SkyFunction.Environment environment,
      boolean debug,
      ImmutableSet<ToolchainTypeInfo> requiredToolchainTypes,
      ImmutableList<ConfiguredTargetKey> availableExecutionPlatformKeys,
      Table<ConfiguredTargetKey, ToolchainTypeInfo, Label> resolvedToolchains) {
    for (ConfiguredTargetKey executionPlatformKey : availableExecutionPlatformKeys) {
      if (!resolvedToolchains.containsRow(executionPlatformKey)) {
        continue;
      }

      Map<ToolchainTypeInfo, Label> toolchains = resolvedToolchains.row(executionPlatformKey);
      if (!toolchains.keySet().containsAll(requiredToolchainTypes)) {
        // Not all toolchains are present, ignore this execution platform.
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
                        "ToolchainResolution: Selected execution platform %s, %s",
                        executionPlatformKey.getLabel(), selectedToolchains)));
      }
      return Optional.of(executionPlatformKey);
    }

    return Optional.empty();
  }

  @Nullable
  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
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

  /** Used to indicate errors during the computation of an {@link UnloadedToolchainContext}. */
  private static final class ToolchainResolutionFunctionException extends SkyFunctionException {
    public ToolchainResolutionFunctionException(ToolchainException e) {
      super(e, Transience.PERSISTENT);
    }
  }
}
