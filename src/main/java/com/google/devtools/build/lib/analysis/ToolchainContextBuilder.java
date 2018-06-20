package com.google.devtools.build.lib.analysis;

import static com.google.common.collect.ImmutableList.toImmutableList;
import static java.util.stream.Collectors.joining;

import com.google.auto.value.AutoValue;
import com.google.common.base.Joiner;
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
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.skyframe.BuildConfigurationValue;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetAndData;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import com.google.devtools.build.lib.skyframe.ConstraintValueLookupFunction.InvalidConstraintValueException;
import com.google.devtools.build.lib.skyframe.ConstraintValueLookupValue;
import com.google.devtools.build.lib.skyframe.PlatformLookupFunction.InvalidPlatformException;
import com.google.devtools.build.lib.skyframe.PlatformLookupValue;
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
import java.util.stream.StreamSupport;
import javax.annotation.Nullable;

@AutoValue
public abstract class ToolchainContextBuilder {

  @Nullable
  public static ToolchainContextBuilder create(
      Environment env,
      String targetDescription,
      Set<Label> requiredToolchainTypes,
      Set<Label> execConstraintLabels,
      @Nullable BuildConfigurationValue.Key configurationKey)
      throws InterruptedException, ToolchainException {

    // In some cases this is called with a missing configuration, so we skip toolchain context.
    if (configurationKey == null) {
      return null;
    }

    // This call could be combined with the call below, but this is evaluated so rarely it's not
    // worth optimizing.
    BuildConfigurationValue value = (BuildConfigurationValue) env.getValue(configurationKey);
    if (value == null) {
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
    if (env.valuesMissing()) {
      return null;
    }

    // Load all available execution platforms. This will find any errors in the execution platform
    // definitions.
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

    // Determine the actual toolchain implementations to use.
    ResolvedToolchains resolvedToolchains =
        resolveToolchainLabels(
            env,
            requiredToolchainTypes,
            configurationKey,
            hostPlatformKey,
            availableExecutionPlatformKeys,
            targetPlatformKey,
            debug);
    if (resolvedToolchains == null) {
      return null;
    }

    Map<ConfiguredTargetKey, PlatformInfo> platforms =
        getPlatformInfo(
            ImmutableList.of(
                resolvedToolchains.executionPlatformKey(), resolvedToolchains.targetPlatformKey()),
            env);

    if (platforms == null) {
      return null;
    }

    return new AutoValue_ToolchainContextBuilder(
        targetDescription,
        ImmutableSet.copyOf(requiredToolchainTypes),
        platforms.get(resolvedToolchains.executionPlatformKey()),
        platforms.get(resolvedToolchains.targetPlatformKey()),
        resolvedToolchains.toolchains());
  }

  private static RegisteredExecutionPlatformsValue loadRegisteredExecutionPlatforms(
      Environment env, BuildConfigurationValue.Key configurationKey)
      throws InterruptedException, InvalidPlatformException {
      RegisteredExecutionPlatformsValue registeredExecutionPlatforms =
          (RegisteredExecutionPlatformsValue)
              env.getValueOrThrow(
                  RegisteredExecutionPlatformsValue.key(configurationKey),
                  InvalidPlatformException.class);
      if (registeredExecutionPlatforms == null) {
        return null;
      }
      return registeredExecutionPlatforms;
  }

  @Nullable
  private static ImmutableList<ConfiguredTargetKey> filterPlatforms(
      ImmutableList<ConfiguredTargetKey> platformKeys,
      ImmutableList<ConfiguredTargetKey> constraintKeys,
      Environment env,
      boolean debug)
      throws InterruptedException, InvalidPlatformException, InvalidConstraintValueException {

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

  @Nullable
  private static Map<ConfiguredTargetKey, PlatformInfo> getPlatformInfo(
      Iterable<ConfiguredTargetKey> platformKeys, Environment env)
      throws InterruptedException, InvalidPlatformException {

    PlatformLookupValue.Key key = PlatformLookupValue.key(platformKeys);
    PlatformLookupValue platforms =
        (PlatformLookupValue) env.getValueOrThrow(key, InvalidPlatformException.class);
    if (platforms == null) {
      return null;
    }
    return platforms.platforms();
  }

  @Nullable
  private static List<ConstraintValueInfo> getConstraintValueInfo(
      Iterable<ConfiguredTargetKey> constraintValueKeys, Environment env)
      throws InterruptedException, InvalidConstraintValueException {

    ConstraintValueLookupValue.Key key = ConstraintValueLookupValue.key(constraintValueKeys);
    ConstraintValueLookupValue constraintValues =
        (ConstraintValueLookupValue) env.getValueOrThrow(key, InvalidConstraintValueException.class);
    if (constraintValues == null) {
      return null;
    }
    return constraintValues.constraintValues();
  }

  @AutoValue
  protected abstract static class ResolvedToolchains {

    abstract ConfiguredTargetKey executionPlatformKey();

    abstract ConfiguredTargetKey targetPlatformKey();

    abstract ImmutableBiMap<Label, Label> toolchains();

    protected static ResolvedToolchains create(
        ConfiguredTargetKey executionPlatformKey,
        ConfiguredTargetKey targetPlatformKey,
        Map<Label, Label> toolchains) {
      return new AutoValue_ToolchainContextBuilder_ResolvedToolchains(
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
      throws InterruptedException, ToolchainException {

    // Find the toolchains for the required toolchain types.
    List<ToolchainResolutionValue.Key> registeredToolchainKeys = new ArrayList<>();
    for (Label toolchainType : requiredToolchains) {
      registeredToolchainKeys.add(
          ToolchainResolutionValue.key(
              configurationKey, toolchainType, targetPlatformKey, availableExecutionPlatformKeys));
    }

    Map<SkyKey, ValueOrException2<NoToolchainFoundException, InvalidToolchainLabelException>>
        results =
            env.getValuesOrThrow(
                registeredToolchainKeys,
                NoToolchainFoundException.class,
                InvalidToolchainLabelException.class);
    boolean valuesMissing = false;

    // Determine the potential set of toolchains.
    Table<ConfiguredTargetKey, Label, Label> resolvedToolchains = HashBasedTable.create();
    List<Label> missingToolchains = new ArrayList<>();
    for (Map.Entry<
            SkyKey, ValueOrException2<NoToolchainFoundException, InvalidToolchainLabelException>>
        entry : results.entrySet()) {
      try {
        Label requiredToolchainType =
            ((ToolchainResolutionValue.Key) entry.getKey().argument()).toolchainType();
        ValueOrException2<NoToolchainFoundException, InvalidToolchainLabelException>
            valueOrException = entry.getValue();
        ToolchainResolutionValue toolchainResolutionValue =
            (ToolchainResolutionValue) valueOrException.get();
        if (toolchainResolutionValue == null) {
          valuesMissing = true;
          continue;
        }

        addPlatformsAndLabels(resolvedToolchains, requiredToolchainType, toolchainResolutionValue);
      } catch (NoToolchainFoundException e) {
        // Save the missing type and continue looping to check for more.
        missingToolchains.add(e.missingToolchainType());
      }
    }

    if (!missingToolchains.isEmpty()) {
      throw new UnresolvedToolchainsException(missingToolchains);
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
      throw new NoMatchingPlatformException(
          requiredToolchains, availableExecutionPlatformKeys, targetPlatformKey);
    }

    return ResolvedToolchains.create(
        selectedExecutionPlatformKey.get(),
        targetPlatformKey,
        resolvedToolchains.row(selectedExecutionPlatformKey.get()));
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

    return Optional.empty();
  }

  /** Returns a description of the target being used, for error messaging. */
  abstract String targetDescription();

  /** Returns the toolchain types that were requested. */
  public abstract ImmutableSet<Label> requiredToolchainTypes();

  /** Returns the selected execution platform that these toolchains use. */
  public abstract PlatformInfo executionPlatform();

  /** Returns the target platform that these toolchains generate output for. */
  public abstract PlatformInfo targetPlatform();

  // DO NOT USE: Internal only.
  abstract ImmutableBiMap<Label, Label> toolchainTypeToResolved();

  /** Returns the labels of the specific toolchains being used. */
  public ImmutableSet<Label> resolvedToolchainLabels() {
    return ImmutableSet.copyOf(toolchainTypeToResolved().values());
  }

  /** Filter the given {@link Label labels} and return only those that are toolchains being used. */
  public Set<Label> filterToolchainLabels(Iterable<Label> labels) {
    return StreamSupport.stream(labels.spliterator(), false)
        .filter(this::isToolchainLabel)
        .collect(ImmutableSet.toImmutableSet());
  }

  private boolean isToolchainLabel(Label label) {
    return toolchainTypeToResolved().containsValue(label);
  }

  /** Perform final checks and return a newly created {@link ToolchainContext} ready to be used. */
  public ToolchainContext loadToolchainProviders(
      OrderedSetMultimap<Attribute, ConfiguredTargetAndData> prerequisiteMap) {
    // Find the prerequisites associated with PlatformSemantics.RESOLVED_TOOLCHAINS_ATTR.
    Optional<Attribute> toolchainAttribute =
        prerequisiteMap
            .keys()
            .stream()
            .filter(Objects::nonNull)
            .filter(
                attribute -> attribute.getName().equals(PlatformSemantics.RESOLVED_TOOLCHAINS_ATTR))
            .findFirst();
    ImmutableMap.Builder<Label, ToolchainInfo> toolchains = new ImmutableMap.Builder<>();
    if (toolchainAttribute.isPresent()) {
      for (ConfiguredTargetAndData target : prerequisiteMap.get(toolchainAttribute.get())) {
        Label discoveredLabel = target.getTarget().getLabel();
        Label toolchainType = toolchainTypeToResolved().inverse().get(discoveredLabel);
        if (toolchainType != null) {
          ToolchainInfo toolchainInfo =
              PlatformProviderUtils.toolchain(target.getConfiguredTarget());
          toolchains.put(toolchainType, toolchainInfo);
        }

        // Find any template variables present for this toolchain.
        // TODO(jcater): save this somewhere.
      }
    }

    return ToolchainContext.create(
        targetDescription(),
        executionPlatform(),
        targetPlatform(),
        requiredToolchainTypes(),
        toolchainTypeToResolved(),
        toolchains.build());
  }

  /** Exception used when no execution platform can be found. */
  static final class NoMatchingPlatformException extends ToolchainException {
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

  /** Exception used when a toolchain type is required but no matching toolchain is found. */
  static final class UnresolvedToolchainsException extends ToolchainException {
    UnresolvedToolchainsException(List<Label> missingToolchainTypes) {
      super(
          String.format(
              "no matching toolchains found for types %s",
              Joiner.on(", ").join(missingToolchainTypes)));
    }
  }
}
