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

package com.google.devtools.build.lib.skyframe;

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.packages.ConfiguredAttributeMapper;
import com.google.devtools.build.lib.packages.InputFile;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.OutputFile;
import com.google.devtools.build.lib.packages.RequiredProviders;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.TargetData;
import com.google.devtools.build.lib.packages.TestTimeout;
import com.google.devtools.build.lib.skyframe.config.BuildConfigurationKey;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyframeLookupResult;
import java.util.Comparator;
import java.util.Set;
import javax.annotation.Nullable;
import net.starlark.java.syntax.Location;

/**
 * A container class for a {@link ConfiguredTarget} and associated data, {@link Target}, {@link
 * BuildConfigurationValue}, and transition keys. In the future, {@link ConfiguredTarget} objects
 * will no longer contain their associated {@link BuildConfigurationValue}. Consumers that need the
 * {@link Target} or {@link BuildConfigurationValue} must therefore have access to one of these
 * objects.
 *
 * <p>These objects are intended to be short-lived, never stored in Skyframe, since they pair three
 * heavyweight objects, a {@link ConfiguredTarget}, a {@link Target} (which holds a {@link
 * com.google.devtools.build.lib.packages.Package}), and a {@link BuildConfigurationValue}.
 */
public class ConfiguredTargetAndData {
  /**
   * Orders split dependencies by configuration.
   *
   * <p>Requires non-null configurations.
   */
  public static final Comparator<ConfiguredTargetAndData> SPLIT_DEP_ORDERING =
      new SplitDependencyComparator();

  private final ConfiguredTarget configuredTarget;

  // TODO(b/297857068): Remove transient, serializing this by creating a projection of the
  // underlying target.
  /** A {@link Target} when locally derived but a lightweight projection when fetched remotely. */
  private final transient TargetData target;

  @Nullable // Null iff configuredTarget's configuration key is null.
  private final BuildConfigurationValue configuration;
  private final ImmutableList<String> transitionKeys;

  @VisibleForTesting
  public ConfiguredTargetAndData(
      ConfiguredTarget configuredTarget,
      TargetData target,
      @Nullable BuildConfigurationValue configuration,
      ImmutableList<String> transitionKeys) {
    this(configuredTarget, target, configuration, transitionKeys, /*checkConsistency=*/ true);
  }

  private ConfiguredTargetAndData(
      ConfiguredTarget configuredTarget,
      TargetData target,
      @Nullable BuildConfigurationValue configuration,
      ImmutableList<String> transitionKeys,
      boolean checkConsistency) {
    this.configuredTarget = configuredTarget;
    this.target = target;
    this.configuration = configuration;
    this.transitionKeys = transitionKeys;
    if (!checkConsistency) {
      return;
    }
    checkState(
        configuredTarget.getLabel().equals(target.getLabel()),
        "Unable to construct ConfiguredTargetAndData:"
            + " ConfiguredTarget's label %s is not equal to Target's label %s",
        configuredTarget.getLabel(),
        target.getLabel());
    BuildConfigurationKey innerConfigurationKey = configuredTarget.getConfigurationKey();
    if (configuration == null) {
      checkState(
          innerConfigurationKey == null,
          "Non-null configuration key for %s but configuration is null (%s)",
          configuredTarget,
          target);
    } else {
      checkState(
          innerConfigurationKey.getOptions().equals(configuration.getOptions()),
          "Configurations don't match: %s %s (%s %s)",
          innerConfigurationKey,
          configuration,
          configuredTarget,
          target);
    }
  }

  /**
   * Wraps an existing {@link ConfiguredTarget} by looking up auxiliary data in Skyframe.
   *
   * <p>Assumes that since the given {@link ConfiguredTarget} is done, then its associated {@link
   * PackageValue} and {@link BuildConfigurationValue} (if applicable) are done too.
   */
  static ConfiguredTargetAndData fromExistingConfiguredTargetInSkyframe(
      ConfiguredTarget ct, SkyFunction.Environment env) throws InterruptedException {
    BuildConfigurationValue configuration = null;
    ImmutableSet<SkyKey> packageAndMaybeConfiguration;
    PackageIdentifier packageKey = ct.getLabel().getPackageIdentifier();
    BuildConfigurationKey configurationKeyMaybe = ct.getConfigurationKey();
    if (configurationKeyMaybe == null) {
      packageAndMaybeConfiguration = ImmutableSet.of(packageKey);
    } else {
      packageAndMaybeConfiguration = ImmutableSet.of(packageKey, configurationKeyMaybe);
    }
    SkyframeLookupResult packageAndMaybeConfigurationValues =
        env.getValuesAndExceptions(packageAndMaybeConfiguration);
    // Don't test env.valuesMissing(), because values may already be missing from the caller.
    PackageValue packageValue = (PackageValue) packageAndMaybeConfigurationValues.get(packageKey);
    checkNotNull(packageValue, "Missing package for %s", ct);
    if (configurationKeyMaybe != null) {
      configuration =
          (BuildConfigurationValue) packageAndMaybeConfigurationValues.get(configurationKeyMaybe);
      checkNotNull(configuration, "Missing configuration for %s", ct);
    }
    try {
      return new ConfiguredTargetAndData(
          ct, packageValue.getPackage().getTarget(ct.getLabel().getName()), configuration, null);
    } catch (NoSuchTargetException e) {
      throw new IllegalStateException("Failed to retrieve target for " + ct, e);
    }
  }

  /**
   * For use with {@code MergedConfiguredTarget} and similar, where we create a virtual {@link
   * ConfiguredTarget} corresponding to the same {@link Target}.
   */
  public ConfiguredTargetAndData fromConfiguredTarget(ConfiguredTarget maybeNew) {
    if (configuredTarget.equals(maybeNew)) {
      return this;
    }
    return new ConfiguredTargetAndData(maybeNew, target, configuration, transitionKeys);
  }

  /**
   * Variation of {@link #fromConfiguredTarget} that doesn't check the new target has the same
   * configuration as the original.
   *
   * <p>Intended for trimming (like {@code --trim_test_configuration}).
   */
  public ConfiguredTargetAndData fromConfiguredTargetNoCheck(ConfiguredTarget maybeNew) {
    if (configuredTarget.equals(maybeNew)) {
      return this;
    }
    return new ConfiguredTargetAndData(maybeNew, target, configuration, transitionKeys, false);
  }

  @Nullable
  public BuildConfigurationValue getConfiguration() {
    return configuration;
  }

  @Nullable
  public BuildConfigurationKey getConfigurationKey() {
    return configuredTarget.getConfigurationKey();
  }

  public ConfiguredTarget getConfiguredTarget() {
    return configuredTarget;
  }

  public ImmutableList<String> getTransitionKeys() {
    return transitionKeys;
  }

  public Label getTargetLabel() {
    return target.getLabel();
  }

  public Location getLocation() {
    return target.getLocation();
  }

  public String getTargetKind() {
    return target.getTargetKind();
  }

  public boolean isForDependencyResolution() {
    return target.isForDependencyResolution();
  }

  /** Returns the rule class name if the target is a rule and "" otherwise. */
  public String getRuleClass() {
    return target.getRuleClass();
  }

  /** Returns the rule tags if the target is a rule and an empty set otherwise. */
  public Set<String> getRuleTags() {
    return target.getRuleTags();
  }

  public boolean isTargetRule() {
    return target.isRule();
  }

  public boolean isTargetFile() {
    return target.isFile();
  }

  public boolean isTargetInputFile() {
    return target.isInputFile();
  }

  public boolean isTargetOutputFile() {
    return target.isOutputFile();
  }

  /** The generating rule's label if the target is an {@link OutputFile} otherwise null. */
  @Nullable
  public Label getGeneratingRuleLabel() {
    return target.getGeneratingRuleLabel();
  }

  /** The input file path if the target is an {@link InputFile} otherwise null. */
  @Nullable
  public Path getInputPath() {
    return target.getInputPath();
  }

  /** Any deprecation warning of the associated rule (maybe generating) otherwise null. */
  @Nullable
  public String getDeprecationWarning() {
    return target.getDeprecationWarning();
  }

  /** True if the target is a testonly rule or an output file generated by a testonly rule. */
  public boolean isTestOnly() {
    return target.isTestOnly();
  }

  /**
   * True if the underlying target advertises the required providers.
   *
   * <p>This is used to determine whether an aspect should propagate to this configured target.
   */
  public boolean satisfies(RequiredProviders required) {
    // TODO(shahan): If this is an output file, refers to the providers of the generating rule.
    // However, in such cases, aspects are not permitted to have required providers. Consider
    // short-circuiting the logic for that case.

    // NOTE: it is tempting to use providers of `configuredTarget` instead, however, it may contain
    // providers that are not advertised and can lead to illegal aspect propagation.
    return target.satisfies(required);
  }

  @Nullable
  public TestTimeout getTestTimeout() {
    return target.getTestTimeout();
  }

  public ConfiguredTargetAndData copyWithClearedTransitionKeys() {
    if (transitionKeys.isEmpty()) {
      return this;
    }
    return copyWithTransitionKeys(ImmutableList.of());
  }

  public ConfiguredTargetAndData copyWithTransitionKeys(ImmutableList<String> keys) {
    return new ConfiguredTargetAndData(configuredTarget, target, configuration, keys);
  }

  /**
   * This should only be used in testing.
   *
   * <p>Distributed implementations of prerequisites do not contain targets, but only a bare minimum
   * of fields needed by consumers.
   */
  @VisibleForTesting
  public Target getTargetForTesting() {
    return (Target) target;
  }

  @VisibleForTesting
  public ConfiguredAttributeMapper getAttributeMapperForTesting() {
    return ConfiguredAttributeMapper.of(
        (Rule) target, configuredTarget.getConfigConditions(), configuration);
  }

  private static final class SplitDependencyComparator
      implements Comparator<ConfiguredTargetAndData> {
    @Override
    public int compare(ConfiguredTargetAndData o1, ConfiguredTargetAndData o2) {
      BuildConfigurationValue first = o1.getConfiguration();
      BuildConfigurationValue second = o2.getConfiguration();
      int result = first.getMnemonic().compareTo(second.getMnemonic());
      if (result != 0) {
        return result;
      }
      return first.checksum().compareTo(second.checksum());
    }
  }
}
