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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.PlatformConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.platform.PlatformInfo;
import com.google.devtools.build.lib.analysis.platform.PlatformProviderUtils;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.pkgcache.FilteringPolicy;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetFunction.ConfiguredValueCreationException;
import com.google.devtools.build.lib.skyframe.ToolchainUtil.InvalidPlatformException;
import com.google.devtools.build.lib.skyframe.TargetPatternValue.TargetPatternKey;
import com.google.devtools.build.lib.skyframe.TargetPatternValue.TargetPatternSkyKeyOrException;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.ValueOrException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

/** {@link SkyFunction} that returns all registered execution platforms available. */
public class RegisteredExecutionPlatformsFunction implements SkyFunction {

  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws SkyFunctionException, InterruptedException {

    BuildConfigurationValue buildConfigurationValue =
        (BuildConfigurationValue)
            env.getValue(((RegisteredExecutionPlatformsValue.Key) skyKey).getConfigurationKey());
    if (env.valuesMissing()) {
      return null;
    }
    BuildConfiguration configuration = buildConfigurationValue.getConfiguration();

    ImmutableList.Builder<String> targetPatterns = new ImmutableList.Builder<>();

    // Get the execution platforms from the configuration.
    PlatformConfiguration platformConfiguration =
        configuration.getFragment(PlatformConfiguration.class);
    if (platformConfiguration != null) {
      targetPatterns.addAll(platformConfiguration.getExtraExecutionPlatforms());
    }

    // Get the registered execution platforms from the WORKSPACE.
    List<String> workspaceExecutionPlatforms = getWorkspaceExecutionPlatforms(env);
    if (workspaceExecutionPlatforms == null) {
      return null;
    }
    targetPatterns.addAll(workspaceExecutionPlatforms);

    // Expand target patterns.
    ImmutableList<Label> platformLabels = expandTargetPatterns(env, targetPatterns.build());
    if (env.valuesMissing()) {
      return null;
    }

    // Load the configured target for each, and get the declared execution platforms providers.
    ImmutableList<ConfiguredTargetKey> registeredExecutionPlatformKeys =
        configureRegisteredExecutionPlatforms(env, configuration, platformLabels);
    if (env.valuesMissing()) {
      return null;
    }

    return RegisteredExecutionPlatformsValue.create(registeredExecutionPlatformKeys);
  }

  /**
   * Loads the external package and then returns the registered execution platform labels.
   *
   * @param env the environment to use for lookups
   */
  @Nullable
  @VisibleForTesting
  public static List<String> getWorkspaceExecutionPlatforms(Environment env)
      throws InterruptedException {
    PackageValue externalPackageValue =
        (PackageValue) env.getValue(PackageValue.key(Label.EXTERNAL_PACKAGE_IDENTIFIER));
    if (externalPackageValue == null) {
      return null;
    }

    Package externalPackage = externalPackageValue.getPackage();
    return externalPackage.getRegisteredExecutionPlatforms();
  }

  private static class PlatformFilteringPolicy extends FilteringPolicy {
    private static final PlatformFilteringPolicy INSTANCE = new PlatformFilteringPolicy();

    @Override
    public boolean shouldRetain(Target target, boolean explicit) {
      if (explicit) {
        return true;
      }

      if (target.getAssociatedRule().getRuleClass().equals("platform")) {
        return true;
      }

      return false;
    }
  }

  @Nullable
  private static ImmutableList<Label> expandTargetPatterns(Environment env, List<String> targetPatterns)
      throws RegisteredExecutionPlatformsFunctionException, InterruptedException {
    // First parse the patterns, and throw any errors immediately.
    List<TargetPatternKey> patternKeys = new ArrayList<>();
    for (TargetPatternSkyKeyOrException keyOrException :
        TargetPatternValue.keys(targetPatterns, PlatformFilteringPolicy.INSTANCE, "")) {

      try {
        patternKeys.add(keyOrException.getSkyKey());
      } catch (TargetParsingException e) {
        throw new RegisteredExecutionPlatformsFunctionException(
            new InvalidExecutionPlatformLabelException(keyOrException.getOriginalPattern(), e),
            Transience.PERSISTENT);
      }
    }

    // Then, resolve the patterns.
    ImmutableList.Builder<Label> labels = new ImmutableList.Builder<>();
    Map<SkyKey, ValueOrException<TargetParsingException>> resolvedPatterns =
        env.getValuesOrThrow(patternKeys, TargetParsingException.class);
    if (env.valuesMissing()) {
      return null;
    }

    for (TargetPatternKey pattern : patternKeys) {
      TargetPatternValue value;
      try {
        value = (TargetPatternValue) resolvedPatterns.get(pattern).get();
        labels.addAll(value.getTargets().getTargets());
      } catch (TargetParsingException e) {
        throw new RegisteredExecutionPlatformsFunctionException(
            new InvalidExecutionPlatformLabelException(pattern.getPattern(), e), Transience.PERSISTENT);
      }
    }

    return labels.build();
  }

  private ImmutableList<ConfiguredTargetKey> configureRegisteredExecutionPlatforms(
      Environment env, BuildConfiguration configuration, List<Label> labels)
      throws InterruptedException, RegisteredExecutionPlatformsFunctionException {
    ImmutableList<ConfiguredTargetKey> keys =
        labels
            .stream()
            .map(label -> ConfiguredTargetKey.of(label, configuration))
            .collect(ImmutableList.toImmutableList());

    // Load the actual configured targets and ensure that they have real, valid PlatformInfo
    // instances. These are loaded later during toolchain resolution (see
    // ToolchainUtil#getPlatformInfo), so this is work that needs to be done anyway, but here we can
    // fail fast on an error.
    Map<SkyKey, ValueOrException<ConfiguredValueCreationException>> values =
        env.getValuesOrThrow(keys, ConfiguredValueCreationException.class);
    boolean valuesMissing = false;
    for (SkyKey key : keys) {
      ConfiguredTargetKey configuredTargetKey = (ConfiguredTargetKey) key.argument();
      Label platformLabel = configuredTargetKey.getLabel();
      try {
        ValueOrException<ConfiguredValueCreationException> valueOrException = values.get(key);
        if (valueOrException.get() == null) {
          valuesMissing = true;
          continue;
        }
        ConfiguredTarget target =
            ((ConfiguredTargetValue) valueOrException.get()).getConfiguredTarget();
        PlatformInfo platformInfo = PlatformProviderUtils.platform(target);

        if (platformInfo == null) {
          throw new RegisteredExecutionPlatformsFunctionException(
              new InvalidPlatformException(platformLabel), Transience.PERSISTENT);
        }
      } catch (ConfiguredValueCreationException e) {
        throw new RegisteredExecutionPlatformsFunctionException(
            new InvalidPlatformException(platformLabel, e), Transience.PERSISTENT);
      }
    }

    if (valuesMissing) {
      return null;
    }
    return keys;
  }

  @Nullable
  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }

  /**
   * Used to indicate that the given {@link Label} represents a {@link ConfiguredTarget} which is
   * not a valid {@link PlatformInfo} provider.
   */
  static final class InvalidExecutionPlatformLabelException extends Exception {

    private InvalidExecutionPlatformLabelException(Label invalidLabel) {
      super(
          String.format(
              "invalid registered execution platform '%s': "
                  + "target does not provide the PlatformInfo provider",
              invalidLabel));
    }

    public InvalidExecutionPlatformLabelException(String invalidPattern, TargetParsingException e) {
      super(
          String.format("invalid registered execution platform '%s': %s", invalidPattern, e.getMessage()),
          e);
    }

    private InvalidExecutionPlatformLabelException(
        Label invalidLabel, ConfiguredValueCreationException e) {
      super(
          String.format(
              "invalid registered execution platform '%s': %s", invalidLabel, e.getMessage()),
          e);
    }
  }

  /**
   * Used to declare all the exception types that can be wrapped in the exception thrown by {@link
   * #compute}.
   */
  private static class RegisteredExecutionPlatformsFunctionException extends SkyFunctionException {

    private RegisteredExecutionPlatformsFunctionException(
        InvalidExecutionPlatformLabelException cause, Transience transience) {
      super(cause, transience);
    }
    private RegisteredExecutionPlatformsFunctionException(
        InvalidPlatformException cause, Transience transience) {
      super(cause, transience);
    }
  }
}
