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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.PlatformConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.platform.DeclaredToolchainInfo;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.pkgcache.FilteringPolicies;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetFunction.ConfiguredValueCreationException;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.ValueOrException;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * {@link SkyFunction} that returns all registered toolchains available for toolchain resolution.
 */
public class RegisteredToolchainsFunction implements SkyFunction {

  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws SkyFunctionException, InterruptedException {

    BuildConfigurationValue buildConfigurationValue =
        (BuildConfigurationValue)
            env.getValue(((RegisteredToolchainsValue.Key) skyKey).getConfigurationKey());
    if (env.valuesMissing()) {
      return null;
    }
    BuildConfiguration configuration = buildConfigurationValue.getConfiguration();

    ImmutableList.Builder<String> targetPatternBuilder = new ImmutableList.Builder<>();

    // Get the toolchains from the configuration.
    PlatformConfiguration platformConfiguration =
        configuration.getFragment(PlatformConfiguration.class);
    targetPatternBuilder.addAll(platformConfiguration.getExtraToolchains());

    // Get the registered toolchains from the WORKSPACE.
    targetPatternBuilder.addAll(getWorkspaceToolchains(env));
    if (env.valuesMissing()) {
      return null;
    }
    ImmutableList<String> targetPatterns = targetPatternBuilder.build();

    // Expand target patterns.
    ImmutableList<Label> toolchainLabels;
    try {
      toolchainLabels =
          TargetPatternUtil.expandTargetPatterns(
              env, targetPatterns, FilteringPolicies.ruleType("toolchain", true));
      if (env.valuesMissing()) {
        return null;
      }
    } catch (TargetPatternUtil.InvalidTargetPatternException e) {
      throw new RegisteredToolchainsFunctionException(
          new InvalidToolchainLabelException(e), Transience.PERSISTENT);
    }

    // Load the configured target for each, and get the declared toolchain providers.
    ImmutableList<DeclaredToolchainInfo> registeredToolchains =
        configureRegisteredToolchains(env, configuration, targetPatterns, toolchainLabels);
    if (env.valuesMissing()) {
      return null;
    }

    return RegisteredToolchainsValue.create(registeredToolchains);
  }

  private Iterable<? extends String> getWorkspaceToolchains(Environment env)
      throws InterruptedException {
    List<String> patterns = getRegisteredToolchains(env);
    if (patterns == null) {
      return ImmutableList.of();
    }
    return patterns;
  }

  /**
   * Loads the external package and then returns the registered toolchains.
   *
   * @param env the environment to use for lookups
   */
  @Nullable
  @VisibleForTesting
  public static List<String> getRegisteredToolchains(Environment env) throws InterruptedException {
    PackageValue externalPackageValue =
        (PackageValue) env.getValue(PackageValue.key(LabelConstants.EXTERNAL_PACKAGE_IDENTIFIER));
    if (externalPackageValue == null) {
      return null;
    }

    Package externalPackage = externalPackageValue.getPackage();
    return externalPackage.getRegisteredToolchains();
  }

  private ImmutableList<DeclaredToolchainInfo> configureRegisteredToolchains(
      Environment env,
      BuildConfiguration configuration,
      ImmutableList<String> targetPatterns,
      List<Label> labels)
      throws InterruptedException, RegisteredToolchainsFunctionException {
    ImmutableList<SkyKey> keys =
        labels
            .stream()
            .map(label -> ConfiguredTargetKey.of(label, configuration))
            .collect(toImmutableList());

    Map<SkyKey, ValueOrException<ConfiguredValueCreationException>> values =
        env.getValuesOrThrow(keys, ConfiguredValueCreationException.class);
    ImmutableList.Builder<DeclaredToolchainInfo> toolchains = new ImmutableList.Builder<>();
    boolean valuesMissing = false;
    for (SkyKey key : keys) {
      ConfiguredTargetKey configuredTargetKey = (ConfiguredTargetKey) key.argument();
      Label toolchainLabel = configuredTargetKey.getLabel();
      try {
        ValueOrException<ConfiguredValueCreationException> valueOrException = values.get(key);
        if (valueOrException.get() == null) {
          valuesMissing = true;
          continue;
        }
        ConfiguredTarget target =
            ((ConfiguredTargetValue) valueOrException.get()).getConfiguredTarget();
        DeclaredToolchainInfo toolchainInfo = target.getProvider(DeclaredToolchainInfo.class);

        if (toolchainInfo == null) {
          if (TargetPatternUtil.isTargetExplicit(targetPatterns, toolchainLabel)) {
            // Only report an error if the label was explicitly requested.
            throw new RegisteredToolchainsFunctionException(
                new InvalidToolchainLabelException(toolchainLabel), Transience.PERSISTENT);
          }
          continue;
        }
        toolchains.add(toolchainInfo);
      } catch (ConfiguredValueCreationException e) {
        if (TargetPatternUtil.isTargetExplicit(targetPatterns, toolchainLabel)) {
          // Only report an error if the label was explicitly requested.
          throw new RegisteredToolchainsFunctionException(
              new InvalidToolchainLabelException(toolchainLabel, e), Transience.PERSISTENT);
        }
      }
    }

    if (valuesMissing) {
      return null;
    }
    return toolchains.build();
  }

  @Nullable
  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }

  /**
   * Used to indicate that the given {@link Label} represents a {@link ConfiguredTarget} which is
   * not a valid {@link DeclaredToolchainInfo} provider.
   */
  public static final class InvalidToolchainLabelException extends ToolchainException {

    public InvalidToolchainLabelException(Label invalidLabel) {
      super(
          formatMessage(
              invalidLabel.getCanonicalForm(),
              "target does not provide the DeclaredToolchainInfo provider"));
    }

    public InvalidToolchainLabelException(TargetPatternUtil.InvalidTargetPatternException e) {
      this(e.getInvalidPattern(), e.getTpe());
    }

    public InvalidToolchainLabelException(String invalidPattern, TargetParsingException e) {
      super(formatMessage(invalidPattern, e.getMessage()), e);
    }

    public InvalidToolchainLabelException(Label invalidLabel, ConfiguredValueCreationException e) {
      super(formatMessage(invalidLabel.getCanonicalForm(), e.getMessage()), e);
    }

    private static String formatMessage(String invalidPattern, String reason) {
      return String.format("invalid registered toolchain '%s': %s", invalidPattern, reason);
    }
  }

  /**
   * Used to declare all the exception types that can be wrapped in the exception thrown by {@link
   * #compute}.
   */
  public static class RegisteredToolchainsFunctionException extends SkyFunctionException {

    public RegisteredToolchainsFunctionException(
        InvalidToolchainLabelException cause, Transience transience) {
      super(cause, transience);
    }
  }
}
