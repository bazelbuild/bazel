// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe.config;

import static com.google.devtools.build.lib.analysis.constraints.ConstraintConstants.CPU_CONSTRAINT_SETTING;

import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.PlatformOptions;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.ConfigurationValueEvent;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.analysis.config.FragmentFactory;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.analysis.config.StarlarkExecTransitionLoader.StarlarkExecTransitionLoadingException;
import com.google.devtools.build.lib.analysis.config.transitions.BaselineOptionsValue;
import com.google.devtools.build.lib.analysis.platform.PlatformValue;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.RuleClassProvider;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Optional;
import javax.annotation.Nullable;
import net.starlark.java.eval.StarlarkSemantics;

/** A builder for {@link BuildConfigurationValue} instances. */
public final class BuildConfigurationFunction implements SkyFunction {

  private final BlazeDirectories directories;
  private final ConfiguredRuleClassProvider ruleClassProvider;
  private final FragmentFactory fragmentFactory = new FragmentFactory();

  public BuildConfigurationFunction(
      BlazeDirectories directories, RuleClassProvider ruleClassProvider) {
    this.directories = directories;
    this.ruleClassProvider = (ConfiguredRuleClassProvider) ruleClassProvider;
  }

  @Override
  @Nullable
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws InterruptedException, BuildConfigurationFunctionException {
    StarlarkSemantics starlarkSemantics = PrecomputedValue.STARLARK_SEMANTICS.get(env);
    if (starlarkSemantics == null) {
      return null;
    }
    BuildConfigurationKey key = (BuildConfigurationKey) skyKey.argument();

    BuildOptions targetOptions = key.getOptions();
    Optional<BuildOptions> baselineOptions = getBaselineOptions(env, targetOptions);
    if (baselineOptions == null) {
      return null;
    }

    String targetCpu = getTargetCpu(env, targetOptions);
    if (targetCpu == null) {
      return null;
    }

    try {
      var configurationValue =
          BuildConfigurationValue.create(
              targetOptions,
              baselineOptions.orElse(null),
              starlarkSemantics.getBool(
                  BuildLanguageOptions.EXPERIMENTAL_SIBLING_REPOSITORY_LAYOUT),
              targetCpu,
              // Arguments below this are server-global.
              directories,
              ruleClassProvider,
              fragmentFactory);
      env.getListener().post(ConfigurationValueEvent.create(configurationValue));
      return configurationValue;
    } catch (InvalidConfigurationException e) {
      throw new BuildConfigurationFunctionException(e);
    }
  }

  @Nullable
  private static String getTargetCpu(Environment env, BuildOptions targetOptions)
      throws InterruptedException {
    CoreOptions coreOptions = targetOptions.get(CoreOptions.class);
    if (!coreOptions.incompatibleTargetCpuFromPlatform) {
      return coreOptions.cpu;
    }

    if (targetOptions.get(PlatformOptions.class) == null) {
      return "";
    }

    var platformLabel = targetOptions.get(PlatformOptions.class).computeTargetPlatform();

    Optional<String> overridePlatformCpuName =
        coreOptions.getPlatformCpuNameOverride(platformLabel);
    if (overridePlatformCpuName.isPresent()) {
      return overridePlatformCpuName.get();
    }

    PlatformValue platformValue = (PlatformValue) env.getValue(PlatformValue.key(platformLabel));
    if (platformValue == null) {
      return null;
    }

    var cpuConstraint = platformValue.platformInfo().constraints().get(CPU_CONSTRAINT_SETTING);
    if (cpuConstraint == null) {
      return "";
    }

    return cpuConstraint.label().getName();
  }

  /**
   * Determine the baseline options to use for tracking changes.
   *
   * <p>Returns {@code null} if a Skyframe restart is needed, or an {@link Optional} with either the
   * baseline options to use, or none if there is no valid baseline.
   */
  @Nullable
  private static Optional<BuildOptions> getBaselineOptions(
      Environment env, BuildOptions targetOptions)
      throws InterruptedException, BuildConfigurationFunctionException {

    if (targetOptions.hasNoConfig()) {
      return Optional.empty();
    }

    CoreOptions coreOptions = targetOptions.get(CoreOptions.class);
    if (!coreOptions.useBaselineForOutputDirectoryNamingScheme()) {
      return Optional.empty();
    }

    var platformOptions = targetOptions.get(PlatformOptions.class);

    // Determine whether this is part of the exec transition, or if we need to calculate a target
    // platform.
    boolean useDynamicBaseline =
        coreOptions.outputDirectoryNamingScheme.equals(
            CoreOptions.OutputDirectoryNamingScheme.DIFF_AGAINST_DYNAMIC_BASELINE);
    boolean applyExecTransitionToBaseline = useDynamicBaseline && coreOptions.isExec;
    // In practice, platforms should always be 'well-formed' and contain at most one Label.
    Label newPlatform = null;
    if (useDynamicBaseline
        && coreOptions.platformInOutputDir
        && platformOptions != null
        && platformOptions.platforms != null // this may be overly defensive
        && platformOptions.platforms.size() <= 1) {
      newPlatform = platformOptions.computeTargetPlatform();
    }

    try {
      var baselineOptionsValue =
          (BaselineOptionsValue)
              env.getValueOrThrow(
                  BaselineOptionsValue.key(applyExecTransitionToBaseline, newPlatform),
                  StarlarkExecTransitionLoadingException.class);
      if (baselineOptionsValue == null) {
        return null;
      }
      return Optional.of(baselineOptionsValue.toOptions());
    } catch (StarlarkExecTransitionLoadingException e) {
      throw new BuildConfigurationFunctionException(new InvalidConfigurationException(e));
    }
  }

  private static final class BuildConfigurationFunctionException extends SkyFunctionException {
    BuildConfigurationFunctionException(Exception e) {
      super(e, Transience.PERSISTENT);
    }
  }
}
