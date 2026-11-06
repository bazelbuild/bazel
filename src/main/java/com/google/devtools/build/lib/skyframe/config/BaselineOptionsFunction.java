// Copyright 2023 The Bazel Authors. All rights reserved.
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

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.PlatformOptions;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.transitions.BaselineOptionsValue;
import com.google.devtools.build.lib.analysis.test.TestTrimmingLogic;
import com.google.devtools.build.lib.skyframe.PrecomputedValue.Precomputed;
import com.google.devtools.build.lib.skyframe.toolchains.PlatformLookupUtil.InvalidPlatformException;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.Version;
import com.google.devtools.common.options.OptionsParsingException;
import javax.annotation.Nullable;

/** A builder for {@link BaselineOptionsValue} instances. */
public final class BaselineOptionsFunction implements SkyFunction {

  // Don't use these directly. Instead, use the BuildOptions obtained from this function, which
  // applies the appropriate trimming and transition logic to reduce Skyframe invalidation.
  // Unsharable because of complications in deserializing BuildOptions on startup due to caching.
  public static final Precomputed<BuildOptions> BASELINE_CONFIGURATION =
      Precomputed.createUnshareable("baseline_configuration");
  public static final Precomputed<BuildOptions> BASELINE_EXEC_CONFIGURATION =
      Precomputed.createUnshareable("baseline_exec_configuration");

  private final Version minimalVersionToInject;

  public BaselineOptionsFunction(Version minimalVersionToInject) {
    this.minimalVersionToInject = checkNotNull(minimalVersionToInject);
  }

  @Override
  @Nullable
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws InterruptedException, BaselineOptionsFunctionException {
    env.injectVersion(minimalVersionToInject);

    BaselineOptionsValue.Key key = (BaselineOptionsValue.Key) skyKey.argument();

    BuildOptions rawBaselineOptions;
    if (key.afterExecTransition()) {
      // Use the precomputed baseline exec
      rawBaselineOptions = BASELINE_EXEC_CONFIGURATION.get(env);
    } else {
      // Use the standard baseline
      rawBaselineOptions = BASELINE_CONFIGURATION.get(env);
    }

    // Some test infrastructure only creates mock or partial top-level BuildOptions such that
    // PlatformOptions or even CoreOptions might not be included.
    // In that case, is not worth doing any special processing of the baseline.
    if (rawBaselineOptions.hasNoConfig()) {
      return BaselineOptionsValue.create(rawBaselineOptions);
    }

    if (key.trimTestOptions()) {
      rawBaselineOptions = TestTrimmingLogic.trim(rawBaselineOptions);
    }

    // Make sure platform-based flags (the platform() rule's flags attribute) and platform mappings
    // are applied to the baseline, since these are de facto practical baselines.
    //
    // If this is a target configuration, BuildTool.addPlatformFlags already handled platform-based
    // flags but not platform mappings. Re-applying platform-based flags is redundant but harmless.
    //
    // If this is an exec configuration, the platform is likely different than the target platform.
    // So both platform-based flags and platform mappings must be applied.
    //
    // The reason we don't do everything in BuildTool.addPlatformFlags is because that method
    // supplies the values of BASELINE_CONFIGURATION and BASELINE_EXEC_CONFIGURATION. That happens
    // before analysis, when we can't call BuildConfigurationKeyValue like below because that
    // analyzes platform() configured targets.
    BuildOptions mappedBaselineOptions = mapBuildOptions(env, rawBaselineOptions);
    if (mappedBaselineOptions == null) {
      return null;
    }
    BuildOptions adjustedBaselineOptions = mappedBaselineOptions;

    if (key.newPlatform() == null) {
      return BaselineOptionsValue.create(mappedBaselineOptions);
    }

    // Callers only set key.newPlatform() when the platform is directly part of
    // blaze-out/<platform_id>-.../ paths. This is important for the remapping logic below.

    // Clone for safety as-is the standard for all transitions.
    adjustedBaselineOptions = adjustedBaselineOptions.clone();
    adjustedBaselineOptions
        .get(PlatformOptions.class)
        .setPlatforms(ImmutableList.of(key.newPlatform()));

    // Re-apply platform_mappings if the current platform is different than the baseline platform.
    //
    // This is because the new platform may set flags, like --foo=bar. The baseline doesn't have
    // --foo=bar so a naive diff(baseline, currentFlags) would notice --foo is in that diff and add
    // an ST-<hash>. But --foo=bar is an intrinsic property of newPlatform, so
    // blaze-out/<new_platform_id>/ is already safely unique.
    BuildOptions remappedForNewPlatformOptions = mapBuildOptions(env, adjustedBaselineOptions);
    if (remappedForNewPlatformOptions == null) {
      return null;
    }
    return BaselineOptionsValue.create(remappedForNewPlatformOptions);
  }

  @Nullable
  private static BuildOptions mapBuildOptions(Environment env, BuildOptions rawBaselineOptions)
      throws InterruptedException, BaselineOptionsFunctionException {
    var bckvk = BuildConfigurationKeyValue.Key.createForBaseline(rawBaselineOptions);
    try {
      var buildConfigurationKeyValue =
          (BuildConfigurationKeyValue)
              env.getValueOrThrow(
                  bckvk,
                  OptionsParsingException.class,
                  PlatformMappingException.class,
                  InvalidPlatformException.class);
      if (buildConfigurationKeyValue == null) {
        return null;
      }
      return buildConfigurationKeyValue.buildConfigurationKey().getOptions();
    } catch (PlatformMappingException | OptionsParsingException | InvalidPlatformException e) {
      throw new BaselineOptionsFunctionException(e);
    }
  }

  private static final class BaselineOptionsFunctionException extends SkyFunctionException {
    BaselineOptionsFunctionException(Exception e) {
      super(e, Transience.PERSISTENT);
    }
  }
}
