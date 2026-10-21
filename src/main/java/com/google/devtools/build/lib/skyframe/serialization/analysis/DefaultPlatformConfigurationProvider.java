// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe.serialization.analysis;

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSortedMap;
import com.google.devtools.build.lib.analysis.PlatformOptions;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.analysis.config.OutputPathMnemonicComputer;
import com.google.devtools.build.lib.analysis.test.TestConfiguration.TestOptions;
import com.google.devtools.build.lib.analysis.test.TestTrimmingLogic;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skyframe.serialization.PlatformConfigurationProvider;
import com.google.devtools.common.options.Options;

/**
 * Default implementation of {@link PlatformConfigurationProvider} with option trimming and mnemonic
 * resolution.
 */
public final class DefaultPlatformConfigurationProvider implements PlatformConfigurationProvider {
  private final Label topLevelPlatformLabel;
  private final BuildOptions targetBaseline;
  private final BuildOptions targetBaselineTrimmed;
  private final BuildOptions targetBaselineWithDefaultTestOptions;
  private final BuildOptions execBaseline;
  private final BuildOptions execBaselineTrimmed;
  private final BuildOptions execBaselineWithDefaultTestOptions;

  public DefaultPlatformConfigurationProvider(
      Label topLevelPlatformLabel, BuildOptions targetBaseline, BuildOptions execBaseline) {
    this.topLevelPlatformLabel = checkNotNull(topLevelPlatformLabel);
    this.targetBaseline = checkNotNull(targetBaseline);
    this.targetBaselineTrimmed = TestTrimmingLogic.trim(targetBaseline);
    this.targetBaselineWithDefaultTestOptions = withDefaultTestOptions(targetBaselineTrimmed);
    this.execBaseline = checkNotNull(execBaseline);
    this.execBaselineTrimmed = TestTrimmingLogic.trim(execBaseline);
    this.execBaselineWithDefaultTestOptions = withDefaultTestOptions(execBaselineTrimmed);
  }

  /**
   * Returns {@code trimmedBaseline} with {@link TestOptions} at their default values.
   *
   * <p>This is the baseline that untrimmed configurations are diffed against. It must not contain
   * this build's {@link TestOptions} or its untrimmed {@code --run_under}: those are not part of
   * the configuration checksum in {@code FrontierNodeVersion}, which is computed from the
   * test-trimmed top-level options. If the baseline carried them, they would cancel out of the
   * diff, and builds that differ only in test flags would serialize their untrimmed keys to
   * identical bytes under the same {@code FrontierNodeVersion}, sharing cache entries that they
   * must not share (b/570095568). The trimmed part of the baseline is covered by {@code
   * FrontierNodeVersion}, and the defaults are the same for every build of the same Blaze binary,
   * so with this baseline every test option that differs from its default is recorded in the diff.
   *
   * <p>This build's own test options still determine the output directory mnemonics of untrimmed
   * configurations (see {@link #getActualBaseOptionsForPlatform}), so the codec records them
   * separately for untrimmed keys.
   */
  private static BuildOptions withDefaultTestOptions(BuildOptions trimmedBaseline) {
    return trimmedBaseline.toBuilder()
        .addFragmentOptions(Options.getDefaults(TestOptions.class))
        .build();
  }

  @Override
  public Label getTopLevelPlatformLabel() {
    return topLevelPlatformLabel;
  }

  @Override
  public BuildOptions getBaseOptionsForPlatform(
      Label platformLabel, boolean isExec, boolean trimTestOptions) {
    Preconditions.checkNotNull(platformLabel);
    BuildOptions base =
        isExec
            ? (trimTestOptions ? execBaselineTrimmed : execBaselineWithDefaultTestOptions)
            : (trimTestOptions ? targetBaselineTrimmed : targetBaselineWithDefaultTestOptions);
    return withPlatform(base, platformLabel);
  }

  /**
   * Returns this build's baseline, which, unlike the one returned by {@link
   * #getBaseOptionsForPlatform}, includes this build's {@link TestOptions} and full {@code
   * --run_under} for untrimmed configurations. Output directory mnemonics depend on these, since
   * Skyframe computes them against this build's baseline.
   */
  @Override
  public BuildOptions getActualBaseOptionsForPlatform(
      Label platformLabel, boolean isExec, boolean trimTestOptions) {
    Preconditions.checkNotNull(platformLabel);
    BuildOptions base =
        isExec
            ? (trimTestOptions ? execBaselineTrimmed : execBaseline)
            : (trimTestOptions ? targetBaselineTrimmed : targetBaseline);
    return withPlatform(base, platformLabel);
  }

  private static BuildOptions withPlatform(BuildOptions base, Label platformLabel) {
    BuildOptions clonedOptions = base.clone();
    clonedOptions.get(PlatformOptions.class).setPlatforms(ImmutableList.of(platformLabel));
    return clonedOptions;
  }

  @Override
  public String resolveMnemonic(BuildOptions targetOptions) {
    Preconditions.checkNotNull(targetOptions);
    PlatformOptions platformOptions = targetOptions.get(PlatformOptions.class);
    Label platformLabel = platformOptions != null ? platformOptions.computeTargetPlatform() : null;
    CoreOptions coreOptions = targetOptions.get(CoreOptions.class);
    Preconditions.checkNotNull(coreOptions);
    Preconditions.checkNotNull(platformLabel);
    Preconditions.checkArgument(
        coreOptions.usePlatformInOutputDir(platformLabel),
        "Platform %s is not used in output directory",
        platformLabel);
    boolean isExec = coreOptions.getIsExec();
    boolean trimTestOptions = !targetOptions.contains(TestOptions.class);
    BuildOptions baselineOptions =
        getActualBaseOptionsForPlatform(platformLabel, isExec, trimTestOptions);
    try {
      return OutputPathMnemonicComputer.computeMnemonic(
              targetOptions, baselineOptions, ImmutableSortedMap.of())
          .mnemonic();
    } catch (OutputPathMnemonicComputer.InvalidMnemonicException e) {
      throw new IllegalStateException("Invalid mnemonic for " + targetOptions, e);
    }
  }
}
