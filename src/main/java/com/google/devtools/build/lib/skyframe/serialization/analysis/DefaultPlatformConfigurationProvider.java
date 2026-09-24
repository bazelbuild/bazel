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

/**
 * Default implementation of {@link PlatformConfigurationProvider} with option trimming and mnemonic
 * resolution.
 */
public final class DefaultPlatformConfigurationProvider implements PlatformConfigurationProvider {
  private final Label topLevelPlatformLabel;
  private final BuildOptions targetBaseline;
  private final BuildOptions targetBaselineTrimmed;
  private final BuildOptions execBaseline;
  private final BuildOptions execBaselineTrimmed;

  public DefaultPlatformConfigurationProvider(
      Label topLevelPlatformLabel, BuildOptions targetBaseline, BuildOptions execBaseline) {
    this.topLevelPlatformLabel = checkNotNull(topLevelPlatformLabel);
    this.targetBaseline = checkNotNull(targetBaseline);
    this.targetBaselineTrimmed = TestTrimmingLogic.trim(targetBaseline);
    this.execBaseline = checkNotNull(execBaseline);
    this.execBaselineTrimmed = TestTrimmingLogic.trim(execBaseline);
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
            ? (trimTestOptions ? execBaselineTrimmed : execBaseline)
            : (trimTestOptions ? targetBaselineTrimmed : targetBaseline);

    BuildOptions clonedOptions = base.clone();
    clonedOptions.get(PlatformOptions.class).setPlatforms(ImmutableList.of(platformLabel));
    return clonedOptions;
  }

  @Override
  public boolean trimTestOptions(BuildOptions options) {
    return !options.contains(TestOptions.class);
  }

  @Override
  public boolean usePlatformInOutputDir() {
    CoreOptions coreOptions = targetBaseline.get(CoreOptions.class);
    return coreOptions != null && coreOptions.usePlatformInOutputDir(topLevelPlatformLabel);
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
        getBaseOptionsForPlatform(platformLabel, isExec, trimTestOptions);
    try {
      return OutputPathMnemonicComputer.computeMnemonic(
          targetOptions, baselineOptions, ImmutableSortedMap.of());
    } catch (OutputPathMnemonicComputer.InvalidMnemonicException e) {
      throw new IllegalStateException("Invalid mnemonic for " + targetOptions, e);
    }
  }
}
