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

package com.google.devtools.build.lib.rules.objc;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.packages.Attribute.SplitTransition;
import com.google.devtools.build.lib.rules.apple.AppleCommandLineOptions;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration.ConfigurationDistinguisher;
import java.util.List;

/**
 * Transition that produces a configuration that causes c++ toolchain selection to use the
 * CROSSTOOL given in apple_crosstool_top.
 *
 * <p>Duplicates {@link AppleCrosstoolTransition} as a {@link SplitTransition}.  This is necessary
 * for the top level configuration hook, until the top level configuration hook is supported for
 * dynamic configurations.
 *
 * <p>TODO(b/34241319): Use AppleCrosstoolTransition at the top level, and retire this class.
 */
public class AppleCrosstoolSplitTransition implements SplitTransition<BuildOptions> {

  /**
   * A singleton instance of {@link AppleCrosstoolSplitTransition}, since the class must be
   * stateless.  Use in BuildConfigurationCollection.Transitions#topLevelConfigurationHook.
   */
  public static final AppleCrosstoolSplitTransition APPLE_CROSSTOOL_SPLIT_TRANSITION =
      new AppleCrosstoolSplitTransition();

  @Override
  public boolean defaultsToSelf() {
    return true;
  }

  @Override
  public List<BuildOptions> split(BuildOptions buildOptions) {
    BuildOptions result = buildOptions.clone();
    result.get(AppleCommandLineOptions.class).configurationDistinguisher =
        ConfigurationDistinguisher.APPLE_CROSSTOOL;

    // TODO(b/29355778): Once ios_cpu is retired, introduce another top-level flag (perhaps
    // --apple_cpu) for toolchain selection in top-level consuming rules.
    String cpu = "ios_" + buildOptions.get(AppleCommandLineOptions.class).iosCpu;
    AppleCrosstoolTransition.setAppleCrosstoolTransitionConfiguration(buildOptions, result, cpu);

    return ImmutableList.of(result);
  }
}
