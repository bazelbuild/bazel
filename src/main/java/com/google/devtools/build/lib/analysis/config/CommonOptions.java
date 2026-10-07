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
package com.google.devtools.build.lib.analysis.config;

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.common.options.Options;

/** Common sets of option objects for use in core processing. */
public final class CommonOptions {

  /**
   * The values of the flags that configure Bazel's analysis phase rather than rule logic and are
   * thus inherited by the no-config configuration.
   */
  private record AnalysisPhaseFlags(
      boolean checkVisibility,
      boolean verboseVisibilityErrors,
      boolean enforceTransitiveVisibility,
      boolean checkTestonlyForOutputFiles) {
    static AnalysisPhaseFlags of(CoreOptions options) {
      return new AnalysisPhaseFlags(
          options.getCheckVisibility(),
          options.getVerboseVisibilityErrors(),
          options.getEnforceTransitiveVisibility(),
          options.getCheckTestonlyForOutputFiles());
    }
  }

  // Ideally the no-config options should be actually empty: no fragment options and no flags. But
  // core Bazel code assumes CoreOptions exists. For example CoreOptions.check_visibility is
  // required for basic configured target graph evaluation. So we provide CoreOptions with default
  // values. The flags that configure Bazel's analysis phase rather than rule logic are inherited
  // from the parent configuration via noConfigOptions, so there is one instance for each
  // combination of their values.
  // TODO(bazel-team): break out flags that configure Bazel's analysis phase into their own
  // FragmentOptions. Those flags should also be ineligible outputs for other transitions because
  // they're not meant for rule logic.
  private static final ImmutableMap<AnalysisPhaseFlags, BuildOptions> NO_CONFIG_OPTIONS =
      createNoConfigOptions();

  private static ImmutableMap<AnalysisPhaseFlags, BuildOptions> createNoConfigOptions() {
    var noConfigOptions = ImmutableMap.<AnalysisPhaseFlags, BuildOptions>builder();
    var values = new boolean[] {false, true};
    for (boolean checkVisibility : values) {
      for (boolean verboseVisibilityErrors : values) {
        for (boolean enforceTransitiveVisibility : values) {
          for (boolean checkTestonlyForOutputFiles : values) {
            BuildOptions baseOptions =
                BuildOptions.builder()
                    .addFragmentOptions(Options.getDefaults(CoreOptions.class))
                    .build();
            var coreOptions = baseOptions.get(CoreOptions.class);
            // Disable the exec transition. Since this config is empty it shouldn't trigger any exec
            // transitions. More important, the default value this would otherwise propagate may not
            // exist in the repo (if the repo remaps with a repo-wide bazelrc).
            coreOptions.setStarlarkExecConfig(null);
            coreOptions.setCheckVisibility(checkVisibility);
            coreOptions.setVerboseVisibilityErrors(verboseVisibilityErrors);
            coreOptions.setEnforceTransitiveVisibility(enforceTransitiveVisibility);
            coreOptions.setCheckTestonlyForOutputFiles(checkTestonlyForOutputFiles);
            noConfigOptions.put(AnalysisPhaseFlags.of(coreOptions), baseOptions);
          }
        }
      }
    }
    return noConfigOptions.buildOrThrow();
  }

  /**
   * Returns the options of the no-config configuration (see {@link
   * com.google.devtools.build.lib.analysis.config.transitions.NoConfigTransition}) for targets
   * reached from a configuration with the given options.
   *
   * <p>These only contain {@link CoreOptions} with default values, except that flags that configure
   * Bazel's analysis phase rather than rule logic, such as {@code --check_visibility}, are
   * inherited. This keeps their effect consistent across all targets while still preventing
   * forking: there is usually at most one such configuration per build, and it is the same instance
   * for all configurations that agree on these flags.
   */
  public static BuildOptions noConfigOptions(BuildOptions options) {
    return NO_CONFIG_OPTIONS.get(
        AnalysisPhaseFlags.of(checkNotNull(options.get(CoreOptions.class))));
  }

  /**
   * Returns all possible results of {@link #noConfigOptions} for registration as serialization
   * constants.
   */
  public static ImmutableCollection<BuildOptions> allNoConfigOptions() {
    return NO_CONFIG_OPTIONS.values();
  }

  private CommonOptions() {}
}
