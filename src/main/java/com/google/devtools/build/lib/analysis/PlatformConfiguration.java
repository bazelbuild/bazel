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

package com.google.devtools.build.lib.analysis;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.Fragment;
import com.google.devtools.build.lib.analysis.config.RequiresOptions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.starlarkbuildapi.platform.PlatformConfigurationApi;
import com.google.devtools.build.lib.util.RegexFilter;
import java.util.List;
import java.util.Map;

/** A configuration fragment describing the current platform configuration. */
@ThreadSafety.Immutable
@RequiresOptions(options = {PlatformOptions.class})
public class PlatformConfiguration extends Fragment implements PlatformConfigurationApi {
  private final Label hostPlatform;
  private final ImmutableList<String> extraExecutionPlatforms;
  private final Label targetPlatform;
  private final ImmutableList<String> extraToolchains;
  private final List<Map.Entry<RegexFilter, List<Label>>> targetFilterToAdditionalExecConstraints;

  public PlatformConfiguration(BuildOptions buildOptions) {
    PlatformOptions platformOptions = buildOptions.get(PlatformOptions.class);

    this.hostPlatform = platformOptions.computeHostPlatform();
    this.extraExecutionPlatforms = ImmutableList.copyOf(platformOptions.extraExecutionPlatforms);
    this.targetPlatform = platformOptions.computeTargetPlatform();
    this.extraToolchains = ImmutableList.copyOf(platformOptions.extraToolchains);
    this.targetFilterToAdditionalExecConstraints =
        platformOptions.targetFilterToAdditionalExecConstraints;
  }

  @Override
  public void reportInvalidOptions(EventHandler reporter, BuildOptions buildOptions) {
    PlatformOptions platformOptions = buildOptions.get(PlatformOptions.class);
    // TODO(https://github.com/bazelbuild/bazel/issues/6519): Implement true multiplatform builds.
    if (platformOptions.platforms.size() > 1) {
      reporter.handle(
          Event.warn(
              String.format(
                  "--platforms only supports a single target platform: using the first option %s",
                  this.targetPlatform)));
    }
  }

  @Override
  public Label getHostPlatform() {
    return hostPlatform;
  }

  /**
   * Target patterns that select additional platforms that will be made available for action
   * execution.
   */
  public ImmutableList<String> getExtraExecutionPlatforms() {
    return extraExecutionPlatforms;
  }

  /**
   * Returns the single target platform used in this configuration. The flag is multi-valued for
   * future handling of multiple target platforms but any given configuration should only be
   * concerned with a single target platform.
   */
  @Override
  public Label getTargetPlatform() {
    return targetPlatform;
  }

  @Override
  public ImmutableList<Label> getTargetPlatforms() {
    return ImmutableList.of(targetPlatform);
  }

  /**
   * Target patterns that select additional toolchains that will be considered during toolchain
   * resolution.
   */
  public ImmutableList<String> getExtraToolchains() {
    return extraToolchains;
  }

  /**
   * Returns a list of labels referring to additional constraint value targets which should be taken
   * into account when resolving the toolchains/execution platform for the target with the given
   * label.
   */
  public List<Label> getAdditionalExecutionConstraintsFor(Label label) {
    ImmutableList.Builder<Label> constraints = ImmutableList.builder();
    for (Map.Entry<RegexFilter, List<Label>> filter : targetFilterToAdditionalExecConstraints) {
      if (filter.getKey().isIncluded(label.getCanonicalForm())) {
        constraints.addAll(filter.getValue());
      }
    }
    return constraints.build();
  }
}
