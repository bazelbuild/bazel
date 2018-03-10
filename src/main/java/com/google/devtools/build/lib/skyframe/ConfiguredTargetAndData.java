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

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Target;

/**
 * A container class for a {@link ConfiguredTarget} and associated data, {@link Target} and {@link
 * BuildConfiguration}. In the future, {@link ConfiguredTarget} objects will no longer contain their
 * associated {@link BuildConfiguration}. Consumers that need the {@link Target} or {@link
 * BuildConfiguration} must therefore have access to one of these objects.
 *
 * <p>These objects are intended to be short-lived, never stored in Skyframe, since they pair three
 * heavyweight objects, a {@link ConfiguredTarget}, a {@link Target} (which holds a {@link
 * Package}), and a {@link BuildConfiguration}.
 */
public class ConfiguredTargetAndData {
  private final ConfiguredTarget configuredTarget;
  private final Target target;
  private final BuildConfiguration configuration;

  ConfiguredTargetAndData(
      ConfiguredTarget configuredTarget, Target target, BuildConfiguration configuration) {
    this.configuredTarget = configuredTarget;
    this.target = target;
    this.configuration = configuration;
    Preconditions.checkState(
        configuredTarget.getLabel().equals(target.getLabel()),
        "Unable to construct ConfiguredTargetAndData:"
            + " ConfiguredTarget's label %s is not equal to Target's label %s",
        configuredTarget.getLabel(),
        target.getLabel());
    Preconditions.checkState(
        configuration == configuredTarget.getConfiguration(),
        "Configurations don't match: %s %s (%s %s)",
        configuration,
        configuredTarget.getConfiguration(),
        configuredTarget,
        target);
  }

  /**
   * For use with {@code MergedConfiguredTarget} and similar, where we create a virtual {@link
   * ConfiguredTarget} corresponding to the same {@link Target}.
   */
  public ConfiguredTargetAndData fromConfiguredTarget(ConfiguredTarget maybeNew) {
    if (configuredTarget.equals(maybeNew)) {
      return this;
    }
    return new ConfiguredTargetAndData(maybeNew, this.target, configuration);
  }

  public Target getTarget() {
    return target;
  }

  public BuildConfiguration getConfiguration() {
    return configuration;
  }

  public ConfiguredTarget getConfiguredTarget() {
    return configuredTarget;
  }
}
