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
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import java.util.List;

/** A configuration fragment describing the current platform configuration. */
@ThreadSafety.Immutable
@SkylarkModule(
  name = "platform",
  doc = "The platform configuration.",
  category = SkylarkModuleCategory.CONFIGURATION_FRAGMENT
)
public class PlatformConfiguration extends BuildConfiguration.Fragment {

  private final Label executionPlatform;
  private final ImmutableList<Label> targetPlatforms;
  private final ImmutableList<Label> extraToolchains;

  public PlatformConfiguration(
      Label executionPlatform, List<Label> targetPlatforms, List<Label> extraToolchains) {

    this.executionPlatform = executionPlatform;
    this.targetPlatforms = ImmutableList.copyOf(targetPlatforms);
    this.extraToolchains = ImmutableList.copyOf(extraToolchains);
  }

  @SkylarkCallable(
    name = "execution_platform",
    structField = true,
    doc = "The current execution platform"
  )
  public Label getExecutionPlatform() {
    return executionPlatform;
  }

  @SkylarkCallable(name = "platforms", structField = true, doc = "The current target platforms")
  public ImmutableList<Label> getTargetPlatforms() {
    return targetPlatforms;
  }

  /** Additional toolchains that should be considered during toolchain resolution. */
  public ImmutableList<Label> getExtraToolchains() {
    return extraToolchains;
  }
}
