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

package com.google.devtools.build.lib.skylarkbuildapi.platform;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import java.util.List;

/**
 * The platform configuration.
 */
@SkylarkModule(
    name = "platform",
    doc = "The platform configuration.",
    category = SkylarkModuleCategory.CONFIGURATION_FRAGMENT
)
public interface PlatformConfigurationApi {

  @SkylarkCallable(name = "host_platform", structField = true, doc = "The current host platform")
  Label getHostPlatform();

  @SkylarkCallable(name = "platform", structField = true, doc = "The current target platform")
  Label getTargetPlatform();

  @SkylarkCallable(
      name = "platforms",
      structField = true,
      doc = "The current target platforms",
      documented = false)
  @Deprecated
  ImmutableList<Label> getTargetPlatforms();

  @SkylarkCallable(
      name = "enabled_toolchain_types",
      structField = true,
      doc = "The set of toolchain types enabled for platform-based toolchain selection.")
  List<Label> getEnabledToolchainTypes();
}
