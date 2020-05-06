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
import com.google.devtools.build.lib.skylarkinterface.StarlarkBuiltin;
import com.google.devtools.build.lib.skylarkinterface.StarlarkDocumentationCategory;
import com.google.devtools.build.lib.syntax.StarlarkValue;

/** The platform configuration. */
@StarlarkBuiltin(
    name = "platform",
    doc = "The platform configuration.",
    category = StarlarkDocumentationCategory.CONFIGURATION_FRAGMENT)
public interface PlatformConfigurationApi extends StarlarkValue {

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
}
