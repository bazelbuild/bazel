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

package com.google.devtools.build.lib.skylarkbuildapi.apple;

import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.skylarkinterface.SkylarkValue;

/**
 * Interface for a descriptor for an Apple platform type, such as such as iOS or macOS.
 */
@SkylarkModule(
  name = "apple_platform_type",
  category = SkylarkModuleCategory.NONE,
  doc = "Describes an Apple \"platform type\", such as iOS, macOS, tvOS, or watchOS. This is "
      + "distinct from a \"platform\", which is the platform type combined with one or more CPU "
      + "architectures.<p>"
      + "Specific instances of this type can be retrieved by accessing the fields of the "
      + "<a href='apple_common.html#platform_type'>apple_common.platform_type</a>:<br><ul>"
      + "<li><code>apple_common.platform_type.ios</code></li>"
      + "<li><code>apple_common.platform_type.macos</code></li>"
      + "<li><code>apple_common.platform_type.tvos</code></li>"
      + "<li><code>apple_common.platform_type.watchos</code></li>"
      + "</ul><p>"
      + "Likewise, the platform type of an existing platform value can be retrieved using its "
      + "<code>platform_type</code> field.<p>"
      + "Platform types can be converted to a lowercase string (e.g., <code>ios</code> or "
      + "<code>macos</code>) using the <a href='globals.html#str'>str</a> function."
)
public interface ApplePlatformTypeApi extends SkylarkValue {}
