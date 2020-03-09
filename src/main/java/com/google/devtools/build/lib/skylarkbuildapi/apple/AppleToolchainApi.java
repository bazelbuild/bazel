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

import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.syntax.StarlarkValue;

/** Interface for a utility module involving the Apple toolchain. */
@SkylarkModule(
    name = "apple_toolchain",
    doc = "Utilities for resolving items from the Apple toolchain.")
public interface AppleToolchainApi<AppleConfigurationApiT extends AppleConfigurationApi<?>>
    extends StarlarkValue {

  @SkylarkCallable(
      name = "sdk_dir",
      doc = "Returns the platform directory inside of Xcode for a given configuration.")
  String sdkDirConstant();

  @SkylarkCallable(
      name = "developer_dir",
      doc = "Returns the Developer directory inside of Xcode for a given configuration.")
  String developerDirConstant();

  @SkylarkCallable(
      name = "platform_developer_framework_dir",
      doc = "Returns the platform frameworks directory inside of Xcode for a given configuration.",
      parameters = {
        @Param(
            name = "configuration",
            positional = true,
            named = false,
            type = AppleConfigurationApi.class,
            doc = "The apple configuration fragment.")
      })
  String platformFrameworkDirFromConfig(AppleConfigurationApiT configuration);
}
