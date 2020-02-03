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

import com.google.devtools.build.lib.skylarkbuildapi.core.ProviderApi;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.syntax.StarlarkValue;

/** Module containing functions to interact with the platform APIs. */
@SkylarkModule(
    name = "platform_common",
    doc = "Functions for Starlark to interact with the platform APIs.")
public interface PlatformCommonApi extends StarlarkValue {
  @SkylarkCallable(
      name = "TemplateVariableInfo",
      doc =
          "The provider used to retrieve the provider that contains the template variables defined"
              + " by a particular toolchain, for example by calling "
              + "ctx.attr._cc_toolchain[platform_common.TemplateVariableInfo].make_variables[<name>]",
      structField = true)
  ProviderApi getMakeVariableProvider();

  @SkylarkCallable(
      name = "ToolchainInfo",
      doc = "The provider used to retrieve information about a toolchain.",
      structField = true)
  ProviderApi getToolchainInfoConstructor();

  @SkylarkCallable(
      name = "PlatformInfo",
      doc = "The provider used to retrieve the information about a platform.",
      structField = true)
  ProviderApi getPlatformInfoConstructor();

  @SkylarkCallable(
      name = "ConstraintSettingInfo",
      doc = "The provider used to retrieve the information about a constraint setting.",
      structField = true)
  ProviderApi getConstraintSettingInfoConstructor();

  @SkylarkCallable(
      name = "ConstraintValueInfo",
      doc = "The provider used to retrieve the information about a constraint value.",
      structField = true)
  ProviderApi getConstraintValueInfoConstructor();
}
