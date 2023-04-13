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

package com.google.devtools.build.lib.starlarkbuildapi.platform;

import com.google.devtools.build.docgen.annot.DocCategory;
import com.google.devtools.build.lib.starlarkbuildapi.core.ProviderApi;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.StarlarkValue;

/** Module containing functions to interact with the platform APIs. */
@StarlarkBuiltin(
    name = "platform_common",
    category = DocCategory.TOP_LEVEL_MODULE,
    doc = "Functions for Starlark to interact with the platform APIs.")
public interface PlatformCommonApi extends StarlarkValue {
  @StarlarkMethod(
      name = "TemplateVariableInfo",
      doc =
          "The constructor/key for the <a href='../providers/TemplateVariableInfo.html'>"
              + "TemplateVariableInfo</a> provider.",
      structField = true)
  ProviderApi getMakeVariableProvider();

  @StarlarkMethod(
      name = "ToolchainInfo",
      doc =
          "The constructor/key for the <a href='../providers/ToolchainInfo.html'>ToolchainInfo</a>"
              + " provider.",
      structField = true)
  ProviderApi getToolchainInfoConstructor();

  @StarlarkMethod(
      name = "PlatformInfo",
      doc =
          "The constructor/key for the <a href='../providers/PlatformInfo.html'>PlatformInfo</a>"
              + " provider."
              + PlatformInfoApi.EXPERIMENTAL_WARNING,
      structField = true)
  ProviderApi getPlatformInfoConstructor();

  @StarlarkMethod(
      name = "ConstraintSettingInfo",
      doc =
          "The constructor/key for the <a href='../providers/ConstraintSettingInfo.html'>"
              + "ConstraintSettingInfo</a> provider."
              + PlatformInfoApi.EXPERIMENTAL_WARNING,
      structField = true)
  ProviderApi getConstraintSettingInfoConstructor();

  @StarlarkMethod(
      name = "ConstraintValueInfo",
      doc =
          "The constructor/key for the <a href='../providers/ConstraintValueInfo.html'>"
              + "ConstraintValueInfo</a> provider."
              + PlatformInfoApi.EXPERIMENTAL_WARNING,
      structField = true)
  ProviderApi getConstraintValueInfoConstructor();
}
