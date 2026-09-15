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
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.starlarkbuildapi.core.StructApi;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;

/** Info object representing data about a specific platform. */
@StarlarkBuiltin(
    name = "PlatformInfo",
    doc =
        "Provides access to data about a specific platform. See "
            + "<a href='${link platforms#constraints-platforms}'>Defining "
            + "Constraints and Platforms</a> for more information."
            + PlatformInfoApi.EXPERIMENTAL_WARNING,
    category = DocCategory.PROVIDER)
public interface PlatformInfoApi<
        ConstraintSettingInfoT extends ConstraintSettingInfoApi,
        ConstraintValueInfoT extends ConstraintValueInfoApi>
    extends StructApi {

  String EXPERIMENTAL_WARNING =
      "<br/><i>Note: This API is experimental and may change at any time. It is disabled by"
          + " default, but may be enabled with <code>--experimental_platforms_api</code></i>";

  @StarlarkMethod(
      name = "label",
      doc = "The label of the target that created this platform.",
      structField = true,
      enableOnlyWithFlag = BuildLanguageOptions.EXPERIMENTAL_PLATFORMS_API)
  Label label();

  @StarlarkMethod(
      name = "constraints",
      doc =
          "The <a href=\"ConstraintValueInfo\">ConstraintValueInfo</a> instances that define "
              + "this platform.",
      structField = true,
      enableOnlyWithFlag = BuildLanguageOptions.EXPERIMENTAL_PLATFORMS_API)
  ConstraintCollectionApi<ConstraintSettingInfoT, ConstraintValueInfoT> constraints();
}
