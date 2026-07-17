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

/**
 * Info object representing a value for a constraint setting that can be used to define a platform.
 */
@StarlarkBuiltin(
    name = "ConstraintValueInfo",
    doc =
        "A value for a constraint setting that can be used to define a platform. See "
            + "<a href='${link platforms#constraints-platforms}'>Defining "
            + "Constraints and Platforms</a> for more information."
            + PlatformInfoApi.EXPERIMENTAL_WARNING,
    category = DocCategory.PROVIDER)
public interface ConstraintValueInfoApi extends StructApi {

  @StarlarkMethod(
      name = "constraint",
      doc =
          "The <a href=\"ConstraintSettingInfo\">ConstraintSettingInfo</a> this value can be "
              + "applied to.",
      structField = true,
      enableOnlyWithFlag = BuildLanguageOptions.EXPERIMENTAL_PLATFORMS_API)
  ConstraintSettingInfoApi constraint();

  @StarlarkMethod(
      name = "label",
      doc = "The label of the target that created this constraint value.",
      structField = true,
      enableOnlyWithFlag = BuildLanguageOptions.EXPERIMENTAL_PLATFORMS_API)
  Label label();
}
