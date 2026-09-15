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
import javax.annotation.Nullable;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;

/** Info object representing a specific constraint setting that may be used to define a platform. */
@StarlarkBuiltin(
    name = "ConstraintSettingInfo",
    doc =
        "A specific constraint setting that may be used to define a platform. See "
            + "<a href='${link platforms#constraints-platforms}'>Defining "
            + "Constraints and Platforms</a> for more information."
            + PlatformInfoApi.EXPERIMENTAL_WARNING,
    category = DocCategory.PROVIDER)
public interface ConstraintSettingInfoApi extends StructApi {

  @StarlarkMethod(
      name = "label",
      doc = "The label of the target that created this constraint.",
      structField = true,
      enableOnlyWithFlag = BuildLanguageOptions.EXPERIMENTAL_PLATFORMS_API)
  Label label();

  @StarlarkMethod(
      name = "default_constraint_value",
      doc = "The default constraint_value for this setting.",
      structField = true,
      allowReturnNones = true,
      enableOnlyWithFlag = BuildLanguageOptions.EXPERIMENTAL_PLATFORMS_API)
  @Nullable
  ConstraintValueInfoApi defaultConstraintValue();

  @StarlarkMethod(
      name = "has_default_constraint_value",
      doc = "Whether there is a default constraint_value for this setting.",
      structField = true)
  boolean hasDefaultConstraintValue();
}
