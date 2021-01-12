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
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import javax.annotation.Nullable;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.StarlarkIndexable;
import net.starlark.java.eval.StarlarkValue;

/** Info object representing data about a specific platform. */
@StarlarkBuiltin(
    name = "ConstraintCollection",
    doc =
        "Provides access to data about a collection of ConstraintValueInfo providers. "
            + PlatformInfoApi.EXPERIMENTAL_WARNING,
    category = DocCategory.PROVIDER)
public interface ConstraintCollectionApi<
        ConstraintSettingInfoT extends ConstraintSettingInfoApi,
        ConstraintValueInfoT extends ConstraintValueInfoApi>
    extends StarlarkIndexable, StarlarkValue {

  @StarlarkMethod(
      name = "constraint_settings",
      doc = "The ConstraintSettingInfo values that this collection directly references.",
      structField = true,
      enableOnlyWithFlag = BuildLanguageOptions.EXPERIMENTAL_PLATFORMS_API)
  Sequence<ConstraintSettingInfoT> constraintSettings();

  @StarlarkMethod(
      name = "get",
      doc = "Returns the specific ConstraintValueInfo for a specific ConstraintSettingInfo.",
      allowReturnNones = true,
      parameters = {
        @Param(
            name = "constraint",
            named = true,
            doc = "The constraint setting to fetch the value for.")
      },
      enableOnlyWithFlag = BuildLanguageOptions.EXPERIMENTAL_PLATFORMS_API)
  @Nullable
  ConstraintValueInfoT get(ConstraintSettingInfoT constraint);

  @StarlarkMethod(
      name = "has",
      doc = "Returns whether the specific ConstraintSettingInfo is set.",
      parameters = {
        @Param(
            name = "constraint",
            named = true,
            doc = "The constraint setting to check.")
      },
      enableOnlyWithFlag = BuildLanguageOptions.EXPERIMENTAL_PLATFORMS_API)
  boolean has(ConstraintSettingInfoT constraint);

  @StarlarkMethod(
      name = "has_constraint_value",
      doc = "Returns whether the specific ConstraintValueInfo is set.",
      parameters = {
        @Param(
            name = "constraint_value",
            named = true,
            doc = "The constraint value to check.")
      },
      enableOnlyWithFlag = BuildLanguageOptions.EXPERIMENTAL_PLATFORMS_API)
  boolean hasConstraintValue(ConstraintValueInfoT constraintValue);
}
