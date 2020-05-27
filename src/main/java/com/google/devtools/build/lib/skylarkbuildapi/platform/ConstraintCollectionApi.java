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

import com.google.devtools.build.lib.syntax.Sequence;
import com.google.devtools.build.lib.syntax.StarlarkIndexable;
import com.google.devtools.build.lib.syntax.StarlarkSemantics.FlagIdentifier;
import com.google.devtools.build.lib.syntax.StarlarkValue;
import javax.annotation.Nullable;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkDocumentationCategory;
import net.starlark.java.annot.StarlarkMethod;

/** Info object representing data about a specific platform. */
@StarlarkBuiltin(
    name = "ConstraintCollection",
    doc =
        "Provides access to data about a collection of ConstraintValueInfo providers. "
            + PlatformInfoApi.EXPERIMENTAL_WARNING,
    category = StarlarkDocumentationCategory.PROVIDER)
public interface ConstraintCollectionApi<
        ConstraintSettingInfoT extends ConstraintSettingInfoApi,
        ConstraintValueInfoT extends ConstraintValueInfoApi>
    extends StarlarkIndexable, StarlarkValue {

  @StarlarkMethod(
      name = "constraint_settings",
      doc = "The ConstraintSettingInfo values that this collection directly references.",
      structField = true,
      enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_PLATFORM_API)
  Sequence<ConstraintSettingInfoT> constraintSettings();

  @StarlarkMethod(
      name = "get",
      doc = "Returns the specific ConstraintValueInfo for a specific ConstraintSettingInfo.",
      allowReturnNones = true,
      parameters = {
        @Param(
            name = "constraint",
            type = ConstraintSettingInfoApi.class,
            named = true,
            doc = "The constraint setting to fetch the value for.")
      },
      enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_PLATFORM_API)
  @Nullable
  ConstraintValueInfoT get(ConstraintSettingInfoT constraint);

  @StarlarkMethod(
      name = "has",
      doc = "Returns whether the specific ConstraintSettingInfo is set.",
      parameters = {
        @Param(
            name = "constraint",
            type = ConstraintSettingInfoApi.class,
            named = true,
            doc = "The constraint setting to check.")
      },
      enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_PLATFORM_API)
  boolean has(ConstraintSettingInfoT constraint);

  @StarlarkMethod(
      name = "has_constraint_value",
      doc = "Returns whether the specific ConstraintValueInfo is set.",
      parameters = {
        @Param(
            name = "constraint_value",
            type = ConstraintValueInfoApi.class,
            named = true,
            doc = "The constraint value to check.")
      },
      enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_PLATFORM_API)
  boolean hasConstraintValue(ConstraintValueInfoT constraintValue);
}
