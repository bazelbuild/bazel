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

import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.skylarkinterface.SkylarkValue;
import com.google.devtools.build.lib.syntax.SkylarkIndexable;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.syntax.SkylarkSemantics.FlagIdentifier;
import javax.annotation.Nullable;

/** Info object representing data about a specific platform. */
@SkylarkModule(
    name = "ConstraintCollection",
    doc =
        "Provides access to data about a collection of ConstraintValueInfo providers. "
            + PlatformInfoApi.EXPERIMENTAL_WARNING,
    category = SkylarkModuleCategory.PROVIDER)
public interface ConstraintCollectionApi<
        ConstraintSettingInfoT extends ConstraintSettingInfoApi,
        ConstraintValueInfoT extends ConstraintValueInfoApi>
    extends SkylarkIndexable, SkylarkValue {

  @SkylarkCallable(
      name = "constraint_settings",
      doc = "The ConstraintSettingInfo values that this collection directly references.",
      structField = true,
      enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_PLATFORM_API)
  SkylarkList<ConstraintSettingInfoT> constraintSettings();

  @SkylarkCallable(
      name = "get",
      doc = "Returns the specific ConstraintValueInfo for a specific ConstraintSettingInfo.",
      allowReturnNones = true,
      parameters = {
        @Param(
            name = "constraint",
            type = ConstraintSettingInfoApi.class,
            defaultValue = "None",
            noneable = true,
            named = true,
            doc = "The constraint setting to fetch the value for.")
      },
      enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_PLATFORM_API)
  @Nullable
  ConstraintValueInfoT get(ConstraintSettingInfoT constraint);

  @SkylarkCallable(
      name = "has",
      doc = "Returns whether the specific ConstraintSettingInfo is set.",
      parameters = {
        @Param(
            name = "constraint",
            type = ConstraintSettingInfoApi.class,
            defaultValue = "None",
            noneable = true,
            named = true,
            doc = "The constraint setting to check.")
      })
  boolean has(ConstraintSettingInfoT constraint);
}
