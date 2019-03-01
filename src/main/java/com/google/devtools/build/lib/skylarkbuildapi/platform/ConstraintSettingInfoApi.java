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

import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skylarkbuildapi.StructApi;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.syntax.StarlarkSemantics.FlagIdentifier;
import javax.annotation.Nullable;

/** Info object representing a specific constraint setting that may be used to define a platform. */
@SkylarkModule(
    name = "ConstraintSettingInfo",
    doc =
        "A specific constraint setting that may be used to define a platform. "
            + PlatformInfoApi.EXPERIMENTAL_WARNING,
    category = SkylarkModuleCategory.PROVIDER)
public interface ConstraintSettingInfoApi extends StructApi {

  @SkylarkCallable(
      name = "label",
      doc = "The label of the target that created this constraint.",
      structField = true,
      enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_PLATFORM_API)
  Label label();

  @SkylarkCallable(
      name = "default_constraint_value",
      doc = "The default constraint_value for this setting.",
      structField = true,
      allowReturnNones = true,
      enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_PLATFORM_API)
  @Nullable
  ConstraintValueInfoApi defaultConstraintValue();

  @SkylarkCallable(
      name = "has_default_constraint_value",
      doc = "Whether there is a default constraint_value for this setting.",
      structField = true)
  boolean hasDefaultConstraintValue();
}
