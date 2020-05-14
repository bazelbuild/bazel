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
import com.google.devtools.build.lib.skylarkbuildapi.core.StructApi;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.StarlarkBuiltin;
import com.google.devtools.build.lib.skylarkinterface.StarlarkDocumentationCategory;
import com.google.devtools.build.lib.syntax.StarlarkSemantics.FlagIdentifier;

/**
 * Info object representing a value for a constraint setting that can be used to define a platform.
 */
@StarlarkBuiltin(
    name = "ConstraintValueInfo",
    doc =
        "A value for a constraint setting that can be used to define a platform. See "
            + "<a href='../../platforms.html#defining-constraints-and-platforms'>Defining "
            + "Constraints and Platforms</a> for more information."
            + PlatformInfoApi.EXPERIMENTAL_WARNING,
    category = StarlarkDocumentationCategory.PROVIDER)
public interface ConstraintValueInfoApi extends StructApi {

  @SkylarkCallable(
      name = "constraint",
      doc =
          "The <a href=\"ConstraintSettingInfo.html\">ConstraintSettingInfo</a> this value can be "
              + "applied to.",
      structField = true,
      enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_PLATFORM_API)
  ConstraintSettingInfoApi constraint();

  @SkylarkCallable(
      name = "label",
      doc = "The label of the target that created this constraint value.",
      structField = true,
      enableOnlyWithFlag = FlagIdentifier.EXPERIMENTAL_PLATFORM_API)
  Label label();
}
