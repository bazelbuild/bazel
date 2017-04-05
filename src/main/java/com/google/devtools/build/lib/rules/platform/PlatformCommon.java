// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.platform;

import com.google.devtools.build.lib.packages.ClassObjectConstructor;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;

/** Skylark namespace used to interact with Blaze's platform APIs. */
@SkylarkModule(
  name = "platform_common",
  doc = "Functions for Skylark to interact with Blaze's platform APIs."
)
public class PlatformCommon {

  @SkylarkCallable(
    name = PlatformInfo.SKYLARK_NAME,
    doc = "The key used to retrieve the provider containing platform_info's value.",
    structField = true
  )
  public ClassObjectConstructor getPlatformInfoConstructor() {
    return PlatformInfo.SKYLARK_CONSTRUCTOR;
  }

  @SkylarkCallable(
    name = ConstraintSettingInfo.SKYLARK_NAME,
    doc = "The key used to retrieve the provider containing constraint_setting_info's value.",
    structField = true
  )
  public ClassObjectConstructor getConstraintSettingInfoConstructor() {
    return ConstraintSettingInfo.SKYLARK_CONSTRUCTOR;
  }

  @SkylarkCallable(
    name = ConstraintValueInfo.SKYLARK_NAME,
    doc = "The key used to retrieve the provider containing constraint_value_info's value.",
    structField = true
  )
  public ClassObjectConstructor getConstraintValueInfoConstructor() {
    return ConstraintValueInfo.SKYLARK_CONSTRUCTOR;
  }
}
