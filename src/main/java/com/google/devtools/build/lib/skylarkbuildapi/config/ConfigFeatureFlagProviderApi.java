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

package com.google.devtools.build.lib.skylarkbuildapi.config;

import com.google.devtools.build.lib.skylarkbuildapi.core.StructApi;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;

/** An info object for config_feature_flag rules. */
@SkylarkModule(
    name = "FeatureFlagInfo",
    category = SkylarkModuleCategory.PROVIDER,
    doc = "A provider used to access information about config_feature_flag rules.")
public interface ConfigFeatureFlagProviderApi extends StructApi {

  @SkylarkCallable(
      name = "value",
      doc = "The current value of the flag in the flag's current configuration.",
      structField = true)
  String getFlagValue();

  @SkylarkCallable(
      name = "is_valid_value",
      doc = "The value of the flag in the configuration used by the flag rule.",
      parameters = {
        @Param(
            name = "value",
            type = String.class,
            doc = "String, the value to check for validity for this flag."),
      })
  boolean isValidValue(String value);
}
