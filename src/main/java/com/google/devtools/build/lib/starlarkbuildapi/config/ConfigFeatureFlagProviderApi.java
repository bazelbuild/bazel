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

package com.google.devtools.build.lib.starlarkbuildapi.config;

import com.google.devtools.build.docgen.annot.DocCategory;
import com.google.devtools.build.lib.starlarkbuildapi.core.StructApi;
import javax.annotation.Nullable;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;

/** An info object for config_feature_flag rules. */
@StarlarkBuiltin(
    name = "FeatureFlagInfo",
    category = DocCategory.PROVIDER,
    doc = "A provider used to access information about config_feature_flag rules.")
public interface ConfigFeatureFlagProviderApi extends StructApi {

  @StarlarkMethod(
      name = "value",
      doc =
          "The current value of the flag in the flag's current configuration. None if there is an"
              + " error.",
      structField = true,
      allowReturnNones = true)
  @Nullable
  String getFlagValue();

  // TODO(blaze-configurability-team, brandjon): Make part of the struct?
  //   Technically, this is in union with value as both should never be non-empty.
  @StarlarkMethod(
      name = "error",
      doc = "If non-None, this error was generated when trying to compute current value of flag.",
      structField = true,
      allowReturnNones = true)
  @Nullable
  String getError();

  @StarlarkMethod(
      name = "is_valid_value",
      doc = "The value of the flag in the configuration used by the flag rule.",
      parameters = {
        @Param(name = "value", doc = "String, the value to check for validity for this flag."),
      })
  boolean isValidValue(String value);
}
