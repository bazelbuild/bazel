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

import com.google.devtools.build.lib.skylarkbuildapi.core.ProviderApi;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.syntax.StarlarkValue;

/** Helper utility containing functions regarding configurations.ss */
@SkylarkModule(
    name = "config_common",
    doc = "Functions for Starlark to interact with Blaze's configurability APIs.")
public interface ConfigSkylarkCommonApi extends StarlarkValue {

  @SkylarkCallable(
      name = "FeatureFlagInfo",
      doc = "The key used to retrieve the provider containing config_feature_flag's value.",
      structField = true)
  ProviderApi getConfigFeatureFlagProviderConstructor();
}
