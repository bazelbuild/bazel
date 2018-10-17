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

import com.google.common.collect.ImmutableMap.Builder;
import com.google.devtools.build.lib.skylarkbuildapi.Bootstrap;
import com.google.devtools.build.lib.skylarkbuildapi.SkylarkConfigApi;
import com.google.devtools.build.lib.syntax.Runtime;

/**
 * A {@link Bootstrap} for config-related libraries of the build API.
 */
public class ConfigBootstrap implements Bootstrap {

  private final ConfigSkylarkCommonApi configSkylarkCommonApi;
  private final SkylarkConfigApi skylarkConfigApi;
  private final ConfigGlobalLibraryApi configGlobalLibrary;

  public ConfigBootstrap(
      ConfigSkylarkCommonApi configSkylarkCommonApi,
      SkylarkConfigApi skylarkConfigApi,
      ConfigGlobalLibraryApi configGlobalLibrary) {
    this.configSkylarkCommonApi = configSkylarkCommonApi;
    this.skylarkConfigApi = skylarkConfigApi;
    this.configGlobalLibrary = configGlobalLibrary;
  }

  @Override
  public void addBindingsToBuilder(Builder<String, Object> builder) {
    builder.put("config_common", configSkylarkCommonApi);
    builder.put("config", skylarkConfigApi);
    Runtime.setupSkylarkLibrary(builder, configGlobalLibrary);
  }
}
