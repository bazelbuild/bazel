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

package com.google.devtools.build.lib.skylarkbuildapi;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.skylarkbuildapi.DefaultInfoApi.DefaultInfoApiProvider;
import com.google.devtools.build.lib.skylarkbuildapi.OutputGroupInfoApi.OutputGroupInfoApiProvider;
import com.google.devtools.build.lib.syntax.Runtime;

/**
 * A {@link Bootstrap} for top-level libraries of the build API.
 */
public class TopLevelBootstrap implements Bootstrap {
  private final Class<? extends SkylarkBuildApiGlobals> skylarkBuildApiGlobals;
  private final Class<? extends SkylarkAttrApi> skylarkAttrApi;
  private final Class<? extends SkylarkCommandLineApi> skylarkCommandLineApi;
  private final Class<? extends SkylarkNativeModuleApi> skylarkNativeModuleApi;
  private final Class<? extends SkylarkRuleFunctionsApi<?>> skylarkRuleFunctionsApi;
  private final StructApi.StructProviderApi structProvider;
  private final OutputGroupInfoApiProvider outputGroupInfoProvider;
  private final ActionsInfoProviderApi actionsInfoProviderApi;
  private final DefaultInfoApiProvider<?, ?> defaultInfoProvider;

  public TopLevelBootstrap(
      Class<? extends SkylarkBuildApiGlobals> skylarkBuildApiGlobals,
      Class<? extends SkylarkAttrApi> skylarkAttrApi,
      Class<? extends SkylarkCommandLineApi> skylarkCommandLineApi,
      Class<? extends SkylarkNativeModuleApi> skylarkNativeModuleApi,
      Class<? extends SkylarkRuleFunctionsApi<?>> skylarkRuleFunctionsApi,
      StructApi.StructProviderApi structProvider,
      OutputGroupInfoApiProvider outputGroupInfoProvider,
      ActionsInfoProviderApi actionsInfoProviderApi,
      DefaultInfoApiProvider<?, ?> defaultInfoProvider) {
    this.skylarkAttrApi = skylarkAttrApi;
    this.skylarkBuildApiGlobals = skylarkBuildApiGlobals;
    this.skylarkCommandLineApi = skylarkCommandLineApi;
    this.skylarkNativeModuleApi = skylarkNativeModuleApi;
    this.skylarkRuleFunctionsApi = skylarkRuleFunctionsApi;
    this.structProvider = structProvider;
    this.outputGroupInfoProvider = outputGroupInfoProvider;
    this.actionsInfoProviderApi = actionsInfoProviderApi;
    this.defaultInfoProvider = defaultInfoProvider;
  }

  @Override
  public void addBindingsToBuilder(ImmutableMap.Builder<String, Object> builder) {
    Runtime.setupModuleGlobals(builder, skylarkAttrApi);
    Runtime.setupModuleGlobals(builder, skylarkBuildApiGlobals);
    Runtime.setupModuleGlobals(builder, skylarkCommandLineApi);
    Runtime.setupModuleGlobals(builder, skylarkNativeModuleApi);
    Runtime.setupModuleGlobals(builder, skylarkRuleFunctionsApi);
    builder.put("struct", structProvider);
    builder.put("OutputGroupInfo", outputGroupInfoProvider);
    builder.put("Actions", actionsInfoProviderApi);
    builder.put("DefaultInfo", defaultInfoProvider);
  }
}
