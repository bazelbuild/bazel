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
import com.google.devtools.build.lib.skylarkbuildapi.core.Bootstrap;
import com.google.devtools.build.lib.skylarkbuildapi.core.StructApi;
import com.google.devtools.build.lib.syntax.Starlark;

/**
 * A {@link Bootstrap} for top-level libraries of the build API.
 */
public class TopLevelBootstrap implements Bootstrap {
  private final StarlarkBuildApiGlobals starlarkBuildApiGlobals;
  private final StarlarkAttrModuleApi starlarkAttrModuleApi;
  private final StarlarkCommandLineApi starlarkCommandLineApi;
  private final StarlarkNativeModuleApi starlarkNativeModuleApi;
  private final StarlarkRuleFunctionsApi<?> starlarkRuleFunctionsApi;
  private final StructApi.StructProviderApi structProvider;
  private final OutputGroupInfoApiProvider outputGroupInfoProvider;
  private final ActionsInfoProviderApi actionsInfoProviderApi;
  private final DefaultInfoApiProvider<?, ?> defaultInfoProvider;

  public TopLevelBootstrap(
      StarlarkBuildApiGlobals starlarkBuildApiGlobals,
      StarlarkAttrModuleApi starlarkAttrModuleApi,
      StarlarkCommandLineApi starlarkCommandLineApi,
      StarlarkNativeModuleApi starlarkNativeModuleApi,
      StarlarkRuleFunctionsApi<?> starlarkRuleFunctionsApi,
      StructApi.StructProviderApi structProvider,
      OutputGroupInfoApiProvider outputGroupInfoProvider,
      ActionsInfoProviderApi actionsInfoProviderApi,
      DefaultInfoApiProvider<?, ?> defaultInfoProvider) {
    this.starlarkAttrModuleApi = starlarkAttrModuleApi;
    this.starlarkBuildApiGlobals = starlarkBuildApiGlobals;
    this.starlarkCommandLineApi = starlarkCommandLineApi;
    this.starlarkNativeModuleApi = starlarkNativeModuleApi;
    this.starlarkRuleFunctionsApi = starlarkRuleFunctionsApi;
    this.structProvider = structProvider;
    this.outputGroupInfoProvider = outputGroupInfoProvider;
    this.actionsInfoProviderApi = actionsInfoProviderApi;
    this.defaultInfoProvider = defaultInfoProvider;
  }

  @Override
  public void addBindingsToBuilder(ImmutableMap.Builder<String, Object> builder) {
    Starlark.addMethods(builder, starlarkBuildApiGlobals);
    Starlark.addMethods(builder, starlarkRuleFunctionsApi);
    Starlark.addModule(builder, starlarkAttrModuleApi); // "attr"
    Starlark.addModule(builder, starlarkCommandLineApi); // "cmd_helper"
    Starlark.addModule(builder, starlarkNativeModuleApi); // "native"
    builder.put("struct", structProvider);
    builder.put("OutputGroupInfo", outputGroupInfoProvider);
    builder.put("Actions", actionsInfoProviderApi);
    builder.put("DefaultInfo", defaultInfoProvider);
  }
}
