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

import com.google.devtools.build.lib.analysis.IncompatiblePlatformProvider;
import com.google.devtools.build.lib.analysis.TemplateVariableInfo;
import com.google.devtools.build.lib.analysis.platform.ConstraintSettingInfo;
import com.google.devtools.build.lib.analysis.platform.ConstraintValueInfo;
import com.google.devtools.build.lib.analysis.platform.PlatformInfo;
import com.google.devtools.build.lib.analysis.platform.ToolchainInfo;
import com.google.devtools.build.lib.packages.Provider;
import com.google.devtools.build.lib.starlarkbuildapi.platform.PlatformCommonApi;

/** Starlark namespace used to interact with the platform APIs. */
public class PlatformCommon implements PlatformCommonApi<IncompatiblePlatformProvider> {

  @Override
  public Provider getPlatformInfoConstructor() {
    return PlatformInfo.PROVIDER;
  }

  @Override
  public Provider getConstraintSettingInfoConstructor() {
    return ConstraintSettingInfo.PROVIDER;
  }

  @Override
  public Provider getConstraintValueInfoConstructor() {
    return ConstraintValueInfo.PROVIDER;
  }

  @Override
  public Provider getMakeVariableProvider() {
    return TemplateVariableInfo.PROVIDER;
  }

  @Override
  public Provider getToolchainInfoConstructor() {
    return ToolchainInfo.PROVIDER;
  }

  @Override
  public IncompatiblePlatformProvider incompatibleTarget() {
    return IncompatiblePlatformProvider.incompatibleDueToStarlark(null, "Hello, World!");
  }
}
