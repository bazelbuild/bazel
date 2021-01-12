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

package com.google.devtools.build.skydoc.fakebuildapi.platform;

import com.google.devtools.build.lib.starlarkbuildapi.core.ProviderApi;
import com.google.devtools.build.lib.starlarkbuildapi.platform.PlatformCommonApi;
import com.google.devtools.build.skydoc.fakebuildapi.FakeProviderApi;

/**
 * Fake implementation of {@link PlatformCommonApi}.
 */
public class FakePlatformCommon implements PlatformCommonApi {

  @Override
  public ProviderApi getMakeVariableProvider() {
    return new FakeProviderApi("TemplateVariableInfo");
  }

  @Override
  public ProviderApi getToolchainInfoConstructor() {
    return new FakeProviderApi("ToolchainInfo");
  }

  @Override
  public ProviderApi getPlatformInfoConstructor() {
    return new FakeProviderApi("PlatformInfo");
  }

  @Override
  public ProviderApi getConstraintSettingInfoConstructor() {
    return new FakeProviderApi("ConstraintSettingInfo");
  }

  @Override
  public ProviderApi getConstraintValueInfoConstructor() {
    return new FakeProviderApi("ConstraintValueInfo");
  }
}
