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

package com.google.devtools.build.lib.skylarkbuildapi.cpp;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.skylarkbuildapi.Bootstrap;
import com.google.devtools.build.lib.skylarkbuildapi.FileApi;
import com.google.devtools.build.lib.skylarkbuildapi.SkylarkActionFactoryApi;
import com.google.devtools.build.lib.skylarkbuildapi.SkylarkRuleContextApi;

/**
 * {@link Bootstrap} for skylark objects related to cpp rules.
 */
public class CcBootstrap implements Bootstrap {
  private final BazelCcModuleApi<
          ? extends SkylarkActionFactoryApi,
          ? extends FileApi,
          ? extends SkylarkRuleContextApi,
          ? extends CcToolchainProviderApi<? extends FeatureConfigurationApi>,
          ? extends FeatureConfigurationApi,
          ? extends CcCompilationContextApi,
          ? extends CcCompilationOutputsApi<? extends FileApi>,
          ? extends CcLinkingOutputsApi<? extends FileApi>,
          ? extends CcLinkingContextApi,
          ? extends LibraryToLinkApi,
          ? extends CcToolchainVariablesApi,
          ? extends CcToolchainConfigInfoApi>
      ccModule;

  public CcBootstrap(
      BazelCcModuleApi<
              ? extends SkylarkActionFactoryApi,
              ? extends FileApi,
              ? extends SkylarkRuleContextApi,
              ? extends CcToolchainProviderApi<? extends FeatureConfigurationApi>,
              ? extends FeatureConfigurationApi,
              ? extends CcCompilationContextApi,
              ? extends CcCompilationOutputsApi<? extends FileApi>,
              ? extends CcLinkingOutputsApi<? extends FileApi>,
              ? extends CcLinkingContextApi,
              ? extends LibraryToLinkApi,
              ? extends CcToolchainVariablesApi,
              ? extends CcToolchainConfigInfoApi>
          ccModule) {
    this.ccModule = ccModule;
  }

  @Override
  public void addBindingsToBuilder(ImmutableMap.Builder<String, Object> builder) {
    builder.put("cc_common", ccModule);
  }
}
