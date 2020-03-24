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
import com.google.devtools.build.lib.skylarkbuildapi.FileApi;
import com.google.devtools.build.lib.skylarkbuildapi.SkylarkActionFactoryApi;
import com.google.devtools.build.lib.skylarkbuildapi.SkylarkRuleContextApi;
import com.google.devtools.build.lib.skylarkbuildapi.core.Bootstrap;
import com.google.devtools.build.lib.skylarkbuildapi.platform.ConstraintValueInfoApi;
import com.google.devtools.build.lib.syntax.FlagGuardedValue;
import com.google.devtools.build.lib.syntax.StarlarkSemantics.FlagIdentifier;

/**
 * {@link Bootstrap} for skylark objects related to cpp rules.
 */
public class CcBootstrap implements Bootstrap {
  private final CcModuleApi<
          ? extends SkylarkActionFactoryApi,
          ? extends FileApi,
          ? extends CcToolchainProviderApi<? extends FeatureConfigurationApi>,
          ? extends FeatureConfigurationApi,
          ? extends CcCompilationContextApi<? extends FileApi>,
          ? extends
              LinkerInputApi<? extends LibraryToLinkApi<? extends FileApi>, ? extends FileApi>,
          ? extends CcLinkingContextApi<? extends FileApi>,
          ? extends LibraryToLinkApi<? extends FileApi>,
          ? extends CcToolchainVariablesApi,
          ? extends ConstraintValueInfoApi,
          ? extends SkylarkRuleContextApi<? extends ConstraintValueInfoApi>,
          ? extends CcToolchainConfigInfoApi,
          ? extends CcCompilationOutputsApi<? extends FileApi>>
      ccModule;

  private final CcInfoApi.Provider<? extends FileApi> ccInfoProvider;
  private final CcToolchainConfigInfoApi.Provider ccToolchainConfigInfoProvider;
  private final PyWrapCcHelperApi<?, ?, ?, ?, ?, ?, ?, ?, ?> pyWrapCcHelper;
  private final GoWrapCcHelperApi<?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?> goWrapCcHelper;
  private final PyWrapCcInfoApi.Provider pyWrapCcInfoProvider;
  private final PyCcLinkParamsProviderApi.Provider pyCcLinkInfoParamsInfoProvider;

  public CcBootstrap(
      CcModuleApi<
              ? extends SkylarkActionFactoryApi,
              ? extends FileApi,
              ? extends CcToolchainProviderApi<? extends FeatureConfigurationApi>,
              ? extends FeatureConfigurationApi,
              ? extends CcCompilationContextApi<? extends FileApi>,
              ? extends
                  LinkerInputApi<? extends LibraryToLinkApi<? extends FileApi>, ? extends FileApi>,
              ? extends CcLinkingContextApi<? extends FileApi>,
              ? extends LibraryToLinkApi<? extends FileApi>,
              ? extends CcToolchainVariablesApi,
              ? extends ConstraintValueInfoApi,
              ? extends SkylarkRuleContextApi<? extends ConstraintValueInfoApi>,
              ? extends CcToolchainConfigInfoApi,
              ? extends CcCompilationOutputsApi<? extends FileApi>>
          ccModule,
      CcInfoApi.Provider<? extends FileApi> ccInfoProvider,
      CcToolchainConfigInfoApi.Provider ccToolchainConfigInfoProvider,
      PyWrapCcHelperApi<?, ?, ?, ?, ?, ?, ?, ?, ?> pyWrapCcHelper,
      GoWrapCcHelperApi<?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?> goWrapCcHelper,
      PyWrapCcInfoApi.Provider pyWrapCcInfoProvider,
      PyCcLinkParamsProviderApi.Provider pyCcLinkInfoParamsInfoProvider) {
    this.ccModule = ccModule;
    this.ccInfoProvider = ccInfoProvider;
    this.ccToolchainConfigInfoProvider = ccToolchainConfigInfoProvider;
    this.pyWrapCcHelper = pyWrapCcHelper;
    this.goWrapCcHelper = goWrapCcHelper;
    this.pyWrapCcInfoProvider = pyWrapCcInfoProvider;
    this.pyCcLinkInfoParamsInfoProvider = pyCcLinkInfoParamsInfoProvider;
  }

  @Override
  public void addBindingsToBuilder(ImmutableMap.Builder<String, Object> builder) {
    builder.put("cc_common", ccModule);
    builder.put("CcInfo", ccInfoProvider);
    builder.put("CcToolchainConfigInfo", ccToolchainConfigInfoProvider);
    builder.put(
        "py_wrap_cc_helper_do_not_use",
        FlagGuardedValue.onlyWhenExperimentalFlagIsTrue(
            FlagIdentifier.EXPERIMENTAL_GOOGLE_LEGACY_API, pyWrapCcHelper));
    builder.put(
        "go_wrap_cc_helper_do_not_use",
        FlagGuardedValue.onlyWhenExperimentalFlagIsTrue(
            FlagIdentifier.EXPERIMENTAL_GOOGLE_LEGACY_API, goWrapCcHelper));
    builder.put("PyWrapCcInfo", pyWrapCcInfoProvider);
    builder.put("PyCcLinkParamsProvider", pyCcLinkInfoParamsInfoProvider);
  }
}
