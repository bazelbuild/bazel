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

package com.google.devtools.build.lib.starlarkbuildapi.cpp;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.starlarkbuildapi.FileApi;
import com.google.devtools.build.lib.starlarkbuildapi.StarlarkActionFactoryApi;
import com.google.devtools.build.lib.starlarkbuildapi.StarlarkRuleContextApi;
import com.google.devtools.build.lib.starlarkbuildapi.core.Bootstrap;
import com.google.devtools.build.lib.starlarkbuildapi.core.ContextAndFlagGuardedValue;
import com.google.devtools.build.lib.starlarkbuildapi.platform.ConstraintValueInfoApi;
import com.google.devtools.build.lib.starlarkbuildapi.python.PyBootstrap;
import net.starlark.java.eval.FlagGuardedValue;

/** {@link Bootstrap} for Starlark objects related to cpp rules. */
public class CcBootstrap implements Bootstrap {
  private static final ImmutableSet<PackageIdentifier> allowedRepositories =
      ImmutableSet.of(
          PackageIdentifier.createUnchecked("_builtins", ""),
          PackageIdentifier.createUnchecked("bazel_tools", ""),
          PackageIdentifier.createUnchecked("local_config_cc", ""),
          PackageIdentifier.createUnchecked("rules_cc", ""),
          PackageIdentifier.createUnchecked("", "tools/build_defs/cc"));

  private final CcModuleApi<
          ? extends StarlarkActionFactoryApi,
          ? extends FileApi,
          ? extends FdoContextApi<?>,
          ? extends
              CcToolchainProviderApi<
                  ? extends FeatureConfigurationApi, ?, ? extends FdoContextApi<?>>,
          ? extends FeatureConfigurationApi,
          ? extends CcCompilationContextApi<? extends FileApi>,
          ? extends LtoBackendArtifactsApi<? extends FileApi>,
          ? extends
              LinkerInputApi<
                  ? extends
                      LibraryToLinkApi<
                          ? extends FileApi, ? extends LtoBackendArtifactsApi<? extends FileApi>>,
                  ? extends LtoBackendArtifactsApi<? extends FileApi>,
                  ? extends FileApi>,
          ? extends CcLinkingContextApi<? extends FileApi>,
          ? extends
              LibraryToLinkApi<
                  ? extends FileApi, ? extends LtoBackendArtifactsApi<? extends FileApi>>,
          ? extends CcToolchainVariablesApi,
          ? extends ConstraintValueInfoApi,
          ? extends StarlarkRuleContextApi<? extends ConstraintValueInfoApi>,
          ? extends CcToolchainConfigInfoApi,
          ? extends CcCompilationOutputsApi<? extends FileApi>,
          ? extends CcDebugInfoContextApi,
          ? extends CppModuleMapApi<? extends FileApi>>
      ccModule;

  private final CcInfoApi.Provider<? extends FileApi> ccInfoProvider;
  private final DebugPackageInfoApi.Provider<? extends FileApi> debugPackageInfoProvider;
  private final CcToolchainConfigInfoApi.Provider ccToolchainConfigInfoProvider;
  private final PyWrapCcHelperApi<?, ?, ?, ?, ?, ?, ?, ?, ?> pyWrapCcHelper;
  private final GoWrapCcHelperApi<?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?> goWrapCcHelper;
  private final PyWrapCcInfoApi.Provider pyWrapCcInfoProvider;
  private final PyCcLinkParamsProviderApi.Provider pyCcLinkInfoParamsInfoProvider;

  public CcBootstrap(
      CcModuleApi<
              ? extends StarlarkActionFactoryApi,
              ? extends FileApi,
              ? extends FdoContextApi<?>,
              ? extends
                  CcToolchainProviderApi<
                      ? extends FeatureConfigurationApi, ?, ? extends FdoContextApi<?>>,
              ? extends FeatureConfigurationApi,
              ? extends CcCompilationContextApi<? extends FileApi>,
              ? extends LtoBackendArtifactsApi<? extends FileApi>,
              ? extends
                  LinkerInputApi<
                      ? extends
                          LibraryToLinkApi<
                              ? extends FileApi,
                              ? extends LtoBackendArtifactsApi<? extends FileApi>>,
                      ? extends LtoBackendArtifactsApi<? extends FileApi>,
                      ? extends FileApi>,
              ? extends CcLinkingContextApi<? extends FileApi>,
              ? extends
                  LibraryToLinkApi<
                      ? extends FileApi, ? extends LtoBackendArtifactsApi<? extends FileApi>>,
              ? extends CcToolchainVariablesApi,
              ? extends ConstraintValueInfoApi,
              ? extends StarlarkRuleContextApi<? extends ConstraintValueInfoApi>,
              ? extends CcToolchainConfigInfoApi,
              ? extends CcCompilationOutputsApi<? extends FileApi>,
              ? extends CcDebugInfoContextApi,
              ? extends CppModuleMapApi<? extends FileApi>>
          ccModule,
      CcInfoApi.Provider<? extends FileApi> ccInfoProvider,
      DebugPackageInfoApi.Provider<? extends FileApi> debugPackageInfoProvider,
      CcToolchainConfigInfoApi.Provider ccToolchainConfigInfoProvider,
      PyWrapCcHelperApi<?, ?, ?, ?, ?, ?, ?, ?, ?> pyWrapCcHelper,
      GoWrapCcHelperApi<?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?> goWrapCcHelper,
      PyWrapCcInfoApi.Provider pyWrapCcInfoProvider,
      PyCcLinkParamsProviderApi.Provider pyCcLinkInfoParamsInfoProvider) {
    this.ccModule = ccModule;
    this.ccInfoProvider = ccInfoProvider;
    this.debugPackageInfoProvider = debugPackageInfoProvider;
    this.ccToolchainConfigInfoProvider = ccToolchainConfigInfoProvider;
    this.pyWrapCcHelper = pyWrapCcHelper;
    this.goWrapCcHelper = goWrapCcHelper;
    this.pyWrapCcInfoProvider = pyWrapCcInfoProvider;
    this.pyCcLinkInfoParamsInfoProvider = pyCcLinkInfoParamsInfoProvider;
  }

  @Override
  public void addBindingsToBuilder(ImmutableMap.Builder<String, Object> builder) {
    builder.put(
        "cc_common",
        ContextAndFlagGuardedValue.onlyInAllowedReposOrWhenIncompatibleFlagIsFalse(
            BuildLanguageOptions.INCOMPATIBLE_STOP_EXPORTING_LANGUAGE_MODULES,
            ccModule,
            allowedRepositories));
    builder.put(
        "CcInfo",
        ContextAndFlagGuardedValue.onlyInAllowedReposOrWhenIncompatibleFlagIsFalse(
            BuildLanguageOptions.INCOMPATIBLE_STOP_EXPORTING_LANGUAGE_MODULES,
            ccInfoProvider,
            allowedRepositories));
    builder.put(
        "DebugPackageInfo",
        ContextAndFlagGuardedValue.onlyInAllowedReposOrWhenIncompatibleFlagIsFalse(
            BuildLanguageOptions.INCOMPATIBLE_STOP_EXPORTING_LANGUAGE_MODULES,
            debugPackageInfoProvider,
            allowedRepositories));
    builder.put(
        "CcToolchainConfigInfo",
        ContextAndFlagGuardedValue.onlyInAllowedReposOrWhenIncompatibleFlagIsFalse(
            BuildLanguageOptions.INCOMPATIBLE_STOP_EXPORTING_LANGUAGE_MODULES,
            ccToolchainConfigInfoProvider,
            allowedRepositories));
    builder.put(
        "py_wrap_cc_helper_do_not_use",
        FlagGuardedValue.onlyWhenExperimentalFlagIsTrue(
            BuildLanguageOptions.EXPERIMENTAL_GOOGLE_LEGACY_API, pyWrapCcHelper));
    builder.put(
        "go_wrap_cc_helper_do_not_use",
        FlagGuardedValue.onlyWhenExperimentalFlagIsTrue(
            BuildLanguageOptions.EXPERIMENTAL_GOOGLE_LEGACY_API, goWrapCcHelper));
    builder.put(
        "PyWrapCcInfo",
        ContextAndFlagGuardedValue.onlyInAllowedReposOrWhenIncompatibleFlagIsFalse(
            BuildLanguageOptions.INCOMPATIBLE_STOP_EXPORTING_LANGUAGE_MODULES,
            pyWrapCcInfoProvider,
            PyBootstrap.allowedRepositories));
    builder.put(
        "PyCcLinkParamsProvider",
        ContextAndFlagGuardedValue.onlyInAllowedReposOrWhenIncompatibleFlagIsFalse(
            BuildLanguageOptions.INCOMPATIBLE_STOP_EXPORTING_LANGUAGE_MODULES,
            pyCcLinkInfoParamsInfoProvider,
            PyBootstrap.allowedRepositories));
  }
}
