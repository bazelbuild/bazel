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
import net.starlark.java.eval.Starlark;

/** {@link Bootstrap} for Starlark objects related to cpp rules. */
public class CcBootstrap implements Bootstrap {
  private static final ImmutableSet<PackageIdentifier> allowedRepositories =
      ImmutableSet.of(
          PackageIdentifier.createUnchecked("_builtins", ""),
          PackageIdentifier.createUnchecked("bazel_tools", ""),
          PackageIdentifier.createUnchecked("local_config_cc", ""),
          PackageIdentifier.createUnchecked("rules_cc", ""),
          PackageIdentifier.createUnchecked("", "tools/build_defs/cc"));

  private final CcInfoApi.Provider<? extends FileApi> ccInfoProvider;
  private final DebugPackageInfoApi.Provider<? extends FileApi> debugPackageInfoProvider;
  private final CcToolchainConfigInfoApi.Provider ccToolchainConfigInfoProvider;

  public CcBootstrap(
      CcModuleApi<
              ? extends StarlarkActionFactoryApi,
              ? extends FileApi,
              ? extends FeatureConfigurationApi,
              ? extends
                  CcCompilationContextApi<
                      ? extends FileApi, ? extends CppModuleMapApi<? extends FileApi>>,
              ? extends LtoBackendArtifactsApi<? extends FileApi>,
              ? extends CcLinkingContextApi<? extends FileApi>,
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
      CcToolchainConfigInfoApi.Provider ccToolchainConfigInfoProvider) {
    this.ccInfoProvider = ccInfoProvider;
    this.debugPackageInfoProvider = debugPackageInfoProvider;
    this.ccToolchainConfigInfoProvider = ccToolchainConfigInfoProvider;
  }

  @Override
  public void addBindingsToBuilder(ImmutableMap.Builder<String, Object> builder) {
    builder.put(
        "cc_common",
        ContextAndFlagGuardedValue.onlyInAllowedReposOrWhenIncompatibleFlagIsFalse(
            BuildLanguageOptions.INCOMPATIBLE_STOP_EXPORTING_LANGUAGE_MODULES,
            Starlark.NONE,
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
  }
}
