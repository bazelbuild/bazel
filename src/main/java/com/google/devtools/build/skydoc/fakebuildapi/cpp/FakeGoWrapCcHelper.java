// Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.skydoc.fakebuildapi.cpp;

import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.starlarkbuildapi.FileApi;
import com.google.devtools.build.lib.starlarkbuildapi.RunfilesApi;
import com.google.devtools.build.lib.starlarkbuildapi.StarlarkRuleContextApi;
import com.google.devtools.build.lib.starlarkbuildapi.core.TransitiveInfoCollectionApi;
import com.google.devtools.build.lib.starlarkbuildapi.cpp.CcCompilationContextApi;
import com.google.devtools.build.lib.starlarkbuildapi.cpp.CcInfoApi;
import com.google.devtools.build.lib.starlarkbuildapi.cpp.CcLinkingContextApi;
import com.google.devtools.build.lib.starlarkbuildapi.cpp.CcToolchainProviderApi;
import com.google.devtools.build.lib.starlarkbuildapi.cpp.CompilationInfoApi;
import com.google.devtools.build.lib.starlarkbuildapi.cpp.FeatureConfigurationApi;
import com.google.devtools.build.lib.starlarkbuildapi.cpp.GoCcLinkParamsInfoApi;
import com.google.devtools.build.lib.starlarkbuildapi.cpp.GoWrapCcHelperApi;
import com.google.devtools.build.lib.starlarkbuildapi.cpp.GoWrapCcInfoApi;
import com.google.devtools.build.lib.starlarkbuildapi.cpp.WrapCcIncludeProviderApi;
import com.google.devtools.build.lib.starlarkbuildapi.go.GoConfigurationApi;
import com.google.devtools.build.lib.starlarkbuildapi.go.GoContextInfoApi;
import com.google.devtools.build.lib.starlarkbuildapi.go.GoPackageInfoApi;
import com.google.devtools.build.lib.starlarkbuildapi.platform.ConstraintValueInfoApi;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.Tuple;

/** Fake implementation of {@link GoWrapCcHelperApi}. */
public class FakeGoWrapCcHelper
    implements GoWrapCcHelperApi<
        FileApi,
        ConstraintValueInfoApi,
        StarlarkRuleContextApi<ConstraintValueInfoApi>,
        CcInfoApi<FileApi>,
        FeatureConfigurationApi,
        CcToolchainProviderApi<FeatureConfigurationApi>,
        CcLinkingContextApi<FileApi>,
        GoConfigurationApi,
        GoContextInfoApi,
        TransitiveInfoCollectionApi,
        CompilationInfoApi<FileApi>,
        CcCompilationContextApi<FileApi>,
        WrapCcIncludeProviderApi> {

  @Override
  public RunfilesApi starlarkGetGoRunfiles(
      StarlarkRuleContextApi<ConstraintValueInfoApi> starlarkRuleContext) {
    return null;
  }

  @Override
  public int getArchIntSize(GoConfigurationApi goConfig) {
    return 0;
  }

  @Override
  public GoContextInfoApi starlarkCollectTransitiveGoContextGopkg(
      StarlarkRuleContextApi<ConstraintValueInfoApi> starlarkRuleContext,
      FileApi export,
      FileApi pkg,
      FileApi gopkg,
      Object starlarkWrapContext,
      CcInfoApi<FileApi> ccInfo) {
    return null;
  }

  @Override
  public GoWrapCcInfoApi<FileApi> getGoWrapCcInfo(
      StarlarkRuleContextApi<ConstraintValueInfoApi> starlarkRuleContext,
      CcInfoApi<FileApi> ccInfo) {
    return null;
  }

  @Override
  public GoCcLinkParamsInfoApi getGoCcLinkParamsProvider(
      StarlarkRuleContextApi<ConstraintValueInfoApi> ruleContext,
      CcLinkingContextApi<FileApi> ccLinkingContext) {
    return null;
  }

  @Override
  public Tuple /* of FileApi */ createGoCompileActions(
      StarlarkRuleContextApi<ConstraintValueInfoApi> starlarkRuleContext,
      CcToolchainProviderApi<FeatureConfigurationApi> ccToolchainProvider,
      Sequence<?> srcs,
      Sequence<?> deps) {
    return null;
  }

  @Override
  public Tuple /* of FileApi */ createGoCompileActionsGopkg(
      StarlarkRuleContextApi<ConstraintValueInfoApi> starlarkRuleContext,
      CcToolchainProviderApi<FeatureConfigurationApi> ccToolchainProvider,
      Sequence<?> srcs,
      Sequence<?> deps) {
    return null;
  }

  @Override
  public GoPackageInfoApi createTransitiveGopackageInfo(
      StarlarkRuleContextApi<ConstraintValueInfoApi> starlarkRuleContext,
      FileApi starlarkGopkg,
      FileApi export,
      FileApi swigOutGo) {
    return null;
  }

  @Override
  public Depset /*<FileApi>*/ getGopackageFilesForStarlark(
      StarlarkRuleContextApi<ConstraintValueInfoApi> starlarkRuleContext, FileApi starlarkGopkg) {
    return null;
  }

  @Override
  public FeatureConfigurationApi starlarkGetFeatureConfiguration(
      StarlarkRuleContextApi<ConstraintValueInfoApi> starlarkRuleContext,
      CcToolchainProviderApi<FeatureConfigurationApi> ccToolchain) {
    return null;
  }

  @Override
  public Depset starlarkCollectTransitiveSwigIncludes(
      StarlarkRuleContextApi<ConstraintValueInfoApi> starlarkRuleContext) {
    return null;
  }

  @Override
  public CompilationInfoApi<FileApi> starlarkCreateCompileActions(
      StarlarkRuleContextApi<ConstraintValueInfoApi> starlarkRuleContext,
      FeatureConfigurationApi featureConfiguration,
      CcToolchainProviderApi<FeatureConfigurationApi> ccToolchain,
      FileApi ccFile,
      FileApi headerFile,
      Sequence<?> depCcCompilationContexts,
      Sequence<?> targetCopts) {
    return null;
  }

  @Override
  public String starlarkGetMangledTargetName(
      StarlarkRuleContextApi<ConstraintValueInfoApi> starlarkRuleContext) {
    return null;
  }

  @Override
  public WrapCcIncludeProviderApi getWrapCcIncludeProvider(
      StarlarkRuleContextApi<ConstraintValueInfoApi> starlarkRuleContext, Depset swigIncludes) {
    return null;
  }

  @Override
  public void registerSwigAction(
      StarlarkRuleContextApi<ConstraintValueInfoApi> starlarkRuleContext,
      CcToolchainProviderApi<FeatureConfigurationApi> ccToolchain,
      FeatureConfigurationApi featureConfiguration,
      CcCompilationContextApi<FileApi> wrapperCcCompilationContext,
      Depset swigIncludes,
      FileApi swigSource,
      Sequence<?> subParameters,
      FileApi ccFile,
      FileApi headerFile,
      Sequence<?> outputFiles,
      Object outDir,
      Object javaDir,
      Depset auxiliaryInputs,
      String swigAttributeName,
      Object zipTool) {}
}
