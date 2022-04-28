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

package com.google.devtools.build.lib.rules.cpp;

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
import com.google.devtools.build.lib.starlarkbuildapi.cpp.PyCcLinkParamsProviderApi;
import com.google.devtools.build.lib.starlarkbuildapi.cpp.PyWrapCcHelperApi;
import com.google.devtools.build.lib.starlarkbuildapi.cpp.PyWrapCcInfoApi;
import com.google.devtools.build.lib.starlarkbuildapi.cpp.WrapCcHelperApi;
import com.google.devtools.build.lib.starlarkbuildapi.cpp.WrapCcIncludeProviderApi;
import com.google.devtools.build.lib.starlarkbuildapi.go.GoConfigurationApi;
import com.google.devtools.build.lib.starlarkbuildapi.go.GoContextInfoApi;
import com.google.devtools.build.lib.starlarkbuildapi.go.GoPackageInfoApi;
import com.google.devtools.build.lib.starlarkbuildapi.platform.ConstraintValueInfoApi;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Printer;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.Tuple;

/**
 * Fake stub implementations for C++-related Starlark API which are unsupported without use of
 * --experimental_google_legacy_api.
 */
public final class GoogleLegacyStubs {

  private GoogleLegacyStubs() {}

  private static class WrapCcHelper
      implements WrapCcHelperApi<
          FeatureConfigurationApi,
          ConstraintValueInfoApi,
          StarlarkRuleContextApi<ConstraintValueInfoApi>,
          CcToolchainProviderApi<FeatureConfigurationApi, ?, ?>,
          CompilationInfoApi<FileApi>,
          FileApi,
          CcCompilationContextApi<FileApi>,
          WrapCcIncludeProviderApi> {

    @Override
    public FeatureConfigurationApi starlarkGetFeatureConfiguration(
        StarlarkRuleContextApi<ConstraintValueInfoApi> starlarkRuleContext,
        CcToolchainProviderApi<FeatureConfigurationApi, ?, ?> ccToolchain)
        throws EvalException, InterruptedException {
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
        CcToolchainProviderApi<FeatureConfigurationApi, ?, ?> ccToolchain,
        FileApi ccFile,
        FileApi headerFile,
        Sequence<?> depCcCompilationContexts, // <CcCompilationContextApi>
        Sequence<?> targetCopts /* <String> */)
        throws EvalException, InterruptedException {
      return null;
    }

    @Override
    public String starlarkGetMangledTargetName(
        StarlarkRuleContextApi<ConstraintValueInfoApi> starlarkRuleContext)
        throws EvalException, InterruptedException {
      return null;
    }

    @Override
    public WrapCcIncludeProviderApi getWrapCcIncludeProvider(
        StarlarkRuleContextApi<ConstraintValueInfoApi> starlarkRuleContext, Depset swigIncludes)
        throws EvalException, InterruptedException {
      return null;
    }

    @Override
    public void registerSwigAction(
        StarlarkRuleContextApi<ConstraintValueInfoApi> starlarkRuleContext,
        CcToolchainProviderApi<FeatureConfigurationApi, ?, ?> ccToolchain,
        FeatureConfigurationApi featureConfiguration,
        CcCompilationContextApi<FileApi> wrapperCcCompilationContext,
        Depset swigIncludes,
        FileApi swigSource,
        Sequence<?> subParameters, // <String>
        FileApi ccFile,
        FileApi headerFile,
        Sequence<?> outputFiles, // <FileApi>
        Object outDir,
        Object javaDir,
        Depset auxiliaryInputs,
        String swigAttributeName,
        Object zipTool)
        throws EvalException, InterruptedException {}
  }

  /**
   * Fake no-op implementation of {@link PyWrapCcHelperApi}. This implementation should be
   * unreachable without (discouraged) use of --experimental_google_legacy_api.
   */
  public static class PyWrapCcHelper extends WrapCcHelper
      implements PyWrapCcHelperApi<
          FileApi,
          ConstraintValueInfoApi,
          StarlarkRuleContextApi<ConstraintValueInfoApi>,
          CcInfoApi<FileApi>,
          FeatureConfigurationApi,
          CcToolchainProviderApi<FeatureConfigurationApi, ?, ?>,
          CompilationInfoApi<FileApi>,
          CcCompilationContextApi<FileApi>,
          WrapCcIncludeProviderApi> {

    @Override
    public Sequence<String> getPyExtensionLinkopts(
        StarlarkRuleContextApi<ConstraintValueInfoApi> starlarkRuleContext) {
      return null;
    }

    @Override
    public Depset getTransitivePythonSources(
        StarlarkRuleContextApi<ConstraintValueInfoApi> starlarkRuleContext, FileApi pyFile) {
      return null;
    }

    @Override
    public RunfilesApi getPythonRunfiles(
        StarlarkRuleContextApi<ConstraintValueInfoApi> starlarkRuleContext, Depset filesToBuild) {
      return null;
    }

    @Override
    public PyWrapCcInfoApi<FileApi> getPyWrapCcInfo(
        StarlarkRuleContextApi<ConstraintValueInfoApi> starlarkRuleContext,
        CcInfoApi<FileApi> ccInfo) {
      return null;
    }
  }

  /**
   * Fake no-op implementation of {@link GoWrapCcHelperApi}. This implementation should be
   * unreachable without (discouraged) use of --experimental_google_legacy_api.
   */
  public static class GoWrapCcHelper extends WrapCcHelper
      implements GoWrapCcHelperApi<
          FileApi,
          ConstraintValueInfoApi,
          StarlarkRuleContextApi<ConstraintValueInfoApi>,
          CcInfoApi<FileApi>,
          FeatureConfigurationApi,
          CcToolchainProviderApi<FeatureConfigurationApi, ?, ?>,
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
    public Tuple createGoCompileActions(
        StarlarkRuleContextApi<ConstraintValueInfoApi> starlarkRuleContext,
        CcToolchainProviderApi<FeatureConfigurationApi, ?, ?> ccToolchainProvider,
        Sequence<?> srcs, // <FileApi>
        Sequence<?> deps /* <TransitiveInfoCollectionApi> */) {
      return null;
    }

    @Override
    public Tuple createGoCompileActionsGopkg(
        StarlarkRuleContextApi<ConstraintValueInfoApi> starlarkRuleContext,
        CcToolchainProviderApi<FeatureConfigurationApi, ?, ?> ccToolchainProvider,
        Sequence<?> srcs, // <FileApi>
        Sequence<?> deps /* <TransitiveInfoCollectionApi> */) {
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
  }

  /**
   * Fake no-op implementation of {@link PyWrapCcInfoApi.Provider}. This implementation should be
   * unreachable without (discouraged) use of --experimental_google_legacy_api.
   */
  public static class PyWrapCcInfoProvider implements PyWrapCcInfoApi.Provider {

    @Override
    public void repr(Printer printer) {
      printer.append("<unknown object>");
    }
  }

  /**
   * Fake no-op implementation of {@link PyCcLinkParamsProviderApi.Provider}. This implementation
   * should be unreachable without (discouraged) use of --experimental_google_legacy_api.
   */
  public static class PyCcLinkParamsProvider implements PyCcLinkParamsProviderApi.Provider {

    @Override
    public void repr(Printer printer) {
      printer.append("<unknown object>");
    }
  }
}
