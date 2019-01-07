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

package com.google.devtools.build.skydoc.fakebuildapi.cpp;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.skylark.SkylarkRuleContext;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.skylarkbuildapi.ProviderApi;
import com.google.devtools.build.lib.skylarkbuildapi.cpp.BazelCcModuleApi;
import com.google.devtools.build.lib.skylarkbuildapi.cpp.CcCompilationContextApi;
import com.google.devtools.build.lib.skylarkbuildapi.cpp.CcCompilationOutputsApi;
import com.google.devtools.build.lib.skylarkbuildapi.cpp.CcInfoApi;
import com.google.devtools.build.lib.skylarkbuildapi.cpp.CcLinkingInfoApi;
import com.google.devtools.build.lib.skylarkbuildapi.cpp.CcModuleApi;
import com.google.devtools.build.lib.skylarkbuildapi.cpp.CcToolchainProviderApi;
import com.google.devtools.build.lib.skylarkbuildapi.cpp.CcToolchainVariablesApi;
import com.google.devtools.build.lib.skylarkbuildapi.cpp.CompilationInfoApi;
import com.google.devtools.build.lib.skylarkbuildapi.cpp.FeatureConfigurationApi;
import com.google.devtools.build.lib.skylarkbuildapi.cpp.LinkingInfoApi;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.SkylarkDict;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.skydoc.fakebuildapi.FakeProviderApi;

/** Fake implementation of {@link CcModuleApi}. */
public class FakeCcModule
    implements BazelCcModuleApi<
        CcToolchainProviderApi,
        FeatureConfigurationApi,
        CompilationInfoApi,
        CcCompilationContextApi,
        CcCompilationOutputsApi,
        LinkingInfoApi,
        CcLinkingInfoApi,
        CcToolchainVariablesApi> {

  @Override
  public ProviderApi getCcToolchainProvider() {
    return new FakeProviderApi();
  }

  @Override
  public FeatureConfigurationApi configureFeatures(CcToolchainProviderApi toolchain,
      SkylarkList<String> requestedFeatures, SkylarkList<String> unsupportedFeatures)
      throws EvalException {
    return null;
  }

  @Override
  public String getToolForAction(FeatureConfigurationApi featureConfiguration, String actionName) {
    return "";
  }

  @Override
  public boolean isEnabled(FeatureConfigurationApi featureConfiguration, String featureName) {
    return false;
  }

  @Override
  public boolean actionIsEnabled(FeatureConfigurationApi featureConfiguration, String actionName) {
    return false;
  }

  @Override
  public SkylarkList<String> getCommandLine(FeatureConfigurationApi featureConfiguration,
      String actionName, CcToolchainVariablesApi variables) {
    return null;
  }

  @Override
  public SkylarkDict<String, String> getEnvironmentVariable(
      FeatureConfigurationApi featureConfiguration, String actionName,
      CcToolchainVariablesApi variables) {
    return null;
  }

  @Override
  public CcToolchainVariablesApi getCompileBuildVariables(
      CcToolchainProviderApi ccToolchainProvider, FeatureConfigurationApi featureConfiguration,
      Object sourceFile, Object outputFile, Object userCompileFlags, Object includeDirs,
      Object quoteIncludeDirs, Object systemIncludeDirs, Object defines, boolean usePic,
      boolean addLegacyCxxOptions) throws EvalException {
    return null;
  }

  @Override
  public CcToolchainVariablesApi getLinkBuildVariables(CcToolchainProviderApi ccToolchainProvider,
      FeatureConfigurationApi featureConfiguration, Object librarySearchDirectories,
      Object runtimeLibrarySearchDirectories, Object userLinkFlags, Object outputFile,
      Object paramFile, Object defFile, boolean isUsingLinkerNotArchiver,
      boolean isCreatingSharedLibrary, boolean mustKeepDebug, boolean useTestOnlyFlags,
      boolean isStaticLinkingMode) throws EvalException {
    return null;
  }

  @Override
  public CcToolchainVariablesApi getVariables() {
    return null;
  }

  @Override
  public Object createLibraryLinkerInput(
      Object actions,
      Object featureConfiguration,
      Object ccToolchainProvider,
      Object staticLibrary,
      Object picStaticLibrary,
      Object dynamicLibrary,
      Object interfaceLibrary,
      boolean alwayslink,
      Location location,
      Environment environment) {
    return null;
  }

  @Override
  public Object createCcLinkingInfo(
      Object librariesToLinkObject,
      Object userLinkFlagsObject,
      Location location,
      Environment environment) {
    return null;
  }

  @Override
  public CcInfoApi mergeCcInfos(SkylarkList<CcInfoApi> ccInfos) {
    return null;
  }

  @Override
  public CcCompilationContextApi createCcCompilationContext(
      Object headers, Object systemIncludes, Object includes, Object quoteIncludes, Object defines)
      throws EvalException {
    return null;
  }

  @Override
  public String legacyCcFlagsMakeVariable(CcToolchainProviderApi ccToolchain) {
    return "";
  }

  @Override
  public CompilationInfoApi compile(
      SkylarkRuleContext skylarkRuleContext,
      FeatureConfigurationApi skylarkFeatureConfiguration,
      CcToolchainProviderApi skylarkCcToolchainProvider,
      SkylarkList<Artifact> sources,
      SkylarkList<Artifact> headers,
      Object skylarkIncludes,
      Object skylarkCopts,
      SkylarkList<CcCompilationContextApi> ccCompilationContexts) {
    return null;
  }

  @Override
  public LinkingInfoApi link(
      SkylarkRuleContext skylarkRuleContext,
      FeatureConfigurationApi skylarkFeatureConfiguration,
      CcToolchainProviderApi skylarkCcToolchainProvider,
      CcCompilationOutputsApi ccCompilationOutputs,
      Object skylarkLinkopts,
      Object dynamicLibrary,
      SkylarkList<CcLinkingInfoApi> skylarkCcLinkingInfos,
      boolean neverLink) {
    return null;
  }
}
