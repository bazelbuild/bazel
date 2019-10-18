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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.skylarkbuildapi.FileApi;
import com.google.devtools.build.lib.skylarkbuildapi.ProviderApi;
import com.google.devtools.build.lib.skylarkbuildapi.SkylarkActionFactoryApi;
import com.google.devtools.build.lib.skylarkbuildapi.SkylarkRuleContextApi;
import com.google.devtools.build.lib.skylarkbuildapi.cpp.BazelCcModuleApi;
import com.google.devtools.build.lib.skylarkbuildapi.cpp.CcCompilationContextApi;
import com.google.devtools.build.lib.skylarkbuildapi.cpp.CcCompilationOutputsApi;
import com.google.devtools.build.lib.skylarkbuildapi.cpp.CcInfoApi;
import com.google.devtools.build.lib.skylarkbuildapi.cpp.CcLinkingContextApi;
import com.google.devtools.build.lib.skylarkbuildapi.cpp.CcLinkingOutputsApi;
import com.google.devtools.build.lib.skylarkbuildapi.cpp.CcModuleApi;
import com.google.devtools.build.lib.skylarkbuildapi.cpp.CcToolchainConfigInfoApi;
import com.google.devtools.build.lib.skylarkbuildapi.cpp.CcToolchainProviderApi;
import com.google.devtools.build.lib.skylarkbuildapi.cpp.CcToolchainVariablesApi;
import com.google.devtools.build.lib.skylarkbuildapi.cpp.FeatureConfigurationApi;
import com.google.devtools.build.lib.skylarkbuildapi.cpp.LibraryToLinkApi;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.SkylarkDict;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.syntax.SkylarkList.Tuple;
import com.google.devtools.build.lib.syntax.StarlarkThread;
import com.google.devtools.build.skydoc.fakebuildapi.FakeProviderApi;

/** Fake implementation of {@link CcModuleApi}. */
public class FakeCcModule
    implements BazelCcModuleApi<
        SkylarkActionFactoryApi,
        FileApi,
        SkylarkRuleContextApi,
        CcToolchainProviderApi<FeatureConfigurationApi>,
        FeatureConfigurationApi,
        CcCompilationContextApi,
        CcCompilationOutputsApi<FileApi>,
        CcLinkingOutputsApi<FileApi>,
        LibraryToLinkApi<FileApi>,
        CcLinkingContextApi<FileApi>,
        CcToolchainVariablesApi,
        CcToolchainConfigInfoApi> {

  @Override
  public ProviderApi getCcToolchainProvider() {
    return new FakeProviderApi();
  }

  @Override
  public FeatureConfigurationApi configureFeatures(
      Object ruleContextOrNone,
      CcToolchainProviderApi<FeatureConfigurationApi> toolchain,
      SkylarkList<?> requestedFeatures,
      SkylarkList<?> unsupportedFeatures)
      throws EvalException {
    return null;
  }

  @Override
  public String getToolForAction(FeatureConfigurationApi featureConfiguration, String actionName) {
    return "";
  }

  @Override
  public SkylarkList<String> getExecutionRequirements(
      FeatureConfigurationApi featureConfiguration, String actionName) {
    return SkylarkList.createImmutable(ImmutableList.of());
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
      CcToolchainProviderApi ccToolchainProvider,
      FeatureConfigurationApi featureConfiguration,
      Object sourceFile,
      Object outputFile,
      Object userCompileFlags,
      Object includeDirs,
      Object quoteIncludeDirs,
      Object systemIncludeDirs,
      Object frameworkIncludeDirs,
      Object defines,
      boolean usePic,
      boolean addLegacyCxxOptions)
      throws EvalException {
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
  public LibraryToLinkApi createLibraryLinkerInput(
      Object actions,
      Object featureConfiguration,
      Object ccToolchainProvider,
      Object staticLibrary,
      Object picStaticLibrary,
      Object dynamicLibrary,
      Object interfaceLibrary,
      boolean alwayslink,
      Location location,
      StarlarkThread thread) {
    return null;
  }

  @Override
  public CcLinkingContextApi createCcLinkingInfo(
      Object librariesToLinkObject,
      Object userLinkFlagsObject,
      SkylarkList<?> nonCodeInputs,
      Location location,
      StarlarkThread thread) {
    return null;
  }

  @Override
  public CcInfoApi mergeCcInfos(SkylarkList<?> ccInfos) {
    return null;
  }

  @Override
  public CcCompilationContextApi createCcCompilationContext(
      Object headers,
      Object systemIncludes,
      Object includes,
      Object quoteIncludes,
      Object frameworkIncludes,
      Object defines,
      Object localDefines)
      throws EvalException {
    return null;
  }

  @Override
  public String legacyCcFlagsMakeVariable(CcToolchainProviderApi ccToolchain) {
    return "";
  }

  @Override
  public boolean isCcToolchainResolutionEnabled(SkylarkRuleContextApi ruleContext) {
    return false;
  }

  @Override
  public Tuple<Object> compile(
      SkylarkActionFactoryApi skylarkActionFactoryApi,
      FeatureConfigurationApi skylarkFeatureConfiguration,
      CcToolchainProviderApi<FeatureConfigurationApi> skylarkCcToolchainProvider,
      SkylarkList<?> sources,
      SkylarkList<?> publicHeaders,
      SkylarkList<?> privateHeaders,
      SkylarkList<?> includes,
      SkylarkList<?> quoteIncludes,
      SkylarkList<?> defines,
      SkylarkList<?> localDefines,
      SkylarkList<?> systemIncludes,
      SkylarkList<?> frameworkIncludes,
      SkylarkList<?> userCompileFlags,
      SkylarkList<?> ccCompilationContexts,
      String name,
      boolean disallowPicOutputs,
      boolean disallowNopicOutputs,
      SkylarkList<?> additionalInputs,
      Location location,
      StarlarkThread thread)
      throws EvalException, InterruptedException {
    return null;
  }

  @Override
  public Tuple<Object> createLinkingContextFromCompilationOutputs(
      SkylarkActionFactoryApi skylarkActionFactoryApi,
      FeatureConfigurationApi skylarkFeatureConfiguration,
      CcToolchainProviderApi<FeatureConfigurationApi> skylarkCcToolchainProvider,
      CcCompilationOutputsApi<FileApi> compilationOutputs,
      SkylarkList<?> userLinkFlags,
      SkylarkList<?> ccLinkingContextApis,
      String name,
      String language,
      boolean alwayslink,
      SkylarkList<?> nonCodeInputs,
      boolean disallowStaticLibraries,
      boolean disallowDynamicLibraries,
      Object grepIncludes,
      Location location,
      StarlarkThread thread)
      throws InterruptedException, EvalException {
    return null;
  }

  @Override
  public CcLinkingOutputsApi<FileApi> link(
      SkylarkActionFactoryApi skylarkActionFactoryApi,
      FeatureConfigurationApi skylarkFeatureConfiguration,
      CcToolchainProviderApi<FeatureConfigurationApi> skylarkCcToolchainProvider,
      Object compilationOutputs,
      SkylarkList<?> userLinkFlags,
      SkylarkList<?> linkingContexts,
      String name,
      String language,
      String outputType,
      boolean linkDepsStatically,
      SkylarkList<?> additionalInputs,
      Location location,
      StarlarkThread thread)
      throws InterruptedException, EvalException {
    return null;
  }

  @Override
  public CcToolchainConfigInfoApi ccToolchainConfigInfoFromSkylark(
      SkylarkRuleContextApi skylarkRuleContext,
      SkylarkList<?> features,
      SkylarkList<?> actionConfigs,
      SkylarkList<?> artifactNamePatterns,
      SkylarkList<?> cxxBuiltInIncludeDirectories,
      String toolchainIdentifier,
      String hostSystemName,
      String targetSystemName,
      String targetCpu,
      String targetLibc,
      String compiler,
      String abiVersion,
      String abiLibcVersion,
      SkylarkList<?> toolPaths,
      SkylarkList<?> makeVariables,
      Object builtinSysroot,
      Object ccTargetOs)
      throws EvalException {
    return null;
  }

  @Override
  public CcCompilationOutputsApi<FileApi> createCompilationOutputsFromSkylark(
      Object objectsObject, Object picObjectsObject, Location location) {
    return null;
  }

  @Override
  public CcCompilationOutputsApi<FileApi> mergeCcCompilationOutputsFromSkylark(
      SkylarkList<?> compilationOutputs) {
    return null;
  }
}
