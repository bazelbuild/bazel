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

import com.google.devtools.build.lib.skylarkbuildapi.FileApi;
import com.google.devtools.build.lib.skylarkbuildapi.RunfilesApi;
import com.google.devtools.build.lib.skylarkbuildapi.SkylarkRuleContextApi;
import com.google.devtools.build.lib.skylarkbuildapi.cpp.CcCompilationContextApi;
import com.google.devtools.build.lib.skylarkbuildapi.cpp.CcInfoApi;
import com.google.devtools.build.lib.skylarkbuildapi.cpp.CcToolchainProviderApi;
import com.google.devtools.build.lib.skylarkbuildapi.cpp.CompilationInfoApi;
import com.google.devtools.build.lib.skylarkbuildapi.cpp.FeatureConfigurationApi;
import com.google.devtools.build.lib.skylarkbuildapi.cpp.PyWrapCcHelperApi;
import com.google.devtools.build.lib.skylarkbuildapi.cpp.PyWrapCcInfoApi;
import com.google.devtools.build.lib.skylarkbuildapi.cpp.WrapCcIncludeProviderApi;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;

/** Fake implementation of {@link PyWrapCcHelperApi}. */
public class FakePyWrapCcHelper
    implements PyWrapCcHelperApi<
        FileApi,
        SkylarkRuleContextApi,
        CcInfoApi,
        FeatureConfigurationApi,
        CcToolchainProviderApi<FeatureConfigurationApi>,
        CompilationInfoApi,
        CcCompilationContextApi,
        WrapCcIncludeProviderApi> {

  @Override
  public SkylarkList<String> getPyExtensionLinkopts(SkylarkRuleContextApi skylarkRuleContext) {
    return null;
  }

  @Override
  public SkylarkNestedSet getTransitivePythonSources(
      SkylarkRuleContextApi skylarkRuleContext, FileApi pyFile) {
    return null;
  }

  @Override
  public RunfilesApi getPythonRunfiles(
      SkylarkRuleContextApi skylarkRuleContext, SkylarkNestedSet filesToBuild) {
    return null;
  }

  @Override
  public PyWrapCcInfoApi getPyWrapCcInfo(
      SkylarkRuleContextApi skylarkRuleContext, CcInfoApi ccInfo) {
    return null;
  }

  @Override
  public FeatureConfigurationApi skylarkGetFeatureConfiguration(
      SkylarkRuleContextApi skylarkRuleContext,
      CcToolchainProviderApi<FeatureConfigurationApi> ccToolchain) {
    return null;
  }

  @Override
  public SkylarkNestedSet skylarkCollectTransitiveSwigIncludes(
      SkylarkRuleContextApi skylarkRuleContext) {
    return null;
  }

  @Override
  public String skylarkGetMangledTargetName(SkylarkRuleContextApi skylarkRuleContext) {
    return null;
  }

  @Override
  public WrapCcIncludeProviderApi getWrapCcIncludeProvider(
      SkylarkRuleContextApi skylarkRuleContext, SkylarkNestedSet swigIncludes) {
    return null;
  }

  @Override
  public void registerSwigAction(
      SkylarkRuleContextApi skylarkRuleContext,
      CcToolchainProviderApi<FeatureConfigurationApi> ccToolchain,
      FeatureConfigurationApi featureConfiguration,
      CcCompilationContextApi wrapperCcCompilationContext,
      SkylarkNestedSet swigIncludes,
      FileApi swigSource,
      SkylarkList<String> subParameters,
      FileApi ccFile,
      FileApi headerFile,
      SkylarkList<FileApi> outputFiles,
      Object outDir,
      Object javaDir,
      SkylarkNestedSet auxiliaryInputs,
      String swigAttributeName,
      Object zipTool) {}

  @Override
  public CompilationInfoApi skylarkCreateCompileActions(
      SkylarkRuleContextApi skylarkRuleContext,
      FeatureConfigurationApi featureConfiguration,
      CcToolchainProviderApi<FeatureConfigurationApi> ccToolchain,
      FileApi ccFile,
      FileApi headerFile,
      SkylarkList<CcCompilationContextApi> depCcCompilationContexts,
      SkylarkList<String> targetCopts) {
    return null;
  }
}
