// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.objc;

import static com.google.devtools.build.lib.rules.objc.CompilationSupport.IncludeProcessingType.INCLUDE_SCANNING;

import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.build.lib.rules.cpp.CcCompilationContext;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.cpp.CppCompileActionBuilder;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration.HeadersCheckingMode;
import com.google.devtools.build.lib.rules.cpp.CppSemantics;
import com.google.devtools.build.lib.rules.cpp.HeaderDiscovery.DotdPruningMode;
import com.google.devtools.build.lib.rules.cpp.IncludeProcessing;
import com.google.devtools.build.lib.rules.objc.CompilationSupport.IncludeProcessingType;

/**
 * CppSemantics for objc builds.
 */
public class ObjcCppSemantics implements CppSemantics {

  private final IncludeProcessingType includeProcessingType;
  private final IncludeProcessing includeProcessing;
  private final ObjcConfiguration config;
  private final IntermediateArtifacts intermediateArtifacts;
  private final BuildConfiguration buildConfiguration;
  private final boolean enableModules;

  /**
   * Creates an instance of ObjcCppSemantics
   *
   * @param includeProcessingType The type of include processing to be run.
   * @param includeProcessing the closure providing the strategy for processing of includes for
   *     actions
   * @param config the ObjcConfiguration for this build
   * @param intermediateArtifacts used to create headers_list artifacts
   * @param buildConfiguration the build configuration for this build
   * @param enableModules whether modules are enabled
   */
  public ObjcCppSemantics(
      IncludeProcessingType includeProcessingType,
      IncludeProcessing includeProcessing,
      ObjcConfiguration config,
      IntermediateArtifacts intermediateArtifacts,
      BuildConfiguration buildConfiguration,
      boolean enableModules) {
    this.includeProcessingType = includeProcessingType;
    this.includeProcessing = includeProcessing;
    this.config = config;
    this.intermediateArtifacts = intermediateArtifacts;
    this.buildConfiguration = buildConfiguration;
    this.enableModules = enableModules;
  }

  @Override
  public void finalizeCompileActionBuilder(
      BuildConfiguration configuration,
      FeatureConfiguration featureConfiguration,
      CppCompileActionBuilder actionBuilder) {
    actionBuilder
        // Without include scanning, we need the entire crosstool filegroup, including header files,
        // as opposed to just the "compile" filegroup.  Even with include scanning, we need the
        // system framework headers since they are not being scanned right now.
        // TODO(waltl): do better with include scanning.
        .addTransitiveMandatoryInputs(actionBuilder.getToolchain().getAllFilesMiddleman())
        .setShouldScanIncludes(includeProcessingType == INCLUDE_SCANNING);
  }

  @Override
  public HeadersCheckingMode determineHeadersCheckingMode(RuleContext ruleContext) {
    return HeadersCheckingMode.STRICT;
  }

  @Override
  public IncludeProcessing getIncludeProcessing() {
    return includeProcessing;
  }

  @Override
  public boolean needsDotdInputPruning() {
    return config.getDotdPruningPlan() == DotdPruningMode.USE;
  }

  @Override
  public void validateAttributes(RuleContext ruleContext) {
  }

  @Override
  public boolean needsIncludeValidation() {
    // We disable include valication when modules are enabled, because Apple uses absolute paths in
    // its module maps, which include validation does not recognize.  Modules should only be used
    // rarely and in third party code anyways.
    return (includeProcessingType == INCLUDE_SCANNING) && !enableModules;
  }

  /**
   * Gets the purpose for the {@code CcCompilationContext}.
   *
   * @see CcCompilationContext.Builder#setPurpose
   */
  public String getPurpose() {
    // ProtoSupport creates multiple {@code CcCompilationContext}s for a single rule,
    // potentially
    // multiple archives per build configuration. This covers that worst case.
    return "ObjcCppSemantics_build_arch_"
        + buildConfiguration.getMnemonic()
        + "_with_suffix_"
        + intermediateArtifacts.archiveFileNameSuffix();
  }

  /** cc_shared_library is not supported with Objective-C */
  @Override
  public StructImpl getCcSharedLibraryInfo(TransitiveInfoCollection dep) {
    return null;
  }

  @Override
  public void validateLayeringCheckFeatures(RuleContext ruleContext) {}
}
