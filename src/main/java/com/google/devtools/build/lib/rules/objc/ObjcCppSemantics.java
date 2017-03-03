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

import static com.google.devtools.build.lib.rules.objc.ObjcProvider.DYNAMIC_FRAMEWORK_FILE;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.HEADER;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.STATIC_FRAMEWORK_FILE;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.rules.cpp.CppCompilationContext.Builder;
import com.google.devtools.build.lib.rules.cpp.CppCompileActionBuilder;
import com.google.devtools.build.lib.rules.cpp.CppCompileActionContext;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration.HeadersCheckingMode;
import com.google.devtools.build.lib.rules.cpp.CppFileTypes;
import com.google.devtools.build.lib.rules.cpp.CppSemantics;
import com.google.devtools.build.lib.rules.cpp.HeaderDiscovery.DotdPruningMode;
import com.google.devtools.build.lib.rules.cpp.IncludeProcessing;
import com.google.devtools.build.lib.util.FileTypeSet;
import com.google.devtools.build.lib.vfs.PathFragment;

/**
 * CppSemantics for objc builds.
 */
public class ObjcCppSemantics implements CppSemantics {

  private final IncludeProcessing includeProcessing;
  private final ObjcProvider objcProvider;
  private final ObjcConfiguration config;
  private final boolean isHeaderThinningEnabled;
  private final IntermediateArtifacts intermediateArtifacts;

  /**
   * Set of {@link com.google.devtools.build.lib.util.FileType} of source artifacts that are
   * compatible with header thinning.
   */
  private static final FileTypeSet SOURCES_FOR_HEADER_THINNING =
      FileTypeSet.of(
          CppFileTypes.OBJC_SOURCE,
          CppFileTypes.OBJCPP_SOURCE,
          CppFileTypes.CPP_SOURCE,
          CppFileTypes.C_SOURCE);

  /**
   * Creates an instance of ObjcCppSemantics
   *
   * @param objcProvider the provider that should be used in determining objc-specific inputs to
   *     actions
   * @param includeProcessing the closure providing the strategy for processing of includes for
   *     actions
   * @param config the ObjcConfiguration for this build
   * @param isHeaderThinningEnabled true if headers_list artifacts should be generated and added as
   *     input to compiling actions
   * @param intermediateArtifacts used to create headers_list artifacts
   */
  public ObjcCppSemantics(
      ObjcProvider objcProvider,
      IncludeProcessing includeProcessing,
      ObjcConfiguration config,
      boolean isHeaderThinningEnabled,
      IntermediateArtifacts intermediateArtifacts) {
    this.objcProvider = objcProvider;
    this.includeProcessing = includeProcessing;
    this.config = config;
    this.isHeaderThinningEnabled = isHeaderThinningEnabled;
    this.intermediateArtifacts = intermediateArtifacts;
  }

  @Override
  public PathFragment getEffectiveSourcePath(Artifact source) {
    return source.getRootRelativePath();
  }

  @Override
  public void finalizeCompileActionBuilder(
      RuleContext ruleContext, CppCompileActionBuilder actionBuilder) {
    actionBuilder.setCppConfiguration(ruleContext.getFragment(CppConfiguration.class));
    actionBuilder.setActionContext(CppCompileActionContext.class);
    // Because Bazel does not support include scanning, we need the entire crosstool filegroup,
    // including header files, as opposed to just the "compile" filegroup.
    actionBuilder.addTransitiveMandatoryInputs(actionBuilder.getToolchain().getCrosstool());
    actionBuilder.setShouldScanIncludes(false);

    actionBuilder.addTransitiveMandatoryInputs(objcProvider.get(STATIC_FRAMEWORK_FILE));
    actionBuilder.addTransitiveMandatoryInputs(objcProvider.get(DYNAMIC_FRAMEWORK_FILE));

    if (isHeaderThinningEnabled) {
      Artifact sourceFile = actionBuilder.getSourceFile();
      if (!sourceFile.isTreeArtifact()
          && SOURCES_FOR_HEADER_THINNING.matches(sourceFile.getFilename())) {
        actionBuilder.addMandatoryInputs(
            ImmutableList.of(intermediateArtifacts.headersListFile(sourceFile)));
      }
    }
  }

  @Override
  public void setupCompilationContext(RuleContext ruleContext, Builder contextBuilder) {
    // For objc builds, no extra setup is required.
  }

  @Override
  public NestedSet<Artifact> getAdditionalPrunableIncludes() {
    return objcProvider.get(HEADER);
  }

  @Override
  public HeadersCheckingMode determineHeadersCheckingMode(RuleContext ruleContext) {
    // Currently, objc builds do not enforce strict deps.  To begin enforcing strict deps in objc,
    // switch this flag to STRICT.
    return HeadersCheckingMode.WARN;
  }

  @Override
  public IncludeProcessing getIncludeProcessing() {
    return includeProcessing;
  }

  @Override
  public boolean needsIncludeScanning(RuleContext ruleContext) {
    return false;
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
    return false;
  }
}
