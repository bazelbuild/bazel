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

import static com.google.devtools.build.lib.rules.objc.ObjcProvider.IMPORTED_LIBRARY;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;
import com.google.devtools.build.lib.rules.apple.Platform;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.Variables.ValueSequence;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.Variables.VariablesExtension;

/** Build variable extensions for templating a toolchain for objc builds. */
class ObjcVariablesExtension implements VariablesExtension {

  static final String PCH_FILE_VARIABLE_NAME = "pch_file";
  static final String FRAMEWORKS_VARIABLE_NAME = "framework_paths";
  static final String VERSION_MIN_VARIABLE_NAME = "version_min";
  static final String MODULES_MAPS_DIR_NAME = "module_maps_dir";
  static final String OBJC_MODULE_CACHE_DIR_NAME = "_objc_module_cache";
  static final String OBJC_MODULE_CACHE_KEY = "modules_cache_path";
  static final String OBJ_LIST_PATH_VARIABLE_NAME = "obj_list_path";
  static final String ARCHIVE_PATH_VARIABLE_NAME = "archive_path";
  static final String FULLY_LINKED_ARCHIVE_PATH_VARIABLE_NAME = "fully_linked_archive_path";
  static final String OBJC_LIBRARY_EXEC_PATHS_VARIABLE_NAME = "objc_library_exec_paths";
  static final String CC_LIBRARY_EXEC_PATHS_VARIABLE_NAME = "cc_library_exec_paths";
  static final String IMPORTED_LIBRARY_EXEC_PATHS_VARIABLE_NAME = "imported_library_exec_paths";

  private final RuleContext ruleContext;
  private final ObjcProvider objcProvider;
  private final CompilationArtifacts compilationArtifacts;
  private final Artifact fullyLinkArchive;
  private final IntermediateArtifacts intermediateArtifacts;

  private final BuildConfiguration buildConfiguration;
  private final AppleConfiguration appleConfiguration;
  private final ObjcConfiguration objcConfiguration;

  public ObjcVariablesExtension(
      RuleContext ruleContext,
      ObjcProvider objcProvider,
      CompilationArtifacts compilationArtifacts,
      Artifact fullyLinkArchive,
      IntermediateArtifacts intermediateArtifacts,
      BuildConfiguration buildConfiguration) {
    this.ruleContext = ruleContext;
    this.objcProvider = objcProvider;
    this.compilationArtifacts = compilationArtifacts;
    this.fullyLinkArchive = fullyLinkArchive;
    this.intermediateArtifacts = intermediateArtifacts;
    this.buildConfiguration = buildConfiguration;
    this.objcConfiguration = buildConfiguration.getFragment(ObjcConfiguration.class);
    this.appleConfiguration = buildConfiguration.getFragment(AppleConfiguration.class);
  }

  @Override
  public void addVariables(CcToolchainFeatures.Variables.Builder builder) {
    addPchVariables(builder);
    addFrameworkVariables(builder);
    addArchVariables(builder);
    if (ObjcCommon.shouldUseObjcModules(ruleContext)) {
      addModuleMapVariables(builder);
    }
    if (compilationArtifacts.getArchive().isPresent()) {
      addArchiveVariables(builder);
    }
    addFullyLinkArchiveVariables(builder);
  }

  private void addPchVariables(CcToolchainFeatures.Variables.Builder builder) {
    if (ruleContext.getPrerequisiteArtifact("pch", Mode.TARGET) != null) {
      builder.addVariable(
          PCH_FILE_VARIABLE_NAME,
          ruleContext.getPrerequisiteArtifact("pch", Mode.TARGET).getExecPathString());
    }
  }

  private void addFrameworkVariables(CcToolchainFeatures.Variables.Builder builder) {
    ValueSequence.Builder frameworkSequence = new ValueSequence.Builder();
    for (String framework :
        CompilationSupport.commonFrameworkNames(objcProvider, appleConfiguration)) {
      frameworkSequence.addValue(framework);
    }
    builder.addSequence(FRAMEWORKS_VARIABLE_NAME, frameworkSequence.build());
  }

  private void addModuleMapVariables(CcToolchainFeatures.Variables.Builder builder) {
    builder.addVariable(
        MODULES_MAPS_DIR_NAME,
        intermediateArtifacts
            .moduleMap()
            .getArtifact()
            .getExecPath()
            .getParentDirectory()
            .toString());
    builder.addVariable(
        OBJC_MODULE_CACHE_KEY,
        buildConfiguration.getGenfilesFragment() + "/" + OBJC_MODULE_CACHE_DIR_NAME);
  }

  private void addArchVariables(CcToolchainFeatures.Variables.Builder builder) {
    Platform platform = appleConfiguration.getSingleArchPlatform();
    switch (platform.getType()) {
      case IOS:
        builder.addVariable(VERSION_MIN_VARIABLE_NAME, objcConfiguration.getMinimumOs().toString());
        break;
      case WATCHOS:
        builder.addVariable(
            VERSION_MIN_VARIABLE_NAME,
            appleConfiguration.getSdkVersionForPlatform(platform).toString());
        break;
      default:
        throw new IllegalArgumentException("Unhandled platform: " + platform);
    }
  }

  private void addArchiveVariables(CcToolchainFeatures.Variables.Builder builder) {
    builder.addVariable(
        OBJ_LIST_PATH_VARIABLE_NAME,
        ObjcRuleClasses.intermediateArtifacts(ruleContext).archiveObjList().getExecPathString());
    builder.addVariable(
        ARCHIVE_PATH_VARIABLE_NAME, compilationArtifacts.getArchive().get().getExecPathString());
  }

  private void addFullyLinkArchiveVariables(CcToolchainFeatures.Variables.Builder builder) {
    builder.addVariable(
        FULLY_LINKED_ARCHIVE_PATH_VARIABLE_NAME, fullyLinkArchive.getExecPathString());
    builder.addSequenceVariable(
        OBJC_LIBRARY_EXEC_PATHS_VARIABLE_NAME,
        ImmutableList.copyOf(Artifact.toExecPaths(objcProvider.getObjcLibraries())));
    builder.addSequenceVariable(
        CC_LIBRARY_EXEC_PATHS_VARIABLE_NAME,
        ImmutableList.copyOf(Artifact.toExecPaths(objcProvider.getCcLibraries())));
    builder.addSequenceVariable(
        IMPORTED_LIBRARY_EXEC_PATHS_VARIABLE_NAME,
        ImmutableList.copyOf(Artifact.toExecPaths(objcProvider.get(IMPORTED_LIBRARY))));
  }
}
