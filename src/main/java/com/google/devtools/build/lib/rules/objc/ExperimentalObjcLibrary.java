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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;
import com.google.devtools.build.lib.rules.cpp.CcLibraryHelper;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.Variables.Builder;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.Variables.ValueSequence;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.Variables.VariablesExtension;
import com.google.devtools.build.lib.rules.cpp.CcToolchainProvider;
import com.google.devtools.build.lib.rules.cpp.CppRuleClasses;
import com.google.devtools.build.lib.rules.cpp.PrecompiledFiles;

import java.util.Collection;

/**
 * Implementation for experimental_objc_library.
 */
public class ExperimentalObjcLibrary implements RuleConfiguredTargetFactory {

  private static final String PCH_FILE_VARIABLE_NAME = "pch_file";
  private static final String FRAMEWORKS_VARIABLE_NAME = "framework_paths";
  private static final String MODULES_MAPS_DIR_NAME = "module_maps_dir";
  private static final String OBJC_MODULE_CACHE_DIR_NAME = "_objc_module_cache";
  private static final String OBJC_MODULE_CACHE_KEY = "modules_cache_path";
  private static final String OBJC_MODULE_FEATURE_NAME = "use_objc_modules";
  private static final Iterable<String> ACTIVATED_ACTIONS =
      ImmutableList.of("objc-compile", "objc++-compile");

  /**
   * Build variable extensions for templating a toolchain for objc builds.
   */
  private static class ObjcVariablesExtension implements VariablesExtension {

    private final RuleContext ruleContext;
    private final ObjcProvider objcProvider;

    public ObjcVariablesExtension(RuleContext ruleContext, ObjcProvider objcProvider) {
      this.ruleContext = ruleContext;
      this.objcProvider = objcProvider;
    }

    @Override
    public void addVariables(Builder builder) {
      addPchVariables(builder);
      addFrameworkVariables(builder);
      if (ObjcCommon.shouldUseObjcModules(ruleContext)) {
        addModuleMapVariables(builder);
      }
    }

    private void addPchVariables(Builder builder) {
       if (ruleContext.getPrerequisiteArtifact("pch", Mode.TARGET) != null) {
        builder.addVariable(
            PCH_FILE_VARIABLE_NAME,
            ruleContext.getPrerequisiteArtifact("pch", Mode.TARGET).getExecPathString());
      }
    }

    private void addFrameworkVariables(Builder builder) {
      ValueSequence.Builder frameworkSequence = new ValueSequence.Builder();
      AppleConfiguration appleConfig = ruleContext.getFragment(AppleConfiguration.class);
      for (String framework :
          CompilationSupport.commonFrameworkNames(objcProvider, appleConfig)) {
        frameworkSequence.addValue(framework);
      }
      builder.addSequence(FRAMEWORKS_VARIABLE_NAME, frameworkSequence.build());
    }

    private void addModuleMapVariables(Builder builder) {
      builder.addVariable(
          MODULES_MAPS_DIR_NAME,
          ObjcRuleClasses.intermediateArtifacts(ruleContext)
              .moduleMap()
              .getArtifact()
              .getExecPath()
              .getParentDirectory()
              .toString());
      builder.addVariable(
          OBJC_MODULE_CACHE_KEY,
          ruleContext.getConfiguration().getGenfilesFragment() + "/" + OBJC_MODULE_CACHE_DIR_NAME);
    }
  }

  @Override
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException {
    validateAttributes(ruleContext);

    CompilationArtifacts compilationArtifacts =
        CompilationSupport.compilationArtifacts(ruleContext);
    CompilationAttributes compilationAttributes =
        CompilationAttributes.Builder.fromRuleContext(ruleContext).build();
    PrecompiledFiles precompiledFiles = new PrecompiledFiles(ruleContext);

    ObjcCommon common = common(ruleContext, compilationAttributes, compilationArtifacts);

    Collection<Artifact> arcSources = Sets.newHashSet(compilationArtifacts.getSrcs());
    Collection<Artifact> nonArcSources = Sets.newHashSet(compilationArtifacts.getNonArcSrcs());
    Collection<Artifact> privateHdrs = Sets.newHashSet(compilationArtifacts.getPrivateHdrs());
    Collection<Artifact> publicHdrs = Sets.newHashSet(compilationAttributes.hdrs());

    CcLibraryHelper helper =
        new CcLibraryHelper(
                ruleContext,
                new ObjcCppSemantics(common.getObjcProvider()),
                getFeatureConfiguration(ruleContext),
                CcLibraryHelper.SourceCategory.CC_AND_OBJC)
            .addSources(arcSources, ImmutableMap.of("objc_arc", ""))
            .addSources(nonArcSources, ImmutableMap.of("no_objc_arc", ""))
            .addSources(privateHdrs)
            .enableCompileProviders()
            .addPublicHeaders(publicHdrs)
            .addPrecompiledFiles(precompiledFiles)
            .addDeps(ruleContext.getPrerequisites("deps", Mode.TARGET))
            .addVariableExtension(
                new ObjcVariablesExtension(ruleContext, common.getObjcProvider()));

    if (ObjcCommon.shouldUseObjcModules(ruleContext)) {
      helper.setCppModuleMap(ObjcRuleClasses.intermediateArtifacts(ruleContext).moduleMap());
    }

    CcLibraryHelper.Info info = helper.build();

    NestedSetBuilder<Artifact> filesToBuild =
        NestedSetBuilder.<Artifact>stableOrder().addAll(common.getCompiledArchive().asSet());

    return ObjcRuleClasses.ruleConfiguredTarget(ruleContext, filesToBuild.build())
        .addProviders(info.getProviders())
        .build();
  }

  private FeatureConfiguration getFeatureConfiguration(RuleContext ruleContext) {
    CcToolchainProvider toolchain =
        ruleContext
            .getPrerequisite(":cc_toolchain", Mode.TARGET)
            .getProvider(CcToolchainProvider.class);

    ImmutableList.Builder<String> activatedCrosstoolSelectables =
        ImmutableList.<String>builder().addAll(ACTIVATED_ACTIONS);

    if (ruleContext.getPrerequisiteArtifact("pch", Mode.TARGET) != null) {
      activatedCrosstoolSelectables.add("pch");
    }

    if (ObjcCommon.shouldUseObjcModules(ruleContext)) {
      activatedCrosstoolSelectables.add(OBJC_MODULE_FEATURE_NAME);
    }

    activatedCrosstoolSelectables.addAll(
        ruleContext.getFragment(AppleConfiguration.class).getBitcodeMode().getFeatureNames());

    // We create a module map by default to allow for swift interop.
    activatedCrosstoolSelectables.add(CppRuleClasses.MODULE_MAPS);
    activatedCrosstoolSelectables.add(CppRuleClasses.COMPILE_ACTION_FLAGS_IN_FLAG_SET);
    activatedCrosstoolSelectables.add(CppRuleClasses.DEPENDENCY_FILE);

    return toolchain.getFeatures().getFeatureConfiguration(activatedCrosstoolSelectables.build());
  }

  /**
   * Throws errors or warnings for bad attribute state.
   */
  private void validateAttributes(RuleContext ruleContext) {
    for (String copt : ObjcCommon.getNonCrosstoolCopts(ruleContext)) {
      if (copt.contains("-fmodules-cache-path")) {
        ruleContext.ruleWarning(CompilationSupport.MODULES_CACHE_PATH_WARNING);
      }
    }
  }

  private static ObjcCommon common(
      RuleContext ruleContext,
      CompilationAttributes compilationAttributes,
      CompilationArtifacts compilationArtifacts) {
    return new ObjcCommon.Builder(ruleContext)
        .setCompilationAttributes(compilationAttributes)
        .setCompilationArtifacts(compilationArtifacts)
        .addDepObjcProviders(ruleContext.getPrerequisites("deps", Mode.TARGET, ObjcProvider.class))
        .build();
  }
}
