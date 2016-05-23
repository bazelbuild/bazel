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
import com.google.devtools.build.lib.rules.cpp.PrecompiledFiles;
import com.google.devtools.build.lib.rules.objc.ObjcCommon.CompilationAttributes;

import java.util.Collection;

/**
 * Implementation for experimental_objc_library.
 */
public class ExperimentalObjcLibrary implements RuleConfiguredTargetFactory {

  private static final String PCH_FILE_VARIABLE_NAME = "pch_file";
  private static final String FRAMEWORKS_VARIABLE_NAME = "framework_paths";
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
      for (String framework :
          CompilationSupport.commonFrameworkNames(
              objcProvider, ruleContext.getFragment(AppleConfiguration.class))) {
        frameworkSequence.addValue(framework);
      }
      builder.addSequence(FRAMEWORKS_VARIABLE_NAME, frameworkSequence.build());      
    }
  }

  @Override
  public ConfiguredTarget create(RuleContext ruleContext) throws InterruptedException {

    CompilationArtifacts compilationArtifacts =
        CompilationSupport.compilationArtifacts(ruleContext);
    CompilationAttributes compilationAttributes = new CompilationAttributes(ruleContext);
    PrecompiledFiles precompiledFiles = new PrecompiledFiles(ruleContext);

    ObjcCommon common = common(ruleContext, compilationAttributes, compilationArtifacts);

    CcToolchainProvider toolchain =
        ruleContext
            .getPrerequisite(":cc_toolchain", Mode.TARGET)
            .getProvider(CcToolchainProvider.class);

    ImmutableList.Builder<String> activatedCrosstoolSelectables =
        ImmutableList.<String>builder().addAll(ACTIVATED_ACTIONS);
    if (ruleContext.getPrerequisiteArtifact("pch", Mode.TARGET) != null) {
      activatedCrosstoolSelectables.add("pch");
    }

    FeatureConfiguration featureConfiguration =
        toolchain.getFeatures().getFeatureConfiguration(activatedCrosstoolSelectables.build());

    Collection<Artifact> sources = Sets.newHashSet(compilationArtifacts.getSrcs());
    Collection<Artifact> privateHdrs = Sets.newHashSet(compilationArtifacts.getPrivateHdrs());
    Collection<Artifact> publicHdrs = Sets.newHashSet(compilationAttributes.hdrs());

    CcLibraryHelper helper =
        new CcLibraryHelper(
                ruleContext,
                new ObjcCppSemantics(common.getObjcProvider()),
                featureConfiguration,
                CcLibraryHelper.SourceCategory.CC_AND_OBJC)
            .addSources(sources)
            .addSources(privateHdrs)
            .enableCompileProviders()
            .addPublicHeaders(publicHdrs)
            .addPrecompiledFiles(precompiledFiles)
            .addDeps(ruleContext.getPrerequisites("deps", Mode.TARGET))
            .addVariableExtension(
                new ObjcVariablesExtension(ruleContext, common.getObjcProvider()));

    CcLibraryHelper.Info info = helper.build();

    NestedSetBuilder<Artifact> filesToBuild =
        NestedSetBuilder.<Artifact>stableOrder().addAll(common.getCompiledArchive().asSet());

    return ObjcRuleClasses.ruleConfiguredTarget(ruleContext, filesToBuild.build())
        .addProviders(info.getProviders())
        .build();
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
