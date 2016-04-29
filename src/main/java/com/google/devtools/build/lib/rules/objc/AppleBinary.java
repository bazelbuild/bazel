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
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.LIBRARY;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableListMultimap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;
import com.google.devtools.build.lib.rules.cpp.CcToolchainProvider;
import com.google.devtools.build.lib.rules.cpp.CppCompilationContext;
import com.google.devtools.build.lib.rules.objc.CompilationSupport.ExtraLinkArgs;
import com.google.devtools.build.lib.rules.objc.ObjcCommon.CompilationAttributes;
import com.google.devtools.build.lib.rules.objc.ObjcCommon.ResourceAttributes;

import java.util.List;
import java.util.Set;

/**
 * Implementation for the "apple_binary" rule.
 */
public class AppleBinary implements RuleConfiguredTargetFactory {
  @VisibleForTesting
  static final String REQUIRES_AT_LEAST_ONE_LIBRARY_OR_SOURCE_FILE =
      "At least one library dependency or source file is required.";

  @Override
  public final ConfiguredTarget create(RuleContext ruleContext) throws InterruptedException {
    ImmutableListMultimap<BuildConfiguration, ObjcProvider> configurationToNonPropagatedObjcMap =
        ruleContext.getPrerequisitesByConfiguration("non_propagated_deps", Mode.SPLIT,
            ObjcProvider.class);
    ImmutableListMultimap<BuildConfiguration, TransitiveInfoCollection> configToDepsCollectionMap =
        ruleContext.getPrerequisitesByConfiguration("deps", Mode.SPLIT);
    
    Set<BuildConfiguration> childConfigurations = getChildConfigurations(ruleContext);
    
    IntermediateArtifacts ruleIntermediateArtifacts =
        ObjcRuleClasses.intermediateArtifacts(ruleContext);

    NestedSetBuilder<Artifact> binariesToLipo =
        NestedSetBuilder.<Artifact>stableOrder();
    NestedSetBuilder<Artifact> filesToBuild =
        NestedSetBuilder.<Artifact>stableOrder()
            .add(ruleIntermediateArtifacts.combinedArchitectureBinary());
    
    for (BuildConfiguration childConfig : childConfigurations) {
      IntermediateArtifacts intermediateArtifacts =
          ObjcRuleClasses.intermediateArtifacts(ruleContext, childConfig);
      ObjcCommon common = common(ruleContext, childConfig, intermediateArtifacts,
          nullToEmptyList(configToDepsCollectionMap.get(childConfig)),
          nullToEmptyList(configurationToNonPropagatedObjcMap.get(childConfig)));
      ObjcProvider objcProvider = common.getObjcProvider();
      if (!hasLibraryOrSources(objcProvider)) {
        ruleContext.ruleError(REQUIRES_AT_LEAST_ONE_LIBRARY_OR_SOURCE_FILE);
        return null;
      }
      if (ruleContext.hasErrors()) {
        return null;
      }
      binariesToLipo.add(intermediateArtifacts.strippedSingleArchitectureBinary());
      new CompilationSupport(ruleContext, childConfig)
          .registerCompileAndArchiveActions(common)
          .registerLinkActions(
              objcProvider, new ExtraLinkArgs(), ImmutableList.<Artifact>of(),
              DsymOutputType.APP)
          .validateAttributes();

      if (ruleContext.hasErrors()) {
        return null;
      }
    }

    AppleConfiguration appleConfiguration = ruleContext.getFragment(AppleConfiguration.class);

    new LipoSupport(ruleContext).registerCombineArchitecturesAction(
        binariesToLipo.build(),
        ruleIntermediateArtifacts.combinedArchitectureBinary(),
        appleConfiguration.getIosCpuPlatform());

    RuleConfiguredTargetBuilder targetBuilder =
        ObjcRuleClasses.ruleConfiguredTarget(ruleContext, filesToBuild.build());

    return targetBuilder.build();
  }

  private boolean hasLibraryOrSources(ObjcProvider objcProvider) {
    return !Iterables.isEmpty(objcProvider.get(LIBRARY)) // Includes sources from this target.
        || !Iterables.isEmpty(objcProvider.get(IMPORTED_LIBRARY));
  }

  private ObjcCommon common(RuleContext ruleContext, BuildConfiguration buildConfiguration,
      IntermediateArtifacts intermediateArtifacts,
      List<TransitiveInfoCollection> propagatedDeps,
      List<ObjcProvider> nonPropagatedObjcDeps) {
    ImmutableList.Builder<ObjcProvider> propagatedObjcDeps = ImmutableList.<ObjcProvider>builder();
    ImmutableList.Builder<CppCompilationContext> cppDeps =
        ImmutableList.<CppCompilationContext>builder();
    
    for (TransitiveInfoCollection dep : propagatedDeps) {
      if (dep.getProvider(ObjcProvider.class) != null) {
        propagatedObjcDeps.add(dep.getProvider(ObjcProvider.class));
      }
      if (dep.getProvider(CppCompilationContext.class) != null) {
        cppDeps.add(dep.getProvider(CppCompilationContext.class));
      }
    }

    CompilationArtifacts compilationArtifacts =
        CompilationSupport.compilationArtifacts(ruleContext, intermediateArtifacts);

    return new ObjcCommon.Builder(ruleContext, buildConfiguration)
        .setCompilationAttributes(new CompilationAttributes(ruleContext))
        .setResourceAttributes(new ResourceAttributes(ruleContext))
        .setCompilationArtifacts(compilationArtifacts)
        .addDefines(ruleContext.getTokenizedStringListAttr("defines"))
        .addDepObjcProviders(propagatedObjcDeps.build())
        .addDepCcHeaderProviders(cppDeps.build())
        .addDepCcLinkProviders(propagatedDeps)
        .addDepObjcProviders(
            ruleContext.getPrerequisites("bundles", Mode.TARGET, ObjcProvider.class))
        .addNonPropagatedDepObjcProviders(nonPropagatedObjcDeps)
        .setIntermediateArtifacts(intermediateArtifacts)
        .setAlwayslink(false)
        .setHasModuleMap()
        .setLinkedBinary(intermediateArtifacts.strippedSingleArchitectureBinary())
        .build();
  }
  
  private <T> List<T> nullToEmptyList(List<T> inputList) {
    return inputList != null ? inputList : ImmutableList.<T>of();
  }
  
  private Set<BuildConfiguration> getChildConfigurations(RuleContext ruleContext) {
    // This is currently a hack to obtain all child configurations regardless of the attribute
    // values of this rule -- this rule does not currently use the actual info provided by
    // this attribute. b/28403953 tracks cc toolchain usage.
    ImmutableListMultimap<BuildConfiguration, CcToolchainProvider> configToProvider =
        ruleContext.getPrerequisitesByConfiguration(":cc_toolchain", Mode.SPLIT,
            CcToolchainProvider.class);
    
    return configToProvider.keySet();
  }
}