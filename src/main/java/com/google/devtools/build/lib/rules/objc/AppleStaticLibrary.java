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
import com.google.common.collect.ImmutableListMultimap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.OutputGroupInfo;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;
import com.google.devtools.build.lib.rules.apple.ApplePlatform;
import com.google.devtools.build.lib.rules.apple.ApplePlatform.PlatformType;
import com.google.devtools.build.lib.rules.cpp.CcInfo;
import com.google.devtools.build.lib.rules.cpp.CcToolchainProvider;
import com.google.devtools.build.lib.rules.cpp.CppSemantics;
import com.google.devtools.build.lib.rules.objc.ObjcProvider.Key;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/**
 * Implementation for the "apple_static_library" rule.
 */
public class AppleStaticLibrary implements RuleConfiguredTargetFactory {
  /**
   * Attribute name for dependent libraries which should not be linked into the outputs of this
   * rule.
   */
  public static final String AVOID_DEPS_ATTR_NAME = "avoid_deps";

  private final CppSemantics cppSemantics;

  protected AppleStaticLibrary(CppSemantics cppSemantics) {
    this.cppSemantics = cppSemantics;
  }

  /**
   * Set of {@link ObjcProvider} values which are propagated from dependencies to dependers by this
   * rule.
   */
  private static final ImmutableSet<Key<?>> PROPAGATE_KEYS =
      ImmutableSet.<Key<?>>of(
          ObjcProvider.SDK_DYLIB, ObjcProvider.SDK_FRAMEWORK, ObjcProvider.WEAK_SDK_FRAMEWORK);

  @Override
  public final ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException, ActionConflictException {
    MultiArchSplitTransitionProvider.validateMinimumOs(ruleContext);
    PlatformType platformType = MultiArchSplitTransitionProvider.getPlatformType(ruleContext);

    ImmutableListMultimap<String, TransitiveInfoCollection> cpuToDepsCollectionMap =
        MultiArchBinarySupport.transformMap(ruleContext.getPrerequisitesByConfiguration("deps"));

    ImmutableListMultimap<String, ObjcProvider> cpuToObjcAvoidDepsMap =
        MultiArchBinarySupport.transformMap(
            ruleContext.getPrerequisitesByConfiguration(
                AVOID_DEPS_ATTR_NAME, ObjcProvider.STARLARK_CONSTRUCTOR));

    ImmutableListMultimap<String, CcInfo> cpuToCcAvoidDepsMap =
        MultiArchBinarySupport.transformMap(
            ruleContext.getPrerequisitesByConfiguration(AVOID_DEPS_ATTR_NAME, CcInfo.PROVIDER));

    Map<BuildConfiguration, CcToolchainProvider> childConfigurationsAndToolchains =
        MultiArchBinarySupport.getChildConfigurationsAndToolchains(ruleContext);
    IntermediateArtifacts ruleIntermediateArtifacts =
        ObjcRuleClasses.intermediateArtifacts(ruleContext);

    NestedSetBuilder<Artifact> librariesToLipo =
        NestedSetBuilder.<Artifact>stableOrder();
    NestedSetBuilder<Artifact> filesToBuild =
        NestedSetBuilder.<Artifact>stableOrder()
            .add(ruleIntermediateArtifacts.combinedArchitectureArchive());

    ObjcProvider.Builder objcProviderBuilder =
        new ObjcProvider.Builder(ruleContext.getAnalysisEnvironment().getStarlarkSemantics());

    Map<String, NestedSet<Artifact>> outputGroupCollector = new TreeMap<>();
    for (Map.Entry<BuildConfiguration, CcToolchainProvider> entry :
        childConfigurationsAndToolchains.entrySet()) {
      BuildConfiguration childToolchainConfig = entry.getKey();
      String childCpu = entry.getKey().getCpu();
      CcToolchainProvider childToolchain = entry.getValue();

      IntermediateArtifacts intermediateArtifacts =
          ObjcRuleClasses.intermediateArtifacts(ruleContext, childToolchainConfig);

      ObjcCommon common =
          common(
              ruleContext,
              childToolchainConfig,
              intermediateArtifacts,
              nullToEmptyList(cpuToDepsCollectionMap.get(childCpu)));
      ObjcProvider objcProvider =
          common
              .getObjcProvider()
              .subtractSubtrees(
                  cpuToObjcAvoidDepsMap.get(childCpu),
                  cpuToCcAvoidDepsMap.get(childCpu).stream()
                      .map(CcInfo::getCcLinkingContext)
                      .collect(ImmutableList.toImmutableList()));

      librariesToLipo.add(intermediateArtifacts.strippedSingleArchitectureLibrary());

      CompilationSupport compilationSupport =
          new CompilationSupport.Builder(ruleContext, cppSemantics)
              .setConfig(childToolchainConfig)
              .setToolchainProvider(childToolchain)
              .setOutputGroupCollector(outputGroupCollector)
              .build();

      compilationSupport
          .registerCompileAndArchiveActions(
              common.getCompilationArtifacts().get(), ObjcCompilationContext.EMPTY)
          .registerFullyLinkAction(
              objcProvider, intermediateArtifacts.strippedSingleArchitectureLibrary())
          .validateAttributes();
      ruleContext.assertNoErrors();

      addTransitivePropagatedKeys(objcProviderBuilder, objcProvider);
    }

    ImmutableListMultimap<BuildConfiguration, CcInfo> buildConfigToCcInfoMap =
        ruleContext.getPrerequisitesByConfiguration("deps", CcInfo.PROVIDER);
    NestedSetBuilder<Artifact> headerTokens = NestedSetBuilder.stableOrder();
    for (Map.Entry<BuildConfiguration, CcInfo> entry : buildConfigToCcInfoMap.entries()) {
      CcInfo dep = entry.getValue();
      headerTokens.addTransitive(dep.getCcCompilationContext().getHeaderTokens());
    }
    outputGroupCollector.put(OutputGroupInfo.VALIDATION, headerTokens.build());

    AppleConfiguration appleConfiguration = ruleContext.getFragment(AppleConfiguration.class);
    ApplePlatform platform = null;
    try {
      platform = appleConfiguration.getMultiArchPlatform(platformType);
    } catch (IllegalArgumentException e) {
      ruleContext.throwWithRuleError(e);
    }
    new LipoSupport(ruleContext)
        .registerCombineArchitecturesAction(
            librariesToLipo.build(),
            ruleIntermediateArtifacts.combinedArchitectureArchive(),
            platform);

    RuleConfiguredTargetBuilder targetBuilder =
        ObjcRuleClasses.ruleConfiguredTarget(ruleContext, filesToBuild.build());

    ObjcProvider objcProvider = objcProviderBuilder.build();

    if (appleConfiguration.shouldLinkingRulesPropagateObjc()) {
      targetBuilder.addNativeDeclaredProvider(objcProvider);
      targetBuilder.addStarlarkTransitiveInfo(ObjcProvider.STARLARK_NAME, objcProvider);
    }

    targetBuilder
        .addNativeDeclaredProvider(
            new AppleStaticLibraryInfo(
                ruleIntermediateArtifacts.combinedArchitectureArchive(),
                objcProvider))
        .addOutputGroups(outputGroupCollector);
    return targetBuilder.build();
  }

  private void addTransitivePropagatedKeys(ObjcProvider.Builder objcProviderBuilder,
      ObjcProvider provider) {
    for (Key<?> key : PROPAGATE_KEYS) {
      objcProviderBuilder.addTransitiveAndPropagate(key, provider);
    }
  }

  private ObjcCommon common(
      RuleContext ruleContext,
      BuildConfiguration buildConfiguration,
      IntermediateArtifacts intermediateArtifacts,
      List<TransitiveInfoCollection> propagatedDeps)
      throws InterruptedException {

    CompilationArtifacts compilationArtifacts = new CompilationArtifacts.Builder().build();

    return new ObjcCommon.Builder(ObjcCommon.Purpose.LINK_ONLY, ruleContext, buildConfiguration)
        .setCompilationAttributes(
            CompilationAttributes.Builder.fromRuleContext(ruleContext).build())
        .setCompilationArtifacts(compilationArtifacts)
        .addDeps(propagatedDeps)
        .setIntermediateArtifacts(intermediateArtifacts)
        .setAlwayslink(false)
        .build();
  }

  private <T> List<T> nullToEmptyList(List<T> inputList) {
    return inputList != null ? inputList : ImmutableList.<T>of();
  }
}
