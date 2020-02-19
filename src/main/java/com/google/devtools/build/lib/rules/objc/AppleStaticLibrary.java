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

import static com.google.devtools.build.lib.rules.objc.ObjcProvider.MULTI_ARCH_LINKED_ARCHIVES;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Optional;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableListMultimap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Multimap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;
import com.google.devtools.build.lib.rules.apple.ApplePlatform.PlatformType;
import com.google.devtools.build.lib.rules.cpp.CcInfo;
import com.google.devtools.build.lib.rules.cpp.CcToolchainProvider;
import com.google.devtools.build.lib.rules.objc.ObjcProvider.Key;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetAndData;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/**
 * Implementation for the "apple_static_library" rule.
 */
public class AppleStaticLibrary implements RuleConfiguredTargetFactory {

  /**
   * Set of {@link ObjcProvider} values which are propagated from dependencies to dependers by this
   * rule.
   */
  private static final ImmutableSet<Key<?>> PROPAGATE_KEYS =
      ImmutableSet.<Key<?>>of(
          ObjcProvider.SDK_DYLIB, ObjcProvider.SDK_FRAMEWORK, ObjcProvider.WEAK_SDK_FRAMEWORK);

  @VisibleForTesting
  static final String UNSUPPORTED_PLATFORM_TYPE_ERROR_FORMAT =
      "Unsupported platform type \"%s\"";

  @Override
  public final ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException, ActionConflictException {
    MultiArchSplitTransitionProvider.validateMinimumOs(ruleContext);
    PlatformType platformType = MultiArchSplitTransitionProvider.getPlatformType(ruleContext);

    ImmutableListMultimap<String, ConfiguredTargetAndData> cpuToCTATDepsCollectionMap =
        MultiArchBinarySupport.transformMap(
            ruleContext.getPrerequisiteCofiguredTargetAndTargetsByConfiguration(
                "deps", Mode.SPLIT));

    ImmutableListMultimap<String, ObjcProvider> cpuToObjcAvoidDepsMap =
        MultiArchBinarySupport.transformMap(
            ruleContext.getPrerequisitesByConfiguration(
                AppleStaticLibraryRule.AVOID_DEPS_ATTR_NAME,
                Mode.SPLIT,
                ObjcProvider.SKYLARK_CONSTRUCTOR));

    ImmutableListMultimap<String, CcInfo> cpuToCcAvoidDepsMap =
        MultiArchBinarySupport.transformMap(
            ruleContext.getPrerequisitesByConfiguration(
                AppleStaticLibraryRule.AVOID_DEPS_ATTR_NAME, Mode.SPLIT, CcInfo.PROVIDER));

    Iterable<ObjcProtoProvider> avoidProtoProviders =
        ruleContext.getPrerequisites(AppleStaticLibraryRule.AVOID_DEPS_ATTR_NAME, Mode.TARGET,
            ObjcProtoProvider.SKYLARK_CONSTRUCTOR);
    NestedSet<Artifact> protosToAvoid = protoArtifactsToAvoid(avoidProtoProviders);

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
        new ObjcProvider.Builder(ruleContext.getAnalysisEnvironment().getSkylarkSemantics());

    ImmutableListMultimap<BuildConfiguration, ObjcProtoProvider> objcProtoProvidersByConfig =
        ruleContext.getPrerequisitesByConfiguration(
            "deps", Mode.SPLIT, ObjcProtoProvider.SKYLARK_CONSTRUCTOR);
    Multimap<String, ObjcProtoProvider> objcProtoProvidersMap =
        MultiArchBinarySupport.transformMap(objcProtoProvidersByConfig);

    Map<String, NestedSet<Artifact>> outputGroupCollector = new TreeMap<>();
    for (Map.Entry<BuildConfiguration, CcToolchainProvider> entry :
        childConfigurationsAndToolchains.entrySet()) {
      BuildConfiguration childToolchainConfig = entry.getKey();
      String childCpu = entry.getKey().getCpu();
      CcToolchainProvider childToolchain = entry.getValue();

      Optional<ObjcProvider> protosObjcProvider;
      if (ObjcRuleClasses.objcConfiguration(ruleContext).enableAppleBinaryNativeProtos()) {
        Collection<ObjcProtoProvider> objcProtoProviders = objcProtoProvidersMap.get(childCpu);
        ProtobufSupport protoSupport =
            new ProtobufSupport(
                    ruleContext,
                    childToolchainConfig,
                    protosToAvoid,
                    objcProtoProviders,
                    ProtobufSupport.getTransitivePortableProtoFilters(objcProtoProviders),
                    childToolchain)
                .registerGenerationAction()
                .registerCompilationAction();

        protosObjcProvider = protoSupport.getObjcProvider();
      } else {
        protosObjcProvider = Optional.absent();
      }

      IntermediateArtifacts intermediateArtifacts =
          ObjcRuleClasses.intermediateArtifacts(ruleContext, childToolchainConfig);

      ObjcCommon common =
          common(
              ruleContext,
              childToolchainConfig,
              intermediateArtifacts,
              nullToEmptyList(cpuToCTATDepsCollectionMap.get(childCpu)),
              protosObjcProvider);
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
          new CompilationSupport.Builder()
              .setRuleContext(ruleContext)
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

    AppleConfiguration appleConfiguration = ruleContext.getFragment(AppleConfiguration.class);

    new LipoSupport(ruleContext)
        .registerCombineArchitecturesAction(
            librariesToLipo.build(),
            ruleIntermediateArtifacts.combinedArchitectureArchive(),
            appleConfiguration.getMultiArchPlatform(platformType));

    RuleConfiguredTargetBuilder targetBuilder =
        ObjcRuleClasses.ruleConfiguredTarget(ruleContext, filesToBuild.build());

    objcProviderBuilder.add(
        MULTI_ARCH_LINKED_ARCHIVES, ruleIntermediateArtifacts.combinedArchitectureArchive());

    ObjcProvider objcProvider = objcProviderBuilder.build();

    if (appleConfiguration.shouldLinkingRulesPropagateObjc()) {
      targetBuilder.addNativeDeclaredProvider(objcProvider);
      targetBuilder.addSkylarkTransitiveInfo(ObjcProvider.SKYLARK_NAME, objcProvider);
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
      List<ConfiguredTargetAndData> propagatedConfigredTargetAndTargetDeps,
      Optional<ObjcProvider> protosObjcProvider) throws InterruptedException {

    CompilationArtifacts compilationArtifacts = new CompilationArtifacts.Builder().build();

    return new ObjcCommon.Builder(ruleContext, buildConfiguration)
        .setCompilationAttributes(
            CompilationAttributes.Builder.fromRuleContext(ruleContext).build())
        .setCompilationArtifacts(compilationArtifacts)
        .addDeps(propagatedConfigredTargetAndTargetDeps)
        .addDepObjcProviders(protosObjcProvider.asSet())
        .setIntermediateArtifacts(intermediateArtifacts)
        .setAlwayslink(false)
        .build();
  }

  private <T> List<T> nullToEmptyList(List<T> inputList) {
    return inputList != null ? inputList : ImmutableList.<T>of();
  }

  private static NestedSet<Artifact> protoArtifactsToAvoid(
      Iterable<ObjcProtoProvider> avoidedProviders) {
    NestedSetBuilder<Artifact> avoidArtifacts = NestedSetBuilder.stableOrder();
    for (ObjcProtoProvider avoidProvider : avoidedProviders) {
      avoidArtifacts.addTransitive(avoidProvider.getProtoFiles());
    }
    return avoidArtifacts.build();
  }
}
