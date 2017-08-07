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
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;
import com.google.devtools.build.lib.rules.apple.ApplePlatform.PlatformType;
import com.google.devtools.build.lib.rules.cpp.CcLinkParamsProvider;
import com.google.devtools.build.lib.rules.cpp.CcToolchainProvider;
import com.google.devtools.build.lib.rules.cpp.CppHelper;
import com.google.devtools.build.lib.rules.objc.ObjcCommon.ResourceAttributes;
import com.google.devtools.build.lib.rules.objc.ObjcProvider.Key;
import com.google.devtools.build.lib.rules.proto.ProtoSourcesProvider;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/**
 * Implementation for the "apple_static_library" rule.
 */
public class AppleStaticLibrary implements RuleConfiguredTargetFactory {

  /**
   * Set of {@link ObjcProvider} values which are propagated from dependencies to dependers by
   * this rule.
   */
  private static final ImmutableSet<Key<?>> PROPAGATE_KEYS =
      ImmutableSet.<Key<?>>of(
          ObjcProvider.ASSET_CATALOG,
          ObjcProvider.BUNDLE_FILE,
          ObjcProvider.GENERAL_RESOURCE_DIR,
          ObjcProvider.GENERAL_RESOURCE_FILE,
          ObjcProvider.SDK_DYLIB,
          ObjcProvider.SDK_FRAMEWORK,
          ObjcProvider.STORYBOARD,
          ObjcProvider.STRINGS,
          ObjcProvider.WEAK_SDK_FRAMEWORK,
          ObjcProvider.XCDATAMODEL,
          ObjcProvider.XIB,
          ObjcProvider.XCASSETS_DIR);

  @VisibleForTesting
  static final String UNSUPPORTED_PLATFORM_TYPE_ERROR_FORMAT =
      "Unsupported platform type \"%s\"";

  @Override
  public final ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException {
    MultiArchSplitTransitionProvider.validateMinimumOs(ruleContext);
    PlatformType platformType = MultiArchSplitTransitionProvider.getPlatformType(ruleContext);

    ImmutableListMultimap<BuildConfiguration, TransitiveInfoCollection> configToDepsCollectionMap =
        ruleContext.getPrerequisitesByConfiguration("deps", Mode.SPLIT);
    ImmutableListMultimap<BuildConfiguration, ObjcProvider> configToObjcAvoidDepsMap =
        ruleContext.getPrerequisitesByConfiguration(AppleStaticLibraryRule.AVOID_DEPS_ATTR_NAME,
            Mode.SPLIT, ObjcProvider.SKYLARK_CONSTRUCTOR);
    ImmutableListMultimap<BuildConfiguration, CcLinkParamsProvider> configToCcAvoidDepsMap =
        ruleContext.getPrerequisitesByConfiguration(AppleStaticLibraryRule.AVOID_DEPS_ATTR_NAME,
            Mode.SPLIT, CcLinkParamsProvider.CC_LINK_PARAMS);
    Iterable<ObjcProtoProvider> avoidProtoProviders =
        ruleContext.getPrerequisites(AppleStaticLibraryRule.AVOID_DEPS_ATTR_NAME, Mode.TARGET,
            ObjcProtoProvider.class);
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

    ObjcProvider.Builder objcProviderBuilder = new ObjcProvider.Builder();

    ImmutableListMultimap<BuildConfiguration, ObjcProtoProvider> objcProtoProvidersMap =
        ruleContext.getPrerequisitesByConfiguration("deps", Mode.SPLIT, ObjcProtoProvider.class);

    Map<String, NestedSet<Artifact>> outputGroupCollector = new TreeMap<>();
    for (BuildConfiguration childConfig : childConfigurationsAndToolchains.keySet()) {
      Iterable<ObjcProtoProvider> objcProtoProviders = objcProtoProvidersMap.get(childConfig);
      ProtobufSupport protoSupport =
          new ProtobufSupport(
                  ruleContext,
                  childConfig,
                  protosToAvoid,
                  ImmutableList.<ProtoSourcesProvider>of(),
                  objcProtoProviders,
                  ProtobufSupport.getTransitivePortableProtoFilters(objcProtoProviders))
              .registerGenerationActions()
              .registerCompilationActions();

      Optional<ObjcProvider> protosObjcProvider = protoSupport.getObjcProvider();

      IntermediateArtifacts intermediateArtifacts =
          ObjcRuleClasses.intermediateArtifacts(ruleContext, childConfig);

      ObjcCommon common =
          common(
              ruleContext,
              childConfig,
              intermediateArtifacts,
              nullToEmptyList(configToDepsCollectionMap.get(childConfig)),
              protosObjcProvider);
      ObjcProvider objcProvider =
          common.getObjcProvider().subtractSubtrees(configToObjcAvoidDepsMap.get(childConfig),
              configToCcAvoidDepsMap.get(childConfig));

      librariesToLipo.add(intermediateArtifacts.strippedSingleArchitectureLibrary());

      CompilationSupport compilationSupport =
          new CompilationSupport.Builder()
              .setRuleContext(ruleContext)
              .setConfig(childConfig)
              .setOutputGroupCollector(outputGroupCollector)
              .build();

      compilationSupport
          .registerCompileAndArchiveActions(
              common.getCompilationArtifacts().get(),
              objcProvider,
              childConfigurationsAndToolchains.get(childConfig))
          .registerFullyLinkAction(
              objcProvider,
              intermediateArtifacts.strippedSingleArchitectureLibrary(),
              childConfigurationsAndToolchains.get(childConfig),
              CppHelper.getFdoSupportUsingDefaultCcToolchainAttribute(ruleContext))
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

    targetBuilder
        // TODO(cparsons): Remove ObjcProvider as a direct provider.
        .addNativeDeclaredProvider(objcProvider)
        .addNativeDeclaredProvider(
            new AppleStaticLibraryProvider(
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
      List<TransitiveInfoCollection> propagatedDeps,
      Optional<ObjcProvider> protosObjcProvider) {

    CompilationArtifacts compilationArtifacts =
        CompilationSupport.compilationArtifacts(ruleContext, intermediateArtifacts);

    return new ObjcCommon.Builder(ruleContext, buildConfiguration)
        .setCompilationAttributes(
            CompilationAttributes.Builder.fromRuleContext(ruleContext).build())
        .setCompilationArtifacts(compilationArtifacts)
        .setResourceAttributes(new ResourceAttributes(ruleContext))
        .addDefines(ruleContext.getTokenizedStringListAttr("defines"))
        .addDeps(propagatedDeps)
        .addDepObjcProviders(
            ruleContext.getPrerequisites("bundles", Mode.TARGET, ObjcProvider.SKYLARK_CONSTRUCTOR))
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
      for (NestedSet<Artifact> avoidProviderOutputGroup : avoidProvider.getProtoGroups()) {
        avoidArtifacts.addTransitive(avoidProviderOutputGroup);
      }
    }
    return avoidArtifacts.build();
  }
}
