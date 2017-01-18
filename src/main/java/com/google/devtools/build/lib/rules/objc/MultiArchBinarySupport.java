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

import com.google.common.base.Optional;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableListMultimap;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.rules.apple.Platform;
import com.google.devtools.build.lib.rules.objc.CompilationSupport.ExtraLinkArgs;
import com.google.devtools.build.lib.rules.objc.ObjcCommon.ResourceAttributes;

import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Support utility for creating multi-arch Apple binaries.
 */
public class MultiArchBinarySupport {
  private final RuleContext ruleContext;
  
  /**
   * @param ruleContext the current rule context
   */
  public MultiArchBinarySupport(RuleContext ruleContext) {
    this.ruleContext = ruleContext;
  }

  /**
   * Registers actions to create a multi-arch Apple binary.
   *
   * @param platform the platform for which the binary is targeted
   * @param extraLinkArgs the extra linker args to add to link actions linking single-architecture
   *     binaries together
   * @param configurationToObjcProvider a map from from dependency configuration to the
   *     {@link ObjcProvider} which comprises all information about the dependencies in that
   *     configuration. Can be obtained via {@link #objcProviderByDepConfiguration}
   * @param extraLinkInputs the extra linker inputs to be made available during link actions
   * @param configToDepsCollectionMap a multimap from dependency configuration to the
   *     list of provider collections which are propagated from the dependencies of that
   *     configuration
   * @param outputLipoBinary the artifact (lipo'ed binary) which should be output as a result of
   *     this support
   * @throws RuleErrorException if there are attribute errors in the current rule context
   */
  public void registerActions(
      Platform platform,
      ExtraLinkArgs extraLinkArgs,
      Map<BuildConfiguration, ObjcProvider> configurationToObjcProvider,
      Iterable<Artifact> extraLinkInputs,
      ImmutableListMultimap<BuildConfiguration, TransitiveInfoCollection> configToDepsCollectionMap,
      Artifact outputLipoBinary)
      throws RuleErrorException, InterruptedException {

    NestedSetBuilder<Artifact> binariesToLipo =
        NestedSetBuilder.<Artifact>stableOrder();
    for (BuildConfiguration childConfig : configurationToObjcProvider.keySet()) {
      IntermediateArtifacts intermediateArtifacts =
          ObjcRuleClasses.intermediateArtifacts(ruleContext, childConfig);
      ImmutableList.Builder<J2ObjcMappingFileProvider> j2ObjcMappingFileProviders =
          ImmutableList.builder();
      J2ObjcEntryClassProvider.Builder j2ObjcEntryClassProviderBuilder =
          new J2ObjcEntryClassProvider.Builder();
      for (TransitiveInfoCollection dep : configToDepsCollectionMap.get(childConfig)) {
        if (dep.getProvider(J2ObjcMappingFileProvider.class) != null) {
          j2ObjcMappingFileProviders.add(dep.getProvider(J2ObjcMappingFileProvider.class));
        }
        if (dep.getProvider(J2ObjcEntryClassProvider.class) != null) {
          j2ObjcEntryClassProviderBuilder.addTransitive(
              dep.getProvider(J2ObjcEntryClassProvider.class));
        }
      }
      J2ObjcMappingFileProvider j2ObjcMappingFileProvider =
          J2ObjcMappingFileProvider.union(j2ObjcMappingFileProviders.build());
      J2ObjcEntryClassProvider j2ObjcEntryClassProvider = j2ObjcEntryClassProviderBuilder.build();

      binariesToLipo.add(intermediateArtifacts.strippedSingleArchitectureBinary());

      ObjcProvider objcProvider = configurationToObjcProvider.get(childConfig);
      CompilationArtifacts compilationArtifacts =
          CompilationSupport.compilationArtifacts(
              ruleContext, ObjcRuleClasses.intermediateArtifacts(ruleContext, childConfig));
      CompilationSupport.createForConfig(ruleContext, childConfig)
          .registerCompileAndArchiveActions(compilationArtifacts, objcProvider)
          .registerLinkActions(
              objcProvider,
              j2ObjcMappingFileProvider,
              j2ObjcEntryClassProvider,
              extraLinkArgs,
              extraLinkInputs,
              DsymOutputType.APP)
          .validateAttributes();
      ruleContext.assertNoErrors();
    }

    new LipoSupport(ruleContext)
        .registerCombineArchitecturesAction(
            binariesToLipo.build(),
            outputLipoBinary,
            platform);
  }

  /**
   * Returns a map from from dependency configuration to the {@link ObjcCommon} which comprises all
   * information about the dependencies in that configuration. This can be used both to register
   * actions in {@link #registerActions} and collect provider information to be propagated upstream.
   *
   * @param childConfigurations the set of configurations in which dependencies of the current rule
   *     are built
   * @param configToDepsCollectionMap a map from child configuration to providers that "deps" of the
   *     current rule have propagated in that configuration
   * @param configurationToNonPropagatedObjcMap a map from child configuration to providers that
   *     "non_propagated_deps" of the current rule have propagated in that configuration
   * @param dylibObjcProviders {@link ObjcProvider}s that dynamic library dependencies of the
   *     current rule have propagated
   * @param dylibProtoProviders {@link ObjcProtoProvider} providers that dynamic library
   *     dependencies of the current rule have propagated
   * @param bundleLoaderObjcProvider Optional ObjcProvider containing artifacts and paths to be
   *     included in this binary's compilation actions
   * @throws RuleErrorException if there are attribute errors in the current rule context
   */
  public Map<BuildConfiguration, ObjcProvider> objcProviderByDepConfiguration(
      Set<BuildConfiguration> childConfigurations,
      ImmutableListMultimap<BuildConfiguration, TransitiveInfoCollection> configToDepsCollectionMap,
      ImmutableListMultimap<BuildConfiguration, ObjcProvider> configurationToNonPropagatedObjcMap,
      Iterable<ObjcProvider> dylibObjcProviders,
      Iterable<ObjcProtoProvider> dylibProtoProviders,
      Optional<ObjcProvider> bundleLoaderObjcProvider)
      throws RuleErrorException, InterruptedException {
    ImmutableMap.Builder<BuildConfiguration, ObjcProvider> configurationToObjcProviderBuilder =
        ImmutableMap.builder();

    for (BuildConfiguration childConfig : childConfigurations) {
      Optional<ObjcProvider> protosObjcProvider;
      if (ObjcRuleClasses.objcConfiguration(ruleContext).enableAppleBinaryNativeProtos()) {
        ProtobufSupport protoSupport =
            new ProtobufSupport(ruleContext, childConfig,
                    protoArtifactsToAvoid(dylibProtoProviders))
                .registerGenerationActions()
                .registerCompilationActions();
        protosObjcProvider = protoSupport.getObjcProvider();
      } else {
        protosObjcProvider = Optional.absent();
      }

      IntermediateArtifacts intermediateArtifacts =
          ObjcRuleClasses.intermediateArtifacts(ruleContext, childConfig);

      Iterable<ObjcProvider> additionalDepProviders =
          Iterables.concat(
              dylibObjcProviders,
              ruleContext.getPrerequisites("bundles", Mode.TARGET, ObjcProvider.class),
              protosObjcProvider.asSet(),
              bundleLoaderObjcProvider.asSet());

      ObjcCommon common =
          common(
              ruleContext,
              childConfig,
              intermediateArtifacts,
              nullToEmptyList(configToDepsCollectionMap.get(childConfig)),
              nullToEmptyList(configurationToNonPropagatedObjcMap.get(childConfig)),
              additionalDepProviders);
      ObjcProvider objcProvider = common.getObjcProvider().subtractSubtrees(dylibObjcProviders);

      configurationToObjcProviderBuilder.put(childConfig, objcProvider);
    }

    return configurationToObjcProviderBuilder.build();
  }

  private ObjcCommon common(
      RuleContext ruleContext,
      BuildConfiguration buildConfiguration,
      IntermediateArtifacts intermediateArtifacts,
      List<TransitiveInfoCollection> propagatedDeps,
      List<ObjcProvider> nonPropagatedObjcDeps,
      Iterable<ObjcProvider> additionalDepProviders) {

    CompilationArtifacts compilationArtifacts =
        CompilationSupport.compilationArtifacts(ruleContext, intermediateArtifacts);

    ObjcCommon.Builder commonBuilder = new ObjcCommon.Builder(ruleContext, buildConfiguration)
        .setCompilationAttributes(
            CompilationAttributes.Builder.fromRuleContext(ruleContext).build())
        .setCompilationArtifacts(compilationArtifacts)
        .setResourceAttributes(new ResourceAttributes(ruleContext))
        .addDefines(ruleContext.getTokenizedStringListAttr("defines"))
        .addDeps(propagatedDeps)
        .addDepObjcProviders(additionalDepProviders)
        .addNonPropagatedDepObjcProviders(nonPropagatedObjcDeps)
        .setIntermediateArtifacts(intermediateArtifacts)
        .setAlwayslink(false)
        // TODO(b/29152500): Enable module map generation.
        .setLinkedBinary(intermediateArtifacts.strippedSingleArchitectureBinary());

    if (ObjcRuleClasses.objcConfiguration(ruleContext).generateDsym()) {
      commonBuilder.addDebugArtifacts(DsymOutputType.APP);
    }
    return commonBuilder.build();
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
