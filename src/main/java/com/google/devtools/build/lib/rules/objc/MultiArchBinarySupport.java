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

import com.google.auto.value.AutoValue;
import com.google.common.base.Optional;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableListMultimap;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Streams;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.platform.ToolchainInfo;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.Info;
import com.google.devtools.build.lib.packages.NativeProvider;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.rules.apple.ApplePlatform;
import com.google.devtools.build.lib.rules.cpp.CcLinkParamsInfo;
import com.google.devtools.build.lib.rules.cpp.CcToolchainProvider;
import com.google.devtools.build.lib.rules.objc.CompilationSupport.ExtraLinkArgs;
import com.google.devtools.build.lib.rules.proto.ProtoSourcesProvider;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Support utility for creating multi-arch Apple binaries.
 */
public class MultiArchBinarySupport {
  private final RuleContext ruleContext;

  /**
   * Returns all child configurations for this multi-arch target, mapped to the toolchains that they
   * should use.
   */
  static ImmutableMap<BuildConfiguration, CcToolchainProvider> getChildConfigurationsAndToolchains(
      RuleContext ruleContext) {
    // This is currently a hack to obtain all child configurations regardless of the attribute
    // values of this rule -- this rule does not currently use the actual info provided by
    // this attribute. b/28403953 tracks cc toolchain usage.
    ImmutableListMultimap<BuildConfiguration, ToolchainInfo> configToProvider =
        ruleContext.getPrerequisitesByConfiguration(
            ObjcRuleClasses.CHILD_CONFIG_ATTR, Mode.SPLIT, ToolchainInfo.PROVIDER);

    ImmutableMap.Builder<BuildConfiguration, CcToolchainProvider> result = ImmutableMap.builder();
    for (BuildConfiguration config : configToProvider.keySet()) {
      CcToolchainProvider toolchain =
          (CcToolchainProvider) Iterables.getOnlyElement(configToProvider.get(config));
      result.put(config, toolchain);
    }

    return result.build();
  }

  /**
   * A tuple of values about dependency trees in a specific child configuration.
   */
  @AutoValue
  abstract static class DependencySpecificConfiguration {
    static DependencySpecificConfiguration create(
        BuildConfiguration config, CcToolchainProvider toolchain, ObjcProvider objcLinkProvider,
        ObjcProvider objcPropagateProvider) {
      return new AutoValue_MultiArchBinarySupport_DependencySpecificConfiguration(
          config, toolchain, objcLinkProvider, objcPropagateProvider);
    }

    /**
     * Returns the child configuration for this tuple.
     */
    abstract BuildConfiguration config();

    /**
     * Returns the cc toolchain for this configuration.
     */
    abstract CcToolchainProvider toolchain();

    /**
     * Returns the {@link ObjcProvider} to use as input to the support controlling link actoins;
     * dylib symbols should be subtracted from this provider.
     */
    abstract ObjcProvider objcLinkProvider();
    
    /**
     * Returns the {@link ObjcProvider} to propagate up to dependers; this will not have dylib
     * symbols subtracted, thus signaling that this target is still responsible for those symbols.
     */
    abstract ObjcProvider objcProviderWithDylibSymbols();
  }


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
   * @param dependencySpecificConfigurations a set of {@link DependencySpecificConfiguration} that
   *     corresponds to child configurations for this target. Can be obtained via {@link
   *     #getDependencySpecificConfigurations}
   * @param extraLinkInputs the extra linker inputs to be made available during link actions
   * @param configToDepsCollectionMap a multimap from dependency configuration to the list of
   *     provider collections which are propagated from the dependencies of that configuration
   * @param outputLipoBinary the artifact (lipo'ed binary) which should be output as a result of
   *     this support
   * @param outputMapCollector a map to which output groups created by compile action generation are
   *     added
   * @throws RuleErrorException if there are attribute errors in the current rule context
   */
  public void registerActions(
      ApplePlatform platform,
      ExtraLinkArgs extraLinkArgs,
      Set<DependencySpecificConfiguration> dependencySpecificConfigurations,
      Iterable<Artifact> extraLinkInputs,
      ImmutableListMultimap<BuildConfiguration, TransitiveInfoCollection> configToDepsCollectionMap,
      Artifact outputLipoBinary,
      Map<String, NestedSet<Artifact>> outputMapCollector)
      throws RuleErrorException, InterruptedException {

    NestedSetBuilder<Artifact> binariesToLipo =
        NestedSetBuilder.<Artifact>stableOrder();
    for (DependencySpecificConfiguration dependencySpecificConfiguration :
        dependencySpecificConfigurations) {
      IntermediateArtifacts intermediateArtifacts =
          ObjcRuleClasses.intermediateArtifacts(
              ruleContext, dependencySpecificConfiguration.config());
      Iterable<TransitiveInfoCollection> infoCollections =
          configToDepsCollectionMap.get(dependencySpecificConfiguration.config());
      J2ObjcMappingFileProvider j2ObjcMappingFileProvider =
          J2ObjcMappingFileProvider.union(
              getTypedProviders(infoCollections, J2ObjcMappingFileProvider.class));
      J2ObjcEntryClassProvider j2ObjcEntryClassProvider =
          new J2ObjcEntryClassProvider.Builder()
              .addTransitive(getTypedProviders(infoCollections, J2ObjcEntryClassProvider.class))
              .build();

      binariesToLipo.add(intermediateArtifacts.strippedSingleArchitectureBinary());

      ObjcProvider objcProvider = dependencySpecificConfiguration.objcLinkProvider();
      CompilationArtifacts compilationArtifacts = new CompilationArtifacts.Builder()
          .setIntermediateArtifacts(ObjcRuleClasses.intermediateArtifacts(
                  ruleContext, dependencySpecificConfiguration.config()))
          .build();

      CompilationSupport compilationSupport =
          new CompilationSupport.Builder()
              .setRuleContext(ruleContext)
              .setConfig(dependencySpecificConfiguration.config())
              .setToolchainProvider(dependencySpecificConfiguration.toolchain())
              .setOutputGroupCollector(outputMapCollector)
              .build();

      compilationSupport
          .registerCompileAndArchiveActions(
              compilationArtifacts, objcProvider, dependencySpecificConfiguration.toolchain())
          .registerLinkActions(
              objcProvider,
              j2ObjcMappingFileProvider,
              j2ObjcEntryClassProvider,
              extraLinkArgs,
              extraLinkInputs,
              DsymOutputType.APP,
              dependencySpecificConfiguration.toolchain())
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
   * Returns a set of {@link DependencySpecificConfiguration} instances that comprise all
   * information about the dependencies for each child configuration. This can be used both to
   * register actions in {@link #registerActions} and collect provider information to be propagated
   * upstream.
   *
   * @param childConfigurationsAndToolchains the set of configurations and toolchains for which
   *     dependencies of the current rule are built
   * @param configToDepsCollectionMap a map from child configuration to providers that "deps" of the
   *     current rule have propagated in that configuration
   * @param configurationToNonPropagatedObjcMap a map from child configuration to providers that
   *     "non_propagated_deps" of the current rule have propagated in that configuration
   * @param dylibProviders {@link TransitiveInfoCollection}s that dynamic library dependencies of
   *     the current rule have propagated
   * @throws RuleErrorException if there are attribute errors in the current rule context
   */
  public ImmutableSet<DependencySpecificConfiguration> getDependencySpecificConfigurations(
      Map<BuildConfiguration, CcToolchainProvider> childConfigurationsAndToolchains,
      ImmutableListMultimap<BuildConfiguration, TransitiveInfoCollection> configToDepsCollectionMap,
      ImmutableListMultimap<BuildConfiguration, ObjcProvider> configurationToNonPropagatedObjcMap,
      Iterable<TransitiveInfoCollection> dylibProviders)
      throws RuleErrorException, InterruptedException {
    Iterable<ObjcProvider> dylibObjcProviders = getDylibObjcProviders(dylibProviders);
    Iterable<ObjcProtoProvider> dylibProtoProviders =
        getTypedProviders(dylibProviders, ObjcProtoProvider.SKYLARK_CONSTRUCTOR);
    NestedSet<Artifact> protosToAvoid = protoArtifactsToAvoid(dylibProtoProviders);
    ImmutableSet.Builder<DependencySpecificConfiguration> childInfoBuilder = ImmutableSet.builder();

    for (BuildConfiguration childConfig : childConfigurationsAndToolchains.keySet()) {
      Iterable<TransitiveInfoCollection> infoCollections =
          configToDepsCollectionMap.get(childConfig);
      Iterable<ObjcProtoProvider> depProtoProviders =
          getTypedProviders(infoCollections, ObjcProtoProvider.SKYLARK_CONSTRUCTOR);
      Optional<ObjcProvider> protosObjcProvider;
      if (ObjcRuleClasses.objcConfiguration(ruleContext).enableAppleBinaryNativeProtos()) {
        ProtobufSupport protoSupport =
            new ProtobufSupport(
                    ruleContext,
                    childConfig,
                    protosToAvoid,
                    ImmutableList.<ProtoSourcesProvider>of(),
                    depProtoProviders,
                    ProtobufSupport.getTransitivePortableProtoFilters(depProtoProviders),
                    childConfigurationsAndToolchains.get(childConfig))
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
              protosObjcProvider.asSet());

      ObjcCommon common =
          common(
              ruleContext,
              childConfig,
              intermediateArtifacts,
              nullToEmptyList(configToDepsCollectionMap.get(childConfig)),
              nullToEmptyList(configurationToNonPropagatedObjcMap.get(childConfig)),
              additionalDepProviders);
      ObjcProvider objcProviderWithDylibSymbols = common.getObjcProvider();
      ObjcProvider objcProvider = objcProviderWithDylibSymbols.subtractSubtrees(dylibObjcProviders,
          ImmutableList.<CcLinkParamsInfo>of());

      childInfoBuilder.add(
          DependencySpecificConfiguration.create(
              childConfig,
              childConfigurationsAndToolchains.get(childConfig),
              objcProvider,
              objcProviderWithDylibSymbols));
    }

    return childInfoBuilder.build();
  }

  private static Iterable<ObjcProvider> getDylibObjcProviders(
      Iterable<TransitiveInfoCollection> transitiveInfoCollections) {
    Iterable<ObjcProvider> frameworkObjcProviders =
        Streams.stream(getTypedProviders(transitiveInfoCollections,
            AppleDynamicFrameworkProvider.SKYLARK_CONSTRUCTOR))
        .map(frameworkProvider -> frameworkProvider.getDepsObjcProvider())
        .collect(ImmutableList.toImmutableList());

    return Iterables.concat(frameworkObjcProviders,
        getTypedProviders(transitiveInfoCollections, ObjcProvider.SKYLARK_CONSTRUCTOR));
  }

  private ObjcCommon common(
      RuleContext ruleContext,
      BuildConfiguration buildConfiguration,
      IntermediateArtifacts intermediateArtifacts,
      List<TransitiveInfoCollection> propagatedDeps,
      List<ObjcProvider> nonPropagatedObjcDeps,
      Iterable<ObjcProvider> additionalDepProviders) {

    ObjcCommon.Builder commonBuilder = new ObjcCommon.Builder(ruleContext, buildConfiguration)
        .setCompilationAttributes(
            CompilationAttributes.Builder.fromRuleContext(ruleContext).build())
        .addDeps(propagatedDeps)
        .addDepObjcProviders(additionalDepProviders)
        .addNonPropagatedDepObjcProviders(nonPropagatedObjcDeps)
        .setIntermediateArtifacts(intermediateArtifacts)
        .setAlwayslink(false)
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

  private static <T extends Info> Iterable<T> getTypedProviders(
      Iterable<TransitiveInfoCollection> infoCollections,
      NativeProvider<T> providerClass) {
    return Streams.stream(infoCollections)
        .filter(infoCollection -> infoCollection.get(providerClass) != null)
        .map(infoCollection -> infoCollection.get(providerClass))
        .collect(ImmutableList.toImmutableList());
  }

  private static <T extends TransitiveInfoProvider> Iterable<T> getTypedProviders(
      Iterable<TransitiveInfoCollection> infoCollections,
      Class<T> providerClass) {
    return Streams.stream(infoCollections)
        .filter(infoCollection -> infoCollection.getProvider(providerClass) != null)
        .map(infoCollection -> infoCollection.getProvider(providerClass))
        .collect(ImmutableList.toImmutableList());
  }
}
