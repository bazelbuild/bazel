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

import static com.google.common.collect.ImmutableList.toImmutableList;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableListMultimap;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Multimap;
import com.google.common.collect.Streams;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.Info;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.rules.cpp.CcInfo;
import com.google.devtools.build.lib.rules.cpp.CcLinkingContext;
import com.google.devtools.build.lib.rules.cpp.CcToolchainProvider;
import com.google.devtools.build.lib.rules.cpp.CppSemantics;
import com.google.devtools.build.lib.rules.objc.CompilationSupport.ExtraLinkArgs;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetAndData;
import java.util.List;
import java.util.Map;

/** Support utility for creating multi-arch Apple binaries. */
public class MultiArchBinarySupport {
  private final RuleContext ruleContext;
  private final CppSemantics cppSemantics;

  /**
   * Returns all child configurations for this multi-arch target, mapped to the toolchains that they
   * should use.
   */
  static ImmutableMap<BuildConfiguration, CcToolchainProvider> getChildConfigurationsAndToolchains(
      RuleContext ruleContext) {
    ImmutableListMultimap<BuildConfiguration, CcToolchainProvider> configToProvider =
        ruleContext.getPrerequisitesByConfiguration(
            ObjcRuleClasses.CHILD_CONFIG_ATTR, CcToolchainProvider.PROVIDER);

    ImmutableMap.Builder<BuildConfiguration, CcToolchainProvider> result = ImmutableMap.builder();
    for (BuildConfiguration config : configToProvider.keySet()) {
      CcToolchainProvider toolchain = Iterables.getOnlyElement(configToProvider.get(config));
      result.put(config, toolchain);
    }

    return result.build();
  }

  static <V> ImmutableListMultimap<String, V> transformMap(Multimap<BuildConfiguration, V> input) {
    ImmutableListMultimap.Builder<String, V> result = ImmutableListMultimap.builder();
    for (Map.Entry<BuildConfiguration, V> entry : input.entries()) {
      result.put(entry.getKey().getCpu(), entry.getValue());
    }

    return result.build();
  }

  /** A tuple of values about dependency trees in a specific child configuration. */
  @AutoValue
  abstract static class DependencySpecificConfiguration {
    static DependencySpecificConfiguration create(
        BuildConfiguration config,
        CcToolchainProvider toolchain,
        ObjcProvider objcLinkProvider,
        ObjcProvider objcPropagateProvider) {
      return new AutoValue_MultiArchBinarySupport_DependencySpecificConfiguration(
          config, toolchain, objcLinkProvider, objcPropagateProvider);
    }

    /** Returns the child configuration for this tuple. */
    abstract BuildConfiguration config();

    /** Returns the cc toolchain for this configuration. */
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

  /** @param ruleContext the current rule context */
  public MultiArchBinarySupport(RuleContext ruleContext, CppSemantics cppSemantics) {
    this.ruleContext = ruleContext;
    this.cppSemantics = cppSemantics;
  }

  /**
   * Registers actions to link a single-platform/architecture Apple binary in a specific
   * configuration.
   *
   * @param dependencySpecificConfiguration a single {@link DependencySpecificConfiguration} that
   *     corresponds to the child configuration to link for this target. Can be obtained via {@link
   *     #getDependencySpecificConfigurations}
   * @param extraLinkArgs the extra linker args to add to link actions linking single-architecture
   *     binaries together
   * @param extraLinkInputs the extra linker inputs to be made available during link actions
   * @param isStampingEnabled whether linkstamping is enabled
   * @param infoCollections a list of provider collections which are propagated from the
   *     dependencies in the requested configuration
   * @param outputMapCollector a map to which output groups created by compile action generation are
   *     added
   * @return an {@link Artifact} representing the single-architecture binary linked from this call
   * @throws RuleErrorException if there are attribute errors in the current rule context
   */
  public Artifact registerConfigurationSpecificLinkActions(
      DependencySpecificConfiguration dependencySpecificConfiguration,
      ExtraLinkArgs extraLinkArgs,
      Iterable<Artifact> extraLinkInputs,
      boolean isStampingEnabled,
      Iterable<TransitiveInfoCollection> infoCollections,
      Map<String, NestedSet<Artifact>> outputMapCollector)
      throws RuleErrorException, InterruptedException {
    IntermediateArtifacts intermediateArtifacts =
        ObjcRuleClasses.intermediateArtifacts(
            ruleContext, dependencySpecificConfiguration.config());
    J2ObjcMappingFileProvider j2ObjcMappingFileProvider =
        J2ObjcMappingFileProvider.union(
            getTypedProviders(infoCollections, J2ObjcMappingFileProvider.PROVIDER));
    J2ObjcEntryClassProvider j2ObjcEntryClassProvider =
        new J2ObjcEntryClassProvider.Builder()
            .addTransitive(getTypedProviders(infoCollections, J2ObjcEntryClassProvider.PROVIDER))
            .build();
    ImmutableList<CcLinkingContext> ccLinkingContexts =
        getTypedProviders(infoCollections, CcInfo.PROVIDER).stream()
            .map(CcInfo::getCcLinkingContext)
            .collect(toImmutableList());

    ObjcProvider objcProvider = dependencySpecificConfiguration.objcLinkProvider();
    CompilationArtifacts compilationArtifacts =
        new CompilationArtifacts.Builder()
            .setIntermediateArtifacts(
                ObjcRuleClasses.intermediateArtifacts(
                    ruleContext, dependencySpecificConfiguration.config()))
            .build();

    CompilationSupport compilationSupport =
        new CompilationSupport.Builder(ruleContext, cppSemantics)
            .setConfig(dependencySpecificConfiguration.config())
            .setToolchainProvider(dependencySpecificConfiguration.toolchain())
            .setOutputGroupCollector(outputMapCollector)
            .build();

    compilationSupport
        .registerCompileAndArchiveActions(compilationArtifacts, ObjcCompilationContext.EMPTY)
        .registerLinkActions(
            objcProvider,
            ccLinkingContexts,
            j2ObjcMappingFileProvider,
            j2ObjcEntryClassProvider,
            extraLinkArgs,
            extraLinkInputs,
            isStampingEnabled)
        .validateAttributes();
    ruleContext.assertNoErrors();

    return intermediateArtifacts.strippedSingleArchitectureBinary();
  }

  /**
   * Returns a set of {@link DependencySpecificConfiguration} instances that comprise all
   * information about the dependencies for each child configuration. This can be used both to
   * register actions in {@link #registerConfigurationSpecificLinkActions} and collect provider
   * information to be propagated upstream.
   *
   * @param childConfigurationsAndToolchains the set of configurations and toolchains for which
   *     dependencies of the current rule are built
   * @param cpuToDepsCollectionMap a map from child configuration CPU to providers that "deps" of
   *     the current rule have propagated in that configuration
   * @param dylibProviders {@link TransitiveInfoCollection}s that dynamic library dependencies of
   *     the current rule have propagated
   * @throws RuleErrorException if there are attribute errors in the current rule context
   */
  public ImmutableSet<DependencySpecificConfiguration> getDependencySpecificConfigurations(
      Map<BuildConfiguration, CcToolchainProvider> childConfigurationsAndToolchains,
      ImmutableListMultimap<String, TransitiveInfoCollection> cpuToDepsCollectionMap,
      ImmutableListMultimap<String, ConfiguredTargetAndData> cpuToCTATDepsCollectionMap,
      ImmutableList<TransitiveInfoCollection> dylibProviders)
      throws RuleErrorException, InterruptedException {
    Iterable<ObjcProvider> dylibObjcProviders = getDylibObjcProviders(dylibProviders);
    ImmutableSet.Builder<DependencySpecificConfiguration> childInfoBuilder = ImmutableSet.builder();

    for (BuildConfiguration childToolchainConfig : childConfigurationsAndToolchains.keySet()) {
      String childCpu = childToolchainConfig.getCpu();

      IntermediateArtifacts intermediateArtifacts =
          ObjcRuleClasses.intermediateArtifacts(ruleContext, childToolchainConfig);

      ObjcCommon common =
          common(
              ruleContext,
              childToolchainConfig,
              intermediateArtifacts,
              nullToEmptyList(cpuToCTATDepsCollectionMap.get(childCpu)),
              dylibObjcProviders);
      ObjcProvider objcProviderWithDylibSymbols = common.getObjcProvider();
      ObjcProvider objcProvider =
          objcProviderWithDylibSymbols.subtractSubtrees(dylibObjcProviders, ImmutableList.of());

      childInfoBuilder.add(
          DependencySpecificConfiguration.create(
              childToolchainConfig,
              childConfigurationsAndToolchains.get(childToolchainConfig),
              objcProvider,
              objcProviderWithDylibSymbols));
    }

    return childInfoBuilder.build();
  }

  private static Iterable<ObjcProvider> getDylibObjcProviders(
      ImmutableList<TransitiveInfoCollection> transitiveInfoCollections) {
    // Dylibs.
    ImmutableList<ObjcProvider> frameworkObjcProviders =
        getTypedProviders(transitiveInfoCollections, AppleDynamicFrameworkInfo.STARLARK_CONSTRUCTOR)
            .stream()
            .map(frameworkProvider -> frameworkProvider.getDepsObjcProvider())
            .collect(ImmutableList.toImmutableList());
    // Bundle Loaders.
    ImmutableList<ObjcProvider> executableObjcProviders =
        getTypedProviders(transitiveInfoCollections, AppleExecutableBinaryInfo.STARLARK_CONSTRUCTOR)
            .stream()
            .map(frameworkProvider -> frameworkProvider.getDepsObjcProvider())
            .collect(ImmutableList.toImmutableList());

    return Iterables.concat(
        frameworkObjcProviders,
        executableObjcProviders,
        getTypedProviders(transitiveInfoCollections, ObjcProvider.STARLARK_CONSTRUCTOR));
  }

  private ObjcCommon common(
      RuleContext ruleContext,
      BuildConfiguration buildConfiguration,
      IntermediateArtifacts intermediateArtifacts,
      List<ConfiguredTargetAndData> propagatedConfiguredTargetAndDataDeps,
      Iterable<ObjcProvider> additionalDepProviders)
      throws InterruptedException {

    ObjcCommon.Builder commonBuilder =
        new ObjcCommon.Builder(ObjcCommon.Purpose.LINK_ONLY, ruleContext, buildConfiguration)
            .setCompilationAttributes(
                CompilationAttributes.Builder.fromRuleContext(ruleContext).build())
            .addDeps(propagatedConfiguredTargetAndDataDeps)
            .addObjcProviders(additionalDepProviders)
            .setIntermediateArtifacts(intermediateArtifacts)
            .setAlwayslink(false)
            .setLinkedBinary(intermediateArtifacts.strippedSingleArchitectureBinary());

    return commonBuilder.build();
  }

  private <T> List<T> nullToEmptyList(List<T> inputList) {
    return inputList != null ? inputList : ImmutableList.<T>of();
  }

  private static <T extends Info> ImmutableList<T> getTypedProviders(
      Iterable<TransitiveInfoCollection> infoCollections, BuiltinProvider<T> providerClass) {
    return Streams.stream(infoCollections)
        .filter(infoCollection -> infoCollection.get(providerClass) != null)
        .map(infoCollection -> infoCollection.get(providerClass))
        .collect(ImmutableList.toImmutableList());
  }
}
