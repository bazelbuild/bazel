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
import static com.google.common.collect.Streams.stream;

import com.google.auto.value.AutoValue;
import com.google.common.base.Optional;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.Info;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.StarlarkInfo;
import com.google.devtools.build.lib.packages.StarlarkProvider;
import com.google.devtools.build.lib.packages.StarlarkProviderIdentifier;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;
import com.google.devtools.build.lib.rules.apple.ApplePlatform;
import com.google.devtools.build.lib.rules.cpp.CcInfo;
import com.google.devtools.build.lib.rules.cpp.CcLinkingContext;
import com.google.devtools.build.lib.rules.cpp.CcLinkingContext.LinkerInput;
import com.google.devtools.build.lib.rules.cpp.CcToolchainProvider;
import com.google.devtools.build.lib.rules.cpp.CppSemantics;
import com.google.devtools.build.lib.rules.cpp.LibraryToLink;
import com.google.devtools.build.lib.rules.cpp.UserVariablesExtension;
import com.google.devtools.build.lib.rules.objc.AppleLinkingOutputs.TargetTriplet;
import com.google.devtools.build.lib.rules.objc.CompilationSupport.ExtraLinkArgs;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetAndData;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.StarlarkList;

/** Support utility for creating multi-arch Apple binaries. */
public class MultiArchBinarySupport {
  private final RuleContext ruleContext;
  private final CppSemantics cppSemantics;

  /** A tuple of values about dependency trees in a specific child configuration. */
  @AutoValue
  abstract static class DependencySpecificConfiguration {
    static DependencySpecificConfiguration create(
        BuildConfigurationValue config,
        CcToolchainProvider toolchain,
        CcLinkingContext linkingInfoProvider,
        ObjcProvider objcProviderWithAvoidDepsSymbols,
        CcInfo ccInfoWithAvoidDepsSymbols) {
      return new AutoValue_MultiArchBinarySupport_DependencySpecificConfiguration(
          config,
          toolchain,
          linkingInfoProvider,
          objcProviderWithAvoidDepsSymbols,
          ccInfoWithAvoidDepsSymbols);
    }

    /** Returns the child configuration for this tuple. */
    abstract BuildConfigurationValue config();

    /** Returns the cc toolchain for this configuration. */
    abstract CcToolchainProvider toolchain();

    /**
     * Returns the {@link CcLinkingContext} that has most of the information used for linking, whose
     * avoid deps symbols have been subtracted.
     */
    abstract CcLinkingContext linkingInfoProvider();

    /**
     * Returns the {@link ObjcProvider} to propagate up to dependers; this will not have avoid deps
     * symbols subtracted, thus signaling that this target is still responsible for those symbols.
     */
    abstract ObjcProvider objcProviderWithAvoidDepsSymbols();

    /**
     * Returns the {@link CcInfo} to propagate up to dependers; this will not have avoid deps
     * symbols subtracted, thus signaling that this target is still responsible for those symbols.
     */
    abstract CcInfo ccInfoWithAvoidDepsSymbols();
  }

  /** @param ruleContext the current rule context */
  public MultiArchBinarySupport(RuleContext ruleContext, CppSemantics cppSemantics) {
    this.ruleContext = ruleContext;
    this.cppSemantics = cppSemantics;
  }

  private StarlarkInfo getStarlarkUnionedJ2objcProvider(
      String providerName,
      String unionFunctionName,
      Iterable<? extends TransitiveInfoCollection> infoCollections)
      throws RuleErrorException, InterruptedException {
    ImmutableList<StarlarkInfo> providers =
        getTypedProviders(
            infoCollections,
            StarlarkProviderIdentifier.forKey(
                new StarlarkProvider.Key(
                    Label.parseCanonicalUnchecked("@_builtins//:common/objc/providers.bzl"),
                    providerName)));

    Object starlarkFunc = ruleContext.getStarlarkDefinedBuiltin(unionFunctionName);
    ruleContext.initStarlarkRuleContext();
    return (StarlarkInfo)
        ruleContext.callStarlarkOrThrowRuleError(
            starlarkFunc,
            ImmutableList.of(StarlarkList.immutableCopyOf(providers)),
            new HashMap<>());
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
   * @param userVariablesExtension the UserVariablesExtension to pass to the linker actions
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
      Iterable<String> extraRequestedFeatures,
      Iterable<String> extraDisabledFeatures,
      boolean isStampingEnabled,
      UserVariablesExtension userVariablesExtension,
      Iterable<? extends TransitiveInfoCollection> infoCollections,
      Map<String, NestedSet<Artifact>> outputMapCollector)
      throws RuleErrorException, InterruptedException, EvalException {
    IntermediateArtifacts intermediateArtifacts =
        new IntermediateArtifacts(ruleContext, dependencySpecificConfiguration.config());

    StarlarkInfo j2ObjcEntryClassProvider =
        getStarlarkUnionedJ2objcProvider(
            "J2ObjcEntryClassInfo", "j2objc_entry_class_info_union", infoCollections);

    StarlarkInfo j2ObjcMappingFileProvider =
        getStarlarkUnionedJ2objcProvider(
            "J2ObjcMappingFileInfo", "j2objc_mapping_file_info_union", infoCollections);

    CompilationSupport compilationSupport =
        new CompilationSupport.Builder(ruleContext, cppSemantics)
            .setConfig(dependencySpecificConfiguration.config())
            .setToolchainProvider(dependencySpecificConfiguration.toolchain())
            .build();

    compilationSupport
        .registerLinkActions(
            dependencySpecificConfiguration.linkingInfoProvider(),
            dependencySpecificConfiguration.objcProviderWithAvoidDepsSymbols(),
            j2ObjcMappingFileProvider,
            j2ObjcEntryClassProvider,
            extraLinkArgs,
            extraLinkInputs,
            extraRequestedFeatures,
            extraDisabledFeatures,
            isStampingEnabled,
            userVariablesExtension)
        .validateAttributes();
    ruleContext.assertNoErrors();

    return intermediateArtifacts.strippedSingleArchitectureBinary();
  }

  private static HashSet<PathFragment> buildAvoidLibrarySet(
      Iterable<CcLinkingContext> avoidDepContexts) {
    HashSet<PathFragment> avoidLibrarySet = new HashSet<>();
    for (CcLinkingContext context : avoidDepContexts) {
      for (LinkerInput linkerInput : context.getLinkerInputs().toList()) {
        for (LibraryToLink libraryToLink : linkerInput.getLibraries()) {
          Artifact library = CompilationSupport.getStaticLibraryForLinking(libraryToLink);
          if (library != null) {
            avoidLibrarySet.add(library.getRunfilesPath());
          }
        }
      }
    }

    return avoidLibrarySet;
  }

  public static CcLinkingContext ccLinkingContextSubtractSubtrees(
      RuleContext ruleContext,
      Iterable<CcLinkingContext> depContexts,
      Iterable<CcLinkingContext> avoidDepContexts) {
    CcLinkingContext.Builder outputContext = new CcLinkingContext.Builder();

    outputContext.setOwner(ruleContext.getLabel());

    HashSet<PathFragment> avoidLibrarySet = buildAvoidLibrarySet(avoidDepContexts);
    ImmutableList.Builder<LibraryToLink> filteredLibraryList = new ImmutableList.Builder<>();
    for (CcLinkingContext context : depContexts) {

      for (LinkerInput linkerInput : context.getLinkerInputs().toList()) {
        for (LibraryToLink libraryToLink : linkerInput.getLibraries()) {
          Artifact library = CompilationSupport.getLibraryForLinking(libraryToLink);
          Preconditions.checkNotNull(library);
          if (!avoidLibrarySet.contains(library.getRunfilesPath())) {
            filteredLibraryList.add(libraryToLink);
          }
        }

        outputContext.addUserLinkFlags(linkerInput.getUserLinkFlags());
        outputContext.addNonCodeInputs(linkerInput.getNonCodeInputs());
        outputContext.addLinkstamps(linkerInput.getLinkstamps());
      }

      NestedSetBuilder<LibraryToLink> filteredLibrarySet = NestedSetBuilder.linkOrder();
      filteredLibrarySet.addAll(filteredLibraryList.build());
      outputContext.addLibraries(filteredLibrarySet.build().toList());
    }

    return outputContext.build();
  }

  /**
   * Returns a map of {@link DependencySpecificConfiguration} instances keyed by their split
   * transition key. Each dependency specific configuration comprise all information about the
   * dependencies for each child configuration. This can be used both to register actions in {@link
   * #registerConfigurationSpecificLinkActions} and collect provider information to be propagated
   * upstream.
   *
   * @param splitToolchains a map from split toolchains for which dependencies of the current rule
   *     are built
   * @param splitDeps a map from split "deps" of the current rule.
   * @param avoidDepsProviders {@link TransitiveInfoCollection}s that dependencies of the current
   *     rule have propagated which should not be linked into the binary
   * @throws RuleErrorException if there are attribute errors in the current rule context
   */
  public ImmutableMap<Optional<String>, DependencySpecificConfiguration>
      getDependencySpecificConfigurations(
          Map<Optional<String>, List<ConfiguredTargetAndData>> splitToolchains,
          Map<Optional<String>, List<ConfiguredTargetAndData>> splitDeps,
          ImmutableList<TransitiveInfoCollection> avoidDepsProviders)
          throws RuleErrorException, InterruptedException {
    Iterable<ObjcProvider> avoidDepsObjcProviders = getAvoidDepsObjcProviders(avoidDepsProviders);
    Iterable<CcInfo> avoidDepsCcInfos = getAvoidDepsCcInfos(avoidDepsProviders);

    ImmutableMap.Builder<Optional<String>, DependencySpecificConfiguration> childInfoBuilder =
        ImmutableMap.builder();
    for (Optional<String> splitTransitionKey : splitToolchains.keySet()) {
      ConfiguredTargetAndData ctad =
          Iterables.getOnlyElement(splitToolchains.get(splitTransitionKey));
      BuildConfigurationValue childToolchainConfig = ctad.getConfiguration();

      List<? extends TransitiveInfoCollection> propagatedDeps =
          getProvidersFromCtads(splitDeps.get(splitTransitionKey));

      ObjcCommon common =
          common(
              ruleContext,
              childToolchainConfig,
              propagatedDeps,
              avoidDepsCcInfos,
              avoidDepsObjcProviders);

      ObjcProvider objcProviderWithAvoidDepsSymbols = common.getObjcProvider();
      CcInfo ccInfoWithAvoidDepsSymbols = common.createCcInfo();

      CcLinkingContext linkingInfoProvider =
          ccLinkingContextSubtractSubtrees(
              ruleContext,
              ImmutableList.of(ccInfoWithAvoidDepsSymbols.getCcLinkingContext()),
              stream(avoidDepsCcInfos).map(CcInfo::getCcLinkingContext).collect(toImmutableList()));

      CcToolchainProvider toolchainProvider =
          ctad.getConfiguredTarget().get(CcToolchainProvider.PROVIDER);

      childInfoBuilder.put(
          splitTransitionKey,
          DependencySpecificConfiguration.create(
              childToolchainConfig,
              toolchainProvider,
              linkingInfoProvider,
              objcProviderWithAvoidDepsSymbols,
              ccInfoWithAvoidDepsSymbols));
    }

    return childInfoBuilder.buildOrThrow();
  }

  private static Iterable<ObjcProvider> getAvoidDepsObjcProviders(
      ImmutableList<TransitiveInfoCollection> transitiveInfoCollections) {
    return getTypedProviders(transitiveInfoCollections, ObjcProvider.STARLARK_CONSTRUCTOR);
  }

  private static Iterable<CcInfo> getAvoidDepsCcInfos(
      ImmutableList<TransitiveInfoCollection> transitiveInfoCollections) {
    ImmutableList<CcInfo> frameworkContexts =
        getTypedProviders(transitiveInfoCollections, AppleDynamicFrameworkInfo.STARLARK_CONSTRUCTOR)
            .stream()
            .map(provider -> provider.getDepsCcInfo())
            .collect(toImmutableList());
    ImmutableList<CcInfo> executableContexts =
        getTypedProviders(transitiveInfoCollections, AppleExecutableBinaryInfo.STARLARK_CONSTRUCTOR)
            .stream()
            .map(provider -> provider.getDepsCcInfo())
            .collect(toImmutableList());
    ImmutableList<CcInfo> directContexts =
        getTypedProviders(transitiveInfoCollections, CcInfo.PROVIDER);

    return Iterables.concat(frameworkContexts, executableContexts, directContexts);
  }

  private ObjcCommon common(
      RuleContext ruleContext,
      BuildConfigurationValue buildConfiguration,
      List<? extends TransitiveInfoCollection> propagatedDeps,
      Iterable<CcInfo> additionalDepCcInfos,
      Iterable<ObjcProvider> additionalDepObjcProviders)
      throws InterruptedException {

    ObjcCommon.Builder commonBuilder =
        new ObjcCommon.Builder(ruleContext, buildConfiguration)
            .setCompilationAttributes(
                CompilationAttributes.Builder.fromRuleContext(ruleContext).build())
            .addDeps(propagatedDeps)
            .addCcLinkingContexts(additionalDepCcInfos)
            .addObjcProviders(additionalDepObjcProviders);

    return commonBuilder.build();
  }

  private static <T extends Info> ImmutableList<T> getTypedProviders(
      Iterable<? extends TransitiveInfoCollection> infoCollections,
      BuiltinProvider<T> providerClass) {
    return stream(infoCollections)
        .filter(infoCollection -> infoCollection.get(providerClass) != null)
        .map(infoCollection -> infoCollection.get(providerClass))
        .collect(toImmutableList());
  }

  private static ImmutableList<StarlarkInfo> getTypedProviders(
      Iterable<? extends TransitiveInfoCollection> infoCollections,
      StarlarkProviderIdentifier identifier) {
    return stream(infoCollections)
        .filter(infoCollection -> infoCollection.get(identifier) != null)
        .map(infoCollection -> (StarlarkInfo) infoCollection.get(identifier))
        .collect(toImmutableList());
  }

  /** Returns providers from a list of {@link ConfiguredTargetAndData} */
  public static List<? extends TransitiveInfoCollection> getProvidersFromCtads(
      List<ConfiguredTargetAndData> ctads) {
    if (ctads == null) {
      return ImmutableList.<TransitiveInfoCollection>of();
    }
    return ctads.stream()
        .map(ConfiguredTargetAndData::getConfiguredTarget)
        .collect(Collectors.toList());
  }

  /**
   * Returns an Apple target triplet (arch, platform, environment) for a given {@link
   * BuildConfigurationValue}.
   *
   * @param config {@link BuildConfigurationValue} from rule context
   * @return {@link AppleLinkingOutputs.TargetTriplet}
   */
  public static AppleLinkingOutputs.TargetTriplet getTargetTriplet(BuildConfigurationValue config) {
    // TODO(b/177442911): Use the target platform from platform info coming from split
    // transition outputs instead of inferring this based on the target CPU.
    ApplePlatform cpuPlatform = ApplePlatform.forTargetCpu(config.getCpu());
    AppleConfiguration appleConfig = config.getFragment(AppleConfiguration.class);

    return TargetTriplet.create(
        appleConfig.getSingleArchitecture(),
        cpuPlatform.getTargetPlatform(),
        cpuPlatform.getTargetEnvironment());
  }

  /**
   * Transforms a {@link Map<Optional<String>, List<ConfiguredTargetAndData>>}, to a Starlark Dict
   * keyed by split transition keys with {@link AppleLinkingOutputs.TargetTriplet} Starlark struct
   * definition.
   *
   * @param ctads a {@link Map<Optional<String>, List<ConfiguredTargetAndData>>} from rule context
   * @return a Starlark {@link Dict<String, StructImpl>} representing split transition keys with
   *     their target triplet (architecture, platform, environment)
   */
  public static Dict<String, StructImpl> getSplitTargetTripletFromCtads(
      Map<Optional<String>, List<ConfiguredTargetAndData>> ctads) throws EvalException {
    Dict.Builder<String, StructImpl> result = Dict.builder();
    for (Optional<String> splitTransitionKey : ctads.keySet()) {
      if (!splitTransitionKey.isPresent()) {
        throw new EvalException("unexpected empty key in split transition");
      }
      TargetTriplet targetTriplet =
          getTargetTriplet(
              Iterables.getOnlyElement(ctads.get(splitTransitionKey)).getConfiguration());
      result.put(splitTransitionKey.get(), targetTriplet.toStarlarkStruct());
    }
    return result.buildImmutable();
  }
}
