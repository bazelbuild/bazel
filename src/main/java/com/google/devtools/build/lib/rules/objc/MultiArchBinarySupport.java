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
import com.google.common.collect.ImmutableSortedSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.PlatformOptions;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.BuildOptionsView;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.Info;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.build.lib.rules.apple.AppleCommandLineOptions;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration.ConfigurationDistinguisher;
import com.google.devtools.build.lib.rules.apple.ApplePlatform;
import com.google.devtools.build.lib.rules.apple.ApplePlatform.PlatformType;
import com.google.devtools.build.lib.rules.apple.DottedVersion;
import com.google.devtools.build.lib.rules.cpp.CcInfo;
import com.google.devtools.build.lib.rules.cpp.CcLinkingContext;
import com.google.devtools.build.lib.rules.cpp.CcLinkingContext.LinkerInput;
import com.google.devtools.build.lib.rules.cpp.CcToolchainProvider;
import com.google.devtools.build.lib.rules.cpp.CppSemantics;
import com.google.devtools.build.lib.rules.cpp.LibraryToLink;
import com.google.devtools.build.lib.rules.objc.AppleLinkingOutputs.TargetTriplet;
import com.google.devtools.build.lib.rules.objc.CompilationSupport.ExtraLinkArgs;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetAndData;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import net.starlark.java.eval.Dict;

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
        Object linkingInfoProvider,
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
     * Returns the {@link Object} that has most of the information used for linking. This is either
     * an ObjcProvider or a CcInfo whose avoid deps symbols have been subtracted.
     */
    abstract Object linkingInfoProvider();

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
      Iterable<? extends TransitiveInfoCollection> infoCollections,
      Map<String, NestedSet<Artifact>> outputMapCollector)
      throws RuleErrorException, InterruptedException {
    IntermediateArtifacts intermediateArtifacts =
        new IntermediateArtifacts(ruleContext, dependencySpecificConfiguration.config());
    J2ObjcMappingFileProvider j2ObjcMappingFileProvider =
        J2ObjcMappingFileProvider.union(
            getTypedProviders(infoCollections, J2ObjcMappingFileProvider.PROVIDER));
    J2ObjcEntryClassProvider j2ObjcEntryClassProvider =
        new J2ObjcEntryClassProvider.Builder()
            .addTransitive(getTypedProviders(infoCollections, J2ObjcEntryClassProvider.PROVIDER))
            .build();

    CompilationSupport compilationSupport =
        new CompilationSupport.Builder(ruleContext, cppSemantics)
            .setConfig(dependencySpecificConfiguration.config())
            .setToolchainProvider(dependencySpecificConfiguration.toolchain())
            .build();

    compilationSupport
        .registerLinkActions(
            dependencySpecificConfiguration.linkingInfoProvider(),
            dependencySpecificConfiguration.objcProviderWithAvoidDepsSymbols(),
            dependencySpecificConfiguration.ccInfoWithAvoidDepsSymbols().getCcLinkingContext(),
            j2ObjcMappingFileProvider,
            j2ObjcEntryClassProvider,
            extraLinkArgs,
            extraLinkInputs,
            isStampingEnabled)
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
    ImmutableList<CcLinkingContext> avoidDepsCcLinkingContexts =
        getTypedProviders(avoidDepsProviders, CcInfo.PROVIDER).stream()
            .map(CcInfo::getCcLinkingContext)
            .collect(toImmutableList());

    ImmutableMap.Builder<Optional<String>, DependencySpecificConfiguration> childInfoBuilder =
        ImmutableMap.builder();
    for (Optional<String> splitTransitionKey : splitToolchains.keySet()) {
      ConfiguredTargetAndData ctad =
          Iterables.getOnlyElement(splitToolchains.get(splitTransitionKey));
      BuildConfigurationValue childToolchainConfig = ctad.getConfiguration();
      IntermediateArtifacts intermediateArtifacts =
          new IntermediateArtifacts(ruleContext, childToolchainConfig);

      List<? extends TransitiveInfoCollection> propagatedDeps =
          getProvidersFromCtads(splitDeps.get(splitTransitionKey));

      ObjcCommon common =
          common(
              ruleContext,
              childToolchainConfig,
              intermediateArtifacts,
              propagatedDeps,
              avoidDepsCcInfos,
              avoidDepsObjcProviders);

      ObjcProvider objcProviderWithAvoidDepsSymbols = common.getObjcProvider();
      CcInfo ccInfoWithAvoidDepsSymbols = common.createCcInfo();

      Object linkingInfoProvider;
      if (!childToolchainConfig.getFragment(ObjcConfiguration.class).linkingInfoMigration()) {
        linkingInfoProvider =
            objcProviderWithAvoidDepsSymbols.subtractSubtrees(
                avoidDepsObjcProviders, avoidDepsCcLinkingContexts);
      } else {
        linkingInfoProvider =
            ccLinkingContextSubtractSubtrees(
                ruleContext,
                ImmutableList.of(ccInfoWithAvoidDepsSymbols.getCcLinkingContext()),
                stream(avoidDepsCcInfos)
                    .map(CcInfo::getCcLinkingContext)
                    .collect(toImmutableList()));
      }

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

  /**
   * Returns the ConfigurationDistinguisher that maps directly to the given PlatformType.
   *
   * @throws IllegalArgumentException if the platform type attribute is an unsupported value
   */
  private static ConfigurationDistinguisher configurationDistinguisher(PlatformType platformType) {
    switch (platformType) {
      case IOS:
        return ConfigurationDistinguisher.APPLEBIN_IOS;
      case CATALYST:
        return ConfigurationDistinguisher.APPLEBIN_CATALYST;
      case WATCHOS:
        return ConfigurationDistinguisher.APPLEBIN_WATCHOS;
      case TVOS:
        return ConfigurationDistinguisher.APPLEBIN_TVOS;
      case MACOS:
        return ConfigurationDistinguisher.APPLEBIN_MACOS;
    }
    throw new IllegalArgumentException("Unsupported platform type " + platformType);
  }

  /**
   * Returns the preferred minimum OS version based on information found from inputs.
   *
   * @param buildOptions the build's top-level options
   * @param platformType the platform type attribute found from the given rule being built
   * @param minimumOsVersion a minimum OS version represented to override command line options, if
   *     one has been found
   * @return an {@link DottedVersion.Option} representing the preferred minimum OS version if found,
   *     or null
   * @throws IllegalArgumentException if the platform type attribute is an unsupported value and the
   *     optional minimumOsVersion is not present
   */
  private static DottedVersion.Option minimumOsVersionOption(
      BuildOptionsView buildOptions,
      PlatformType platformType,
      Optional<DottedVersion> minimumOsVersion) {
    if (minimumOsVersion.isPresent()) {
      return DottedVersion.option(minimumOsVersion.get());
    }
    DottedVersion.Option option;
    switch (platformType) {
      case IOS:
      case CATALYST:
        option = buildOptions.get(AppleCommandLineOptions.class).iosMinimumOs;
        break;
      case WATCHOS:
        option = buildOptions.get(AppleCommandLineOptions.class).watchosMinimumOs;
        break;
      case TVOS:
        option = buildOptions.get(AppleCommandLineOptions.class).tvosMinimumOs;
        break;
      case MACOS:
        option = buildOptions.get(AppleCommandLineOptions.class).macosMinimumOs;
        break;
      default:
        throw new IllegalArgumentException("Unsupported platform type " + platformType);
    }
    return option;
  }

  /**
   * Creates a derivative set of build options for the given split transition with default options.
   *
   * @param buildOptions the build's top-level options
   * @param platformType the platform type attribute found from the given rule being built
   * @param minimumOsVersionOption a minimum OS version option for this given split
   * @return an {@link BuildOptionsView} to be used as a basis for a given multi arch binary split
   *     transition
   */
  private static BuildOptionsView defaultBuildOptionsForSplit(
      BuildOptionsView buildOptions,
      PlatformType platformType,
      DottedVersion.Option minimumOsVersionOption) {
    BuildOptionsView splitOptions = buildOptions.clone();

    AppleCommandLineOptions appleCommandLineOptions =
        splitOptions.get(AppleCommandLineOptions.class);
    appleCommandLineOptions.applePlatformType = platformType;
    switch (platformType) {
      case IOS:
      case CATALYST:
        appleCommandLineOptions.iosMinimumOs = minimumOsVersionOption;
        break;
      case WATCHOS:
        appleCommandLineOptions.watchosMinimumOs = minimumOsVersionOption;
        break;
      case TVOS:
        appleCommandLineOptions.tvosMinimumOs = minimumOsVersionOption;
        break;
      case MACOS:
        appleCommandLineOptions.macosMinimumOs = minimumOsVersionOption;
        break;
    }
    return splitOptions;
  }

  /**
   * Creates a split transition mapping based on --apple_platforms and --platforms.
   *
   * @param buildOptions the build's top-level options
   * @param platformType the platform type attribute found from the given rule being built
   * @param minimumOsVersion a minimum OS version represented to override command line options, if
   *     one has been found
   * @param applePlatforms the {@link List} of {@link Label}s representing Apple platforms to split
   *     on
   * @return an {@link ImmutableMap<String, BuildOptions>} representing the split transition for all
   *     platforms found
   */
  public static ImmutableMap<String, BuildOptions> handleApplePlatforms(
      BuildOptionsView buildOptions,
      PlatformType platformType,
      Optional<DottedVersion> minimumOsVersion,
      List<Label> applePlatforms) {
    ImmutableMap.Builder<String, BuildOptions> splitBuildOptions = ImmutableMap.builder();

    ConfigurationDistinguisher configurationDistinguisher =
        configurationDistinguisher(platformType);
    DottedVersion.Option minimumOsVersionOption =
        minimumOsVersionOption(buildOptions, platformType, minimumOsVersion);

    for (Label platform : ImmutableSortedSet.copyOf(applePlatforms)) {
      BuildOptionsView splitOptions =
          defaultBuildOptionsForSplit(buildOptions, platformType, minimumOsVersionOption);

      // Disable multi-platform options for child configurations.
      splitOptions.get(AppleCommandLineOptions.class).applePlatforms = ImmutableList.of();

      // The cpu flag will be set by platform mapping if a mapping exists.
      splitOptions.get(PlatformOptions.class).platforms = ImmutableList.of(platform);
      if (splitOptions.get(ObjcCommandLineOptions.class).enableCcDeps) {
        // Only set the (CC-compilation) configs for dependencies if explicitly required by the
        // user.
        // This helps users of the iOS rules who do not depend on CC rules as these config values
        // require additional flags to work (e.g. a custom crosstool) which now only need to be
        // set if this feature is explicitly requested.
        AppleCrosstoolTransition.setAppleCrosstoolTransitionPlatformConfiguration(
            buildOptions, splitOptions, platform);
      }
      AppleCommandLineOptions appleCommandLineOptions =
          splitOptions.get(AppleCommandLineOptions.class);
      // Set the configuration distinguisher last, as the method
      // setAppleCrosstoolTransitionPlatformConfiguration will set this value to the Apple CROSSTOOL
      // configuration distinguisher, and we want to make sure it's set for the right platform
      // instead in this split transition.
      appleCommandLineOptions.configurationDistinguisher = configurationDistinguisher;

      splitBuildOptions.put(platform.getCanonicalForm(), splitOptions.underlying());
    }

    return splitBuildOptions.buildOrThrow();
  }

  /**
   * Returns a list of supported Apple CPUs given the minimum OS for the target platform.
   *
   * @param minimumOsVersionOption a minimum OS version represented to override command line
   *     options, if one has been found
   * @param cpus list of Apple CPUs requested by the build invocation
   * @param platformType the platform type attribute found from the given rule being built
   * @return a {@link List<String>} representing the supported list of Apple CPUs
   */
  private static List<String> supportedAppleCpusFromMinimumOs(
      DottedVersion.Option minimumOsVersionOption, List<String> cpus, PlatformType platformType) {
    List<String> supportedCpus = cpus;
    DottedVersion actualMinimumOsVersion = DottedVersion.maybeUnwrap(minimumOsVersionOption);
    DottedVersion unsupported32BitMinimumOs;
    switch (platformType) {
      case IOS:
        unsupported32BitMinimumOs = DottedVersion.fromStringUnchecked("11.0");
        break;
      case WATCHOS:
        unsupported32BitMinimumOs = DottedVersion.fromStringUnchecked("9.0");
        break;
      default:
        return supportedCpus;
    }

    if (actualMinimumOsVersion != null
        && actualMinimumOsVersion.compareTo(unsupported32BitMinimumOs) >= 0) {
      ImmutableList<String> non32BitCpus =
          cpus.stream()
              .filter(cpu -> !ApplePlatform.is32Bit(platformType, cpu))
              .collect(toImmutableList());
      if (!non32BitCpus.isEmpty()) {
        supportedCpus = non32BitCpus;
      }
    }
    return supportedCpus;
  }

  /**
   * Creates a split transition mapping based on Apple cpu options.
   *
   * @param buildOptions the build's top-level options
   * @param platformType the platform type attribute found from the given rule being built
   * @param minimumOsVersion a minimum OS version represented to override command line options, if
   *     one has been found
   * @return an {@link ImmutableMap<String, BuildOptions>} representing the split transition for all
   *     architectures found from cpu flags
   */
  public static ImmutableMap<String, BuildOptions> handleAppleCpus(
      BuildOptionsView buildOptions,
      PlatformType platformType,
      Optional<DottedVersion> minimumOsVersion) {
    List<String> cpus;
    ConfigurationDistinguisher configurationDistinguisher =
        configurationDistinguisher(platformType);
    DottedVersion.Option minimumOsVersionOption =
        minimumOsVersionOption(buildOptions, platformType, minimumOsVersion);

    switch (platformType) {
      case IOS:
        cpus = buildOptions.get(AppleCommandLineOptions.class).iosMultiCpus;
        if (cpus.isEmpty()) {
          cpus =
              ImmutableList.of(
                  AppleConfiguration.iosCpuFromCpu(buildOptions.get(CoreOptions.class).cpu));
        }
        cpus = supportedAppleCpusFromMinimumOs(minimumOsVersionOption, cpus, platformType);
        break;
      case WATCHOS:
        cpus = buildOptions.get(AppleCommandLineOptions.class).watchosCpus;
        if (cpus.isEmpty()) {
          cpus = ImmutableList.of(AppleCommandLineOptions.DEFAULT_WATCHOS_CPU);
        }
        cpus = supportedAppleCpusFromMinimumOs(minimumOsVersionOption, cpus, platformType);
        break;
      case TVOS:
        cpus = buildOptions.get(AppleCommandLineOptions.class).tvosCpus;
        if (cpus.isEmpty()) {
          cpus = ImmutableList.of(AppleCommandLineOptions.DEFAULT_TVOS_CPU);
        }
        break;
      case MACOS:
        cpus = buildOptions.get(AppleCommandLineOptions.class).macosCpus;
        if (cpus.isEmpty()) {
          cpus = ImmutableList.of(AppleCommandLineOptions.DEFAULT_MACOS_CPU);
        }
        break;
      case CATALYST:
        cpus = buildOptions.get(AppleCommandLineOptions.class).catalystCpus;
        if (cpus.isEmpty()) {
          cpus = ImmutableList.of(AppleCommandLineOptions.DEFAULT_CATALYST_CPU);
        }
        break;
      default:
        throw new IllegalArgumentException("Unsupported platform type " + platformType);
    }

    // There may be some duplicate flag values.
    cpus = ImmutableSortedSet.copyOf(cpus).asList();
    ImmutableMap.Builder<String, BuildOptions> splitBuildOptions = ImmutableMap.builder();
    for (String cpu : cpus) {
      BuildOptionsView splitOptions =
          defaultBuildOptionsForSplit(buildOptions, platformType, minimumOsVersionOption);

      AppleCommandLineOptions appleCommandLineOptions =
          splitOptions.get(AppleCommandLineOptions.class);

      appleCommandLineOptions.appleSplitCpu = cpu;

      String platformCpu = ApplePlatform.cpuStringForTarget(platformType, cpu);
      if (splitOptions.get(ObjcCommandLineOptions.class).enableCcDeps) {
        // Only set the (CC-compilation) CPU for dependencies if explicitly required by the user.
        // This helps users of the iOS rules who do not depend on CC rules as these CPU values
        // require additional flags to work (e.g. a custom crosstool) which now only need to be
        // set if this feature is explicitly requested.
        AppleCrosstoolTransition.setAppleCrosstoolTransitionCpuConfiguration(
            buildOptions, splitOptions, platformCpu);
      }
      // Set the configuration distinguisher last, as setAppleCrosstoolTransitionCpuConfiguration
      // will set this value to the Apple CROSSTOOL configuration distinguisher, and we want to make
      // sure it's set for the right platform instead in this split transition.
      appleCommandLineOptions.configurationDistinguisher = configurationDistinguisher;

      splitBuildOptions.put(platformCpu, splitOptions.underlying());
    }
    return splitBuildOptions.buildOrThrow();
  }

  private static Iterable<ObjcProvider> getAvoidDepsObjcProviders(
      ImmutableList<TransitiveInfoCollection> transitiveInfoCollections) {
    ImmutableList<ObjcProvider> frameworkObjcProviders =
        getTypedProviders(transitiveInfoCollections, AppleDynamicFrameworkInfo.STARLARK_CONSTRUCTOR)
            .stream()
            .map(frameworkProvider -> frameworkProvider.getDepsObjcProvider())
            .collect(toImmutableList());
    ImmutableList<ObjcProvider> executableObjcProviders =
        getTypedProviders(transitiveInfoCollections, AppleExecutableBinaryInfo.STARLARK_CONSTRUCTOR)
            .stream()
            .map(frameworkProvider -> frameworkProvider.getDepsObjcProvider())
            .collect(toImmutableList());

    return Iterables.concat(
        frameworkObjcProviders,
        executableObjcProviders,
        getTypedProviders(transitiveInfoCollections, ObjcProvider.STARLARK_CONSTRUCTOR));
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
      IntermediateArtifacts intermediateArtifacts,
      List<? extends TransitiveInfoCollection> propagatedDeps,
      Iterable<CcInfo> additionalDepCcInfos,
      Iterable<ObjcProvider> additionalDepObjcProviders)
      throws InterruptedException {

    ObjcCommon.Builder commonBuilder =
        new ObjcCommon.Builder(ObjcCommon.Purpose.LINK_ONLY, ruleContext, buildConfiguration)
            .setCompilationAttributes(
                CompilationAttributes.Builder.fromRuleContext(ruleContext).build())
            .addDeps(propagatedDeps)
            .addCcLinkingContexts(additionalDepCcInfos)
            .addObjcProviders(additionalDepObjcProviders)
            .setIntermediateArtifacts(intermediateArtifacts)
            .setAlwayslink(false);

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
      Map<Optional<String>, List<ConfiguredTargetAndData>> ctads) {
    Dict.Builder<String, StructImpl> result = Dict.builder();
    for (Optional<String> splitTransitionKey : ctads.keySet()) {
      TargetTriplet targetTriplet =
          getTargetTriplet(
              Iterables.getOnlyElement(ctads.get(splitTransitionKey)).getConfiguration());
      result.put(splitTransitionKey.get(), targetTriplet.toStarlarkStruct());
    }
    return result.buildImmutable();
  }
}
