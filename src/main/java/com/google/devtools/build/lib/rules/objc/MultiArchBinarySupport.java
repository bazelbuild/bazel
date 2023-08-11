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
import com.google.devtools.build.lib.packages.StarlarkInfo;
import com.google.devtools.build.lib.packages.StarlarkProvider;
import com.google.devtools.build.lib.packages.StarlarkProviderIdentifier;
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
import com.google.devtools.build.lib.rules.cpp.CppOptions;
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
      case VISIONOS:
        return ConfigurationDistinguisher.APPLEBIN_VISIONOS;
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
      case VISIONOS:
        // TODO: Replace with CppOptions.minimumOsVersion
        option = DottedVersion.option(DottedVersion.fromStringUnchecked("1.0"));
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
      case VISIONOS:
        // TODO: use CppOptions.minimumOsVersion
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
      setAppleCrosstoolTransitionPlatformConfiguration(buildOptions, splitOptions, platform);
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
      case VISIONOS:
        cpus = buildOptions.get(AppleCommandLineOptions.class).visionosCpus;
        if (cpus.isEmpty()) {
          cpus = ImmutableList.of(AppleCommandLineOptions.DEFAULT_VISIONOS_CPU);
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
      setAppleCrosstoolTransitionCpuConfiguration(buildOptions, splitOptions, platformCpu);
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

  /**
   * Sets configuration fields required for a transition that uses apple_crosstool_top in place of
   * the default CROSSTOOL.
   *
   * @param from options from the originating configuration
   * @param to options for the destination configuration. This instance will be modified to so the
   *     destination configuration uses the apple crosstool
   * @param cpu {@code --cpu} value for toolchain selection in the destination configuration
   */
  private static void setAppleCrosstoolTransitionCpuConfiguration(
      BuildOptionsView from, BuildOptionsView to, String cpu) {
    AppleCommandLineOptions appleOptions = from.get(AppleCommandLineOptions.class);

    CoreOptions toOptions = to.get(CoreOptions.class);
    CppOptions toCppOptions = to.get(CppOptions.class);

    if (toOptions.cpu.equals(cpu)
        && toCppOptions.crosstoolTop.equals(appleOptions.appleCrosstoolTop)) {
      // If neither the CPU nor the Crosstool changes, do nothing. This is so that C++ to
      // Objective-C dependencies work if the top-level configuration is already an Apple one.
      // Removing the configuration distinguisher (which can't be set from the command line) and
      // putting the platform type in the output directory name, which would obviate the need for
      // this hack.
      // TODO(b/112834725): Remove this branch by unifying the distinguisher and the platform type.
      return;
    }

    toOptions.cpu = cpu;
    toCppOptions.crosstoolTop = appleOptions.appleCrosstoolTop;

    setAppleCrosstoolTransitionSharedConfiguration(from, to);

    // Ensure platforms aren't set so that platform mapping can take place.
    to.get(PlatformOptions.class).platforms = ImmutableList.of();
  }

  /**
   * Sets configuration fields required for a transition that uses apple_platforms in place of the
   * default platforms to find the appropriate CROSSTOOL and C++ configuration options.
   *
   * @param from options from the originating configuration
   * @param to options for the destination configuration. This instance will be modified to so the
   *     destination configuration uses the apple crosstool
   * @param platform {@code --platforms} value for toolchain selection in the destination
   *     configuration
   */
  private static void setAppleCrosstoolTransitionPlatformConfiguration(
      BuildOptionsView from, BuildOptionsView to, Label platform) {
    PlatformOptions toPlatformOptions = to.get(PlatformOptions.class);
    ImmutableList<Label> incomingPlatform = ImmutableList.of(platform);

    if (toPlatformOptions.platforms.equals(incomingPlatform)) {
      // If the incoming platform doesn't change, do nothing. This is so that C++ to Objective-C
      // dependencies work if the top-level configuration is already an Apple one.
      // Removing the configuration distinguisher (which can't be set from the command line) and
      // putting the platform type in the output directory name, which would obviate the need for
      // this hack.
      // TODO(b/112834725): Remove this branch by unifying the distinguisher and the platform type.
      return;
    }

    // The cpu flag will be set by platform mapping if a mapping exists.
    to.get(PlatformOptions.class).platforms = incomingPlatform;

    setAppleCrosstoolTransitionSharedConfiguration(from, to);
  }

  /**
   * Sets a common set of configuration fields required for a transition that needs to find the
   * appropriate CROSSTOOL and C++ configuration options.
   *
   * @param from options from the originating configuration
   * @param to options for the destination configuration. This instance will be modified to so the
   *     destination configuration uses the apple crosstool
   */
  private static void setAppleCrosstoolTransitionSharedConfiguration(
      BuildOptionsView from, BuildOptionsView to) {
    to.get(AppleCommandLineOptions.class).configurationDistinguisher =
        ConfigurationDistinguisher.APPLE_CROSSTOOL;

    AppleCommandLineOptions appleOptions = from.get(AppleCommandLineOptions.class);
    CppOptions toCppOptions = to.get(CppOptions.class);
    toCppOptions.cppCompiler = null;
    toCppOptions.libcTopLabel = appleOptions.appleLibcTop;

    // OSX toolchains do not support fission.
    toCppOptions.fissionModes = ImmutableList.of();
  }
}
