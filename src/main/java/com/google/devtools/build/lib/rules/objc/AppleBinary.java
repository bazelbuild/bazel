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

import static com.google.devtools.build.lib.rules.objc.ObjcProvider.MULTI_ARCH_LINKED_BINARIES;
import static com.google.devtools.build.lib.syntax.Type.STRING;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Optional;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableListMultimap;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.Attribute.SplitTransition;
import com.google.devtools.build.lib.packages.Attribute.SplitTransitionProvider;
import com.google.devtools.build.lib.packages.NonconfigurableAttributeMapper;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.rules.apple.AppleCommandLineOptions;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration.ConfigurationDistinguisher;
import com.google.devtools.build.lib.rules.apple.Platform.PlatformType;
import com.google.devtools.build.lib.rules.cpp.CcToolchainProvider;
import com.google.devtools.build.lib.rules.objc.CompilationSupport.ExtraLinkArgs;
import com.google.devtools.build.lib.rules.objc.ObjcCommon.ResourceAttributes;
import java.util.List;
import java.util.Set;

/**
 * Implementation for the "apple_binary" rule.
 */
public class AppleBinary implements RuleConfiguredTargetFactory {

  /**
   * {@link SplitTransitionProvider} instance for the apple binary rule. (This is exposed for
   * convenience as a single static instance as it possesses no internal state.)
   */
  public static final AppleBinaryTransitionProvider SPLIT_TRANSITION_PROVIDER =
      new AppleBinaryTransitionProvider();

  @VisibleForTesting
  static final String REQUIRES_AT_LEAST_ONE_SOURCE_FILE =
      "At least one source file is required (srcs, non_arc_srcs, or precompiled_srcs).";

  @VisibleForTesting
  static final String UNSUPPORTED_PLATFORM_TYPE_ERROR_FORMAT =
      "Unsupported platform type \"%s\"";

  private static final ImmutableSet<PlatformType> SUPPORTED_APPLE_BINARY_PLATFORM_TYPES =
      ImmutableSet.of(PlatformType.IOS, PlatformType.WATCHOS);

  @Override
  public final ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException {
    PlatformType platformType = getPlatformType(ruleContext);
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

    ObjcProvider.Builder objcProviderBuilder = new ObjcProvider.Builder();

    for (BuildConfiguration childConfig : childConfigurations) {
      Optional<ObjcProvider> protosObjcProvider = Optional.absent();
      ObjcConfiguration objcConfiguration = childConfig.getFragment(ObjcConfiguration.class);
      if (objcConfiguration.experimentalAutoTopLevelUnionObjCProtos()) {
        ProtobufSupport protoSupport =
            new ProtobufSupport(ruleContext)
                .registerGenerationActions()
                .registerCompilationActions();

        protosObjcProvider = Optional.of(protoSupport.getObjcProvider());
      }

      IntermediateArtifacts intermediateArtifacts =
          ObjcRuleClasses.intermediateArtifacts(ruleContext, childConfig);

      ObjcCommon common =
          common(
              ruleContext,
              childConfig,
              intermediateArtifacts,
              nullToEmptyList(configToDepsCollectionMap.get(childConfig)),
              nullToEmptyList(configurationToNonPropagatedObjcMap.get(childConfig)),
              protosObjcProvider);
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

      new CompilationSupport(ruleContext, childConfig)
          .registerCompileAndArchiveActions(common)
          .registerLinkActions(
              common.getObjcProvider(),
              j2ObjcMappingFileProvider,
              j2ObjcEntryClassProvider,
              new ExtraLinkArgs(),
              ImmutableList.<Artifact>of(),
              DsymOutputType.APP)
          .validateAttributes();
      ruleContext.assertNoErrors();

      objcProviderBuilder.addTransitiveAndPropagate(common.getObjcProvider());
    }

    AppleConfiguration appleConfiguration = ruleContext.getFragment(AppleConfiguration.class);

    new LipoSupport(ruleContext)
        .registerCombineArchitecturesAction(
            binariesToLipo.build(),
            ruleIntermediateArtifacts.combinedArchitectureBinary(),
            appleConfiguration.getMultiArchPlatform(platformType));

    RuleConfiguredTargetBuilder targetBuilder =
        ObjcRuleClasses.ruleConfiguredTarget(ruleContext, filesToBuild.build());

    objcProviderBuilder.add(
        MULTI_ARCH_LINKED_BINARIES, ruleIntermediateArtifacts.combinedArchitectureBinary());

    targetBuilder.addProvider(ObjcProvider.class, objcProviderBuilder.build());
    return targetBuilder.build();
  }

  private ObjcCommon common(
      RuleContext ruleContext,
      BuildConfiguration buildConfiguration,
      IntermediateArtifacts intermediateArtifacts,
      List<TransitiveInfoCollection> propagatedDeps,
      List<ObjcProvider> nonPropagatedObjcDeps,
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
            ruleContext.getPrerequisites("bundles", Mode.TARGET, ObjcProvider.class))
        .addDepObjcProviders(protosObjcProvider.asSet())
        .addNonPropagatedDepObjcProviders(nonPropagatedObjcDeps)
        .setIntermediateArtifacts(intermediateArtifacts)
        .setAlwayslink(false)
        // TODO(b/29152500): Enable module map generation.
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

  private static PlatformType getPlatformType(RuleContext ruleContext) throws RuleErrorException {
    try {
      return getPlatformType(
          ruleContext.attributes().get(AppleBinaryRule.PLATFORM_TYPE_ATTR_NAME, STRING));
    } catch (IllegalArgumentException exception) {
      throw ruleContext.throwWithAttributeError(AppleBinaryRule.PLATFORM_TYPE_ATTR_NAME,
          String.format(UNSUPPORTED_PLATFORM_TYPE_ERROR_FORMAT,
              ruleContext.attributes().get(AppleBinaryRule.PLATFORM_TYPE_ATTR_NAME, STRING)));
    }
  }

  private static PlatformType getPlatformType(String platformTypeString) {
    PlatformType platformType = PlatformType.fromString(platformTypeString);

    if (!SUPPORTED_APPLE_BINARY_PLATFORM_TYPES.contains(platformType)) {
      throw new IllegalArgumentException(
          String.format(UNSUPPORTED_PLATFORM_TYPE_ERROR_FORMAT, platformTypeString));
    } else {
      return platformType;
    }
  }

  /**
   * {@link SplitTransitionProvider} implementation for the apple binary rule.
   */
  public static class AppleBinaryTransitionProvider implements SplitTransitionProvider {

    private static final ImmutableMap<PlatformType, AppleBinaryTransition>
        SPLIT_TRANSITIONS_BY_TYPE = ImmutableMap.<PlatformType, AppleBinaryTransition>builder()
            .put(PlatformType.IOS, new AppleBinaryTransition(PlatformType.IOS))
            .put(PlatformType.WATCHOS, new AppleBinaryTransition(PlatformType.WATCHOS))
            .build();

    @Override
    public SplitTransition<?> apply(Rule fromRule) {
      String platformTypeString = NonconfigurableAttributeMapper.of(fromRule)
          .get(AppleBinaryRule.PLATFORM_TYPE_ATTR_NAME, STRING);
      PlatformType platformType;
      try {
        platformType = getPlatformType(platformTypeString);
      } catch (IllegalArgumentException exception) {
        // There's no opportunity to propagate exception information up cleanly at the transition
        // provider level. This will later be registered as a rule error during the initialization
        // of the apple_binary target.
        platformType = PlatformType.IOS;
      }

      return SPLIT_TRANSITIONS_BY_TYPE.get(platformType);
    }

    public List<SplitTransition<BuildOptions>> getPotentialSplitTransitions() {
      return ImmutableList.<SplitTransition<BuildOptions>>copyOf(
          SPLIT_TRANSITIONS_BY_TYPE.values());
    }
  }

  /**
   * Transition that results in one configured target per architecture specified in {@code
   * --ios_multi_cpus}.
   */
  protected static class AppleBinaryTransition implements SplitTransition<BuildOptions> {

    private final PlatformType platformType;

    public AppleBinaryTransition(PlatformType platformType) {
      this.platformType = platformType;
    }

    @Override
    public final List<BuildOptions> split(BuildOptions buildOptions) {
      List<String> cpus;
      ConfigurationDistinguisher configurationDistinguisher;
      switch (platformType) {
        case IOS:
          cpus = buildOptions.get(AppleCommandLineOptions.class).iosMultiCpus;
          configurationDistinguisher = ConfigurationDistinguisher.APPLEBIN_IOS;
          break;
        case WATCHOS:
          cpus = buildOptions.get(AppleCommandLineOptions.class).watchosCpus;
          if (cpus.isEmpty()) {
            cpus = ImmutableList.of(AppleCommandLineOptions.DEFAULT_WATCHOS_CPU);
          }
          configurationDistinguisher = ConfigurationDistinguisher.APPLEBIN_WATCHOS;
          break;
        default:
          throw new IllegalArgumentException("Unsupported platform type " + platformType);
      }

      ImmutableList.Builder<BuildOptions> splitBuildOptions = ImmutableList.builder();
      for (String cpu : cpus) {
        BuildOptions splitOptions = buildOptions.clone();

        splitOptions.get(AppleCommandLineOptions.class).applePlatformType = platformType;
        splitOptions.get(AppleCommandLineOptions.class).appleSplitCpu = cpu;
        // Set for backwards compatibility with rules that depend on this flag, even when
        // ios is not the platform type.
        // TODO(b/28958783): Clean this up.
        splitOptions.get(AppleCommandLineOptions.class).iosCpu = cpu;
        if (splitOptions.get(ObjcCommandLineOptions.class).enableCcDeps) {
          // Only set the (CC-compilation) CPU for dependencies if explicitly required by the user.
          // This helps users of the iOS rules who do not depend on CC rules as these CPU values
          // require additional flags to work (e.g. a custom crosstool) which now only need to be
          // set if this feature is explicitly requested.
          splitOptions.get(BuildConfiguration.Options.class).cpu =
              String.format("%s_%s", platformType, cpu);
        }
        splitOptions.get(AppleCommandLineOptions.class).configurationDistinguisher =
            configurationDistinguisher;
        splitBuildOptions.add(splitOptions);
      }
      return splitBuildOptions.build();
    }

    @Override
    public boolean defaultsToSelf() {
      return true;
    }
  }
}
