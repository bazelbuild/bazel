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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Optional;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableListMultimap;
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
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.rules.apple.AppleCommandLineOptions;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration.ConfigurationDistinguisher;
import com.google.devtools.build.lib.rules.apple.Platform.PlatformType;
import com.google.devtools.build.lib.rules.cpp.CcToolchainProvider;
import com.google.devtools.build.lib.rules.objc.CompilationSupport.ExtraLinkArgs;
import com.google.devtools.build.lib.rules.objc.ObjcCommon.ResourceAttributes;
import com.google.devtools.build.lib.rules.objc.ProtoSupport.TargetType;

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

  @Override
  public final ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException {
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
    NestedSetBuilder<Artifact> archivesToLipo =
        NestedSetBuilder.<Artifact>stableOrder();
    NestedSetBuilder<Artifact> filesToBuild =
        NestedSetBuilder.<Artifact>stableOrder()
            .add(ruleIntermediateArtifacts.combinedArchitectureBinary());

    for (BuildConfiguration childConfig : childConfigurations) {
      IntermediateArtifacts intermediateArtifacts =
          ObjcRuleClasses.intermediateArtifacts(ruleContext, childConfig);
      ObjcCommon common = common(ruleContext, childConfig, intermediateArtifacts,
          nullToEmptyList(configToDepsCollectionMap.get(childConfig)),
          nullToEmptyList(configurationToNonPropagatedObjcMap.get(childConfig)));
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

      if (!common.getCompilationArtifacts().get().getArchive().isPresent()) {
        ruleContext.throwWithRuleError(REQUIRES_AT_LEAST_ONE_SOURCE_FILE);
      }

      archivesToLipo.add(common.getCompilationArtifacts().get().getArchive().get());
      binariesToLipo.add(intermediateArtifacts.strippedSingleArchitectureBinary());

      ObjcConfiguration objcConfiguration = childConfig.getFragment(ObjcConfiguration.class);
      if (objcConfiguration.experimentalAutoTopLevelUnionObjCProtos()) {
        ProtoSupport protoSupport =
            new ProtoSupport(ruleContext, TargetType.LINKING_TARGET).registerActions();

        ObjcCommon protoCommon = protoSupport.getCommon();
        new CompilationSupport(
                ruleContext,
                protoSupport.getIntermediateArtifacts(),
                new CompilationAttributes.Builder().build())
            .registerCompileAndArchiveActions(protoCommon, protoSupport.getUserHeaderSearchPaths());
      }

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
    }

    AppleConfiguration appleConfiguration = ruleContext.getFragment(AppleConfiguration.class);

    new LipoSupport(ruleContext)
        .registerCombineArchitecturesAction(
            binariesToLipo.build(),
            ruleIntermediateArtifacts.combinedArchitectureBinary(),
            appleConfiguration.getMultiArchPlatform(PlatformType.IOS))
        .registerCombineArchitecturesAction(
            archivesToLipo.build(),
            ruleContext.getImplicitOutputArtifact(AppleBinaryRule.LIPO_ARCHIVE),
            appleConfiguration.getMultiArchPlatform(PlatformType.IOS));

    RuleConfiguredTargetBuilder targetBuilder =
        ObjcRuleClasses.ruleConfiguredTarget(ruleContext, filesToBuild.build());

    return targetBuilder.build();
  }

  private ObjcCommon common(RuleContext ruleContext, BuildConfiguration buildConfiguration,
      IntermediateArtifacts intermediateArtifacts,
      List<TransitiveInfoCollection> propagatedDeps,
      List<ObjcProvider> nonPropagatedObjcDeps) {

    CompilationArtifacts compilationArtifacts =
        CompilationSupport.compilationArtifacts(ruleContext, intermediateArtifacts);

    Optional<Artifact> protoLib;
    ObjcConfiguration objcConfiguration = buildConfiguration.getFragment(ObjcConfiguration.class);
    if (objcConfiguration.experimentalAutoTopLevelUnionObjCProtos()) {
      ProtoSupport protoSupport = new ProtoSupport(ruleContext, TargetType.LINKING_TARGET);
      protoLib = protoSupport.getCommon().getCompiledArchive();
    } else {
      protoLib = Optional.absent();
    }

    return new ObjcCommon.Builder(ruleContext, buildConfiguration)
        .setCompilationAttributes(
            CompilationAttributes.Builder.fromRuleContext(ruleContext).build())
        .setCompilationArtifacts(compilationArtifacts)
        .setResourceAttributes(new ResourceAttributes(ruleContext))
        .addDefines(ruleContext.getTokenizedStringListAttr("defines"))
        .addDeps(propagatedDeps)
        .addDepObjcProviders(
            ruleContext.getPrerequisites("bundles", Mode.TARGET, ObjcProvider.class))
        .addNonPropagatedDepObjcProviders(nonPropagatedObjcDeps)
        .setIntermediateArtifacts(intermediateArtifacts)
        .setAlwayslink(false)
        .setHasModuleMap()
        .setLinkedBinary(intermediateArtifacts.strippedSingleArchitectureBinary())
        .addExtraImportLibraries(protoLib.asSet())
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

  /**
   * {@link SplitTransitionProvider} implementation for the apple binary rule.
   */
  public static class AppleBinaryTransitionProvider implements SplitTransitionProvider {

    private static final IosMultiCpusTransition IOS_MULTI_CPUS_SPLIT_TRANSITION =
        new IosMultiCpusTransition();

    @Override
    public SplitTransition<?> apply(Rule fromRule) {
      // TODO(cparsons): Support different split transitions based on rule attribute.
      return IOS_MULTI_CPUS_SPLIT_TRANSITION;
    }

    public List<SplitTransition<BuildOptions>> getPotentialSplitTransitions() {
      return ImmutableList.<SplitTransition<BuildOptions>>of(IOS_MULTI_CPUS_SPLIT_TRANSITION);
    }
  }

  /**
   * Transition that results in one configured target per architecture specified in {@code
   * --ios_multi_cpus}.
   */
  protected static class IosMultiCpusTransition implements SplitTransition<BuildOptions> {

    @Override
    public final List<BuildOptions> split(BuildOptions buildOptions) {
      List<String> iosMultiCpus = buildOptions.get(AppleCommandLineOptions.class).iosMultiCpus;

      ImmutableList.Builder<BuildOptions> splitBuildOptions = ImmutableList.builder();
      for (String iosCpu : iosMultiCpus) {
        BuildOptions splitOptions = buildOptions.clone();

        splitOptions.get(AppleCommandLineOptions.class).iosMultiCpus = ImmutableList.of();
        splitOptions.get(AppleCommandLineOptions.class).applePlatformType = PlatformType.IOS;
        splitOptions.get(AppleCommandLineOptions.class).appleSplitCpu = iosCpu;
        splitOptions.get(AppleCommandLineOptions.class).iosCpu = iosCpu;
        if (splitOptions.get(ObjcCommandLineOptions.class).enableCcDeps) {
          // Only set the (CC-compilation) CPU for dependencies if explicitly required by the user.
          // This helps users of the iOS rules who do not depend on CC rules as these CPU values
          // require additional flags to work (e.g. a custom crosstool) which now only need to be
          // set if this feature is explicitly requested.
          splitOptions.get(BuildConfiguration.Options.class).cpu = "ios_" + iosCpu;
        }
        splitOptions.get(AppleCommandLineOptions.class).configurationDistinguisher =
            ConfigurationDistinguisher.APPLEBIN_IOS;
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
