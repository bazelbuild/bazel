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
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableListMultimap;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.OutputGroupInfo;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.rules.cpp.CcInfo;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration;
import com.google.devtools.build.lib.rules.cpp.CppSemantics;
import com.google.devtools.build.lib.rules.cpp.UserVariablesExtension;
import com.google.devtools.build.lib.rules.objc.AppleDebugOutputsInfo.OutputType;
import com.google.devtools.build.lib.rules.objc.AppleLinkingOutputs.TargetTriplet;
import com.google.devtools.build.lib.rules.objc.CompilationSupport.ExtraLinkArgs;
import com.google.devtools.build.lib.rules.objc.MultiArchBinarySupport.DependencySpecificConfiguration;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetAndData;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import net.starlark.java.eval.EvalException;

/** Native support for Apple binary rules. */
public class AppleBinary {
  private AppleBinary() {}

  /**
   * Links a (potentially multi-architecture) binary targeting Apple platforms.
   *
   * <p>This method comprises a bulk of the logic of the {@code apple_binary} rule, and is
   * statically available so that it may be referenced by Starlark APIs that replicate its
   * functionality.
   *
   * @param ruleContext the current rule context
   * @param cppSemantics the cpp semantics to use
   * @param avoidDeps a list of {@code TransitiveInfoColllection} that contain information about
   *     dependencies whose symbols are used by the linked binary but should not be linked into the
   *     binary itself
   * @param extraLinkopts extra linkopts to pass to the linker action
   * @param extraLinkInputs extra input files to pass to the linker action
   * @param isStampingEnabled whether linkstamping is enabled
   * @param userVariablesExtension the UserVariablesExtension to pass to the linker action
   * @return a tuple containing all necessary information about the linked binary
   */
  public static AppleLinkingOutputs linkMultiArchBinary(
      RuleContext ruleContext,
      CppSemantics cppSemantics,
      ImmutableList<TransitiveInfoCollection> avoidDeps,
      Iterable<String> extraLinkopts,
      Iterable<Artifact> extraLinkInputs,
      Iterable<String> extraRequestedFeatures,
      Iterable<String> extraDisabledFeatures,
      boolean isStampingEnabled,
      UserVariablesExtension userVariablesExtension)
      throws InterruptedException, RuleErrorException, EvalException {
    Map<Optional<String>, List<ConfiguredTargetAndData>> splitDeps =
        ruleContext.getSplitPrerequisiteConfiguredTargetAndTargets("deps");
    Map<Optional<String>, List<ConfiguredTargetAndData>> splitToolchains =
        ruleContext.getSplitPrerequisiteConfiguredTargetAndTargets(
            ObjcRuleClasses.CHILD_CONFIG_ATTR);

    Preconditions.checkState(
        splitDeps.keySet().isEmpty() || splitDeps.keySet().equals(splitToolchains.keySet()),
        "Split transition keys are different between 'deps' [%s] and '%s' [%s]",
        splitDeps.keySet(),
        ObjcRuleClasses.CHILD_CONFIG_ATTR,
        splitToolchains.keySet());

    MultiArchBinarySupport multiArchBinarySupport =
        new MultiArchBinarySupport(ruleContext, cppSemantics);
    ImmutableMap<Optional<String>, DependencySpecificConfiguration>
        dependencySpecificConfigurations =
            multiArchBinarySupport.getDependencySpecificConfigurations(
                splitToolchains, splitDeps, avoidDeps);

    Map<String, NestedSet<Artifact>> outputGroupCollector = new TreeMap<>();

    ImmutableList.Builder<Artifact> allLinkInputs = ImmutableList.builder();
    ImmutableList.Builder<String> allLinkopts = ImmutableList.builder();
    allLinkInputs.addAll(extraLinkInputs);
    allLinkopts.addAll(extraLinkopts);

    ImmutableListMultimap<BuildConfigurationValue, CcInfo> buildConfigToCcInfoMap =
        ruleContext.getPrerequisitesByConfiguration("deps", CcInfo.PROVIDER);
    NestedSetBuilder<Artifact> headerTokens = NestedSetBuilder.stableOrder();
    for (Map.Entry<BuildConfigurationValue, CcInfo> entry : buildConfigToCcInfoMap.entries()) {
      CcInfo dep = entry.getValue();
      headerTokens.addTransitive(dep.getCcCompilationContext().getHeaderTokens());
    }
    outputGroupCollector.put(OutputGroupInfo.VALIDATION, headerTokens.build());

    ObjcProvider.Builder objcProviderBuilder = new ObjcProvider.Builder();
    ImmutableList.Builder<CcInfo> ccInfos = new ImmutableList.Builder<>();
    for (DependencySpecificConfiguration dependencySpecificConfiguration :
        dependencySpecificConfigurations.values()) {
      objcProviderBuilder.addTransitiveAndPropagate(
          dependencySpecificConfiguration.objcProviderWithAvoidDepsSymbols());
      ccInfos.add(dependencySpecificConfiguration.ccInfoWithAvoidDepsSymbols());
    }
    CcInfo ccInfo = CcInfo.merge(ccInfos.build());

    AppleDebugOutputsInfo.Builder legacyDebugOutputsBuilder =
        AppleDebugOutputsInfo.Builder.create();
    AppleLinkingOutputs.Builder builder =
        new AppleLinkingOutputs.Builder().addOutputGroups(outputGroupCollector);

    for (Optional<String> splitTransitionKey : dependencySpecificConfigurations.keySet()) {
      DependencySpecificConfiguration dependencySpecificConfiguration =
          dependencySpecificConfigurations.get(splitTransitionKey);
      BuildConfigurationValue childConfig = dependencySpecificConfiguration.config();
      CppConfiguration childCppConfig = childConfig.getFragment(CppConfiguration.class);
      IntermediateArtifacts intermediateArtifacts =
          new IntermediateArtifacts(ruleContext, childConfig);

      List<? extends TransitiveInfoCollection> propagatedDeps =
          MultiArchBinarySupport.getProvidersFromCtads(splitDeps.get(splitTransitionKey));

      Artifact binaryArtifact =
          multiArchBinarySupport.registerConfigurationSpecificLinkActions(
              dependencySpecificConfiguration,
              new ExtraLinkArgs(allLinkopts.build()),
              allLinkInputs.build(),
              extraRequestedFeatures,
              extraDisabledFeatures,
              isStampingEnabled,
              userVariablesExtension,
              propagatedDeps,
              outputGroupCollector);

      TargetTriplet childTriplet = MultiArchBinarySupport.getTargetTriplet(childConfig);
      AppleLinkingOutputs.LinkingOutput.Builder outputBuilder =
          AppleLinkingOutputs.LinkingOutput.builder()
              .setTargetTriplet(childTriplet)
              .setBinary(binaryArtifact);

      if (childCppConfig.appleGenerateDsym()) {
        Artifact dsymBinary =
            childCppConfig.objcShouldStripBinary()
                ? intermediateArtifacts.dsymSymbolForUnstrippedBinary()
                : intermediateArtifacts.dsymSymbolForStrippedBinary();
        outputBuilder.setDsymBinary(dsymBinary);
        legacyDebugOutputsBuilder.addOutput(
            childTriplet.architecture(), OutputType.DSYM_BINARY, dsymBinary);
      }
      if (childCppConfig.objcGenerateLinkmap()) {
        Artifact linkmap = intermediateArtifacts.linkmap();
        outputBuilder.setLinkmap(linkmap);
        legacyDebugOutputsBuilder.addOutput(
            childTriplet.architecture(), OutputType.LINKMAP, linkmap);
      }

      builder.addOutput(outputBuilder.build());
    }

    return builder
        .setDepsObjcProvider(objcProviderBuilder.build())
        .setDepsCcInfo(ccInfo)
        .setLegacyDebugOutputsProvider(legacyDebugOutputsBuilder.build())
        .build();
  }
}
