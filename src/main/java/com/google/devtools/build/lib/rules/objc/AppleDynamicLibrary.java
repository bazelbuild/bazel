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

import static com.google.devtools.build.lib.rules.objc.ObjcProvider.MULTI_ARCH_DYNAMIC_LIBRARIES;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.DylibDependingRule.DYLIBS_ATTR_NAME;

import com.google.common.base.Optional;
import com.google.common.collect.ImmutableListMultimap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;
import com.google.devtools.build.lib.rules.apple.Platform;
import com.google.devtools.build.lib.rules.apple.Platform.PlatformType;
import com.google.devtools.build.lib.rules.cpp.CcToolchainProvider;
import com.google.devtools.build.lib.rules.objc.CompilationSupport.ExtraLinkArgs;
import java.util.Map;
import java.util.Set;

/**
 * Implementation for the "apple_dynamic_library" rule.
 */
public class AppleDynamicLibrary implements RuleConfiguredTargetFactory {

  @Override
  public final ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException {
    PlatformType platformType = MultiArchSplitTransitionProvider.getPlatformType(ruleContext);
    AppleConfiguration appleConfiguration = ruleContext.getFragment(AppleConfiguration.class);

    Platform platform = appleConfiguration.getMultiArchPlatform(platformType);
    ImmutableListMultimap<BuildConfiguration, ObjcProvider> configurationToNonPropagatedObjcMap =
        ruleContext.getPrerequisitesByConfiguration("non_propagated_deps", Mode.SPLIT,
            ObjcProvider.class);
    ImmutableListMultimap<BuildConfiguration, TransitiveInfoCollection> configToDepsCollectionMap =
        ruleContext.getPrerequisitesByConfiguration("deps", Mode.SPLIT);
    Iterable<ObjcProvider> dylibProviders =
        ruleContext.getPrerequisites(DYLIBS_ATTR_NAME, Mode.TARGET, ObjcProvider.class);
    Set<BuildConfiguration> childConfigurations = getChildConfigurations(ruleContext);
    Artifact outputArtifact =
        ObjcRuleClasses.intermediateArtifacts(ruleContext).combinedArchitectureDylib();

    MultiArchBinarySupport multiArchBinarySupport = new MultiArchBinarySupport(ruleContext);
    Map<BuildConfiguration, ObjcProvider> objcProviderByDepConfiguration =
        multiArchBinarySupport.objcProviderByDepConfiguration(
            childConfigurations,
            configToDepsCollectionMap,
            configurationToNonPropagatedObjcMap,
            dylibProviders,
            Optional.<ObjcProvider>absent());

    multiArchBinarySupport.registerActions(
        platform,
        new ExtraLinkArgs("-dynamiclib"),
        objcProviderByDepConfiguration,
        ImmutableSet.<Artifact>of(),
        configToDepsCollectionMap,
        outputArtifact);

    NestedSetBuilder<Artifact> filesToBuild =
        NestedSetBuilder.<Artifact>stableOrder().add(outputArtifact);
    RuleConfiguredTargetBuilder targetBuilder =
        ObjcRuleClasses.ruleConfiguredTarget(ruleContext, filesToBuild.build());

    ObjcProvider.Builder objcProviderBuilder = new ObjcProvider.Builder();
    for (ObjcProvider objcProvider : objcProviderByDepConfiguration.values()) {
      objcProviderBuilder.addTransitiveAndPropagate(objcProvider);
    }
    objcProviderBuilder.add(MULTI_ARCH_DYNAMIC_LIBRARIES, outputArtifact);

    targetBuilder.addProvider(ObjcProvider.class, objcProviderBuilder.build());
    return targetBuilder.build();
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
}
