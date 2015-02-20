// Copyright 2014 Google Inc. All rights reserved.
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

import static com.google.devtools.build.lib.rules.objc.ObjcProvider.MERGE_ZIP;

import com.google.common.base.Optional;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.rules.objc.ReleaseBundlingSupport.LinkedBinary;

/**
 * Implementation for {@code ios_extension}.
 */
public class IosExtension implements RuleConfiguredTargetFactory {

  @Override
  public ConfiguredTarget create(RuleContext ruleContext) throws InterruptedException {
    ObjcCommon common = common(ruleContext);

    XcodeProvider.Builder xcodeProviderBuilder = new XcodeProvider.Builder();
    NestedSetBuilder<Artifact> filesToBuild = NestedSetBuilder.stableOrder();

    ReleaseBundlingSupport releaseBundlingSupport = new ReleaseBundlingSupport(
        ruleContext, common.getObjcProvider(), optionsProvider(ruleContext),
        LinkedBinary.DEPENDENCIES_ONLY, "PlugIns/%s.appex");
    releaseBundlingSupport
        .registerActions()
        .addXcodeSettings(xcodeProviderBuilder)
        .addFilesToBuild(filesToBuild)
        .validateAttributes();

    new XcodeSupport(ruleContext)
        .addFilesToBuild(filesToBuild)
        .addXcodeSettings(
            xcodeProviderBuilder, common.getObjcProvider(), XcodeProductType.EXTENSION)
        .addDependencies(xcodeProviderBuilder, "binary")
        .registerActions(xcodeProviderBuilder.build());

    ObjcProvider nestedBundleProvider = new ObjcProvider.Builder()
        .add(MERGE_ZIP, ruleContext.getImplicitOutputArtifact(ReleaseBundlingSupport.IPA))
        .build();

    return common.configuredTarget(
        filesToBuild.build(),
        Optional.of(xcodeProviderBuilder.build()),
        Optional.of(nestedBundleProvider),
        Optional.<XcTestAppProvider>absent(),
        Optional.<J2ObjcSrcsProvider>absent());
  }

  private OptionsProvider optionsProvider(RuleContext ruleContext) {
    return new OptionsProvider.Builder()
        .addInfoplists(ruleContext.getPrerequisiteArtifacts("infoplist", Mode.TARGET).list())
        .build();
  }

  private ObjcCommon common(RuleContext ruleContext) {
    return new ObjcCommon.Builder(ruleContext)
        .setIntermediateArtifacts(ObjcRuleClasses.intermediateArtifacts(ruleContext))
        .addDepObjcProviders(
            ruleContext.getPrerequisites("binary", Mode.TARGET, ObjcProvider.class))
        .build();
  }
}
