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

import static com.google.devtools.build.lib.rules.objc.XcodeProductType.LIBRARY_STATIC;

import com.google.common.base.Optional;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.view.ConfiguredTarget;
import com.google.devtools.build.lib.view.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.view.RuleContext;

/**
 * Implementation for {@code objc_import}.
 */
public class ObjcImport implements RuleConfiguredTargetFactory {
  @Override
  public ConfiguredTarget create(RuleContext ruleContext) throws InterruptedException {
    ObjcCommon common = new ObjcCommon.Builder(ruleContext)
        .addAssetCatalogs(ruleContext.getPrerequisiteArtifacts("asset_catalogs", Mode.TARGET))
        .addSdkDylibs(ruleContext.attributes().get("sdk_dylibs", Type.STRING_LIST))
        .addHdrs(ruleContext.getPrerequisiteArtifacts("hdrs", Mode.TARGET))
        .addStoryboardInputs(ruleContext.getPrerequisiteArtifacts("storyboards", Mode.TARGET))
        .setIntermediateArtifacts(ObjcBase.intermediateArtifacts(ruleContext))
        .build();
    common.reportErrors();

    OptionsProvider optionsProvider = OptionsProvider.DEFAULT;
    XcodeProvider xcodeProvider = new XcodeProvider.Builder()
        .setLabel(ruleContext.getLabel())
        .addUserHeaderSearchPaths(ObjcCommon.userHeaderSearchPaths(ruleContext.getConfiguration()))
        .addCopts(optionsProvider.getCopts())
        .setProductType(LIBRARY_STATIC)
        .addHeaders(common.getHdrs())
        .setObjcProvider(common.getObjcProvider())
        .build();

    ObjcBase.registerActions(ruleContext, xcodeProvider, common.getStoryboards());

    return common.configuredTarget(
        NestedSetBuilder.<Artifact>stableOrder()
            .add(ruleContext.getImplicitOutputArtifact(ObjcRuleClasses.PBXPROJ))
            .build(),
        Optional.of(xcodeProvider),
        Optional.of(common.getObjcProvider()));
  }
}
