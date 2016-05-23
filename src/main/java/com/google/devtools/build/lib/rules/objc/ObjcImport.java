// Copyright 2014 The Bazel Authors. All rights reserved.
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
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.rules.objc.ObjcCommon.CompilationAttributes;
import com.google.devtools.build.lib.rules.objc.ObjcCommon.ResourceAttributes;
import com.google.devtools.build.lib.syntax.Type;

/**
 * Implementation for {@code objc_import}.
 */
public class ObjcImport implements RuleConfiguredTargetFactory {
  @Override
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException {
    ObjcCommon common =
        new ObjcCommon.Builder(ruleContext)
            .setCompilationAttributes(new CompilationAttributes(ruleContext))
            .setResourceAttributes(new ResourceAttributes(ruleContext))
            .setIntermediateArtifacts(ObjcRuleClasses.intermediateArtifacts(ruleContext))
            .setAlwayslink(ruleContext.attributes().get("alwayslink", Type.BOOLEAN))
            .setHasModuleMap()
            .addExtraImportLibraries(
                ruleContext.getPrerequisiteArtifacts("archives", Mode.TARGET).list())
            .addDepObjcProviders(
                ruleContext.getPrerequisites("bundles", Mode.TARGET, ObjcProvider.class))
            .build();

    XcodeProvider.Builder xcodeProviderBuilder = new XcodeProvider.Builder();
    NestedSetBuilder<Artifact> filesToBuild = NestedSetBuilder.stableOrder();

    new CompilationSupport(ruleContext)
        .registerGenerateModuleMapAction(Optional.<CompilationArtifacts>absent())
        .addXcodeSettings(xcodeProviderBuilder, common)
        .validateAttributes();

    new ResourceSupport(ruleContext)
        .validateAttributes()
        .addXcodeSettings(xcodeProviderBuilder);

    new XcodeSupport(ruleContext)
        .addXcodeSettings(xcodeProviderBuilder, common.getObjcProvider(), LIBRARY_STATIC)
        .addDependencies(xcodeProviderBuilder, new Attribute("bundles", Mode.TARGET))
        .registerActions(xcodeProviderBuilder.build())
        .addFilesToBuild(filesToBuild);

    return ObjcRuleClasses.ruleConfiguredTarget(ruleContext, filesToBuild.build())
        .addProvider(XcodeProvider.class, xcodeProviderBuilder.build())
        .addProvider(ObjcProvider.class, common.getObjcProvider())
        .build();
  }
}
