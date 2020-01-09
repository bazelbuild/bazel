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

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.rules.cpp.CcCommon;
import com.google.devtools.build.lib.rules.cpp.CppModuleMap;

/**
 * Implementation for {@code objc_import}.
 */
public class ObjcImport implements RuleConfiguredTargetFactory {
  @Override
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException, ActionConflictException {
    CcCommon.checkRuleLoadedThroughMacro(ruleContext);
    ObjcCommon common =
        new ObjcCommon.Builder(ruleContext)
            .setCompilationAttributes(
                CompilationAttributes.Builder.fromRuleContext(ruleContext).build())
            .setIntermediateArtifacts(ObjcRuleClasses.intermediateArtifacts(ruleContext))
            .setAlwayslink(ruleContext.attributes().get("alwayslink", Type.BOOLEAN))
            .setHasModuleMap()
            .addExtraImportLibraries(
                ruleContext.getPrerequisiteArtifacts("archives", Mode.TARGET).list())
            .build();

    NestedSetBuilder<Artifact> filesToBuild = NestedSetBuilder.stableOrder();

    CompilationAttributes compilationAttributes =
        CompilationAttributes.Builder.fromRuleContext(ruleContext).build();
    IntermediateArtifacts intermediateArtifacts =
        ObjcRuleClasses.intermediateArtifacts(ruleContext);

    NestedSet<Artifact> publicHeaders = compilationAttributes.hdrs();
    CppModuleMap moduleMap = intermediateArtifacts.moduleMap();

    new CompilationSupport.Builder()
        .setRuleContext(ruleContext)
        .build()
        .registerGenerateModuleMapAction(moduleMap, publicHeaders)
        .validateAttributes();

    ObjcProvider objcProvider = common.getObjcProvider();

    return ObjcRuleClasses.ruleConfiguredTarget(ruleContext, filesToBuild.build())
        .addNativeDeclaredProvider(objcProvider)
        .addSkylarkTransitiveInfo(ObjcProvider.SKYLARK_NAME, objcProvider)
        .build();
  }
}
