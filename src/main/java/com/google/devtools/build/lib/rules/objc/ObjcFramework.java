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

import static com.google.devtools.build.lib.collect.nestedset.Order.STABLE_ORDER;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.rules.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.rules.objc.ObjcCommon.Builder;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.vfs.PathFragment;

/**
 * Implementation for the {@code objc_framework} rule.
 */
public class ObjcFramework implements RuleConfiguredTargetFactory {
  @Override
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException {
    CompilationAttributes compilationAttributes =
        CompilationAttributes.Builder.fromRuleContext(ruleContext).build();

    ObjcCommon.Builder commonBuilder =
        new Builder(ruleContext)
            .addExtraSdkFrameworks(compilationAttributes.sdkFrameworks())
            .addExtraWeakSdkFrameworks(compilationAttributes.weakSdkFrameworks())
            .addExtraSdkDylibs(compilationAttributes.sdkDylibs());

    ImmutableList<Artifact> frameworkImports =
        ruleContext.getPrerequisiteArtifacts("framework_imports", Mode.TARGET).list();
    if (ruleContext.attributes().get("is_dynamic", Type.BOOLEAN)) {
      commonBuilder.addDynamicFrameworkImports(frameworkImports);
    } else {
      commonBuilder.addStaticFrameworkImports(frameworkImports);
    }

    Iterable<String> containerErrors =
        ObjcCommon.notInContainerErrors(frameworkImports, ObjcCommon.FRAMEWORK_CONTAINER_TYPE);
    for (String error : containerErrors) {
      ruleContext.attributeError("framework_imports", error);
    }

    ObjcProvider objcProvider = commonBuilder.build().getObjcProvider();
    Iterable<PathFragment> frameworkDirs =
        ObjcCommon.uniqueContainers(frameworkImports, ObjcCommon.FRAMEWORK_CONTAINER_TYPE);
    AppleDynamicFrameworkProvider frameworkProvider =
        new AppleDynamicFrameworkProvider((Artifact) null, objcProvider,
            NestedSetBuilder.<PathFragment>linkOrder().addAll(frameworkDirs).build(),
            NestedSetBuilder.<Artifact>linkOrder().addAll(frameworkImports).build());
    NestedSet<Artifact> filesToBuild = NestedSetBuilder.emptySet(STABLE_ORDER);
    return ObjcRuleClasses.ruleConfiguredTarget(ruleContext, filesToBuild)
        .addNativeDeclaredProvider(objcProvider)
        .addNativeDeclaredProvider(frameworkProvider)
        .build();
  }
}
