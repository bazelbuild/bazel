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

/**
 * Implementation for {@code objc_bundle}.
 */
public class ObjcBundle implements RuleConfiguredTargetFactory {
  @Override
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException {
    ObjcCommon common = new ObjcCommon.Builder(ruleContext).build();

    ImmutableList<Artifact> bundleImports = ruleContext
        .getPrerequisiteArtifacts("bundle_imports", Mode.TARGET).list();
    Iterable<String> bundleImportErrors =
        ObjcCommon.notInContainerErrors(bundleImports, ObjcCommon.BUNDLE_CONTAINER_TYPE);
    for (String error : bundleImportErrors) {
      ruleContext.attributeError("bundle_imports", error);
    }

    NestedSet<Artifact> filesToBuild = NestedSetBuilder.emptySet(STABLE_ORDER);
    return ObjcRuleClasses.ruleConfiguredTarget(ruleContext, filesToBuild)
        .addProvider(ObjcProvider.class, common.getObjcProvider())
        .addNativeDeclaredProvider(common.getObjcProvider())
        .build();
  }
}
