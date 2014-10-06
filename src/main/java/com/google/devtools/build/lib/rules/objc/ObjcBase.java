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

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.view.FilesToRunProvider;
import com.google.devtools.build.lib.view.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.view.RuleContext;

/**
 * Utility code for all rules that inherit from {@link ObjcRuleClasses.ObjcBaseRule}.
 */
final class ObjcBase {
  private ObjcBase() {}

  /**
   * Object that supplies tools used by rules that inherit from
   * {@link ObjcRuleClasses.ObjcBaseRule}.
   */
  static final class Tools {
    private final RuleContext ruleContext;

    Tools(RuleContext ruleContext) {
      this.ruleContext = Preconditions.checkNotNull(ruleContext);
    }

    Artifact momczipDeployJar() {
      return ruleContext.getPrerequisiteArtifact("$momczip_deploy", Mode.HOST);
    }

    FilesToRunProvider xcodegen() {
      return ruleContext.getExecutablePrerequisite("$xcodegen", Mode.HOST);
    }

    FilesToRunProvider plmerge() {
      return ruleContext.getExecutablePrerequisite("$plmerge", Mode.HOST);
    }
  }

  static IntermediateArtifacts intermediateArtifacts(RuleContext ruleContext) {
    return new IntermediateArtifacts(
        ruleContext.getAnalysisEnvironment(), ruleContext.getBinOrGenfilesDirectory(),
        ruleContext.getLabel());
  }

  static ObjcActionsBuilder actionsBuilder(RuleContext ruleContext) {
    return new ObjcActionsBuilder(ruleContext, intermediateArtifacts(ruleContext),
        objcConfiguration(ruleContext), ruleContext.getConfiguration(), ruleContext);
  }

  static ObjcConfiguration objcConfiguration(RuleContext ruleContext) {
    return ruleContext.getConfiguration().getFragment(ObjcConfiguration.class);
  }

  static void registerActions(RuleContext ruleContext, XcodeProvider xcodeProvider) {
    ObjcActionsBuilder actionsBuilder = actionsBuilder(ruleContext);
    Tools tools = new Tools(ruleContext);
    actionsBuilder.registerResourceActions(
        tools,
        new ObjcActionsBuilder.StringsFiles(CompiledResourceFile.stringsFilesFromRule(ruleContext)),
        new ObjcActionsBuilder.XibFiles(CompiledResourceFile.xibFilesFromRule(ruleContext)),
        Xcdatamodels.xcdatamodels(ruleContext));
    actionsBuilder.registerXcodegenActions(
        tools,
        ruleContext.getImplicitOutputArtifact(ObjcRuleClasses.PBXPROJ),
        xcodeProvider);
  }
}
