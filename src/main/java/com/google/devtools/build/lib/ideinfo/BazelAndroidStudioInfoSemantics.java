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
package com.google.devtools.build.lib.ideinfo;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.ideinfo.AndroidStudioInfoAspect.PrerequisiteAttr;
import com.google.devtools.build.lib.ideinfo.androidstudio.AndroidStudioIdeInfo.CRuleIdeInfo.Builder;
import com.google.devtools.build.lib.ideinfo.androidstudio.AndroidStudioIdeInfo.RuleIdeInfo;
import com.google.devtools.build.lib.rules.cpp.CppCompilationContext;

/**
 * Bazel specific semantics for the {@link AndroidStudioInfoAspect} aspect.
 */
public class BazelAndroidStudioInfoSemantics implements AndroidStudioInfoSemantics {

  @Override
  public void augmentCppRuleInfo(
      Builder builder,
      ConfiguredTarget base,
      RuleContext ruleContext,
      CppCompilationContext cppCompilationContext,
      NestedSetBuilder<Artifact> ideResolveArtifacts) {
  }

  @Override
  public void augmentPrerequisiteAttrs(ImmutableList.Builder<PrerequisiteAttr> builder) {
  }

  @Override
  public void augmentRuleInfo(RuleIdeInfo.Builder builder, ConfiguredTarget base,
      RuleContext ruleContext,
      NestedSetBuilder<Artifact> ideResolveArtifacts) {
  }

  @Override
  public boolean suppressJavaRuleInfo(ConfiguredTarget base) {
    return false;
  }

  @Override
  public boolean checkForAdditionalCppRules(String ruleClass) {
    return false;
  }
}
