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

import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.ToolchainTransitionMode;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration;
import com.google.devtools.build.lib.rules.cpp.CppRuleClasses;

/**
 * Rule definition for objc_library.
 */
public class ObjcLibraryRule implements RuleDefinition {

  @Override
  public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
    return builder
        .requiresConfigurationFragments(
            ObjcConfiguration.class, AppleConfiguration.class, CppConfiguration.class)
        .cfg(AppleCrosstoolTransition.APPLE_CROSSTOOL_TRANSITION)
        .addRequiredToolchains(CppRuleClasses.ccToolchainTypeAttribute(env))
        .useToolchainTransition(ToolchainTransitionMode.ENABLED)
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("objc_library")
        .factoryClass(ObjcLibrary.class)
        .ancestors(
            BaseRuleClasses.NativeBuildRule.class,
            ObjcRuleClasses.CompilingRule.class,
            ObjcRuleClasses.AlwaysLinkRule.class,
            BaseRuleClasses.MakeVariableExpandingRule.class)
        .build();
  }
}

/*<!-- #BLAZE_RULE (NAME = objc_library, TYPE = LIBRARY, FAMILY = Objective-C) -->

<p>This rule produces a static library from the given Objective-C source files.</p>

${IMPLICIT_OUTPUTS}

<!-- #END_BLAZE_RULE -->*/
