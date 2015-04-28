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

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.Type.LABEL;

import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType;

/**
 * Rule definition for {@code experimental_ios_test} rule in Bazel.
 *
 * <p>Note that this will be renamed to {@code ios_test}, and {@link
 * com.google.devtools.build.lib.bazel.rules.objc.BazelIosTestRule} will be removed when it is
 * slightly more feature complete.
 */
public final class ExperimentalIosTestRule implements RuleDefinition {
  @Override
  public RuleClass build(RuleClass.Builder builder, final RuleDefinitionEnvironment env) {
    return builder
        /*<!-- #BLAZE_RULE(experimental_ios_test).IMPLICIT_OUTPUTS -->
        <ul>
          <li><code><var>name</var>.ipa</code>: the test bundle as an
              <code>.ipa</code> file
          <li><code><var>name</var>.xcodeproj/project.pbxproj: An Xcode project file which can be
              used to develop or build on a Mac.</li>
        </ul>
        <!-- #END_BLAZE_RULE.IMPLICIT_OUTPUTS -->*/
        .setImplicitOutputsFunction(
            ImplicitOutputsFunction.fromFunctions(ReleaseBundlingSupport.IPA, XcodeSupport.PBXPROJ))
        .add(attr("$test_template", LABEL)
            .value(env.getLabel("//tools/objc:ios_test.sh.bazel_template")))
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("experimental_ios_test")
        .type(RuleClassType.TEST)
        .ancestors(BaseRuleClasses.BaseRule.class, BaseRuleClasses.TestBaseRule.class,
            ObjcRuleClasses.IosTestBaseRule.class, ObjcRuleClasses.SimulatorRule.class)
        .factoryClass(ExperimentalIosTest.class)
        .build();
  }
}

/*<!-- #BLAZE_RULE (NAME = experimental_ios_test, TYPE = TEST, FAMILY = Objective-C) -->

${ATTRIBUTE_SIGNATURE}

<p>This rule provides a way to build iOS unit tests written in KIF, GTM and XCTest test frameworks
on both iOS simulator and real devices.
</p>

${ATTRIBUTE_DEFINITION}

<!-- #END_BLAZE_RULE -->*/
