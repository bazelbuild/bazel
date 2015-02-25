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

package com.google.devtools.build.lib.bazel.rules.java;

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.Type.LABEL;
import static com.google.devtools.build.lib.packages.Type.STRING;
import static com.google.devtools.build.lib.packages.Type.TRISTATE;

import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.BlazeRule;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.bazel.rules.java.BazelJavaRuleClasses.BaseJavaBinaryRule;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder;
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType;
import com.google.devtools.build.lib.packages.TriState;
import com.google.devtools.build.lib.rules.java.JavaSemantics;

/**
 * Rule definition for the java_test rule.
 */
@BlazeRule(name = "java_test",
             type = RuleClassType.TEST,
             ancestors = { BaseJavaBinaryRule.class,
                           BaseRuleClasses.TestBaseRule.class },
             factoryClass = BazelJavaTest.class)
public final class BazelJavaTestRule implements RuleDefinition {
  
  private static final String JUNIT4_RUNNER = "org.junit.runner.JUnitCore";

  @Override
  public RuleClass build(Builder builder, RuleDefinitionEnvironment env) {
    return builder
        .setImplicitOutputsFunction(BazelJavaRuleClasses.JAVA_BINARY_IMPLICIT_OUTPUTS)
        .override(attr("main_class", STRING).value(JUNIT4_RUNNER))
        .override(attr("stamp", TRISTATE).value(TriState.NO))
        .override(attr(":java_launcher", LABEL).value(JavaSemantics.JAVA_LAUNCHER))
        .build();
  }
}