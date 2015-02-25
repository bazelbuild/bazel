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
package com.google.devtools.build.lib.bazel.rules.common;

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.Type.BOOLEAN;
import static com.google.devtools.build.lib.packages.Type.LABEL_LIST;

import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.BlazeRule;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder;
import com.google.devtools.build.lib.rules.test.TestSuite;

/**
 * Rule object implementing "test_suite".
 */
@BlazeRule(name = "test_suite",
             ancestors = { BaseRuleClasses.BaseRule.class },
             factoryClass = TestSuite.class)
public final class BazelTestSuiteRule implements RuleDefinition {
  @Override
  public RuleClass build(Builder builder, RuleDefinitionEnvironment env) {
    return builder
        .override(attr("testonly", BOOLEAN).value(true)
            .nonconfigurable("policy decision: should be consistent across configurations"))
        .add(attr("tests", LABEL_LIST).orderIndependent().allowedFileTypes()
            .nonconfigurable("policy decision: should be consistent across configurations"))
        .add(attr("suites", LABEL_LIST).orderIndependent().allowedFileTypes()
            .nonconfigurable("policy decision: should be consistent across configurations"))
        // This magic attribute contains all *test rules in the package, iff
        // tests=[] and suites=[]:
        .add(attr("$implicit_tests", LABEL_LIST)
            .nonconfigurable("Accessed in TestTargetUtils without config context"))
        .build();
  }
}
