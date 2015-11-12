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
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType;

/**
 * Rule definition for {@code ios_test} rule in Bazel.
 */
public class IosTestRule extends AbstractIosTestRule {
  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("ios_test")
        .type(RuleClassType.TEST)
        .ancestors(
            BaseRuleClasses.BaseRule.class,
            BaseRuleClasses.TestBaseRule.class,
            ObjcRuleClasses.IosTestBaseRule.class,
            ObjcRuleClasses.SimulatorRule.class)
        .factoryClass(IosTest.class)
        .build();
  }
}
