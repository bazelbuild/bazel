// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.bazel.rules.android;

import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType;
import com.google.devtools.build.lib.rules.android.AndroidInstrumentationTestBaseRule;
import com.google.devtools.build.lib.rules.android.AndroidRuleClasses;

/** Rule definition for Bazel android_instrumentation_test. */
public final class BazelAndroidInstrumentationTestRule implements RuleDefinition {

  @Override
  public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment environment) {
    return builder
        .removeAttribute("deps")
        .removeAttribute("javacopts")
        .removeAttribute("plugins")
        .removeAttribute(":java_plugins")
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("android_instrumentation_test")
        .type(RuleClassType.TEST)
        .ancestors(
            AndroidInstrumentationTestBaseRule.class,
            AndroidRuleClasses.AndroidBaseRule.class,
            BaseRuleClasses.TestBaseRule.class)
        .factoryClass(BazelAndroidInstrumentationTest.class)
        .build();
  }
}
