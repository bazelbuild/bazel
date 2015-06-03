// Copyright 2015 Google Inc. All rights reserved.
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

import static com.google.devtools.build.lib.packages.Attribute.attr;

import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.bazel.rules.cpp.BazelCppRuleClasses;
import com.google.devtools.build.lib.bazel.rules.java.BazelJavaRuleClasses;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.rules.android.AndroidBinaryOnlyRule;
import com.google.devtools.build.lib.rules.android.AndroidRuleClasses;

/**
 * Rule class definition for {@code android_binary}.
 */
public class BazelAndroidBinaryRule implements RuleDefinition {

  @Override
  public RuleClass build(Builder builder, RuleDefinitionEnvironment environment) {
    return builder
        .add(attr(":cc_toolchain_split", Type.LABEL)
            .cfg(AndroidRuleClasses.ANDROID_SPLIT_TRANSITION)
            .value(BazelCppRuleClasses.CC_TOOLCHAIN))
        .setImplicitOutputsFunction(AndroidRuleClasses.ANDROID_BINARY_IMPLICIT_OUTPUTS)
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("android_binary")
        .ancestors(
            AndroidBinaryOnlyRule.class,
            BazelJavaRuleClasses.JavaBaseRule.class,
            BazelCppRuleClasses.CcLinkingRule.class)
        .factoryClass(BazelAndroidBinary.class)
        .build();
  }
}
