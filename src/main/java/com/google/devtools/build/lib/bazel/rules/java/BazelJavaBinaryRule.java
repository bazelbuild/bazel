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
import static com.google.devtools.build.lib.packages.Type.BOOLEAN;

import com.google.devtools.build.lib.analysis.BlazeRule;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.bazel.rules.BazelBaseRuleClasses;
import com.google.devtools.build.lib.bazel.rules.java.BazelJavaRuleClasses.BaseJavaBinaryRule;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder;

/**
 * Rule definition for the java_binary rule.
 */
@BlazeRule(name = "java_binary",
             ancestors = { BaseJavaBinaryRule.class,
                           BazelBaseRuleClasses.BinaryBaseRule.class },
             factoryClass = BazelJavaBinary.class)
public final class BazelJavaBinaryRule implements RuleDefinition {
  @Override
  public RuleClass build(Builder builder, RuleDefinitionEnvironment env) {
    return builder
        .setImplicitOutputsFunction(BazelJavaRuleClasses.JAVA_BINARY_IMPLICIT_OUTPUTS)
        .override(attr("$is_executable", BOOLEAN).nonconfigurable("automatic").value(
            new Attribute.ComputedDefault() {
              @Override
              public Object getDefault(AttributeMap rule) {
                return rule.get("create_executable", BOOLEAN);
              }
            }))
        .build();
  }
}
