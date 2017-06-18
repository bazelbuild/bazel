// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules.android;

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;

import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder;

/** Rule definition for the {@code android_instrumentation} rule. */
public class AndroidInstrumentationRule implements RuleDefinition {
  @Override
  public RuleClass build(Builder builder, RuleDefinitionEnvironment environment) {
    return builder
        .setUndocumented()
        .add(
            attr("target", LABEL)
                .allowedFileTypes(AndroidRuleClasses.APK)
                .allowedRuleClasses("android_binary"))
        .add(
            attr("target_library", LABEL)
                .allowedFileTypes()
                .allowedRuleClasses("android_library"))
        .add(
            attr("instrumentation", LABEL)
                .allowedFileTypes(AndroidRuleClasses.APK)
                .allowedRuleClasses("android_binary"))
        .add(
            attr("instrumentation_library", LABEL)
                .allowedFileTypes()
                .allowedRuleClasses("android_library"))
        .setImplicitOutputsFunction(AndroidInstrumentation.IMPLICIT_OUTPUTS_FUNCTION)
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("android_instrumentation")
        .ancestors(BaseRuleClasses.RuleBase.class)
        .factoryClass(AndroidInstrumentation.class)
        .build();
  }
}
