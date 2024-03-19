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
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;
import static com.google.devtools.build.lib.packages.Type.BOOLEAN;
import static com.google.devtools.build.lib.packages.Types.STRING_LIST;

import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.analysis.config.ExecutionTransitionFactory;
import com.google.devtools.build.lib.packages.RuleClass;

/** Rule definition for the {@code android_host_service_fixture} rule. */
public class AndroidHostServiceFixtureRule implements RuleDefinition {

  private final Class<? extends AndroidHostServiceFixture> factoryClass;

  public AndroidHostServiceFixtureRule(Class<? extends AndroidHostServiceFixture> factoryClass) {
    this.factoryClass = factoryClass;
  }

  @Override
  public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
    return builder
        .setUndocumented()
        .add(
            attr("executable", LABEL)
                .exec()
                .cfg(ExecutionTransitionFactory.createFactory())
                .mandatory()
                .allowedFileTypes())
        .add(attr("service_names", STRING_LIST))
        .add(
            attr("support_apks", LABEL_LIST)
                .allowedFileTypes(AndroidRuleClasses.APK)
                .allowedRuleClasses("android_binary"))
        .add(attr("provides_test_args", BOOLEAN).value(false))
        .add(attr("daemon", BOOLEAN).value(false))
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("android_host_service_fixture")
        .ancestors(BaseRuleClasses.NativeActionCreatingRule.class)
        .factoryClass(factoryClass)
        .build();
  }
}
