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
import static com.google.devtools.build.lib.packages.Type.STRING;

import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.util.FileType;

/** Rule definition for the {@code android_device_script_fixture} rule. */
public class AndroidDeviceScriptFixtureRule implements RuleDefinition {

  static final FileType DEVICE_SCRIPT_FIXTURE = FileType.of(".sh");

  private final Class<? extends AndroidDeviceScriptFixture> factoryClass;

  public AndroidDeviceScriptFixtureRule(Class<? extends AndroidDeviceScriptFixture> factoryClass) {
    this.factoryClass = factoryClass;
  }

  @Override
  public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment environment) {
    return builder
        .setUndocumented()
        .add(
            attr("script", LABEL)
                .exec()
                .allowedFileTypes(DEVICE_SCRIPT_FIXTURE)
                .allowedRuleClasses())
        .add(attr("cmd", STRING))
        .add(
            attr("support_apks", LABEL_LIST)
                .allowedFileTypes(AndroidRuleClasses.APK)
                .allowedRuleClasses("android_binary"))
        .add(attr("daemon", BOOLEAN).value(Boolean.FALSE))
        .add(attr("strict_exit", BOOLEAN).value(Boolean.TRUE))
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("android_device_script_fixture")
        .ancestors(BaseRuleClasses.NativeActionCreatingRule.class)
        .factoryClass(factoryClass)
        .build();
  }
}
