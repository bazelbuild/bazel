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
import static com.google.devtools.build.lib.syntax.Type.STRING_DICT;

import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.analysis.config.HostTransition;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder;
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType;
import com.google.devtools.build.lib.util.FileTypeSet;

/** Rule definition for the {@code android_instrumentation_test} rule. */
public class AndroidInstrumentationTestRule implements RuleDefinition {

  @Override
  public RuleClass build(Builder builder, RuleDefinitionEnvironment environment) {
    return builder
        .setUndocumented()
        .add(
            attr("instrumentation", LABEL)
                .mandatory()
                .allowedFileTypes(FileTypeSet.NO_FILE)
                .allowedRuleClasses("android_binary"))
        .add(
            attr("target_device", LABEL)
                .mandatory()
                .exec()
                .cfg(HostTransition.INSTANCE)
                .allowedFileTypes(FileTypeSet.NO_FILE)
                .allowedRuleClasses("android_device"))
        .add(
            attr("support_apks", LABEL_LIST)
                .allowedFileTypes(AndroidRuleClasses.APK)
                .allowedRuleClasses("android_binary"))
        .add(attr("test_args", STRING_DICT))
        .add(
            attr("fixtures", LABEL_LIST)
                .allowedFileTypes(FileTypeSet.NO_FILE)
                .allowedRuleClasses(
                    "android_device_script_fixture", "android_host_service_fixture"))
        .add(attr("fixture_args", STRING_DICT))
        .add(attr("log_levels", STRING_DICT))
        .add(
            attr("$test_entry_point", LABEL)
                .exec()
                .cfg(HostTransition.INSTANCE)
                .value(
                    environment.getToolsLabel("//tools/android:instrumentation_test_entry_point")))
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
        .ancestors(AndroidRuleClasses.AndroidBaseRule.class, BaseRuleClasses.TestBaseRule.class)
        .factoryClass(AndroidInstrumentationTest.class)
        .build();
  }
}
