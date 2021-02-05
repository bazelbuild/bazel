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
package com.google.devtools.build.lib.rules.android;

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static com.google.devtools.build.lib.packages.Type.STRING;
import static com.google.devtools.build.lib.util.FileTypeSet.ANY_FILE;

import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.analysis.config.ExecutionTransitionFactory;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType;
import com.google.devtools.build.lib.packages.StarlarkProviderIdentifier;
import com.google.devtools.build.lib.rules.java.BootClassPathInfo;
import com.google.devtools.build.lib.rules.java.JavaConfiguration;
import com.google.devtools.build.lib.rules.java.JavaRuleClasses.JavaToolchainBaseRule;
import com.google.devtools.build.lib.rules.java.JavaSemantics;

/** Definition of the {@code android_sdk} rule. */
public class AndroidSdkBaseRule implements RuleDefinition {
    @Override
    public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment environment) {
    return builder
        .requiresConfigurationFragments(JavaConfiguration.class, AndroidConfiguration.class)
        .setUndocumented()
        // build_tools_version is assumed to be the latest version if omitted.
        .add(attr("build_tools_version", STRING))
        // This is the Proguard that comes from the --proguard_top attribute.
        .add(
            attr(":proguard", LABEL)
                .cfg(ExecutionTransitionFactory.create())
                .value(JavaSemantics.PROGUARD)
                .exec())
        // This is the Proguard in the BUILD file that contains the android_sdk rule. Used when
        // --proguard_top is not specified.
        .add(
            attr("proguard", LABEL)
                .mandatory()
                .cfg(ExecutionTransitionFactory.create())
                .allowedFileTypes(ANY_FILE)
                .exec())
        .add(
            attr("aapt", LABEL)
                .mandatory()
                .cfg(ExecutionTransitionFactory.create())
                .allowedFileTypes(ANY_FILE)
                .exec())
        .add(
            attr("aapt2", LABEL)
                .cfg(ExecutionTransitionFactory.create())
                .allowedFileTypes(ANY_FILE)
                .exec())
        .add(
            attr("dx", LABEL)
                .mandatory()
                .cfg(ExecutionTransitionFactory.create())
                .allowedFileTypes(ANY_FILE)
                .exec())
        .add(
            attr("main_dex_list_creator", LABEL)
                .mandatory()
                .cfg(ExecutionTransitionFactory.create())
                .allowedFileTypes(ANY_FILE)
                .exec())
        .add(
            attr("adb", LABEL)
                .mandatory()
                .cfg(ExecutionTransitionFactory.create())
                .allowedFileTypes(ANY_FILE)
                .exec())
        .add(
            attr("framework_aidl", LABEL)
                .mandatory()
                .cfg(ExecutionTransitionFactory.create())
                .allowedFileTypes(ANY_FILE))
        .add(
            attr("aidl", LABEL)
                .mandatory()
                .cfg(ExecutionTransitionFactory.create())
                .allowedFileTypes(ANY_FILE)
                .exec())
        .add(attr("aidl_lib", LABEL).allowedFileTypes(JavaSemantics.JAR))
        .add(
            attr("android_jar", LABEL)
                .mandatory()
                .cfg(ExecutionTransitionFactory.create())
                .allowedFileTypes(JavaSemantics.JAR))
        // TODO(b/67903726): Make this attribute mandatory after updating all android_sdk rules.
        .add(
            attr("source_properties", LABEL)
                .cfg(ExecutionTransitionFactory.create())
                .allowedFileTypes(ANY_FILE))
        .add(
            attr("shrinked_android_jar", LABEL)
                .mandatory()
                .cfg(ExecutionTransitionFactory.create())
                .allowedFileTypes(ANY_FILE))
        .add(
            attr("annotations_jar", LABEL)
                .cfg(ExecutionTransitionFactory.create())
                .allowedFileTypes(ANY_FILE))
        .add(
            attr("main_dex_classes", LABEL)
                .mandatory()
                .cfg(ExecutionTransitionFactory.create())
                .allowedFileTypes(ANY_FILE))
        .add(
            attr("apkbuilder", LABEL)
                .cfg(ExecutionTransitionFactory.create())
                .allowedFileTypes(ANY_FILE)
                .exec())
        .add(
            attr("apksigner", LABEL)
                .mandatory()
                .cfg(ExecutionTransitionFactory.create())
                .allowedFileTypes(ANY_FILE)
                .exec())
        .add(
            attr("zipalign", LABEL)
                .mandatory()
                .cfg(ExecutionTransitionFactory.create())
                .allowedFileTypes(ANY_FILE)
                .exec())
        .add(
            attr("system", LABEL)
                .allowedFileTypes()
                .mandatoryProviders(BootClassPathInfo.PROVIDER.id()))
        .add(
            attr("legacy_main_dex_list_generator", LABEL)
                .cfg(ExecutionTransitionFactory.create())
                .allowedFileTypes(ANY_FILE)
                .exec())
        .advertiseStarlarkProvider(
            StarlarkProviderIdentifier.forKey(AndroidSdkProvider.PROVIDER.getKey()))
        .build();
    }

    @Override
    public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("$android_sdk_base")
        .type(RuleClassType.ABSTRACT)
        .ancestors(JavaToolchainBaseRule.class)
        .build();
    }
  }
