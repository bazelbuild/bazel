// Copyright 2016 The Bazel Authors. All rights reserved.
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

import static com.google.devtools.build.lib.packages.Attribute.ANY_EDGE;
import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;

import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.analysis.config.ExecutionTransitionFactory;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType;
import com.google.devtools.build.lib.packages.StarlarkProviderIdentifier;
import com.google.devtools.build.lib.rules.android.AndroidRuleClasses.AndroidBaseRule;
import com.google.devtools.build.lib.rules.java.JavaConfiguration;
import com.google.devtools.build.lib.rules.java.JavaInfo;
import com.google.devtools.build.lib.rules.java.JavaSemantics;
import com.google.devtools.build.lib.util.FileType;

/** Rule definition for the aar_import rule. */
public class AarImportBaseRule implements RuleDefinition {

  static final String AAR_EMBEDDED_JARS_EXTACTOR = "$aar_embedded_jars_extractor";
  static final String AAR_EMBEDDED_PROGUARD_EXTACTOR = "$aar_embedded_proguard_extractor";
  static final String AAR_NATIVE_LIBS_ZIP_CREATOR = "$aar_native_libs_zip_creator";
  static final String AAR_RESOURCES_EXTRACTOR = "$aar_resources_extractor";
  static final String ZIPPER = "$zipper";

  @Override
  public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
    return builder
        /* <!-- #BLAZE_RULE($aar_import_base).ATTRIBUTE(aar) -->
        The <code>.aar</code> file to provide to the Android targets that depend on this target.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("aar", LABEL).mandatory().allowedFileTypes(FileType.of(".aar")))
        /* <!-- #BLAZE_RULE(aar_import).ATTRIBUTE(exports) -->
        Targets to export to rules that depend on this rule.
        See <a href="${link java_library.exports}">java_library.exports.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(
            attr("exports", LABEL_LIST)
                .allowedRuleClasses("aar_import", "java_import")
                .allowedFileTypes()
                .validityPredicate(ANY_EDGE))
        /* <!-- #BLAZE_RULE(aar_import).ATTRIBUTE(srcjar) -->
        A JAR file that contains source code for the compiled JAR files in the AAR.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(
            attr("srcjar", LABEL)
                .allowedFileTypes(JavaSemantics.SOURCE_JAR, JavaSemantics.JAR)
                .direct_compile_time_input())
        .add(
            attr(AAR_EMBEDDED_JARS_EXTACTOR, LABEL)
                .cfg(ExecutionTransitionFactory.create())
                .exec()
                .value(env.getToolsLabel("//tools/android:aar_embedded_jars_extractor")))
        .add(
            attr(AAR_EMBEDDED_PROGUARD_EXTACTOR, LABEL)
                .cfg(ExecutionTransitionFactory.create())
                .exec()
                .value(env.getToolsLabel("//tools/android:aar_embedded_proguard_extractor")))
        .add(
            attr(AAR_NATIVE_LIBS_ZIP_CREATOR, LABEL)
                .cfg(ExecutionTransitionFactory.create())
                .exec()
                .value(env.getToolsLabel("//tools/android:aar_native_libs_zip_creator")))
        .add(
            attr(AAR_RESOURCES_EXTRACTOR, LABEL)
                .cfg(ExecutionTransitionFactory.create())
                .exec()
                .value(env.getToolsLabel("//tools/android:aar_resources_extractor")))
        .add(
            attr("$import_deps_checker", LABEL)
                .cfg(ExecutionTransitionFactory.create())
                .exec()
                .value(env.getToolsLabel("//tools/android:aar_import_deps_checker")))
        .add(
            attr(ZIPPER, LABEL)
                .cfg(ExecutionTransitionFactory.create())
                .exec()
                .value(env.getToolsLabel("//tools/zip:zipper")))
        .advertiseStarlarkProvider(StarlarkProviderIdentifier.forKey(JavaInfo.PROVIDER.getKey()))
        .requiresConfigurationFragments(AndroidConfiguration.class, JavaConfiguration.class)
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("$aar_import_base")
        .type(RuleClassType.ABSTRACT)
        .ancestors(AndroidBaseRule.class)
        .build();
  }
}
