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
import static com.google.devtools.build.lib.packages.Attribute.ConfigurationTransition.HOST;
import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;

import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder;
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType;
import com.google.devtools.build.lib.rules.android.AndroidRuleClasses.AndroidAaptBaseRule;
import com.google.devtools.build.lib.rules.java.JavaCompilationArgsProvider;
import com.google.devtools.build.lib.util.FileType;

/** Rule definition for the aar_import rule. */
public class AarImportBaseRule implements RuleDefinition {

  @Override
  public RuleClass build(Builder builder, RuleDefinitionEnvironment environment) {
    return builder
        /* <!-- #BLAZE_RULE($aar_import_base).ATTRIBUTE(aar) -->
        The <code>.aar</code> file to provide to the Android targets that depend on this target.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("aar", LABEL)
            .mandatory()
            .allowedFileTypes(FileType.of(".aar")))
        /* <!-- #BLAZE_RULE(aar_import).ATTRIBUTE(exports) -->
        Target to export to rules that depend on this rule.
        See <a href="${link java_library.exports}">java_library.exports.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("exports", LABEL_LIST)
            .allowedRuleClasses("aar_import", "java_import")
            .allowedFileTypes()
            .validityPredicate(ANY_EDGE))
        .add(attr("$aar_embedded_jars_extractor", LABEL)
            .cfg(HOST)
            .exec()
            .value(Label.parseAbsoluteUnchecked(
                environment.getToolsRepository() + "//tools/android:aar_embedded_jars_extractor")))
        .add(attr("$aar_native_libs_zip_creator", LABEL)
            .cfg(HOST)
            .exec()
            .value(Label.parseAbsoluteUnchecked(
                environment.getToolsRepository() + "//tools/android:aar_native_libs_zip_creator")))
        .add(attr("$zip_manifest_creator", LABEL)
            .cfg(HOST)
            .exec()
            .value(Label.parseAbsoluteUnchecked(
                environment.getToolsRepository() + "//tools/android:zip_manifest_creator")))
        .add(attr("$zipper", LABEL)
            .cfg(HOST)
            .exec()
            .value(Label.parseAbsoluteUnchecked(
                environment.getToolsRepository() + "//tools/zip:zipper")))
        .advertiseProvider(JavaCompilationArgsProvider.class)
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("$aar_import_base")
        .type(RuleClassType.ABSTRACT)
        // AndroidAaptBaseRule is needed for $android_manifest_merger which is used by the
        // ApplicationManifest class.
        .ancestors(AndroidAaptBaseRule.class)
        .build();
  }
}
