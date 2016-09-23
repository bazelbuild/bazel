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
package com.google.devtools.build.lib.bazel.rules.android;

import static com.google.devtools.build.lib.packages.Attribute.ANY_EDGE;
import static com.google.devtools.build.lib.packages.Attribute.ConfigurationTransition.HOST;
import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;

import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder;
import com.google.devtools.build.lib.rules.android.AndroidRuleClasses.AndroidAaptBaseRule;
import com.google.devtools.build.lib.rules.java.JavaCompilationArgsProvider;
import com.google.devtools.build.lib.util.FileType;
import com.google.devtools.build.lib.util.FileTypeSet;

/** Rule definition for the aar_import rule. */
public class AarImportRule implements RuleDefinition {

  @Override
  public RuleClass build(Builder builder, RuleDefinitionEnvironment environment) {
    return builder
        .setUndocumented()
        .add(attr("aar", LABEL)
            .mandatory()
            .allowedFileTypes(FileType.of(".aar")))
        .add(attr("exports", LABEL_LIST)
            .allowedRuleClasses("aar_import", "java_import")
            .allowedFileTypes()
            .validityPredicate(ANY_EDGE))
        .add(attr("$zip_manifest_creator", LABEL)
            .cfg(HOST)
            .exec()
            .value(Label.parseAbsoluteUnchecked(
                environment.getToolsRepository() + "//tools/zip:zip_manifest_creator")))
        .add(attr("$unzip", LABEL)
            .cfg(HOST)
            .exec()
            .value(Label.parseAbsoluteUnchecked(
                environment.getToolsRepository() + "//tools/zip:unzip"))
            .allowedFileTypes(FileTypeSet.ANY_FILE))
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
        .name("aar_import")
        // AndroidAaptBaseRule is needed for $android_manifest_merger which is used by the
        // ApplicationManifest class.
        .ancestors(BaseRuleClasses.RuleBase.class, AndroidAaptBaseRule.class)
        .factoryClass(AarImport.class)
        .build();
  }
}
