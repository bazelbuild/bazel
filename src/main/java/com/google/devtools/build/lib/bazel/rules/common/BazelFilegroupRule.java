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
package com.google.devtools.build.lib.bazel.rules.common;

import static com.google.devtools.build.lib.packages.Attribute.ConfigurationTransition.DATA;
import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.Type.LABEL_LIST;
import static com.google.devtools.build.lib.packages.Type.LICENSE;
import static com.google.devtools.build.lib.packages.Type.STRING;

import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.BlazeRule;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder;
import com.google.devtools.build.lib.rules.filegroup.Filegroup;
import com.google.devtools.build.lib.util.FileTypeSet;

/**
 * Rule object implementing "filegroup".
 */
@BlazeRule(name = "filegroup",
             ancestors = { BaseRuleClasses.BaseRule.class },
             factoryClass = Filegroup.class)
public final class BazelFilegroupRule implements RuleDefinition {
  @Override
  public RuleClass build(Builder builder, RuleDefinitionEnvironment env) {
    // filegroup ignores any filtering set with setSrcsAllowedFiles.
    return builder
        .add(attr("srcs", LABEL_LIST).allowedFileTypes(FileTypeSet.ANY_FILE))
        .add(attr("data", LABEL_LIST).cfg(DATA).allowedFileTypes(FileTypeSet.ANY_FILE))
        .add(attr("output_licenses", LICENSE))
        .add(attr("path", STRING))
        .build();
  }
}
