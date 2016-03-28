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

package com.google.devtools.build.lib.rules.cpp;

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;
import static com.google.devtools.build.lib.syntax.Type.STRING;

import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder;
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType;

/**
 * Rule definition for the cc_inc_library class.
 */
public class CcIncLibraryRule implements RuleDefinition { 
  @Override
  public RuleClass build(Builder builder, RuleDefinitionEnvironment env) {
    return builder
        /*<!-- #BLAZE_RULE(cc_inc_library).ATTRIBUTE(prefix) -->
        A prefix for this rule.
        This prefix is the directory that is removed from the beginning of all
        declared header files. The relative paths of all files have to start with
        this prefix. If no trailing slash is present, it is added automatically.
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(attr("prefix", STRING))
        /*<!-- #BLAZE_RULE(cc_inc_library).ATTRIBUTE(hdrs) -->
        A list of header files.
        These are the header files that are exported by this rule. The relative
        paths of all files have to start with the prefix, in the sense that they
        have to be below the directory indicated by the prefix. Only header files
        are allowed.
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(
            attr("hdrs", LABEL_LIST)
                .mandatory()
                .nonEmpty()
                .orderIndependent()
                .direct_compile_time_input()
                .allowedFileTypes(CppFileTypes.CPP_HEADER))
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("$cc_inc_library")
        .ancestors(BaseRuleClasses.RuleBase.class)
        .type(RuleClassType.ABSTRACT)
        .build();
  }
}
