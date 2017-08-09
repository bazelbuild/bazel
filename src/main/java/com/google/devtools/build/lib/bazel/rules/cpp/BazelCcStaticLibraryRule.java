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

package com.google.devtools.build.lib.bazel.rules.cpp;

import static com.google.devtools.build.lib.bazel.rules.cpp.BazelCppRuleClasses.DEPS_ALLOWED_RULES;
import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;
import static com.google.devtools.build.lib.syntax.Type.STRING_LIST;

import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.bazel.rules.cpp.BazelCppRuleClasses.CcBaseRule;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder;

/** Rule definition for the cc_static_library rule. */
public final class BazelCcStaticLibraryRule implements RuleDefinition {
  @Override
  public RuleClass build(Builder builder, RuleDefinitionEnvironment env) {
    return builder
        .add(
            attr("reexport_deps", LABEL_LIST)
                .allowedRuleClasses(DEPS_ALLOWED_RULES)
                .allowedFileTypes())
        .add(attr("linkopts", STRING_LIST))
        .build();
  }

  @Override
  public  Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("cc_shared_library")
        .ancestors(CcBaseRule.class, BaseRuleClasses.MakeVariableExpandingRule.class)
        .factoryClass(BazelCcStaticLibrary.class)
        .build();
  }
}
