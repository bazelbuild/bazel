// Copyright 2015 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.util.FileTypeSet;

/**
 * Definition of the {@code cc_toolchain_suite} rule.
 */
public final class CcToolchainSuiteRule implements RuleDefinition {

  @Override
  public RuleClass build(Builder builder, RuleDefinitionEnvironment environment) {
    return builder
        .setUndocumented()
        .add(attr("toolchains", BuildType.LABEL_DICT_UNARY)
            .mandatory()
            .allowedFileTypes(FileTypeSet.NO_FILE)
            .nonconfigurable("Used during configuration creation"))
        .add(attr("proto", Type.STRING)
            .nonconfigurable("Used during configuration creation"))
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("cc_toolchain_suite")
        .ancestors(BaseRuleClasses.BaseRule.class)
        .factoryClass(CcToolchainSuite.class)
        .build();
  }
}
