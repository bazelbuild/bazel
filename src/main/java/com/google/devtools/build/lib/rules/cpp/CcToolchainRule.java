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
package com.google.devtools.build.lib.rules.cpp;

import static com.google.devtools.build.lib.packages.Attribute.ConfigurationTransition.HOST;
import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.Type.LABEL;
import static com.google.devtools.build.lib.packages.Type.LABEL_LIST;
import static com.google.devtools.build.lib.packages.Type.LICENSE;
import static com.google.devtools.build.lib.packages.Type.STRING;

import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder;
import com.google.devtools.build.lib.view.BaseRuleClasses;
import com.google.devtools.build.lib.view.BlazeRule;
import com.google.devtools.build.lib.view.RuleDefinition;
import com.google.devtools.build.lib.view.RuleDefinitionEnvironment;

/**
 * Rule definition for compiler definition.
 */
@BlazeRule(name = "cc_toolchain",
             ancestors = { BaseRuleClasses.BaseRule.class },
             factoryClass = CcToolchain.class)
public final class CcToolchainRule implements RuleDefinition {
  @Override
  public RuleClass build(Builder builder, RuleDefinitionEnvironment env) {
    return builder
        .setUndocumented()
        .add(attr("output_licenses", LICENSE))
        .add(attr("cpu", STRING).mandatory())
        .add(attr("all_files", LABEL).cfg(HOST).mandatory())
        .add(attr("compiler_files", LABEL).cfg(HOST).mandatory())
        .add(attr("strip_files", LABEL).cfg(HOST).mandatory())
        .add(attr("objcopy_files", LABEL).cfg(HOST).mandatory())
        .add(attr("linker_files", LABEL).cfg(HOST).mandatory())
        .add(attr("dwp_files", LABEL).cfg(HOST).mandatory())
        .add(attr("static_runtime_libs", LABEL_LIST).mandatory())
        .add(attr("dynamic_runtime_libs", LABEL_LIST).mandatory())
        .add(attr("module_map", LABEL).cfg(HOST))
        .build();
  }
}
