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
package com.google.devtools.build.lib.bazel.rules.genrule;

import static com.google.devtools.build.lib.packages.Attribute.ConfigurationTransition.HOST;
import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.Type.BOOLEAN;
import static com.google.devtools.build.lib.packages.Type.LABEL;
import static com.google.devtools.build.lib.packages.Type.LABEL_LIST;
import static com.google.devtools.build.lib.packages.Type.LICENSE;
import static com.google.devtools.build.lib.packages.Type.OUTPUT_LIST;
import static com.google.devtools.build.lib.packages.Type.STRING;

import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.view.BaseRuleClasses;
import com.google.devtools.build.lib.view.BlazeRule;
import com.google.devtools.build.lib.view.RuleDefinition;
import com.google.devtools.build.lib.view.RuleDefinitionEnvironment;

/**
 * Rule definition for the genrule rule.
 */
@BlazeRule(name = "genrule",
             ancestors = { BaseRuleClasses.RuleBase.class },
             factoryClass = GenRule.class)
public final class BazelGenRuleRule implements RuleDefinition {
  public static final String GENRULE_SETUP_LABEL = "//tools/genrule:genrule-setup.sh";

  @Override
  public RuleClass build(Builder builder, RuleDefinitionEnvironment env) {
    return builder
        .setOutputToGenfiles()
        .add(attr("srcs", LABEL_LIST)
            .direct_compile_time_input())
        .add(attr("tools", LABEL_LIST).cfg(HOST))
        .add(attr("$genrule_setup", LABEL).cfg(HOST).value(env.getLabel(GENRULE_SETUP_LABEL)))
        .add(attr("outs", OUTPUT_LIST).mandatory())
        .add(attr("cmd", STRING).mandatory())
        .add(attr("output_to_bindir", BOOLEAN).nonconfigurable().value(false))
        .add(attr("local", BOOLEAN).value(false))
        .add(attr("message", STRING))
        .add(attr("output_licenses", LICENSE).nonconfigurable())
        .add(attr("executable", BOOLEAN).value(false))
        .add(attr("stamp", BOOLEAN).value(false))
        .add(attr("heuristic_label_expansion", BOOLEAN).value(true))
        .add(attr("$is_executable", BOOLEAN).nonconfigurable().value(
            new Attribute.ComputedDefault("outs", "executable") {
              @Override
              public Object getDefault(AttributeMap rule) {
                return (rule.get("outs", Type.OUTPUT_LIST).size() == 1)
                    && rule.get("executable", BOOLEAN);
              }
            }))
        .removeAttribute("data")
        .removeAttribute("deps")
        .build();
  }
}
