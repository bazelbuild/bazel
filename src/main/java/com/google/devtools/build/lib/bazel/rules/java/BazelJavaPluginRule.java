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

package com.google.devtools.build.lib.bazel.rules.java;

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.Type.STRING;

import com.google.devtools.build.lib.analysis.BlazeRule;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder;

/**
 * Rule definition for the java_plugin rule.
 */
@BlazeRule(name = "java_plugin",
             ancestors = { BazelJavaLibraryRule.class },
             factoryClass = BazelJavaPlugin.class)
public final class BazelJavaPluginRule implements RuleDefinition {
  @Override
  public RuleClass build(Builder builder, RuleDefinitionEnvironment env) {
    return builder
        .setImplicitOutputsFunction(BazelJavaRuleClasses.JAVA_LIBRARY_IMPLICIT_OUTPUTS)
        .override(builder.copy("deps").validityPredicate(Attribute.ANY_EDGE))
        .override(builder.copy("srcs").validityPredicate(Attribute.ANY_EDGE))
        .add(attr("processor_class", STRING))
        .removeAttribute("runtime_deps")
        .removeAttribute("exports")
        .removeAttribute("exported_plugins")
        .build();
  }
}

