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

package com.google.devtools.build.lib.bazel.rules.cpp;

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.syntax.Type.BOOLEAN;

import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.bazel.rules.cpp.BazelCppRuleClasses.CcLibraryBaseRule;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration;

/** Rule definition for the cc_library rule. */
public final class BazelCcLibraryRule implements RuleDefinition {
  @Override
  public RuleClass build(Builder builder, RuleDefinitionEnvironment env) {
    return builder
        // TODO: Google cc_library overrides documentation for:
        // deps, data, linkopts, defines, srcs; override here too?

        .requiresConfigurationFragments(CppConfiguration.class)
        /*<!-- #BLAZE_RULE(cc_library).ATTRIBUTE(alwayslink) -->
        If 1, any binary that depends (directly or indirectly) on this C++
        library will link in all the object files for the files listed in
        <code>srcs</code>, even if some contain no symbols referenced by the binary.
        This is useful if your code isn't explicitly called by code in
        the binary, e.g., if your code registers to receive some callback
        provided by some service.
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(
            attr("alwayslink", BOOLEAN)
                .nonconfigurable("value is referenced in an ImplicitOutputsFunction"))
        .override(
            attr("linkstatic", BOOLEAN)
                .value(false)
                .nonconfigurable("value is referenced in an ImplicitOutputsFunction"))
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("cc_library")
        .ancestors(CcLibraryBaseRule.class, BaseRuleClasses.MakeVariableExpandingRule.class)
        .factoryClass(BazelCcLibrary.class)
        .build();
  }
}
