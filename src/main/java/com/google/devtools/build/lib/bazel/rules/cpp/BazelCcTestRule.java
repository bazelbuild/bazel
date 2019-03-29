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
import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static com.google.devtools.build.lib.packages.BuildType.TRISTATE;
import static com.google.devtools.build.lib.syntax.Type.BOOLEAN;

import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.analysis.config.HostTransition;
import com.google.devtools.build.lib.bazel.rules.cpp.BazelCppRuleClasses.CcBinaryBaseRule;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType;
import com.google.devtools.build.lib.packages.TriState;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration;
import com.google.devtools.build.lib.rules.cpp.CppRuleClasses;
import com.google.devtools.build.lib.util.OS;

/** Rule definition for cc_test rules. */
public final class BazelCcTestRule implements RuleDefinition {
  @Override
  public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
    return builder
        .requiresConfigurationFragments(CppConfiguration.class)
        .setImplicitOutputsFunction(CppRuleClasses.CC_BINARY_DEBUG_PACKAGE)
        // We don't want C++ tests to be dynamically linked by default on Windows,
        // because windows_export_all_symbols is not enabled by default, and it cannot solve
        // all symbols visibility issues, for example, users still have to use __declspec(dllimport)
        // to decorate data symbols imported from DLL.
        .override(attr("linkstatic", BOOLEAN).value(OS.getCurrent() == OS.WINDOWS))
        .override(attr("stamp", TRISTATE).value(TriState.NO))
        .add(
            attr("$lcov_merger", LABEL)
                .value(
                    Label.parseAbsoluteUnchecked(
                        "@bazel_tools//tools/test/CoverageOutputGenerator/java/com/google/devtools/coverageoutputgenerator:Main")))
        .add(
            attr("$collect_cc_coverage", LABEL)
                .cfg(HostTransition.createFactory())
                .singleArtifact()
                .value(env.getToolsLabel("//tools/test:collect_cc_coverage")))
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("cc_test")
        .type(RuleClassType.TEST)
        .ancestors(CcBinaryBaseRule.class, BaseRuleClasses.TestBaseRule.class)
        .factoryClass(BazelCcTest.class)
        .build();
  }
}

/*<!-- #BLAZE_RULE (NAME = cc_test, TYPE = TEST, FAMILY = C / C++) -->

<!-- #END_BLAZE_RULE -->*/
