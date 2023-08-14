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
import static com.google.devtools.build.lib.packages.Type.BOOLEAN;

import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.analysis.config.ExecutionTransitionFactory;
import com.google.devtools.build.lib.bazel.rules.cpp.BazelCppRuleClasses.CcBinaryBaseRule;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType;
import com.google.devtools.build.lib.packages.TriState;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration;
import com.google.devtools.build.lib.util.OS;

/**
 * Rule definition for cc_test rules.
 *
 * <p>This rule is implemented in Starlark. This class remains only for doc-gen purposes.
 */
public final class BazelCcTestRule implements RuleDefinition {
  @Override
  public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
    return builder
        .requiresConfigurationFragments(CppConfiguration.class)
        /*<!-- #BLAZE_RULE(cc_test).IMPLICIT_OUTPUTS -->
        <ul>
        <li><code><var>name</var>.stripped</code> (only built if explicitly requested): A stripped
          version of the binary. <code>strip -g</code> is run on the binary to remove debug
          symbols.  Additional strip options can be provided on the command line using
          <code>--stripopt=-foo</code>. This output is only built if explicitly requested.</li>
        <li><code><var>name</var>.dwp</code> (only built if explicitly requested): If
          <a href="https://gcc.gnu.org/wiki/DebugFission">Fission</a> is enabled: a debug
          information package file suitable for debugging remotely deployed binaries. Else: an
          empty file.</li>
        </ul>
        <!-- #END_BLAZE_RULE.IMPLICIT_OUTPUTS -->*/
        .setImplicitOutputsFunction(BazelCppRuleClasses.CC_BINARY_IMPLICIT_OUTPUTS)
        // We don't want C++ tests to be dynamically linked by default on Windows,
        // because windows_export_all_symbols is not enabled by default, and it cannot solve
        // all symbols visibility issues, for example, users still have to use __declspec(dllimport)
        // to decorate data symbols imported from DLL.
        .override(attr("linkstatic", BOOLEAN).value(OS.getCurrent() == OS.WINDOWS))
        .override(attr("stamp", TRISTATE).value(TriState.NO))
        .add(
            attr(":lcov_merger", LABEL)
                .cfg(ExecutionTransitionFactory.createFactory())
                .value(BaseRuleClasses.getCoverageOutputGeneratorLabel()))
        .add(
            attr("$collect_cc_coverage", LABEL)
                .cfg(ExecutionTransitionFactory.createFactory())
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
        .factoryClass(BaseRuleClasses.EmptyRuleConfiguredTargetFactory.class)
        .build();
  }
}

/*<!-- #BLAZE_RULE (NAME = cc_test, TYPE = TEST, FAMILY = C / C++) -->

<!-- #END_BLAZE_RULE -->*/
