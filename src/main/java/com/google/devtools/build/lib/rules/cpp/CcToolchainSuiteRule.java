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
import static com.google.devtools.build.lib.packages.BuildType.NODEP_LABEL;

import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.util.FileTypeSet;

/**
 * Definition of the {@code cc_toolchain_suite} rule.
 */
public final class CcToolchainSuiteRule implements RuleDefinition {

  @Override
  public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment environment) {
    return builder
        .requiresConfigurationFragments(CppConfiguration.class)
        /* <!-- #BLAZE_RULE(cc_toolchain_suite).ATTRIBUTE(toolchains) -->
        A map from "&lt;cpu&gt;" or "&lt;cpu&gt;|&lt;compiler&gt;" strings to
        a <code>cc_toolchain</code> label. "&lt;cpu&gt;" will be used when only <code>--cpu</code>
        is passed to Bazel, and "&lt;cpu&gt;|&lt;compiler&gt;" will be used when both
        <code>--cpu</code> and <code>--compiler</code>  are passed to Bazel. Example:

        <p>
          <pre>
          cc_toolchain_suite(
            name = "toolchain",
            toolchains = {
              "piii|gcc": ":my_cc_toolchain_for_piii_using_gcc",
              "piii": ":my_cc_toolchain_for_piii_using_default_compiler",
            },
          )
          </pre>
        </p>
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(
            attr("toolchains", BuildType.LABEL_DICT_UNARY)
                .mandatory()
                .allowedFileTypes(FileTypeSet.NO_FILE)
                .nonconfigurable("Used during configuration creation"))
        .add(
            attr(CcToolchain.CC_TOOLCHAIN_TYPE_ATTRIBUTE_NAME, NODEP_LABEL)
                .value(CppRuleClasses.ccToolchainTypeAttribute(environment)))
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("cc_toolchain_suite")
        .ancestors(BaseRuleClasses.NativeBuildRule.class)
        .factoryClass(CcToolchainSuite.class)
        .build();
  }
}

/*<!-- #BLAZE_RULE (NAME = cc_toolchain_suite, TYPE = OTHER, FAMILY = C / C++) -->

<p>Represents a collections of C++ toolchains.</p>

<p>
  This rule is responsible for:

  <ul>
    <li>Collecting all relevant C++ toolchains.</li>
    <li>
      Selecting one toolchain depending on <code>--cpu</code> and  <code>--compiler</code> options
      passed to Bazel.
    </li>
  </ul>
</p>

<p>
  See also this
  <a href="https://docs.bazel.build/versions/master/cc-toolchain-config-reference.html">
    page
  </a> for elaborate C++ toolchain configuration and toolchain selection documentation.
</p>
<!-- #END_BLAZE_RULE -->*/
