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
import static com.google.devtools.build.lib.packages.Type.BOOLEAN;

import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.bazel.rules.cpp.BazelCppRuleClasses.CcBinaryBaseRule;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration;

/**
 * Rule definition for cc_binary rules.
 *
 * <p>This rule is implemented in Starlark. This class remains only for doc-gen purposes.
 */
public final class BazelCcBinaryRule implements RuleDefinition {

  @Override
  public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
    return builder
        .requiresConfigurationFragments(CppConfiguration.class)
        /*<!-- #BLAZE_RULE(cc_binary).IMPLICIT_OUTPUTS -->
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
        /*<!-- #BLAZE_RULE(cc_binary).ATTRIBUTE(linkshared) -->
        Create a shared library.
        To enable this attribute, include <code>linkshared=True</code> in your rule. By default
        this option is off.
        <p>
          The presence of this flag means that linking occurs with the <code>-shared</code> flag
          to <code>gcc</code>, and the resulting shared library is suitable for loading into for
          example a Java program. However, for build purposes it will never be linked into the
          dependent binary, as it is assumed that shared libraries built with a
          <a href="#cc_binary">cc_binary</a> rule are only loaded manually by other programs, so
          it should not be considered a substitute for the <a href="#cc_library">cc_library</a>
          rule. For sake of scalability we recommend avoiding this approach altogether and
          simply letting <code>java_library</code> depend on <code>cc_library</code> rules
          instead.
        </p>
        <p>
          If you specify both <code>linkopts=['-static']</code> and <code>linkshared=True</code>,
          you get a single completely self-contained unit. If you specify both
          <code>linkstatic=True</code> and <code>linkshared=True</code>, you get a single, mostly
          self-contained unit.
        </p>
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(
            attr("linkshared", BOOLEAN)
                .value(false)
                .nonconfigurable("used to *determine* the rule's configuration"))
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("cc_binary")
        .ancestors(CcBinaryBaseRule.class, BaseRuleClasses.BinaryBaseRule.class)
        .factoryClass(BaseRuleClasses.EmptyRuleConfiguredTargetFactory.class)
        .build();
  }
}

/*<!-- #BLAZE_RULE (NAME = cc_binary, TYPE = BINARY, FAMILY = C / C++) -->

${IMPLICIT_OUTPUTS}

<!-- #END_BLAZE_RULE -->*/
