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

package com.google.devtools.build.lib.rules.objc;

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.ImplicitOutputsFunction.fromTemplates;
import static com.google.devtools.build.lib.syntax.Type.BOOLEAN;

import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction.SafeImplicitOutputsFunction;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;

/**
 * Rule definition for apple_binary.
 */
public class AppleBinaryRule implements RuleDefinition {

  /**
   * Template for the fat binary output (using Apple's "lipo" tool to combine binaries of
   * multiple architectures).
   */
  private static final SafeImplicitOutputsFunction LIPOBIN = fromTemplates("%{name}_lipobin");

  @Override
  public RuleClass build(Builder builder, RuleDefinitionEnvironment env) {
    MultiArchSplitTransitionProvider splitTransitionProvider =
        new MultiArchSplitTransitionProvider();
    return builder
        .requiresConfigurationFragments(
            ObjcConfiguration.class, J2ObjcConfiguration.class, AppleConfiguration.class)
        .add(attr("$is_executable", BOOLEAN).value(true)
            .nonconfigurable("Called from RunCommand.isExecutable, which takes a Target"))
        .override(builder.copy("deps").cfg(splitTransitionProvider))
        .override(builder.copy("non_propagated_deps").cfg(splitTransitionProvider))
        /*<!-- #BLAZE_RULE(apple_binary).IMPLICIT_OUTPUTS -->
        <ul>
         <li><code><var>name</var>_lipobin</code>: the 'lipo'ed potentially multi-architecture
             binary. All transitive dependencies and <code>srcs</code> are linked.</li>
        </ul>
        <!-- #END_BLAZE_RULE.IMPLICIT_OUTPUTS -->*/
        .setImplicitOutputsFunction(
            ImplicitOutputsFunction.fromFunctions(LIPOBIN))
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("apple_binary")
        .factoryClass(AppleBinary.class)
        .ancestors(BaseRuleClasses.BaseRule.class, ObjcRuleClasses.LinkingRule.class,
            ObjcRuleClasses.MultiArchPlatformRule.class, ObjcRuleClasses.SimulatorRule.class)
        .build();
  }
}

/*<!-- #BLAZE_RULE (NAME = apple_binary, TYPE = BINARY, FAMILY = Objective-C) -->

<p>This rule produces single- or multi-architecture ("fat") Objective-C libraries and/or binaries,
typically used in creating apple bundles, such as frameworks, extensions, or applications.</p>

<p>The <code>lipo</code> tool is used to combine files of multiple architectures. The
<code>--ios_multi_cpus</code> flag controls which architectures are included in the output.</p>

<p>This rule currently only supports building for ios architectures, though more platforms
will be supported in the future.</p>

${IMPLICIT_OUTPUTS}

<!-- #END_BLAZE_RULE -->*/
