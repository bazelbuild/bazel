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
import static com.google.devtools.build.lib.packages.BuildType.LABEL;
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
  
  /**
   * Template for the fat archive output (using Apple's "lipo" tool to combine .a archive files of
   * multiple architectures).
   */
  static final SafeImplicitOutputsFunction LIPO_ARCHIVE = fromTemplates("%{name}_lipo.a");

  @Override
  public RuleClass build(Builder builder, RuleDefinitionEnvironment env) {
    return builder
        .requiresConfigurationFragments(
            ObjcConfiguration.class, J2ObjcConfiguration.class, AppleConfiguration.class)
        .add(attr("$is_executable", BOOLEAN).value(true)
            .nonconfigurable("Called from RunCommand.isExecutable, which takes a Target"))
        .override(builder.copy("deps").cfg(AppleBinary.SPLIT_TRANSITION_PROVIDER))
        .override(builder.copy("non_propagated_deps").cfg(AppleBinary.SPLIT_TRANSITION_PROVIDER))
        // This is currently a hack to obtain all child configurations regardless of the attribute
        // values of this rule -- this rule does not currently use the actual info provided by
        // this attribute.
        .add(attr(":cc_toolchain", LABEL)
            .cfg(AppleBinary.SPLIT_TRANSITION_PROVIDER)
            .value(ObjcRuleClasses.APPLE_TOOLCHAIN))
        /*<!-- #BLAZE_RULE(apple_binary).IMPLICIT_OUTPUTS -->
        <ul>
         <li><code><var>name</var>_lipobin</code>: the 'lipo'ed potentially multi-architecture
             binary. All transitive dependencies and <code>srcs</code> are linked.</li>
         <li><code><var>name</var>_.lipo.a</code>: a 'lipo'ed archive file linking together only
             the <code>srcs</code> of this target.</li>
        </ul>
        <!-- #END_BLAZE_RULE.IMPLICIT_OUTPUTS -->*/
        .setImplicitOutputsFunction(
            ImplicitOutputsFunction.fromFunctions(LIPOBIN, LIPO_ARCHIVE))
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("apple_binary")
        .factoryClass(AppleBinary.class)
        .ancestors(BaseRuleClasses.BaseRule.class, ObjcRuleClasses.LinkingRule.class,
            ObjcRuleClasses.SimulatorRule.class)
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
