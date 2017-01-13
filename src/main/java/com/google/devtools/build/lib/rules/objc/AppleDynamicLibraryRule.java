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

import static com.google.devtools.build.lib.packages.ImplicitOutputsFunction.fromTemplates;

import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction.SafeImplicitOutputsFunction;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;

/**
 * Rule definition for apple_dynamic_library.
 */
// TODO(b/33077308): Remove this rule.
public class AppleDynamicLibraryRule implements RuleDefinition {

  /**
   * Template for the fat dynamic library output (using Apple's "lipo" tool to combine dynamic
   * libraries of multiple architectures).
   */
  private static final SafeImplicitOutputsFunction LIPO_DYLIB = fromTemplates("%{name}_lipo.dylib");

  @Override
  public RuleClass build(Builder builder, RuleDefinitionEnvironment env) {
    MultiArchSplitTransitionProvider splitTransitionProvider =
        new MultiArchSplitTransitionProvider();
    return builder
        .requiresConfigurationFragments(
            ObjcConfiguration.class, J2ObjcConfiguration.class, AppleConfiguration.class)
        .override(builder.copy("deps").cfg(splitTransitionProvider))
        .override(builder.copy("non_propagated_deps").cfg(splitTransitionProvider))
        /*<!-- #BLAZE_RULE(apple_dynamic_library).IMPLICIT_OUTPUTS -->
        <ul>
         <li><code><var>name</var>_lipo.dylib</code>: the 'lipo'ed potentially multi-architecture
             dynamic library. All transitive dependencies and <code>srcs</code> are linked.</li>
        </ul>
        <!-- #END_BLAZE_RULE.IMPLICIT_OUTPUTS -->*/
        .setImplicitOutputsFunction(LIPO_DYLIB)
        .cfg(AppleCrosstoolTransition.APPLE_CROSSTOOL_TRANSITION)
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("apple_dynamic_library")
        .factoryClass(AppleDynamicLibrary.class)
        .ancestors(BaseRuleClasses.BaseRule.class, ObjcRuleClasses.LinkingRule.class,
            ObjcRuleClasses.MultiArchPlatformRule.class, ObjcRuleClasses.SimulatorRule.class,
            ObjcRuleClasses.DylibDependingRule.class)
        .build();
  }
}

/*<!-- #BLAZE_RULE (NAME = apple_dynamic_library, TYPE = BINARY, FAMILY = Objective-C) -->

<p> This rule is deprecated. Please use apple_binary with binary_type = "dylib" instead. </p>

${IMPLICIT_OUTPUTS}

<!-- #END_BLAZE_RULE -->*/
