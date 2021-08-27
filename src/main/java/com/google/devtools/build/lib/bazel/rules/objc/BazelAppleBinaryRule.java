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

package com.google.devtools.build.lib.bazel.rules.objc;

import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.analysis.config.transitions.ComposingTransitionFactory;
import com.google.devtools.build.lib.analysis.config.transitions.TransitionFactory;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleTransitionData;
import com.google.devtools.build.lib.rules.config.ConfigFeatureFlagTransitionFactory;
import com.google.devtools.build.lib.rules.objc.AppleBinaryBaseRule;
import com.google.devtools.build.lib.rules.objc.AppleCrosstoolTransition;
import com.google.devtools.build.lib.rules.objc.ObjcRuleClasses;

/** Rule definition for apple_binary. */
public class BazelAppleBinaryRule implements RuleDefinition {
  @Override
  public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
    return builder
        /*<!-- #BLAZE_RULE(apple_binary).IMPLICIT_OUTPUTS -->
        <ul>
         <li><code><var>name</var>_lipobin</code>: the 'lipo'ed potentially multi-architecture
             binary. All transitive dependencies and <code>srcs</code> are linked.</li>
        </ul>
        <!-- #END_BLAZE_RULE.IMPLICIT_OUTPUTS -->*/
        .setImplicitOutputsFunction(
            ImplicitOutputsFunction.fromFunctions(ObjcRuleClasses.LIPOBIN_OUTPUT))
        .cfg(
            ComposingTransitionFactory.of(
                (TransitionFactory<RuleTransitionData>)
                    (unused) -> AppleCrosstoolTransition.APPLE_CROSSTOOL_TRANSITION,
                new ConfigFeatureFlagTransitionFactory("feature_flags")))
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("apple_binary")
        .factoryClass(BazelAppleBinary.class)
        .ancestors(AppleBinaryBaseRule.class)
        .build();
  }
}

/*<!-- #BLAZE_RULE (NAME = apple_binary, TYPE = BINARY, FAMILY = Objective-C) -->

<p>This rule produces single- or multi-architecture ("fat") Objective-C libraries and/or binaries,
typically used in creating apple bundles, such as frameworks, extensions, or applications.</p>

<p>The <code>lipo</code> tool is used to combine files of multiple architectures. One of several
flags may control which architectures are included in the output, depending on the value of
the "platform_type" attribute.</p>

${IMPLICIT_OUTPUTS}

<!-- #END_BLAZE_RULE -->*/
