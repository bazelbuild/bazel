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

import static com.google.devtools.build.lib.packages.ImplicitOutputsFunction.fromTemplates;

import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.analysis.config.transitions.ComposingTransitionFactory;
import com.google.devtools.build.lib.analysis.config.transitions.TransitionFactory;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction.SafeImplicitOutputsFunction;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleTransitionData;
import com.google.devtools.build.lib.rules.config.ConfigFeatureFlagTransitionFactory;
import com.google.devtools.build.lib.rules.objc.AppleCrosstoolTransition;
import com.google.devtools.build.lib.rules.objc.AppleStaticLibraryBaseRule;

/** Rule definition for apple_static_library. */
public class BazelAppleStaticLibraryRule implements RuleDefinition {

  /**
   * Template for the fat archive output (using Apple's "lipo" tool to combine .a archive files of
   * multiple architectures).
   */
  static final SafeImplicitOutputsFunction LIPO_ARCHIVE = fromTemplates("%{name}_lipo.a");

  @Override
  public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
    return builder
        /*<!-- #BLAZE_RULE(apple_static_library).IMPLICIT_OUTPUTS -->
        <ul>
          <li><code><var>name</var>_lipo.a</code>: a 'lipo'ed archive file. All transitive
          dependencies and <code>srcs</code> are linked, minus all transitive dependencies
          specified in <code>avoid_deps</code>.</li>
        </ul>
        <!-- #END_BLAZE_RULE.IMPLICIT_OUTPUTS -->*/
        .setImplicitOutputsFunction(ImplicitOutputsFunction.fromFunctions(LIPO_ARCHIVE))
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
        .name("apple_static_library")
        .factoryClass(BaseRuleClasses.EmptyRuleConfiguredTargetFactory.class)
        .ancestors(AppleStaticLibraryBaseRule.class)
        .build();
  }
}

/*<!-- #BLAZE_RULE (NAME = apple_static_library, TYPE = BINARY, FAMILY = Objective-C) -->

<p>This rule produces single- or multi-architecture ("fat") Objective-C statically-linked libraries,
typically used in creating static Apple Frameworks for distribution and re-use in
multiple extensions or applications.</p>

<p>The <code>lipo</code> tool is used to combine files of multiple architectures; a build flag
controls which architectures are targeted. The build flag examined depends on the
<code>platform_type</code> attribute for this rule (and is described in its documentation).</p>

${IMPLICIT_OUTPUTS}

<!-- #END_BLAZE_RULE -->*/
