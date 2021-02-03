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
import static com.google.devtools.build.lib.packages.BuildType.LABEL_KEYED_STRING_DICT;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;
import static com.google.devtools.build.lib.packages.ImplicitOutputsFunction.fromTemplates;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.analysis.config.transitions.ComposingTransitionFactory;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction.SafeImplicitOutputsFunction;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;
import com.google.devtools.build.lib.rules.config.ConfigFeatureFlagProvider;
import com.google.devtools.build.lib.rules.config.ConfigFeatureFlagTransitionFactory;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration;
import com.google.devtools.build.lib.rules.cpp.CppRuleClasses;

/**
 * Rule definition for apple_static_library.
 */
public class AppleStaticLibraryRule implements RuleDefinition {
  /**
   * Template for the fat archive output (using Apple's "lipo" tool to combine .a archive files of
   * multiple architectures).
   */
  static final SafeImplicitOutputsFunction LIPO_ARCHIVE = fromTemplates("%{name}_lipo.a");

  /**
   * Attribute name for dependent libraries which should not be linked into the outputs of this
   * rule.
   */
  static final String AVOID_DEPS_ATTR_NAME = "avoid_deps";

  @Override
  public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
    MultiArchSplitTransitionProvider splitTransitionProvider =
        new MultiArchSplitTransitionProvider();

    return builder
        .requiresConfigurationFragments(
            ObjcConfiguration.class,
            J2ObjcConfiguration.class,
            AppleConfiguration.class,
            CppConfiguration.class)
        /* <!-- #BLAZE_RULE(apple_static_library).ATTRIBUTE(avoid_deps) -->
        <p>A list of targets which should not be included (nor their transitive dependencies
        included) in the outputs of this rule -- even if they are otherwise transitively depended
        on via the <code>deps</code> attribute.</p>

        <p>This attribute effectively serves to remove portions of the dependency tree from a static
        library, and is useful most commonly in scenarios where static libraries depend on each
        other.</p>

        <p>That is, suppose static libraries X and C are typically distributed to consumers
        separately. C is a very-common base library, and X contains less-common functionality; X
        depends on C, such that applications seeking to import library X must also import library
        C. The target describing X would set C's target in <code>avoid_deps</code>. In this way,
        X can depend on C without also containing C. Without this <code>avoid_deps</code> usage,
        an application importing both X and C would have duplicate symbols for C.</p>
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(
            attr(AVOID_DEPS_ATTR_NAME, LABEL_LIST)
                .direct_compile_time_input()
                .allowedRuleClasses(ObjcRuleClasses.CompilingRule.ALLOWED_CC_DEPS_RULE_CLASSES)
                .mandatoryProviders(ObjcProvider.STARLARK_CONSTRUCTOR.id())
                .cfg(splitTransitionProvider)
                .allowedFileTypes())
        .add(
            attr("feature_flags", LABEL_KEYED_STRING_DICT)
                .undocumented("the feature flag feature has not yet been launched")
                .allowedRuleClasses("config_feature_flag")
                .allowedFileTypes()
                .nonconfigurable("defines an aspect of configuration")
                .mandatoryProviders(ImmutableList.of(ConfigFeatureFlagProvider.id())))
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
                (rule) -> AppleCrosstoolTransition.APPLE_CROSSTOOL_TRANSITION,
                new ConfigFeatureFlagTransitionFactory("feature_flags")))
        .addRequiredToolchains(CppRuleClasses.ccToolchainTypeAttribute(env))
        .useToolchainTransition(true)
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("apple_static_library")
        .factoryClass(AppleStaticLibrary.class)
        .ancestors(
            BaseRuleClasses.NativeBuildRule.class, ObjcRuleClasses.MultiArchPlatformRule.class)
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
