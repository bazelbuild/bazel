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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType;
import com.google.devtools.build.lib.packages.RuleClass.ToolchainTransitionMode;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;
import com.google.devtools.build.lib.rules.config.ConfigFeatureFlagProvider;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration;
import com.google.devtools.build.lib.rules.cpp.CppRuleClasses;

/**
 * Abstract rule definition for apple_static_library.
 */
public class AppleStaticLibraryBaseRule implements RuleDefinition {
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
            AppleConfiguration.class,
            CppConfiguration.class,
            ObjcConfiguration.class,
            J2ObjcConfiguration.class)
        /* <!-- #BLAZE_RULE($apple_static_library_base_rule).ATTRIBUTE(avoid_deps) -->
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
        .addRequiredToolchains(CppRuleClasses.ccToolchainTypeAttribute(env))
        .useToolchainTransition(ToolchainTransitionMode.ENABLED)
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("$apple_static_library_base_rule")
        .type(RuleClassType.ABSTRACT)
        .ancestors(
            BaseRuleClasses.NativeBuildRule.class, ObjcRuleClasses.MultiArchPlatformRule.class)
        .build();
  }
}
