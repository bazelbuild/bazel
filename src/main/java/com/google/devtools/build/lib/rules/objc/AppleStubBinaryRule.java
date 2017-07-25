// Copyright 2017 The Bazel Authors. All rights reserved.
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
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;
import static com.google.devtools.build.lib.syntax.Type.STRING;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;

/** Rule definition for apple_stub_binary. */
public class AppleStubBinaryRule implements RuleDefinition {

  public static final String XCENV_BASED_PATH_ATTR = "xcenv_based_path";

  @Override
  public RuleClass build(Builder builder, RuleDefinitionEnvironment env) {
    MultiArchSplitTransitionProvider splitTransitionProvider =
        new MultiArchSplitTransitionProvider();
    return builder
        .requiresConfigurationFragments(ObjcConfiguration.class, AppleConfiguration.class)
        /* <!-- #BLAZE_RULE(apple_stub_binary).ATTRIBUTE(xcenv_based_path) -->
        The path to the stub executable within an Xcode platform or SDK bundle. This path must be
        rooted at either <code>$(SDKROOT)</code> or <code>$(PLATFORM_DIR)</code> (written with
        parentheses, as they are "Make" variables, not shell variables).
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(attr(XCENV_BASED_PATH_ATTR, STRING))
        /* <!-- #BLAZE_RULE(apple_stub_binary).ATTRIBUTE(deps) -->
        The list of targets whose resources will be included in the final bundle.
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(
            attr("deps", LABEL_LIST)
                .direct_compile_time_input()
                .mandatoryNativeProviders(
                    ImmutableList.<Class<? extends TransitiveInfoProvider>>of(ObjcProvider.class))
                .allowedFileTypes()
                .cfg(splitTransitionProvider))
        /*<!-- #BLAZE_RULE(apple_stub_binary).IMPLICIT_OUTPUTS -->
        <ul>
         <li><code><var>name</var></code>: the stub executable copied from the Xcode
             platform/SDK location.</li>
        </ul>
        <!-- #END_BLAZE_RULE.IMPLICIT_OUTPUTS -->*/
        .setImplicitOutputsFunction(
            ImplicitOutputsFunction.fromFunctions(ObjcRuleClasses.LIPOBIN_OUTPUT))
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("apple_stub_binary")
        .factoryClass(AppleStubBinary.class)
        .ancestors(
            BaseRuleClasses.BaseRule.class,
            ObjcRuleClasses.PlatformRule.class,
            ObjcRuleClasses.XcrunRule.class)
        .build();
  }
}

/*<!-- #BLAZE_RULE (NAME = apple_stub_binary, TYPE = BINARY, FAMILY = Objective-C) -->

<p>This rule copies a stub executable (such as that used by a watchOS application or an iMessage
sticker pack) from a location in an Xcode platform or SDK bundle to the output.</p>

<p>This rule is meant to be used internally by the Apple bundling rules and does has limited use
outside of that. Its purpose is to provide target uniformity across all Apple bundles whether they
contain a user binary or a stub binary and to ensure that the platform transition is still present
for those stubs so that platform selection works as expected in downstream resource dependencies.
</p>

${IMPLICIT_OUTPUTS}

<!-- #END_BLAZE_RULE -->*/
