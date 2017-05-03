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

package com.google.devtools.build.lib.rules.objc;

import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration;

/**
 * Rule definition for ios_extension_binary.
 *
 * @deprecated The native bundling rules have been deprecated. This class will be removed in the
 *     future.
 */
@Deprecated
public class IosExtensionBinaryRule implements RuleDefinition {
  @Override
  public RuleClass build(Builder builder, RuleDefinitionEnvironment env) {
    return builder
        .requiresConfigurationFragments(ObjcConfiguration.class, J2ObjcConfiguration.class,
            AppleConfiguration.class, CppConfiguration.class)
        .cfg(AppleCrosstoolTransition.APPLE_CROSSTOOL_TRANSITION)
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("ios_extension_binary")
        .factoryClass(IosExtensionBinary.class)
        .ancestors(BaseRuleClasses.BaseRule.class, ObjcRuleClasses.LinkingRule.class,
            ObjcRuleClasses.XcodegenRule.class)
        .build();
  }
}

/*<!-- #BLAZE_RULE (NAME = ios_extension_binary, TYPE = BINARY, FAMILY = Objective-C) -->

<p><strong>This rule is deprecated.</strong> Please use the new Apple build rules
(<a href="https://github.com/bazelbuild/rules_apple">https://github.com/bazelbuild/rules_apple</a>)
to build Apple targets.</p>

<p>This rule produces a binary for an iOS app extension by linking one or more
Objective-C libraries.</p>

${IMPLICIT_OUTPUTS}

<!-- #END_BLAZE_RULE -->*/
