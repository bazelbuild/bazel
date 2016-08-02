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

import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;

/** Rule definition for apple_watch2_extension. */
public class AppleWatch2ExtensionRule implements RuleDefinition {

  @Override
  public RuleClass build(Builder builder, RuleDefinitionEnvironment env) {
    return builder
        .requiresConfigurationFragments(ObjcConfiguration.class, AppleConfiguration.class)
        /* <!-- #BLAZE_RULE(apple_watch2_extension).ATTRIBUTE(binary) -->
        The binary target containing the logic for the watch extension.
        ${SYNOPSIS}
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(
            attr("binary", LABEL)
                .allowedRuleClasses("apple_binary")
                .allowedFileTypes()
                .mandatory()
                .direct_compile_time_input())
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("apple_watch2_extension")
        .factoryClass(AppleWatch2Extension.class)
        .ancestors(
            BaseRuleClasses.BaseRule.class,
            ObjcRuleClasses.XcodegenRule.class,
            ObjcRuleClasses.WatchApplicationBundleRule.class,
            ObjcRuleClasses.WatchExtensionBundleRule.class)
        .build();
  }
}

/*<!-- #BLAZE_RULE (NAME = apple_watch2_extension, TYPE = BINARY, FAMILY = Objective-C) -->

<p>This rule produces an extension bundle for apple watch OS 2.</p>

<p>It requires attributes set for both the watchOS2 application and watchOS2 extension that will be
   present in any final ios application bundle. Application attributes are prefixed with app_, and
   extension attributes prefixed with ext_.</p>

<p>The required 'binary' attribute should contain the apple_binary extension binary (built for
   the watch platform type.</p>

${IMPLICIT_OUTPUTS}

${ATTRIBUTE_DEFINITION}

<!-- #END_BLAZE_RULE -->*/
