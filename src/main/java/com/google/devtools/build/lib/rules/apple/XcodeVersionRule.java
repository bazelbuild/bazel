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

package com.google.devtools.build.lib.rules.apple;

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.Type.STRING;
import static com.google.devtools.build.lib.packages.Types.STRING_LIST;

import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.BaseRuleClasses.EmptyRuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.RuleClass;

/**
 * Rule definition for {@code xcode_version} rule.
 */
public class XcodeVersionRule implements RuleDefinition {

  static final String VERSION_ATTR_NAME = "version";
  static final String ALIASES_ATTR_NAME = "aliases";
  static final String DEFAULT_IOS_SDK_VERSION_ATTR_NAME = "default_ios_sdk_version";
  static final String DEFAULT_VISIONOS_SDK_VERSION_ATTR_NAME = "default_visionos_sdk_version";
  static final String DEFAULT_WATCHOS_SDK_VERSION_ATTR_NAME = "default_watchos_sdk_version";
  static final String DEFAULT_TVOS_SDK_VERSION_ATTR_NAME = "default_tvos_sdk_version";
  static final String DEFAULT_MACOS_SDK_VERSION_ATTR_NAME = "default_macos_sdk_version";

  @Override
  public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
    return builder
        .requiresConfigurationFragments(AppleConfiguration.class)
        .exemptFromConstraintChecking(
            "this rule refines configuration variables and does not build actual content")
        /* <!-- #BLAZE_RULE(xcode_version).ATTRIBUTE(version) -->
        The official version number of a version of Xcode.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(
            attr(VERSION_ATTR_NAME, STRING)
                .mandatory()
                .nonconfigurable("this rule determines configuration"))
        /* <!-- #BLAZE_RULE(xcode_version).ATTRIBUTE(version) -->
        Accepted aliases for this version of Xcode.
        If the value of the <code>xcode_version</code> build flag matches any of the given
        alias strings, this xcode version will be used.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(
            attr(ALIASES_ATTR_NAME, STRING_LIST)
                .nonconfigurable("this rule determines configuration"))

        /* <!-- #BLAZE_RULE(xcode_version).ATTRIBUTE(default_ios_sdk_version) -->
        The ios sdk version that is used by default when this version of xcode is being used.
        The <code>ios_sdk_version</code> build flag will override the value specified here.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(
            attr(DEFAULT_IOS_SDK_VERSION_ATTR_NAME, STRING)
                .nonconfigurable("this rule determines configuration"))
        /* <!-- #BLAZE_RULE(xcode_version).ATTRIBUTE(default_visionos_sdk_version) -->
        The visionos sdk version that is used by default when this version of xcode is being used.
        The <code>visionos_sdk_version</code> build flag will override the value specified here.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(
            attr(DEFAULT_VISIONOS_SDK_VERSION_ATTR_NAME, STRING)
                .nonconfigurable("this rule determines configuration"))
        /* <!-- #BLAZE_RULE(xcode_version).ATTRIBUTE(default_watchos_sdk_version) -->
        The watchos sdk version that is used by default when this version of xcode is being used.
        The <code>watchos_sdk_version</code> build flag will override the value specified here.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(
            attr(DEFAULT_WATCHOS_SDK_VERSION_ATTR_NAME, STRING)
                .nonconfigurable("this rule determines configuration"))
        /* <!-- #BLAZE_RULE(xcode_version).ATTRIBUTE(default_tvos_sdk_version) -->
        The tvos sdk version that is used by default when this version of xcode is being used.
        The <code>tvos_sdk_version</code> build flag will override the value specified here.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(
            attr(DEFAULT_TVOS_SDK_VERSION_ATTR_NAME, STRING)
                .nonconfigurable("this rule determines configuration"))
        /* <!-- #BLAZE_RULE(xcode_version).ATTRIBUTE(default_macos_sdk_version) -->
        The macosx sdk version that is used by default when this version of xcode is being used.
        The <code>macos_sdk_version</code> build flag will override the value specified here.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(
            attr(DEFAULT_MACOS_SDK_VERSION_ATTR_NAME, STRING)
                .nonconfigurable("this rule determines configuration"))
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("xcode_version")
        .ancestors(BaseRuleClasses.NativeBuildRule.class)
        .factoryClass(EmptyRuleConfiguredTargetFactory.class)
        .build();
  }
}

/*<!-- #BLAZE_RULE (NAME = xcode_version, TYPE = OTHER, FAMILY = Objective-C) -->

<p>Represents a single official xcode version with acceptable aliases for that xcode version.
See the <code>xcode_config</code> rule.</p>

<!-- #END_BLAZE_RULE -->*/
