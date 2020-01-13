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
import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;

import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.RuleClass;

/** Rule definition for {@code available_xcodes} rule. */
public class AvailableXcodesRule implements RuleDefinition {

  static final String DEFAULT_ATTR_NAME = "default";
  static final String VERSIONS_ATTR_NAME = "versions";

  @Override
  public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
    return builder
        .requiresConfigurationFragments(AppleConfiguration.class)
        .exemptFromConstraintChecking(
            "this rule refines configuration variables and does not build actual content")
        /* <!-- #BLAZE_RULE(available_xcodes).ATTRIBUTE(versions) -->
        The xcode versions that are available on this platform.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(
            attr(DEFAULT_ATTR_NAME, LABEL)
                .allowedRuleClasses("xcode_version")
                .allowedFileTypes()
                .mandatory()
                .nonconfigurable("this rule determines configuration"))
        /* <!-- #BLAZE_RULE(available_xcodes).ATTRIBUTE(default) -->
        The default xcode version for this platform.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(
            attr(VERSIONS_ATTR_NAME, LABEL_LIST)
                .allowedRuleClasses("xcode_version")
                .allowedFileTypes()
                .nonconfigurable("this rule determines configuration"))
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("available_xcodes")
        .ancestors(BaseRuleClasses.BaseRule.class)
        .factoryClass(AvailableXcodes.class)
        .build();
  }
}

/*<!-- #BLAZE_RULE (NAME = available_xcodes, TYPE = OTHER, FAMILY = Workspace)[GENERIC_RULE] -->

<p>Two targets of this rule can be depended on by an <code>xcode_config</code> rule instance to
indicate the remotely and locally available xcode versions.
This allows selection of an official xcode version from the collectively available xcodes.</p>

<!-- #END_BLAZE_RULE -->*/
