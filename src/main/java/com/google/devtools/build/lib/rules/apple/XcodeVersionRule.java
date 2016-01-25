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
import static com.google.devtools.build.lib.syntax.Type.STRING;
import static com.google.devtools.build.lib.syntax.Type.STRING_LIST;

import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder;

/**
 * Rule definition for {@code xcode_version} rule.
 */
public class XcodeVersionRule implements RuleDefinition {

  static final String VERSION_ATTR_NAME = "version";
  static final String ALIASES_ATTR_NAME = "aliases";
  
  @Override
  public RuleClass build(Builder builder, RuleDefinitionEnvironment env) {
    return builder
        .requiresConfigurationFragments(AppleConfiguration.class)
        .exemptFromConstraintChecking(
            "this rule refines configuration variables and does not build actual content")
        /* <!-- #BLAZE_RULE(proto_library).ATTRIBUTE(version) -->
        The official version number of a version of Xcode.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr(VERSION_ATTR_NAME, STRING).mandatory())
        /* <!-- #BLAZE_RULE(proto_library).ATTRIBUTE(version) -->
        Accepted aliases for this version of Xcode.
        ${SYNOPSIS}
        If the value of the <code>xcode_version</code> build flag matches any of the given
        alias strings, this xcode version will be used.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr(ALIASES_ATTR_NAME, STRING_LIST))
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("xcode_version")
        .ancestors(BaseRuleClasses.BaseRule.class)
        .factoryClass(XcodeVersion.class)
        .build();
  }
}

/*<!-- #BLAZE_RULE (NAME = xcode_version, TYPE = OTHER, FAMILY = Workspace)[GENERIC_RULE] -->

<p>Represents a single official xcode version with acceptable aliases for that xcode version.
See the <code>xcode_config</code> rule.</p>

<!-- #END_BLAZE_RULE -->*/
