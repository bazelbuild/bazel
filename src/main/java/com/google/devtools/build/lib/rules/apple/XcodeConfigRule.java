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
import static com.google.devtools.build.lib.syntax.Type.BOOLEAN;

import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder;

/**
 * Rule definition for {@code xcode_config} rule.
 */
public class XcodeConfigRule implements RuleDefinition {

  static final String DEFAULT_ATTR_NAME = "default";
  static final String VERSIONS_ATTR_NAME = "versions";
  static final String REQUIRE_DEFINED_VERSIONS_ATTR_NAME = "require_defined_version";

  @Override
  public RuleClass build(Builder builder, RuleDefinitionEnvironment env) {
    return builder
        .requiresConfigurationFragments(AppleConfiguration.class)
        .exemptFromConstraintChecking(
            "this rule refines configuration variables and does not build actual content")
        /* <!-- #BLAZE_RULE(proto_library).ATTRIBUTE(version) -->
        The default official version of xcode to use.
        ${SYNOPSIS}
        The version specified by the provided <code>xcode_version</code> target is to be used if
        no <code>xcode_version</code> build flag is specified.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr(DEFAULT_ATTR_NAME, LABEL)
            .allowedRuleClasses("xcode_version")
            .allowedFileTypes())
        /* <!-- #BLAZE_RULE(proto_library).ATTRIBUTE(version) -->
        Accepted <code>xcode_version<code> targets that may be used.
        ${SYNOPSIS}
        If the value of the <code>xcode_version</code> build flag matches one of the aliases
        or version number of any of the given <code>xcode_version</code> targets, the matching
        target will be used.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr(VERSIONS_ATTR_NAME, LABEL_LIST)
            .allowedRuleClasses("xcode_version")
            .allowedFileTypes())
        /* <!-- #BLAZE_RULE(proto_library).ATTRIBUTE(version) -->
        Whether to require the build's xcode version match one of the declared targets.
        ${SYNOPSIS}
        If true, this will raise an error if either the <code>xcode_version</code> flag value
        or <code>default</code> attribute value do not match one of the versions declared
        among <code>xcode_verison</code> targets.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr(REQUIRE_DEFINED_VERSIONS_ATTR_NAME, BOOLEAN).value(false))
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("xcode_config")
        .ancestors(BaseRuleClasses.BaseRule.class)
        .factoryClass(XcodeConfig.class)
        .build();
  }
}

/*<!-- #BLAZE_RULE (NAME = xcode_config, TYPE = OTHER, FAMILY = Workspace)[GENERIC_RULE] -->

${ATTRIBUTE_SIGNATURE}

<p>A single target of this rule can be referenced by the <code>--xcode_config</code> build flag to
translate the <code>--xcode_version</code> flag into an accepted official xcode version. This
allows selection of a an official xcode version from a number of registered aliases.</p>

${ATTRIBUTE_DEFINITION}

<!-- #END_BLAZE_RULE -->*/
