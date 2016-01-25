// Copyright 2014 The Bazel Authors. All rights reserved.
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
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.PLIST_TYPE;

import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder;
import com.google.devtools.build.lib.syntax.Type;

/**
 * Rule definition for {@code objc_options}.
 */
public class ObjcOptionsRule implements RuleDefinition {
  @Override
  public RuleClass build(Builder builder, RuleDefinitionEnvironment env) {
    return builder
        // TODO(bazel-team): Delete this class and merge ObjcOptsRule with CompilingRule.
        .setUndocumented()
        /* <!-- #BLAZE_RULE(objc_options).ATTRIBUTE(xcode_name)[DEPRECATED] -->
        This attribute is ignored and will be removed.
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(attr("xcode_name", Type.STRING))
        /* <!-- #BLAZE_RULE(objc_options).ATTRIBUTE(infoplists) -->
        infoplist files to merge with the final binary's infoplist. This
        corresponds to a single file <i>appname</i>-Info.plist in Xcode
        projects.
        <i>(List of <a href="../build-ref.html#labels">labels</a>; optional)</i>
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(attr("infoplists", BuildType.LABEL_LIST)
            .allowedFileTypes(PLIST_TYPE))
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("objc_options")
        .factoryClass(ObjcOptions.class)
        .ancestors(BaseRuleClasses.BaseRule.class, ObjcRuleClasses.CoptsRule.class)
        .build();
  }
}

/*<!-- #BLAZE_RULE (NAME = objc_options, TYPE = OTHER, FAMILY = Objective-C) -->

${ATTRIBUTE_SIGNATURE}

<p>This rule provides a nameable set of build settings to use when building
Objective-C targets.</p>

${ATTRIBUTE_DEFINITION}

<!-- #END_BLAZE_RULE -->*/
