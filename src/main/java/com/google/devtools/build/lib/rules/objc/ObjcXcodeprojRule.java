// Copyright 2015 Google Inc. All rights reserved.
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
import static com.google.devtools.build.lib.packages.Type.BOOLEAN;
import static com.google.devtools.build.lib.packages.Type.LABEL_LIST;

import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.BlazeRule;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder;

/**
 * Rule definition for {@code objc_xcodeproj}.
 */
@BlazeRule(name = "objc_xcodeproj",
    factoryClass = ObjcXcodeproj.class,
    ancestors = {
        BaseRuleClasses.BaseRule.class,
        ObjcRuleClasses.XcodegenRule.class })
public class ObjcXcodeprojRule implements RuleDefinition {
  @Override
  public RuleClass build(Builder builder, RuleDefinitionEnvironment env) {
    return builder
        /*<!-- #BLAZE_RULE(objc_xcodeproj).IMPLICIT_OUTPUTS -->
        <ul>
        <li><code><var>name</var>.xcodeproj/project.pbxproj</code>: A combined Xcode project file
            containing all the included targets which can be used to develop or build on a Mac.</li>
        </ul>
        <!-- #END_BLAZE_RULE.IMPLICIT_OUTPUTS -->*/
        .setImplicitOutputsFunction(XcodeSupport.PBXPROJ)
        /* <!-- #BLAZE_RULE(objc_xcodeproj).ATTRIBUTE(deps) -->
        The list of targets to include in the combined Xcode project file.
        ${SYNOPSIS}
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(attr("deps", LABEL_LIST)
            .nonEmpty()
            .allowedRuleClasses(
                "objc_binary",
                "ios_extension_binary",
                "ios_test",
                "objc_bundle_library",
                "objc_import",
                "objc_library")
            .allowedFileTypes())
        .override(attr("testonly", BOOLEAN)
            .nonconfigurable("Must support test deps.")
            .value(true))
        .build();
  }
}

/*<!-- #BLAZE_RULE (NAME = objc_xcodeproj, TYPE = OTHER, FAMILY = Objective-C) -->

${ATTRIBUTE_SIGNATURE}

<p>This rule combines build information about several objc targets (and all their transitive
dependencies) into a single Xcode project file, for use in developing on a Mac.</p>

${ATTRIBUTE_DEFINITION}

<!-- #END_BLAZE_RULE -->*/
