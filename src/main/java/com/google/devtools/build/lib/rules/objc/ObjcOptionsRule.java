// Copyright 2014 Google Inc. All rights reserved.
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

import com.google.devtools.build.lib.packages.Attribute.ComputedDefault;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.ObjcOptsRule;
import com.google.devtools.build.lib.view.BaseRuleClasses.BaseRule;
import com.google.devtools.build.lib.view.BlazeRule;
import com.google.devtools.build.lib.view.RuleDefinition;
import com.google.devtools.build.lib.view.RuleDefinitionEnvironment;

/**
 * Rule definition for {@code objc_options}.
 */
@BlazeRule(name = "objc_options",
    factoryClass = ObjcOptions.class,
    ancestors = { BaseRule.class, ObjcOptsRule.class })
public class ObjcOptionsRule implements RuleDefinition {
  @Override
  public RuleClass build(Builder builder, RuleDefinitionEnvironment env) {
    return builder
        /* <!-- #BLAZE_RULE(objc_options).ATTRIBUTE(xcode_name) -->
        The name of the build settings as they will appear in the Xcode
        project, for instance "Release" or "Debug". If omitted, this is set to
        the rule's name.
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(attr("xcode_name", Type.STRING).value(new ComputedDefault() {
          @Override
          public Object getDefault(AttributeMap rule) {
            return rule.getName();
          }
        }))
        .build();
  }
}

/*<!-- #BLAZE_RULE (NAME = objc_options, TYPE = OTHER, FAMILY = Objective-C) -->

${ATTRIBUTE_SIGNATURE}

<p>This rule provides a nameable set of build settings to use when building
Objective-C targets.</p>

${ATTRIBUTE_DEFINITION}

<!-- #END_BLAZE_RULE -->*/
