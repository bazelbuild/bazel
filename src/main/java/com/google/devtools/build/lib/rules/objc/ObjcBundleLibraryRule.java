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

import static com.google.devtools.build.lib.packages.Attribute.ConfigurationTransition.HOST;
import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.Type.LABEL;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.PLIST_TYPE;

import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder;
import com.google.devtools.build.lib.view.BlazeRule;
import com.google.devtools.build.lib.view.RuleDefinition;
import com.google.devtools.build.lib.view.RuleDefinitionEnvironment;

/**
 * Rule definition for objc_bundle_library.
 */
@BlazeRule(name = "objc_bundle_library",
    factoryClass = ObjcBundleLibrary.class,
    ancestors = { ObjcLibraryRule.class })
public class ObjcBundleLibraryRule implements RuleDefinition {
  @Override
  public RuleClass build(Builder builder, RuleDefinitionEnvironment env) {
    return builder
        /* <!-- #BLAZE_RULE(objc_bundle_library).ATTRIBUTE(infoplist) -->
        The infoplist file. This corresponds to <i>appname</i>-Info.plist in Xcode projects.
        ${SYNOPSIS}
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(attr("infoplist", LABEL)
            .allowedFileTypes(PLIST_TYPE))
        .add(attr("$actoolzip_deploy", LABEL).cfg(HOST)
            .value(env.getLabel("//tools/objc:actoolzip_deploy.jar")))
        .build();
  }
}

/*<!-- #BLAZE_RULE (NAME = objc_bundle_library, TYPE = LIBRARY, FAMILY = Objective-C) -->

${ATTRIBUTE_SIGNATURE}

<p>This rule encapsulates a library which is provided to dependers as a bundle.
It is similar to <code>objc_library</code> with the key difference being that
with <code>objc_bundle_libary</code>, the resources and binary are put in a
nested bundle in the final iOS application, whereas with a normal
<code>objc_library</code>, the resources are placed in the same bundle as the
application and the libraries are linked into the main application binary.

${ATTRIBUTE_DEFINITION}

<!-- #END_BLAZE_RULE -->*/
