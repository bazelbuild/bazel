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
import static com.google.devtools.build.lib.packages.Type.LABEL;
import static com.google.devtools.build.lib.packages.Type.LABEL_LIST;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.NON_ARC_SRCS_TYPE;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.SRCS_TYPE;

import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder;
import com.google.devtools.build.lib.util.FileType;
import com.google.devtools.build.lib.view.BlazeRule;
import com.google.devtools.build.lib.view.RuleDefinition;
import com.google.devtools.build.lib.view.RuleDefinitionEnvironment;

/**
 * Rule definition for objc_library.
 */
@BlazeRule(name = "objc_library",
    factoryClass = ObjcLibrary.class,
    ancestors = { ObjcRuleClasses.ObjcBaseRule.class,
                  ObjcRuleClasses.ObjcOptsRule.class })
public class ObjcLibraryRule implements RuleDefinition {
  @Override
  public RuleClass build(Builder builder, RuleDefinitionEnvironment env) {
    return builder
        /*<!-- #BLAZE_RULE(objc_library).IMPLICIT_OUTPUTS -->
        <ul>
         <li><code><var>name</var>.xcodeproj/project.pbxproj: An Xcode project file which can be
             used to develop or build on a Mac.</li>
        </ul>
        <!-- #END_BLAZE_RULE.IMPLICIT_OUTPUTS -->*/
        .setImplicitOutputsFunction(ObjcRuleClasses.PBXPROJ)
        /* <!-- #BLAZE_RULE(objc_library).ATTRIBUTE(srcs) -->
        The list of C, C++, Objective-C, and Objective-C++ files that are
        processed to create the library target.
        ${SYNOPSIS}
        These are your checked-in source files, plus any generated files.
        These are compiled into .o files with Clang, so headers should not go
        here (see the hdrs attribute).
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(attr("srcs", LABEL_LIST)
            .direct_compile_time_input()
            .allowedFileTypes(SRCS_TYPE))
        /* <!-- #BLAZE_RULE(objc_library).ATTRIBUTE(non_arc_srcs) -->
        The list of Objective-C files that are processed to create the
        library target that DO NOT use ARC.
        ${SYNOPSIS}
        The files in this attribute are treated very similar to those in the
        srcs attribute, but are compiled without ARC enabled.
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(attr("non_arc_srcs", LABEL_LIST)
            .direct_compile_time_input()
            .allowedFileTypes(NON_ARC_SRCS_TYPE))
        /* <!-- #BLAZE_RULE(objc_library).ATTRIBUTE(pch) -->
        Header file to prepend to every source file being compiled (both arc
        and non-arc). Note that the file will not be precompiled - this is
        simply a convenience, not a build-speed enhancement.
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(attr("pch", LABEL)
            .direct_compile_time_input()
            .allowedFileTypes(FileType.of(".pch")))
        /* <!-- #BLAZE_RULE(objc_library).ATTRIBUTE(options) -->
        An <code>objc_options</code> target which defines an Xcode build
        configuration profile.
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(attr("options", LABEL)
            .allowedFileTypes()
            .allowedRuleClasses("objc_options"))
        /* <!-- #BLAZE_RULE(objc_library).ATTRIBUTE(deps) -->
        The list of <code>objc_*</code> targets that are linked together to
        form the final bundle.
        ${SYNOPSIS}
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .override(attr("deps", LABEL_LIST)
            .direct_compile_time_input()
            .allowedRuleClasses("objc_library", "objc_import", "objc_bundle", "objc_framework",
                "objc_bundle_library")
            .allowedFileTypes())
        .build();
  }
}

/*<!-- #BLAZE_RULE (NAME = objc_library, TYPE = LIBRARY, FAMILY = Objective-C) -->

${ATTRIBUTE_SIGNATURE}

<p>This rule produces a static library from the given Objective-C source files.</p>

${ATTRIBUTE_DEFINITION}

<!-- #END_BLAZE_RULE -->*/
