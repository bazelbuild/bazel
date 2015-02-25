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
import static com.google.devtools.build.lib.packages.Type.BOOLEAN;
import static com.google.devtools.build.lib.packages.Type.LABEL;
import static com.google.devtools.build.lib.packages.Type.LABEL_LIST;
import static com.google.devtools.build.lib.packages.Type.STRING_LIST;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.NON_ARC_SRCS_TYPE;
import static com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.SRCS_TYPE;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.BlazeRule;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder;
import com.google.devtools.build.lib.rules.objc.ObjcRuleClasses.ObjcCompilationRule;
import com.google.devtools.build.lib.util.FileType;

/**
 * Rule definition for objc_library.
 */
@BlazeRule(name = "objc_library",
    factoryClass = ObjcLibrary.class,
    ancestors = { ObjcCompilationRule.class,
                  ObjcRuleClasses.ObjcOptsRule.class })
public class ObjcLibraryRule implements RuleDefinition {
  private static final Iterable<String> ALLOWED_DEPS_RULE_CLASSES = ImmutableSet.of(
      "objc_library",
      "objc_import",
      "objc_bundle",
      "objc_framework",
      "objc_bundle_library",
      "objc_proto_library",
      "j2objc_library");

  @Override
  public RuleClass build(Builder builder, RuleDefinitionEnvironment env) {
    return builder
        /*<!-- #BLAZE_RULE(objc_library).IMPLICIT_OUTPUTS -->
        <ul>
         <li><code><var>name</var>.xcodeproj/project.pbxproj</code>: An Xcode project file which
             can be used to develop or build on a Mac.</li>
        </ul>
        <!-- #END_BLAZE_RULE.IMPLICIT_OUTPUTS -->*/
        .setImplicitOutputsFunction(XcodeSupport.PBXPROJ)
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
        .add(attr("options", LABEL)
            .undocumented("objc_options will probably be removed")
            .allowedFileTypes()
            .allowedRuleClasses("objc_options"))
        /* <!-- #BLAZE_RULE(objc_library).ATTRIBUTE(alwayslink) -->
        If 1, any bundle or binary that depends (directly or indirectly) on this
        library will link in all the object files for the files listed in
        <code>srcs</code> and <code>non_arc_srcs</code>, even if some contain no
        symbols referenced by the binary.
        ${SYNOPSIS}
        This is useful if your code isn't explicitly called by code in
        the binary, e.g., if your code registers to receive some callback
        provided by some service.
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(attr("alwayslink", BOOLEAN))
        /* <!-- #BLAZE_RULE(objc_library).ATTRIBUTE(deps) -->
        The list of targets that are linked together to form the final bundle.
        ${SYNOPSIS}
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .override(attr("deps", LABEL_LIST)
            .direct_compile_time_input()
            .allowedRuleClasses(ALLOWED_DEPS_RULE_CLASSES)
            .allowedFileTypes())
        /* <!-- #BLAZE_RULE(objc_library).ATTRIBUTE(bundles) -->
        The list of bundle targets that this target requires to be included in the final bundle.
        ${SYNOPSIS}
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(attr("bundles", LABEL_LIST)
            .direct_compile_time_input()
            .allowedRuleClasses("objc_bundle", "objc_bundle_library")
            .allowedFileTypes())
        /* <!-- #BLAZE_RULE(objc_library).ATTRIBUTE(non_propagated_deps) -->
        The list of targets that are required in order to build this target,
        but which are not included in the final bundle.
        <br />
        This attribute should only rarely be used, and probably only for proto
        dependencies.
        ${SYNOPSIS}
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(attr("non_propagated_deps", LABEL_LIST)
            .direct_compile_time_input()
            .allowedRuleClasses(ALLOWED_DEPS_RULE_CLASSES)
            .allowedFileTypes())
        /* <!-- #BLAZE_RULE(objc_library).ATTRIBUTE(defines) -->
        Extra <code>-D</code> flags to pass to the compiler. They should be in
        the form <code>KEY=VALUE</code> or simply <code>KEY</code> and are
        passed not only the compiler for this target (as <code>copts</code>
        are) but also to all <code>objc_</code> dependers of this target.
        ${SYNOPSIS}
        Subject to <a href="#make_variables">"Make variable"</a> substitution and
        <a href="#sh-tokenization">Bourne shell tokenization</a>.
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(attr("defines", STRING_LIST))
        .build();
  }
}

/*<!-- #BLAZE_RULE (NAME = objc_library, TYPE = LIBRARY, FAMILY = Objective-C) -->

${ATTRIBUTE_SIGNATURE}

<p>This rule produces a static library from the given Objective-C source files.</p>

${IMPLICIT_OUTPUTS}

${ATTRIBUTE_DEFINITION}

<!-- #END_BLAZE_RULE -->*/
