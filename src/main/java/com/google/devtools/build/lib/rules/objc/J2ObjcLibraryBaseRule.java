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

package com.google.devtools.build.lib.rules.objc;

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static com.google.devtools.build.lib.syntax.Type.STRING_LIST;

import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder;
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType;

/**
 * Abstract rule definition for j2objc_library.
 */
public class J2ObjcLibraryBaseRule implements RuleDefinition {
  @Override
  public RuleClass build(Builder builder, RuleDefinitionEnvironment env) {
    // TODO(rduan): Add support for package prefixes.
    return builder
          /* <!-- #BLAZE_RULE(j2objc_library).ATTRIBUTE(entry_classes) -->
          The list of Java classes whose translated ObjC counterparts will be referenced directly
          by user ObjC code. This attibute is required if flag <code>--j2objc_dead_code_removal
          </code> is on. The Java classes should be specified in their canonical names as defined by
          <a href="http://docs.oracle.com/javase/specs/jls/se8/html/jls-6.html#jls-6.7">the Java
          Language Specification.</a>
          When flag <code>--j2objc_dead_code_removal</code> is specified, the list of entry classes
          will be collected transitively and used as entry points to perform dead code analysis.
          Unused classes will then be removed from the final ObjC app bundle.
          <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(attr("entry_classes", STRING_LIST))
        .add(attr("$jre_emul_lib", LABEL)
            .value(env.getLabel("//third_party/java/j2objc:jre_emul_lib")))
        .add(attr("$protobuf_lib", LABEL)
            .value(env.getLabel("//third_party/java/j2objc:proto_runtime")))
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("$j2objc_library_base")
        .type(RuleClassType.ABSTRACT)
        .ancestors(BaseRuleClasses.BaseRule.class)
        .build();
  }
}

/*<!-- #BLAZE_RULE (NAME = j2objc_library, TYPE = LIBRARY, FAMILY = Objective-C) -->

<p> This rule uses <a href="https://github.com/google/j2objc">J2ObjC</a> to translate Java source
files to Objective-C, which then can be used used as dependencies of objc_library and objc_binary
rules. Detailed information about J2ObjC itself can be found at  <a href="http://j2objc.org">the
J2ObjC site</a>
</p>
<p>Custom J2ObjC transpilation flags can be specified using the build flag
<code>--j2objc_translation_flags</code> in the command line.
</p>
<p>Please note that currently the translated files included in a j2objc_library target will be
compiled using the same compilation configuration as the top level objc_binary target that depends
on the j2objc_library target.
</p>
<p>Plus, generated code is de-duplicated at target level, not source level. If you have two
different Java targets that include the same Java source files, you may see a duplicate symbol error
at link time. The correct way to resolve this issue is to move the shared Java source files into a
separate common target that can be depended upon.
</p>


<!-- #END_BLAZE_RULE -->*/
