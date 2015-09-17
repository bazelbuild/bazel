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
import static com.google.devtools.build.lib.packages.Type.LABEL;
import static com.google.devtools.build.lib.packages.Type.STRING_LIST;

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
          ${SYNOPSIS}
          When flag <code>--j2objc_dead_code_removal</code> is specified, the list of entry classes
          will be collected transitively and used as entry points to perform dead code analysis.
          Unused classes will then be removed from the final ObjC app bundle.
          <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(attr("entry_classes", STRING_LIST))
        .add(attr("$jre_emul_lib", LABEL)
            .value(env.getLabel("//third_party/java/j2objc:jre_emul_lib")))
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("$j2objc_library_base")
        .type(RuleClassType.ABSTRACT)
        .ancestors(BaseRuleClasses.BaseRule.class, ObjcRuleClasses.CoptsRule.class)
        .build();
  }
}

/*<!-- #BLAZE_RULE (NAME = j2objc_library, TYPE = LIBRARY, FAMILY = Objective-C) -->

${ATTRIBUTE_SIGNATURE}

<p>This rule uses <a href="https://github.com/google/j2objc">J2ObjC</a>
to translate Java source files to Objective-C, which then can be used used as dependencies of
<code>objc_library</code> and <code>objc_binary</code> rules. More information about J2ObjC
can be found <a href="http://j2objc.org">here</a>.
</p>


${ATTRIBUTE_DEFINITION}

<!-- #END_BLAZE_RULE -->*/
