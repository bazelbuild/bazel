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

package com.google.devtools.build.lib.bazel.rules.objc;

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.Type.LABEL_LIST;

import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder;
import com.google.devtools.build.lib.rules.java.J2ObjcConfiguration;
import com.google.devtools.build.lib.rules.objc.J2ObjcAspect;
import com.google.devtools.build.lib.rules.objc.J2ObjcLibrary;
import com.google.devtools.build.lib.rules.objc.J2ObjcLibraryBaseRule;
import com.google.devtools.build.lib.rules.objc.ObjcConfiguration;

/**
 * Concrete implementation of J2ObjCLibraryBaseRule.
 */
public class BazelJ2ObjcLibraryRule implements RuleDefinition {

  @Override
  public RuleClass build(Builder builder, RuleDefinitionEnvironment env) {
    return builder
        .requiresConfigurationFragments(J2ObjcConfiguration.class, ObjcConfiguration.class)
          /* <!-- #BLAZE_RULE(j2objc_library).ATTRIBUTE(deps) -->
          A list of <code>j2objc_library</code>, <code>java_library</code>
          and <code>java_import</code> targets that contain
          Java files to be transpiled to Objective-C.
          ${SYNOPSIS}
          <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .add(attr("deps", LABEL_LIST)
            .aspect(J2ObjcAspect.class)
            .direct_compile_time_input()
            .allowedRuleClasses("j2objc_library", "java_library", "java_import")
            .allowedFileTypes())
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("j2objc_library")
        .factoryClass(J2ObjcLibrary.class)
        .ancestors(J2ObjcLibraryBaseRule.class)
        .build();
  }
}
