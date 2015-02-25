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

package com.google.devtools.build.lib.bazel.rules.java;

import static com.google.devtools.build.lib.packages.Attribute.ConfigurationTransition.HOST;
import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.Type.BOOLEAN;
import static com.google.devtools.build.lib.packages.Type.LABEL_LIST;
import static com.google.devtools.build.lib.packages.Type.STRING_LIST;

import com.google.devtools.build.lib.analysis.BlazeRule;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.bazel.rules.java.BazelJavaRuleClasses.JavaRule;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder;

/**
 * Common attributes for Java rules.
 */
@BlazeRule(name = "java_library",
             ancestors = { JavaRule.class },
             factoryClass = BazelJavaLibrary.class)
public final class BazelJavaLibraryRule implements RuleDefinition {
  @Override
  public RuleClass build(Builder builder, final RuleDefinitionEnvironment env) {

    return builder
        .setImplicitOutputsFunction(BazelJavaRuleClasses.JAVA_LIBRARY_IMPLICIT_OUTPUTS)
        .add(attr("exports", LABEL_LIST)
            .allowedRuleClasses(BazelJavaRuleClasses.ALLOWED_RULES_IN_DEPS)
            .allowedFileTypes(/*May not have files in exports!*/))
        .add(attr("neverlink", BOOLEAN).value(false))
        .override(attr("javacopts", STRING_LIST))
        .add(attr("exported_plugins", LABEL_LIST).cfg(HOST).allowedRuleClasses("java_plugin")
            .legacyAllowAnyFileType())
        .build();
  }
}
