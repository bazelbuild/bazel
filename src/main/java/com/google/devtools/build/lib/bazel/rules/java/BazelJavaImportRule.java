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

package com.google.devtools.build.lib.bazel.rules.java;

import static com.google.devtools.build.lib.packages.Attribute.ANY_EDGE;
import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.rules.java.JavaImportBaseRule;
import com.google.devtools.build.lib.rules.java.JavaRuleClasses.JavaToolchainBaseRule;
import com.google.devtools.build.lib.rules.java.JavaSemantics;

/**
 * Rule definition for the java_import rule.
 *
 * <p>This rule is implemented in Starlark. This class remains only for doc-gen purposes.
 */
public final class BazelJavaImportRule implements RuleDefinition {

  private static final ImmutableSet<String> ALLOWED_DEPS =
      ImmutableSet.of("java_library", "java_import", "cc_library", "cc_binary");

  @Override
  public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
    return builder
        /* <!-- #BLAZE_RULE(java_import).ATTRIBUTE(deps) -->
        The list of other libraries to be linked in to the target.
        See <a href="${link java_library.deps}">java_library.deps</a>.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(
            attr("deps", LABEL_LIST)
                .allowedRuleClasses(ALLOWED_DEPS)
                .allowedFileTypes() // none allowed
                .validityPredicate(ANY_EDGE)
                .mandatoryProvidersList(BazelJavaRuleClasses.MANDATORY_JAVA_PROVIDER_ONLY))
        /* <!-- #BLAZE_RULE(java_import).ATTRIBUTE(exports) -->
        Targets to make available to users of this rule.
        See <a href="${link java_library.exports}">java_library.exports</a>.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(
            attr("exports", LABEL_LIST)
                .allowedRuleClasses(ALLOWED_DEPS)
                .allowedFileTypes() // none allowed
                .validityPredicate(ANY_EDGE)
                .mandatoryProvidersList(BazelJavaRuleClasses.MANDATORY_JAVA_PROVIDER_ONLY))
        /* <!-- #BLAZE_RULE(java_import).ATTRIBUTE(runtime_deps) -->
        Libraries to make available to the final binary or test at runtime only.
        See <a href="${link java_library.runtime_deps}">java_library.runtime_deps</a>.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(
            attr("runtime_deps", LABEL_LIST)
                .allowedFileTypes(JavaSemantics.JAR)
                .allowedRuleClasses(ALLOWED_DEPS)
                .mandatoryProvidersList(BazelJavaRuleClasses.MANDATORY_JAVA_PROVIDER_ONLY)
                .skipAnalysisTimeFileTypeCheck())
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("java_import")
        .ancestors(JavaImportBaseRule.class, JavaToolchainBaseRule.class)
        .factoryClass(BaseRuleClasses.EmptyRuleConfiguredTargetFactory.class)
        .build();
  }
}

/*<!-- #BLAZE_RULE (NAME = java_import, TYPE = LIBRARY, FAMILY = Java) -->

<p>
  This rule allows the use of precompiled <code>.jar</code> files as
  libraries for <code><a href="${link java_library}">java_library</a></code> and
  <code><a href="${link java_binary}">java_binary</a></code> rules.
</p>

<h4 id="java_import_examples">Examples</h4>

<pre class="code">
    java_import(
        name = "maven_model",
        jars = [
            "maven_model/maven-aether-provider-3.2.3.jar",
            "maven_model/maven-model-3.2.3.jar",
            "maven_model/maven-model-builder-3.2.3.jar",
        ],
    )
</pre>

<!-- #END_BLAZE_RULE -->*/
