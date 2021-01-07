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

package com.google.devtools.build.lib.rules.java;

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;

import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType;
import com.google.devtools.build.lib.rules.java.JavaRuleClasses.JavaToolchainBaseRule;

/** A base rule for libraries which can provide proguard specs. */
public final class ProguardLibraryRule implements RuleDefinition {

  @Override
  public RuleClass build(RuleClass.Builder builder, final RuleDefinitionEnvironment environment) {
    return builder
        /* <!-- #BLAZE_RULE($proguard_library).ATTRIBUTE(proguard_specs) -->
        Files to be used as Proguard specification.
        These will describe the set of specifications to be used by Proguard. If specified,
        they will be added to any <code>android_binary</code> target depending on this library.

        The files included here must only have idempotent rules, namely -dontnote, -dontwarn,
        assumenosideeffects, and rules that start with -keep. Other options can only appear in
        <code>android_binary</code>'s proguard_specs, to ensure non-tautological merges.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("proguard_specs", LABEL_LIST).legacyAllowAnyFileType())
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("$proguard_library")
        .ancestors(JavaToolchainBaseRule.class)
        .type(RuleClassType.ABSTRACT)
        .build();
  }
}
