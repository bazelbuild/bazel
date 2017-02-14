// Copyright 2017 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.rules.java;

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LICENSE;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder;
import com.google.devtools.build.lib.util.FileTypeSet;

/** Rule definition for {@code java_runtime_suite} */
public final class JavaRuntimeSuiteRule implements RuleDefinition {
  @Override
  public RuleClass build(Builder builder, RuleDefinitionEnvironment env) {
    return builder
        /* <!-- #BLAZE_RULE(java_runtime_suite).ATTRIBUTE(runtimes) -->
        A map from each supported architecture to the corresponding <code>java_runtime</code>.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("runtimes", BuildType.LABEL_DICT_UNARY).allowedFileTypes(FileTypeSet.NO_FILE))
        /* <!-- #BLAZE_RULE(java_runtime_suite).ATTRIBUTE(default) -->
        The default <code>java_runtime</code>, used if
        <a href="${link java_runtime_suite.runtimes}"><code>runtimes</code></a>
        does not contain an entry for the configured architecture.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(
            attr("default", BuildType.LABEL)
                .mandatoryNativeProviders(
                    ImmutableList.<Class<? extends TransitiveInfoProvider>>of(
                        JavaRuntimeProvider.class))
                .allowedFileTypes(FileTypeSet.NO_FILE))
        .add(attr("output_licenses", LICENSE))
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("java_runtime_suite")
        .ancestors(BaseRuleClasses.BaseRule.class)
        .factoryClass(JavaRuntimeSuite.class)
        .build();
  }
}
/*<!-- #BLAZE_RULE (NAME = java_runtime_suite, TYPE = OTHER, FAMILY = Java) -->

<p>
Specifies the configuration for the Java runtimes for each architecture.
</p>

<h4 id="java_runtime_suite">Example:</h4>

<pre class="code">
java_runtime_suite(
   name = "jdk9",
   runtimes = {
     "k8": ":jdk9-k8",
     "ppc": ":jdk9-ppc",
     "arm": ":jdk9-arm",
   },
)
</pre>

<!-- #END_BLAZE_RULE -->*/
