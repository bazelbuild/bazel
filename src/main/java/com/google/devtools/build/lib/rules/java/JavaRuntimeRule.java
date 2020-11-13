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
import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;
import static com.google.devtools.build.lib.packages.BuildType.LICENSE;
import static com.google.devtools.build.lib.packages.Type.STRING;

import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.analysis.TemplateVariableInfo;
import com.google.devtools.build.lib.analysis.config.ConfigAwareRuleClassBuilder;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.util.FileTypeSet;

/** Rule definition for {@code java_runtime} */
public final class JavaRuntimeRule implements RuleDefinition {
  @Override
  public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
    return ConfigAwareRuleClassBuilder.of(builder)
        .requiresHostConfigurationFragments(JavaConfiguration.class)
        .originalBuilder()
        .requiresConfigurationFragments(JavaConfiguration.class)
        .advertiseProvider(TemplateVariableInfo.class)
        /* <!-- #BLAZE_RULE(java_runtime).ATTRIBUTE(srcs) -->
        All files in the runtime.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("srcs", LABEL_LIST).allowedFileTypes(FileTypeSet.ANY_FILE))
        /* <!-- #BLAZE_RULE(java_runtime).ATTRIBUTE(java) -->
        The path to the java executable.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("java", LABEL).singleArtifact().allowedFileTypes(FileTypeSet.ANY_FILE).exec())
        /* <!-- #BLAZE_RULE(java_runtime).ATTRIBUTE(java_home) -->
        The path to the root of the runtime.
        Subject to <a href="${link make-variables}">"Make" variable</a> substitution.
        If this path is absolute, the rule denotes a non-hermetic Java runtime with a well-known
        path. In that case, the <code>srcs</code> and <code>java</code> attributes must be empty.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("java_home", STRING))
        .add(attr("output_licenses", LICENSE))
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("java_runtime")
        .ancestors(BaseRuleClasses.BaseRule.class)
        .factoryClass(JavaRuntime.class)
        .build();
  }
}
/*<!-- #BLAZE_RULE (NAME = java_runtime, TYPE = OTHER, FAMILY = Java) -->

<p>
Specifies the configuration for a Java runtime.
</p>

<h4 id="java_runtime_example">Example:</h4>

<pre class="code">
java_runtime(
    name = "jdk-9-ea+153",
    srcs = glob(["jdk9-ea+153/**"]),
    java_home = "jdk9-ea+153",
)
</pre>

<!-- #END_BLAZE_RULE -->*/
