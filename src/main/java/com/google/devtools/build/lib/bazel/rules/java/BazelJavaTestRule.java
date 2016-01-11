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

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static com.google.devtools.build.lib.packages.BuildType.TRISTATE;
import static com.google.devtools.build.lib.syntax.Type.STRING;

import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.bazel.rules.java.BazelJavaRuleClasses.BaseJavaBinaryRule;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder;
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType;
import com.google.devtools.build.lib.packages.TriState;
import com.google.devtools.build.lib.rules.java.JavaConfiguration;
import com.google.devtools.build.lib.rules.java.JavaSemantics;
import com.google.devtools.build.lib.rules.java.Jvm;

/**
 * Rule definition for the java_test rule.
 */
public final class BazelJavaTestRule implements RuleDefinition {

  private static final String JUNIT4_RUNNER = "org.junit.runner.JUnitCore";

  @Override
  public RuleClass build(Builder builder, RuleDefinitionEnvironment env) {
    /* <!-- #BLAZE_RULE(java_test).IMPLICIT_OUTPUTS -->
    <ul>
      <li><code><var>name</var>.jar</code>: A Java archive.</li>
      <li><code><var>name</var>_deploy.jar</code>: A Java archive suitable for deployment. (Only
        built if explicitly requested.)</li>
    </ul>
    <!-- #END_BLAZE_RULE.IMPLICIT_OUTPUTS --> */
    return builder
        .requiresConfigurationFragments(JavaConfiguration.class, Jvm.class)
        .setImplicitOutputsFunction(BazelJavaRuleClasses.JAVA_BINARY_IMPLICIT_OUTPUTS)
        .override(attr("main_class", STRING).value(JUNIT4_RUNNER))
        .override(attr("stamp", TRISTATE).value(TriState.NO))
        .override(attr(":java_launcher", LABEL).value(JavaSemantics.JAVA_LAUNCHER))
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("java_test")
        .type(RuleClassType.TEST)
        .ancestors(BaseJavaBinaryRule.class, BaseRuleClasses.TestBaseRule.class)
        .factoryClass(BazelJavaTest.class)
        .build();
  }
}

/*<!-- #BLAZE_RULE (NAME = java_test, TYPE = TEST, FAMILY = Java) -->

${ATTRIBUTE_SIGNATURE}

<p>
A <code>java_test()</code> rule compiles a Java test. A test is a binary wrapper around your
test code. The test runner's main method is invoked instead of the main class being compiled.
</p>

${IMPLICIT_OUTPUTS}

${ATTRIBUTE_DEFINITION}

<p>
See the section on <a href="#java_binary_args">java_binary()</a> arguments, with the <i>caveat</i>
that there is no <code>main_class</code> argument. This rule also supports all
<a href="common-definitions.html#common-attributes-tests">attributes common to all test rules
(*_test)</a>.
</p>

<h4 id="java_test_examples">Examples</h4>

<pre class="code">
java_library(
    name = "tests",
    srcs = glob(["*.java"]),
    deps = [
        "//java/com/foo/base:testResources",
        "//java/com/foo/testing/util",
    ],
)

java_test(
    name = "AllTests",
    size = "small",
    runtime_deps = [
        ":tests",
        "//util/mysql",
    ],
)
</pre>

<!-- #END_BLAZE_RULE -->*/
