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

package com.google.devtools.build.lib.bazel.rules.python;

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static com.google.devtools.build.lib.packages.BuildType.TRISTATE;
import static com.google.devtools.build.lib.syntax.Type.BOOLEAN;

import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.analysis.config.HostTransition;
import com.google.devtools.build.lib.bazel.rules.python.BazelPyRuleClasses.PyBinaryBaseRule;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType;
import com.google.devtools.build.lib.packages.TriState;
import com.google.devtools.build.lib.rules.cpp.CppRuleClasses;
import com.google.devtools.build.lib.rules.python.PyRuleClasses;
import com.google.devtools.build.lib.rules.python.PythonConfiguration;

/**
 * Rule definition for the py_test rule.
 */
public final class BazelPyTestRule implements RuleDefinition {
  @Override
  public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
    return builder
        .requiresConfigurationFragments(PythonConfiguration.class, BazelPythonConfiguration.class)
        .cfg(PyRuleClasses.VERSION_TRANSITION)
        .add(
            attr("$zipper", LABEL)
                .cfg(HostTransition.INSTANCE)
                .exec()
                .value(env.getToolsLabel("//tools/zip:zipper")))
        .override(
            attr("testonly", BOOLEAN)
                .value(true)
                .nonconfigurable("policy decision: should be consistent across configurations"))
        /* <!-- #BLAZE_RULE(py_test).ATTRIBUTE(stamp) -->
        See the section on <a href="${link py_binary_args}">py_binary()</a> arguments, except
        that the stamp argument is set to 0 by default for tests.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .override(attr("stamp", TRISTATE).value(TriState.NO))
        .addRequiredToolchains(CppRuleClasses.ccToolchainTypeAttribute(env))
        .add(
            attr("$launcher", LABEL)
                .cfg(HostTransition.INSTANCE)
                .value(env.getToolsLabel("//tools/launcher:launcher")))
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("py_test")
        .type(RuleClassType.TEST)
        .ancestors(PyBinaryBaseRule.class, BaseRuleClasses.TestBaseRule.class)
        .factoryClass(BazelPyTest.class)
        .build();
  }
}

/*<!-- #BLAZE_RULE (NAME = py_test, TYPE = TEST, FAMILY = Python) -->

<p>
A <code>py_test()</code> rule compiles a test.  A test is a binary wrapper
 around some test code.</p>

<h4 id="py_test_examples">Examples</h4>

<p>
<pre class="code">
py_test(
    name = "runtest_test",
    srcs = ["runtest_test.py"],
    deps = [
        "//path/to/a/py/library",
    ],
)
</pre>

<p>It's also possible to specify a main module:</p>

<pre class="code">
py_test(
    name = "runtest_test",
    srcs = [
        "runtest_main.py",
        "runtest_lib.py",
    ],
    main = "runtest_main.py",
)
</pre>

<!-- #END_BLAZE_RULE -->*/
