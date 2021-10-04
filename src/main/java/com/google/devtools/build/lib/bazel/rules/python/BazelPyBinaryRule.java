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

package com.google.devtools.build.lib.bazel.rules.python;

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;

import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.analysis.config.HostTransition;
import com.google.devtools.build.lib.bazel.rules.python.BazelPyRuleClasses.PyBinaryBaseRule;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.rules.python.PyRuleClasses;
import com.google.devtools.build.lib.rules.python.PythonConfiguration;

/**
 * Rule definition for the {@code py_binary} rule.
 */
public final class BazelPyBinaryRule implements RuleDefinition {
  @Override
  public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
    /* <!-- #BLAZE_RULE(py_binary).NAME -->
    <br/>If <code>main</code> is unspecified, this should be the same as the name
    of the source file that is the main entry point of the application,
    minus the extension.  For example, if your entry point is called
    <code>main.py</code>, then your name should be <code>main</code>.
    <!-- #END_BLAZE_RULE.NAME --> */
    return builder
        .requiresConfigurationFragments(PythonConfiguration.class, BazelPythonConfiguration.class)
        .cfg(PyRuleClasses.VERSION_TRANSITION)
        .add(
            attr("$zipper", LABEL)
                .cfg(HostTransition.createFactory())
                .exec()
                .value(env.getToolsLabel("//tools/zip:zipper")))
        .add(
            attr("$launcher", LABEL)
                .cfg(HostTransition.createFactory())
                .value(env.getToolsLabel("//tools/launcher:launcher")))
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("py_binary")
        .ancestors(PyBinaryBaseRule.class, BaseRuleClasses.BinaryBaseRule.class)
        .factoryClass(BazelPyBinary.class)
        .build();
  }
}

/*<!-- #BLAZE_RULE (NAME = py_binary, TYPE = BINARY, FAMILY = Python) -->

<p>
  A <code>py_binary</code> is an executable Python program consisting
  of a collection of <code>.py</code> source files (possibly belonging
  to other <code>py_library</code> rules), a <code>*.runfiles</code>
  directory tree containing all the code and data needed by the
  program at run-time, and a stub script that starts up the program with
  the correct initial environment and data.
</p>

<h4 id="py_binary_examples">Examples</h4>

<pre class="code">
py_binary(
    name = "foo",
    srcs = ["foo.py"],
    data = [":transform"],  # a cc_binary which we invoke at run time
    deps = [
        ":foolib",  # a py_library
    ],
)
</pre>

<p>If you want to run a <code>py_binary</code> from within another binary or
   test (for example, running a python binary to set up some mock resource from
   within a java_test) then the correct approach is to make the other binary or
   test depend on the <code>py_binary</code> in its data section. The other
   binary can then locate the <code>py_binary</code> relative to the source
   directory.
</p>

<pre class="code">
py_binary(
    name = "test_main",
    srcs = ["test_main.py"],
    deps = [":testlib"],
)

java_library(
    name = "testing",
    srcs = glob(["*.java"]),
    data = [":test_main"]
)
</pre>
<!-- #END_BLAZE_RULE -->*/
