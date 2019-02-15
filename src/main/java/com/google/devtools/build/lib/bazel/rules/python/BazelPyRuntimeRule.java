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

package com.google.devtools.build.lib.bazel.rules.python;

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;
import static com.google.devtools.build.lib.packages.BuildType.LICENSE;
import static com.google.devtools.build.lib.syntax.Type.STRING;

import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.rules.python.PyRuntime;
import com.google.devtools.build.lib.rules.python.PythonConfiguration;
import com.google.devtools.build.lib.util.FileTypeSet;

/** Rule definition for {@code py_runtime} */
public final class BazelPyRuntimeRule implements RuleDefinition {

  @Override
  public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
    return builder
        .requiresConfigurationFragments(PythonConfiguration.class, BazelPythonConfiguration.class)

        /* <!-- #BLAZE_RULE(py_runtime).ATTRIBUTE(files) -->
        The set of files comprising this Python runtime.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("files", LABEL_LIST).allowedFileTypes(FileTypeSet.ANY_FILE))

        /* <!-- #BLAZE_RULE(py_runtime).ATTRIBUTE(interpreter) -->
        The Python interpreter used in this runtime. Binary rules will be executed using this
        binary.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("interpreter", LABEL).allowedFileTypes(FileTypeSet.ANY_FILE).singleArtifact())

        /* <!-- #BLAZE_RULE(py_runtime).ATTRIBUTE(interpreter_path) -->
        The absolute path of a Python interpreter. This attribute and interpreter attribute cannot
        be set at the same time.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("interpreter_path", STRING))
        .add(attr("output_licenses", LICENSE))
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return Metadata.builder()
        .name("py_runtime")
        .ancestors(BaseRuleClasses.BaseRule.class)
        .factoryClass(PyRuntime.class)
        .build();
  }
}
/*<!-- #BLAZE_RULE (NAME = py_runtime, TYPE = OTHER, FAMILY = Python) -->

<p>
Specifies the configuration for a Python runtime. This rule can either describe a Python runtime in
the source tree or one at a well-known absolute path.
</p>

<h4 id="py_runtime">Example:</h4>

<pre class="code">
py_runtime(
    name = "python-2.7.12",
    files = glob(["python-2.7.12/**"]),
    interpreter = "python-2.7.12/bin/python",
)

py_runtime(
    name = "python-3.6.0",
    files = [],
    interpreter_path = "/opt/pyenv/versions/3.6.0/bin/python",
)
</pre>

<!-- #END_BLAZE_RULE -->*/
