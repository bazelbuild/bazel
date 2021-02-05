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

package com.google.devtools.build.lib.rules.python;

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;
import static com.google.devtools.build.lib.packages.BuildType.LICENSE;
import static com.google.devtools.build.lib.packages.Type.STRING;

import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.util.FileTypeSet;

/** Rule definition for {@code py_runtime} */
public final class PyRuntimeRule implements RuleDefinition {

  @Override
  public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
    return builder

        // For --incompatible_py3_is_default.
        .requiresConfigurationFragments(PythonConfiguration.class)

        /* <!-- #BLAZE_RULE(py_runtime).ATTRIBUTE(files) -->
        For an in-build runtime, this is the set of files comprising this runtime. These files will
        be added to the runfiles of Python binaries that use this runtime. For a platform runtime
        this attribute must not be set.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("files", LABEL_LIST).allowedFileTypes(FileTypeSet.ANY_FILE))

        /* <!-- #BLAZE_RULE(py_runtime).ATTRIBUTE(interpreter) -->
        For an in-build runtime, this is the target to invoke as the interpreter. For a platform
        runtime this attribute must not be set.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("interpreter", LABEL).allowedFileTypes(FileTypeSet.ANY_FILE).singleArtifact())

        /* <!-- #BLAZE_RULE(py_runtime).ATTRIBUTE(interpreter_path) -->
        For a platform runtime, this is the absolute path of a Python interpreter on the target
        platform. For an in-build runtime this attribute must not be set.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("interpreter_path", STRING))

        /* <!-- #BLAZE_RULE(py_runtime).ATTRIBUTE(python_version) -->
        Whether this runtime is for Python major version 2 or 3. Valid values are <code>"PY2"</code>
        and <code>"PY3"</code>.

        <p>The default value is controlled by the <code>--incompatible_py3_is_default</code> flag.
        However, in the future this attribute will be mandatory and have no default value.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(
            attr("python_version", STRING)
                .value(PythonVersion._INTERNAL_SENTINEL.toString())
                .allowedValues(PyRuleClasses.TARGET_PYTHON_ATTR_VALUE_SET))
        .add(attr("output_licenses", LICENSE))
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return Metadata.builder()
        .name("py_runtime")
        .ancestors(BaseRuleClasses.NativeBuildRule.class)
        .factoryClass(PyRuntime.class)
        .build();
  }
}
/*<!-- #BLAZE_RULE (NAME = py_runtime, TYPE = OTHER, FAMILY = Python) -->

<p>Represents a Python runtime used to execute Python code.

<p>A <code>py_runtime</code> target can represent either a <em>platform runtime</em> or an
<em>in-build runtime</em>. A platform runtime accesses a system-installed interpreter at a known
path, whereas an in-build runtime points to an executable target that acts as the interpreter. In
both cases, an "interpreter" means any executable binary or wrapper script that is capable of
running a Python script passed on the command line, following the same conventions as the standard
CPython interpreter.

<p>A platform runtime is by its nature non-hermetic. It imposes a requirement on the target platform
to have an interpreter located at a specific path. An in-build runtime may or may not be hermetic,
depending on whether it points to a checked-in interpreter or a wrapper script that accesses the
system interpreter.

<h4 id="py_runtime_example">Example:</h4>

<pre class="code">
py_runtime(
    name = "python-2.7.12",
    files = glob(["python-2.7.12/**"]),
    interpreter = "python-2.7.12/bin/python",
)

py_runtime(
    name = "python-3.6.0",
    interpreter_path = "/opt/pyenv/versions/3.6.0/bin/python",
)
</pre>

<!-- #END_BLAZE_RULE -->*/
