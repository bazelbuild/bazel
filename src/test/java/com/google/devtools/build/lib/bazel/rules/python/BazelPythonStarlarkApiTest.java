// Copyright 2018 The Bazel Authors. All rights reserved.
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

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.actions.util.ActionsTestUtil.prettyArtifactNames;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.packages.SkylarkInfo;
import com.google.devtools.build.lib.rules.python.PyCommon;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;
import java.io.IOException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Python Starlark APi test */
@RunWith(JUnit4.class)
public class BazelPythonStarlarkApiTest extends BuildViewTestCase {
  @Test
  public void pythonProviderWithFields() throws Exception {
    simpleSources();
    assertNoEvents();

    ConfiguredTarget helloTarget = getConfiguredTarget("//py:hello");
    SkylarkInfo provider = getPythonSkylarkProvider(helloTarget);

    assertThat(provider.hasField(PyCommon.TRANSITIVE_PYTHON_SRCS)).isTrue();
    assertThat(provider.hasField(PyCommon.IS_USING_SHARED_LIBRARY)).isTrue();
    assertThat(provider.hasField(PyCommon.IMPORTS)).isTrue();
    assertThat(provider.hasField("srcs")).isFalse();
  }

  @Test
  public void simpleFieldsValues() throws Exception {
    simpleSources();
    assertNoEvents();

    ConfiguredTarget helloTarget = getConfiguredTarget("//py:hello");
    SkylarkInfo provider = getPythonSkylarkProvider(helloTarget);

    SkylarkNestedSet sources =
        provider.getValue(PyCommon.TRANSITIVE_PYTHON_SRCS, SkylarkNestedSet.class);
    assertThat(prettyArtifactNames(sources.getSet(Artifact.class))).containsExactly("py/hello.py");

    assertThat(provider.getValue(PyCommon.IS_USING_SHARED_LIBRARY, Boolean.class)).isFalse();

    SkylarkNestedSet imports = provider.getValue(PyCommon.IMPORTS, SkylarkNestedSet.class);
    assertThat(imports.getSet(String.class)).containsExactly("__main__/py");
  }

  @Test
  public void transitiveFieldsValues() throws Exception {
    simpleSources();
    assertNoEvents();

    ConfiguredTarget helloTarget = getConfiguredTarget("//py:sayHello");
    SkylarkInfo provider = getPythonSkylarkProvider(helloTarget);

    SkylarkNestedSet sources =
        provider.getValue(PyCommon.TRANSITIVE_PYTHON_SRCS, SkylarkNestedSet.class);
    assertThat(prettyArtifactNames(sources.getSet(Artifact.class)))
        .containsExactly("py/hello.py", "py/sayHello.py");

    assertThat(provider.getValue(PyCommon.IS_USING_SHARED_LIBRARY, Boolean.class)).isFalse();

    SkylarkNestedSet imports = provider.getValue(PyCommon.IMPORTS, SkylarkNestedSet.class);
    assertThat(imports.getSet(String.class)).containsExactly("__main__/py");
  }

  private void simpleSources() throws IOException {
    scratch.file(
        "py/hello.py",
        "import os",
        "def Hello():",
        "    print(\"Hello, World!\")",
        "    print(\"Hello, \" + os.getcwd() + \"!\")");
    scratch.file("py/sayHello.py", "from py import hello", "hello.Hello()");
    scratch.file(
        "py/BUILD",
        "py_binary(name=\"sayHello\", srcs=[\"sayHello.py\"], deps=[\":hello\"])",
        "py_library(name=\"hello\", srcs=[\"hello.py\"], imports= [\".\"])");
  }

  private SkylarkInfo getPythonSkylarkProvider(ConfiguredTarget target) {
    Object object = target.get(PyCommon.PYTHON_SKYLARK_PROVIDER_NAME);
    assertThat(object).isInstanceOf(SkylarkInfo.class);
    SkylarkInfo provider = (SkylarkInfo) object;

    assertThat(provider).isNotNull();
    return provider;
  }
}
