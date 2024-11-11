// Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.python;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.testutil.TestConstants;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests that {@code <tools repo>//tools/python:python_version} works, and that users cannot {@code
 * select()} on the native flags.
 */
@RunWith(JUnit4.class)
public class PythonVersionSelectTest extends BuildViewTestCase {

  /**
   * Tests the python_version selectable target, which is the canonical way of determining the
   * Python version from within a select().
   */
  @Test
  public void selectOnPythonVersionTarget() throws Exception {
    // getRuleContext() doesn't populate the information needed to resolve select()s, and
    // ConfiguredTarget doesn't allow us to test an end-to-end view of the behavior of a select().
    // So this test has the select() control srcs and asserts on which one's in the files to build.
    Artifact py2 = getSourceArtifact("pkg/py2");
    Artifact py3 = getSourceArtifact("pkg/py3");
    scratch.file(
        "pkg/BUILD",
        "filegroup(",
        "    name = 'foo',",
        "    srcs = select({",
        "        '" + TestConstants.TOOLS_REPOSITORY + "//tools/python:PY2': ['py2'],",
        "        '" + TestConstants.TOOLS_REPOSITORY + "//tools/python:PY3': ['py3'],",
        "    }),",
        ")");

    // No --python_version, use default value.
    doTestSelectOnPythonVersionTarget(py2, "--incompatible_py3_is_default=false");
    doTestSelectOnPythonVersionTarget(py3, "--incompatible_py3_is_default=true");

    // --python_version is given, use it.
    doTestSelectOnPythonVersionTarget(py2, "--python_version=PY2");
    doTestSelectOnPythonVersionTarget(py3, "--python_version=PY3");
  }

  private void doTestSelectOnPythonVersionTarget(Artifact expected, String... flags)
      throws Exception {
    useConfiguration(flags);
    NestedSet<Artifact> files =
        getConfiguredTarget("//pkg:foo").getProvider(FileProvider.class).getFilesToBuild();
    assertThat(files.toList()).contains(expected);
  }
}
