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

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Unit tests for {@link BazelPyRuntime}.
 */
@RunWith(JUnit4.class)
public final class BazelPyRuntimeTest extends BuildViewTestCase {

  @Rule
  public ExpectedException thrown = ExpectedException.none();

  @Before
  public final void setup() throws Exception {
    scratch.file(
      "py/BUILD",
        "py_runtime(",
        "    name='py-2.7',",
        "    files = [",
        "        'py-2.7/bin/python',",
        "        'py-2.7/lib/libpython2.7.a',",
        "     ],",
        "    interpreter='py-2.7/bin/python',",
        ")",
        "",
        "py_runtime(",
        "    name='py-3.6',",
        "    files = [],",
        "    interpreter_path='/opt/pyenv/versions/3.6.0/bin/python',",
        ")",
        "",
        "py_runtime(",
        "    name='err-interpreter-and-path-both-set',",
        "    files = [],",
        "    interpreter='py-2.7/bin/python',",
        "    interpreter_path='/opt/pyenv/versions/3.6.0/bin/python',",
        ")",
        "",
        "py_runtime(",
        "    name='err-interpreter-and-path-both-unset',",
        "    files = [],",
        ")",
        "",
        "py_runtime(",
        "    name='err-path-not-absolute',",
        "    files = [],",
        "    interpreter_path='py-2.7/bin/python',",
        ")",
        "",
        "py_runtime(",
        "    name='err-non-empty-files-with-path-absolute',",
        "    files = [",
        "        'py-err/bin/python',",
        "        'py-err/lib/libpython2.7.a',",
        "     ],",
        "    interpreter_path='/opt/pyenv/versions/3.6.0/bin/python',",
        ")"
    );

  }

  @Test
  public void testCheckedInPyRuntime() throws Exception {
    useConfiguration("--python_top=//py:py-2.7");
    ConfiguredTarget target = getConfiguredTarget("//py:py-2.7");

    assertThat(
        ActionsTestUtil.prettyArtifactNames(
            target.getProvider(BazelPyRuntimeProvider.class).files()))
        .containsExactly("py/py-2.7/bin/python", "py/py-2.7/lib/libpython2.7.a");
    assertThat(
            target.getProvider(BazelPyRuntimeProvider.class).interpreter().getExecPathString())
        .isEqualTo("py/py-2.7/bin/python");
    assertThat(target.getProvider(BazelPyRuntimeProvider.class).interpreterPath())
        .isEqualTo("");
  }

  @Test
  public void testAbsolutePathPyRuntime() throws Exception {
    useConfiguration("--python_top=//py:py-3.6");
    ConfiguredTarget target = getConfiguredTarget("//py:py-3.6");

    assertThat(
        ActionsTestUtil.prettyArtifactNames(
            target.getProvider(BazelPyRuntimeProvider.class).files()))
        .isEmpty();
    assertThat(
        target.getProvider(BazelPyRuntimeProvider.class).interpreter())
        .isNull();
    assertThat(target.getProvider(BazelPyRuntimeProvider.class).interpreterPath())
        .isEqualTo("/opt/pyenv/versions/3.6.0/bin/python");
  }

  @Test
  public void testErrorWithInterpreterAndPathBothSet() throws Exception {
    useConfiguration("--python_top=//py:err-interpreter-and-path-both-set");
    try {
      getConfiguredTarget("//py:err-interpreter-and-path-both-set");
    } catch (Error e) {
      assertThat(e.getMessage())
          .contains("interpreter and interpreter_path cannot be set at the same time.");
    }
  }

  @Test
  public void testErrorWithInterpreterAndPathBothUnset() throws Exception {
    useConfiguration("--python_top=//py:err-interpreter-and-path-both-unset");
    try {
      getConfiguredTarget("//py:err-interpreter-and-path-both-unset");
    } catch (Error e) {
      assertThat(e.getMessage())
          .contains("interpreter and interpreter_path cannot be empty at the same time.");
    }
  }

  @Test
  public void testErrorWithPathNotAbsolute() throws Exception {
    useConfiguration("--python_top=//py:err-path-not-absolute");
    try {
      getConfiguredTarget("//py:err-path-not-absolute");
    } catch (Error e) {
      assertThat(e.getMessage())
          .contains("must be an absolute path.");
    }
  }

  @Test
  public void testPyRuntimeWithError() throws Exception {
    useConfiguration("--python_top=//py:err-non-empty-files-with-path-absolute");
    try {
      getConfiguredTarget("//py:err-non-empty-files-with-path-absolute");
    } catch (Error e) {
      assertThat(e.getMessage())
          .contains("interpreter with an absolute path requires files to be empty.");
    }
  }

}
