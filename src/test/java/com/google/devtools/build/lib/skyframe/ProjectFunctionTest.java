// Copyright 2024 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skyframe.util.SkyframeExecutorTestUtils;
import com.google.devtools.build.skyframe.EvaluationResult;
import java.util.Collection;
import net.starlark.java.eval.StarlarkInt;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class ProjectFunctionTest extends BuildViewTestCase {

  @Before
  public void setUp() throws Exception {
    setBuildLanguageOptions("--experimental_enable_scl_dialect=true");
  }

  @Test
  public void projectFunction_emptyFile_isValid() throws Exception {
    scratch.file("test/PROJECT.scl", "");
    scratch.file("test/BUILD");
    ProjectValue.Key key = new ProjectValue.Key(Label.parseCanonical("//test:PROJECT.scl"));

    EvaluationResult<ProjectValue> result =
        SkyframeExecutorTestUtils.evaluate(skyframeExecutor, key, false, reporter);
    assertThat(result.hasError()).isFalse();

    ProjectValue value = result.get(key);
    assertThat(value.getOwnedCodePaths()).isEmpty();
  }

  @Test
  public void projectFunction_returnsOwnedCodePaths() throws Exception {
    scratch.file("test/PROJECT.scl", "owned_code_paths = ['a', 'b/c']");
    scratch.file("test/BUILD");
    ProjectValue.Key key = new ProjectValue.Key(Label.parseCanonical("//test:PROJECT.scl"));

    EvaluationResult<ProjectValue> result =
        SkyframeExecutorTestUtils.evaluate(skyframeExecutor, key, false, reporter);
    assertThat(result.hasError()).isFalse();

    ProjectValue value = result.get(key);
    assertThat(value.getOwnedCodePaths()).containsExactly("a", "b/c");
  }

  @Test
  public void projectFunction_incorrectType() throws Exception {
    scratch.file("test/PROJECT.scl", "owned_code_paths = 42");
    scratch.file("test/BUILD");
    ProjectValue.Key key = new ProjectValue.Key(Label.parseCanonical("//test:PROJECT.scl"));

    EvaluationResult<ProjectValue> result =
        SkyframeExecutorTestUtils.evaluate(skyframeExecutor, key, false, reporter);
    assertThat(result.hasError()).isTrue();
    assertThat(result.getError().getException())
        .hasMessageThat()
        .matches("expected a list of strings, got .+Int32");
  }

  @Test
  public void projectFunction_incorrectType_inList() throws Exception {
    scratch.file("test/PROJECT.scl", "owned_code_paths = [42]");
    scratch.file("test/BUILD");
    ProjectValue.Key key = new ProjectValue.Key(Label.parseCanonical("//test:PROJECT.scl"));

    EvaluationResult<ProjectValue> result =
        SkyframeExecutorTestUtils.evaluate(skyframeExecutor, key, false, reporter);
    assertThat(result.hasError()).isTrue();
    assertThat(result.getError().getException())
        .hasMessageThat()
        .matches("expected a list of strings, got element of .+Int32");
  }

  @Test
  public void projectFunction_parsesResidualGlobals() throws Exception {
    scratch.file(
        "test/PROJECT.scl",
        """
        owned_code_paths = ["a", "b/c"]
        foo = [0, 1]
        bar = 'str'
        """);
    scratch.file("test/BUILD");
    ProjectValue.Key key = new ProjectValue.Key(Label.parseCanonical("//test:PROJECT.scl"));

    EvaluationResult<ProjectValue> result =
        SkyframeExecutorTestUtils.evaluate(skyframeExecutor, key, false, reporter);
    assertThat(result.hasError()).isFalse();

    ProjectValue value = result.get(key);
    assertThat(value.getOwnedCodePaths()).containsExactly("a", "b/c");
    assertThat(value.getResidualGlobal("owned_code_paths")).isNull();
    assertThat(value.getResidualGlobal("nonexistent_global")).isNull();

    @SuppressWarnings("unchecked")
    Collection<StarlarkInt> fooValue = (Collection<StarlarkInt>) value.getResidualGlobal("foo");
    assertThat(fooValue).containsExactly(StarlarkInt.of(0), StarlarkInt.of(1));

    String barValue = (String) value.getResidualGlobal("bar");
    assertThat(barValue).isEqualTo("str");
  }

  @Test
  public void projectFunction_catchSyntaxError() throws Exception {
    scratch.file(
        "test/PROJECT.scl",
        """
        something_is_wrong =
        """);
    scratch.file("test/BUILD");
    ProjectValue.Key key = new ProjectValue.Key(Label.parseCanonical("//test:PROJECT.scl"));

    AssertionError e =
        assertThrows(
            AssertionError.class,
            () -> SkyframeExecutorTestUtils.evaluate(skyframeExecutor, key, false, reporter));
    assertThat(e).hasMessageThat().contains("syntax error at 'newline': expected expression");
  }
}
