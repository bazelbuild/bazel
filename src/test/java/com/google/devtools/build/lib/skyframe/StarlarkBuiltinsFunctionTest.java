// Copyright 2020 The Bazel Authors. All rights reserved.
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
import static com.google.devtools.build.skyframe.ErrorInfoSubjectFactory.assertThatErrorInfo;
import static com.google.devtools.build.skyframe.EvaluationResultSubjectFactory.assertThatEvaluationResult;

import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.skyframe.util.SkyframeExecutorTestUtils;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.StarlarkList;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SkyKey;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link StarlarkBuiltinsFunction}, and {@code @builtins} resolution behavior in {@link
 * {@link BzlLoadFunction}.
 */
@RunWith(JUnit4.class)
public class StarlarkBuiltinsFunctionTest extends BuildViewTestCase {

  /** Sets up exports.bzl with the given contents and evaluates the {@code @builtins}. */
  private EvaluationResult<StarlarkBuiltinsValue> evalBuiltins(String... lines) throws Exception {
    scratch.file("tools/builtins_staging/BUILD");
    scratch.file("tools/builtins_staging/exports.bzl", lines);

    SkyKey key = StarlarkBuiltinsValue.key();
    return SkyframeExecutorTestUtils.evaluate(
        getSkyframeExecutor(), key, /*keepGoing=*/ false, reporter);
  }

  /**
   * Sets up exports.bzl with the given contents, evaluates the {@code @builtins}, and returns an
   * expected non-transient Exception.
   */
  private Exception evalBuiltinsToException(String... lines) throws Exception {
    EvaluationResult<StarlarkBuiltinsValue> result = evalBuiltins(lines);

    SkyKey key = StarlarkBuiltinsValue.key();
    assertThatEvaluationResult(result).hasError();
    assertThatErrorInfo(result.getError(key)).isNotTransient();
    return result.getError(key).getException();
  }

  @Test
  public void success() throws Exception {
    EvaluationResult<StarlarkBuiltinsValue> result =
        evalBuiltins(
            "exported_toplevels = {'a': 1, 'b': 2}",
            "exported_rules = {'b': True}",
            "exported_to_java = {'c': [1, 2, 3], 'd': 'foo'}");

    SkyKey key = StarlarkBuiltinsValue.key();
    assertThatEvaluationResult(result).hasNoError();
    StarlarkBuiltinsValue value = result.get(key);
    assertThat(value.exportedToplevels).containsExactly("a", 1, "b", 2).inOrder();
    assertThat(value.exportedRules).containsExactly("b", true);
    assertThat(value.exportedToJava)
        .containsExactly("c", StarlarkList.of(/*mutability=*/ null, 1, 2, 3), "d", "foo")
        .inOrder();
    // No test of the digest.
  }

  @Test
  public void missingDictSymbol() throws Exception {
    Exception ex =
        evalBuiltinsToException(
            "exported_toplevels = {'a': 1, 'b': 2}",
            "# exported_rules missing",
            "exported_to_java = {'c': [1, 2, 3], 'd': 'foo'}");
    assertThat(ex).isInstanceOf(EvalException.class);
    assertThat(ex)
        .hasMessageThat()
        .contains("expected a 'exported_rules' dictionary to be defined");
  }

  @Test
  public void badSymbolType() throws Exception {
    Exception ex =
        evalBuiltinsToException(
            "exported_toplevels = {'a': 1, 'b': 2}",
            "exported_rules = None",
            "exported_to_java = {'c': [1, 2, 3], 'd': 'foo'}");
    assertThat(ex).isInstanceOf(EvalException.class);
    assertThat(ex).hasMessageThat().contains("got NoneType for 'exported_rules dict', want dict");
  }

  @Test
  public void badDictKey() throws Exception {
    Exception ex =
        evalBuiltinsToException(
            "exported_toplevels = {'a': 1, 'b': 2}",
            "exported_rules = {1: 'a'}",
            "exported_to_java = {'c': [1, 2, 3], 'd': 'foo'}");
    assertThat(ex).isInstanceOf(EvalException.class);
    assertThat(ex)
        .hasMessageThat()
        .contains("got dict<int, string> for 'exported_rules dict', want dict<string, unknown>");
  }

  @Test
  public void parseError() throws Exception {
    reporter.removeHandler(failFastHandler);
    Exception ex =
        evalBuiltinsToException(
            "exported_toplevels = {'a': 1, 'b': 2}",
            "exported_rules = {'b': True}",
            "exported_to_java = {'c': [1, 2, 3], 'd': 'foo'}",
            "asdf asdf  # <-- parse error");
    assertThat(ex)
        .hasMessageThat()
        .contains("Extension 'tools/builtins_staging/exports.bzl' has errors");
  }

  @Test
  public void evalError() throws Exception {
    reporter.removeHandler(failFastHandler);
    Exception ex =
        evalBuiltinsToException(
            "exported_toplevels = {'a': 1, 'b': 2}",
            "exported_rules = {'b': True}",
            "exported_to_java = {'c': [1, 2, 3], 'd': 'foo'}",
            "1 // 0  # <-- dynamic error");
    assertThat(ex)
        .hasMessageThat()
        .contains("Extension file 'tools/builtins_staging/exports.bzl' has errors");
  }
}
