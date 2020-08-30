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

import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.analysis.util.MockRule;
import com.google.devtools.build.lib.skyframe.StarlarkBuiltinsFunction.BuiltinsFailedException;
import com.google.devtools.build.lib.skyframe.util.SkyframeExecutorTestUtils;
import com.google.devtools.build.lib.syntax.ClassObject;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SkyKey;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link StarlarkBuiltinsFunction}. */
@RunWith(JUnit4.class)
public class StarlarkBuiltinsFunctionTest extends BuildViewTestCase {

  private static final MockRule OVERRIDABLE_RULE = () -> MockRule.define("overridable_rule");

  @Override
  protected ConfiguredRuleClassProvider createRuleClassProvider() {
    // Add a fake rule and top-level symbol to override.
    ConfiguredRuleClassProvider.Builder builder =
        new ConfiguredRuleClassProvider.Builder()
            .addRuleDefinition(OVERRIDABLE_RULE)
            .addStarlarkAccessibleTopLevels("overridable_symbol", "original_value");
    TestRuleClassProvider.addStandardRules(builder);
    return builder.build();
  }

  // TODO(#11437): Add tests for predeclared env of BUILD (and WORKSPACE?) files, once
  // StarlarkBuiltinsFunction manages that functionality.

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
  public void evalExportsSuccess() throws Exception {
    EvaluationResult<StarlarkBuiltinsValue> result =
        evalBuiltins(
            "exported_toplevels = {'overridable_symbol': 'new_value'}",
            "exported_rules = {'overridable_rule': 'new_rule'}",
            "exported_to_java = {'for_native_code': 'secret_sauce'}");

    SkyKey key = StarlarkBuiltinsValue.key();
    assertThatEvaluationResult(result).hasNoError();
    StarlarkBuiltinsValue value = result.get(key);

    // Universe symbols are omitted (they're added by the interpreter).
    assertThat(value.predeclaredForBuildBzl).doesNotContainKey("print");
    // Generic Bazel symbols are present.
    assertThat(value.predeclaredForBuildBzl).containsKey("rule");
    // Non-overridden symbols are present.
    assertThat(value.predeclaredForBuildBzl).containsKey("CcInfo");
    // Overridden symbol.
    assertThat(value.predeclaredForBuildBzl).containsEntry("overridable_symbol", "new_value");
    // Overridden native field.
    Object nativeField =
        ((ClassObject) value.predeclaredForBuildBzl.get("native")).getValue("overridable_rule");
    assertThat(nativeField).isEqualTo("new_rule");
    // Stuff for native rules.
    assertThat(value.exportedToJava).containsExactly("for_native_code", "secret_sauce").inOrder();
    // No test of the digest.
  }

  @Test
  public void evalExportsSuccess_withLoad() throws Exception {
    // TODO(#11437): Use @builtins//:... syntax, once supported. Don't create a real package.
    scratch.file("builtins_helper/BUILD");
    scratch.file(
        "builtins_helper/dummy.bzl", //
        "toplevels = {'overridable_symbol': 'new_value'}");

    EvaluationResult<StarlarkBuiltinsValue> result =
        evalBuiltins(
            "load('//builtins_helper:dummy.bzl', 'toplevels')",
            "exported_toplevels = toplevels",
            "exported_rules = {}",
            "exported_to_java = {}");

    SkyKey key = StarlarkBuiltinsValue.key();
    assertThatEvaluationResult(result).hasNoError();
    StarlarkBuiltinsValue value = result.get(key);
    assertThat(value.predeclaredForBuildBzl).containsEntry("overridable_symbol", "new_value");
  }

  @Test
  public void evalExportsFails_missingDictSymbol() throws Exception {
    Exception ex =
        evalBuiltinsToException(
            "exported_toplevels = {}", //
            "# exported_rules missing",
            "exported_to_java = {}");
    assertThat(ex).isInstanceOf(BuiltinsFailedException.class);
    assertThat(ex)
        .hasMessageThat()
        .contains(
            "Failed to apply declared builtins: expected a 'exported_rules' dictionary to be "
                + "defined");
  }

  @Test
  public void evalExportsFails_badSymbolType() throws Exception {
    Exception ex =
        evalBuiltinsToException(
            "exported_toplevels = {}", //
            "exported_rules = None",
            "exported_to_java = {}");
    assertThat(ex).isInstanceOf(BuiltinsFailedException.class);
    assertThat(ex)
        .hasMessageThat()
        .contains(
            "Failed to apply declared builtins: got NoneType for 'exported_rules dict', want dict");
  }

  @Test
  public void evalExportsFails_badDictKey() throws Exception {
    Exception ex =
        evalBuiltinsToException(
            "exported_toplevels = {}", //
            "exported_rules = {1: 'a'}",
            "exported_to_java = {}");
    assertThat(ex).isInstanceOf(BuiltinsFailedException.class);
    assertThat(ex)
        .hasMessageThat()
        .contains(
            "Failed to apply declared builtins: got dict<int, string> for 'exported_rules dict', "
                + "want dict<string, unknown>");
  }

  @Test
  public void evalExportsFails_overrideNotAllowed() throws Exception {
    Exception ex =
        evalBuiltinsToException(
            "exported_toplevels = {}", //
            "exported_rules = {'glob': 'new_builtin'}",
            "exported_to_java = {}");
    assertThat(ex).isInstanceOf(BuiltinsFailedException.class);
    assertThat(ex)
        .hasMessageThat()
        .contains(
            "Failed to apply declared builtins: Cannot override native module field 'glob' with an"
                + " injected value");
  }

  @Test
  public void evalExportsFails_parseError() throws Exception {
    reporter.removeHandler(failFastHandler);
    Exception ex =
        evalBuiltinsToException(
            "exported_toplevels = {}",
            "exported_rules = {}",
            "exported_to_java = {}",
            "asdf asdf  # <-- parse error");
    assertThat(ex).isInstanceOf(BuiltinsFailedException.class);
    assertThat(ex)
        .hasMessageThat()
        .contains(
            "Failed to load builtins sources: Extension 'tools/builtins_staging/exports.bzl' has "
                + "errors");
  }

  @Test
  public void evalExportsFails_evalError() throws Exception {
    reporter.removeHandler(failFastHandler);
    Exception ex =
        evalBuiltinsToException(
            "exported_toplevels = {}",
            "exported_rules = {}",
            "exported_to_java = {}",
            "1 // 0  # <-- dynamic error");
    assertThat(ex).isInstanceOf(BuiltinsFailedException.class);
    assertThat(ex)
        .hasMessageThat()
        .contains(
            "Failed to load builtins sources: Extension file 'tools/builtins_staging/exports.bzl' "
                + "has errors");
  }

  @Test
  public void evalExportsFails_errorInDependency() throws Exception {
    reporter.removeHandler(failFastHandler);
    // TODO(#11437): Use @builtins//:... syntax, once supported. Don't create a real package.
    scratch.file("builtins_helper/BUILD");
    scratch.file(
        "builtins_helper/dummy.bzl", //
        "1 // 0  # <-- dynamic error");
    Exception ex =
        evalBuiltinsToException(
            "load('//builtins_helper:dummy.bzl', 'dummy')",
            "exported_toplevels = {}",
            "exported_rules = {}",
            "exported_to_java = {}");
    assertThat(ex).isInstanceOf(BuiltinsFailedException.class);
    assertThat(ex)
        .hasMessageThat()
        .contains(
            "Failed to load builtins sources: in /workspace/tools/builtins_staging/exports.bzl: "
                + "Extension file 'builtins_helper/dummy.bzl' has errors");
  }
}
