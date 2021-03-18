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
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SkyKey;
import net.starlark.java.eval.Structure;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link StarlarkBuiltinsFunction}. */
@RunWith(JUnit4.class)
public class StarlarkBuiltinsFunctionTest extends BuildViewTestCase {

  private static final MockRule OVERRIDABLE_RULE = () -> MockRule.define("overridable_rule");
  private static final MockRule JUST_A_RULE = () -> MockRule.define("just_a_rule");

  @Override
  protected ConfiguredRuleClassProvider createRuleClassProvider() {
    // Add a fake rule and top-level symbol to override.
    ConfiguredRuleClassProvider.Builder builder =
        new ConfiguredRuleClassProvider.Builder()
            .addRuleDefinition(OVERRIDABLE_RULE)
            .addRuleDefinition(JUST_A_RULE)
            .addStarlarkAccessibleTopLevels("overridable_symbol", "original_value")
            .addStarlarkAccessibleTopLevels("just_a_symbol", "another_value");
    TestRuleClassProvider.addStandardRules(builder);
    return builder.build();
  }

  // TODO(#11437): Add tests for predeclared env of BUILD (and WORKSPACE?) files, once
  // StarlarkBuiltinsFunction manages that functionality.

  /** Sets up exports.bzl with the given contents and evaluates the {@code @_builtins}. */
  private EvaluationResult<StarlarkBuiltinsValue> evalBuiltins(String... lines) throws Exception {
    scratch.file("tools/builtins_staging/exports.bzl", lines);
    setBuildLanguageOptions("--experimental_builtins_bzl_path=tools/builtins_staging");

    SkyKey key = StarlarkBuiltinsValue.key();
    return SkyframeExecutorTestUtils.evaluate(
        getSkyframeExecutor(), key, /*keepGoing=*/ false, reporter);
  }

  /**
   * Sets up exports.bzl with the given contents, evaluates the {@code @_builtins}, and asserts that
   * a BuiltinsFailedException is raised with the given message.
   */
  private void assertBuiltinsFailure(String message, String... lines) throws Exception {
    reporter.removeHandler(failFastHandler);
    EvaluationResult<StarlarkBuiltinsValue> result = evalBuiltins(lines);

    SkyKey key = StarlarkBuiltinsValue.key();
    assertThatEvaluationResult(result).hasError();
    assertThatErrorInfo(result.getError(key)).isNotTransient();
    Exception ex = result.getError(key).getException();
    assertThat(ex).isInstanceOf(BuiltinsFailedException.class);
    assertThat(ex).hasMessageThat().contains(message);
  }

  @Test
  public void successfulEvaluation() throws Exception {
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
    // General Bazel symbols are present.
    assertThat(value.predeclaredForBuildBzl).containsKey("rule");
    // Non-overridden rule-specific symbols are present.
    assertThat(value.predeclaredForBuildBzl).containsKey("just_a_symbol");
    // Overridden symbol.
    assertThat(value.predeclaredForBuildBzl).containsEntry("overridable_symbol", "new_value");
    // Overridden native field.
    Structure nativeObject = (Structure) value.predeclaredForBuildBzl.get("native");
    assertThat(nativeObject.getValue("overridable_rule")).isEqualTo("new_rule");
    assertThat(nativeObject.getFieldNames()).contains("just_a_rule");

    // Analogous assertions for build files.
    assertThat(value.predeclaredForBuild).doesNotContainKey("print");
    assertThat(value.predeclaredForBuild).containsKey("glob");
    assertThat(value.predeclaredForBuild).containsEntry("overridable_rule", "new_rule");
    assertThat(value.predeclaredForBuild).containsKey("just_a_rule");

    // Stuff for native rules.
    assertThat(value.exportedToJava).containsExactly("for_native_code", "secret_sauce").inOrder();

    // Digest should be same as the exports file.
    byte[] exportsDigest =
        ((BzlLoadValue)
                SkyframeExecutorTestUtils.evaluate(
                        getSkyframeExecutor(),
                        StarlarkBuiltinsFunction.EXPORTS_ENTRYPOINT_KEY,
                        /*keepGoing=*/ false,
                        reporter)
                    .get(StarlarkBuiltinsFunction.EXPORTS_ENTRYPOINT_KEY))
            .getTransitiveDigest();
    assertThat(value.transitiveDigest).isEqualTo(exportsDigest);
  }

  @Test
  public void exportsDictMustExist() throws Exception {
    assertBuiltinsFailure(
        "Failed to apply declared builtins: expected a 'exported_rules' dictionary to be defined",
        //
        "exported_toplevels = {}",
        "# exported_rules missing",
        "exported_to_java = {}");
  }

  @Test
  public void exportsDictMustBeDict() throws Exception {
    assertBuiltinsFailure(
        "Failed to apply declared builtins: got NoneType for 'exported_rules dict', want dict",
        //
        "exported_toplevels = {}",
        "exported_rules = None",
        "exported_to_java = {}");
  }

  @Test
  public void exportsDictKeyMustBeString() throws Exception {
    assertBuiltinsFailure(
        "Failed to apply declared builtins: got dict<int, string> for 'exported_rules dict', want"
            + " dict<string, unknown>",
        //
        "exported_toplevels = {}",
        "exported_rules = {1: 'a'}",
        "exported_to_java = {}");
  }

  @Test
  public void cannotOverrideGeneralSymbol() throws Exception {
    assertBuiltinsFailure(
        "Failed to apply declared builtins: Cannot override 'glob' with an injected rule",
        //
        "exported_toplevels = {}", //
        "exported_rules = {'glob': 'new_builtin'}",
        "exported_to_java = {}");
  }

  @Test
  public void parseErrorInExportsHandledGracefully() throws Exception {
    assertBuiltinsFailure(
        "Failed to load builtins sources: compilation of module 'exports.bzl' (internal) failed",
        //
        "exported_toplevels = {}",
        "exported_rules = {}",
        "exported_to_java = {}",
        "asdf asdf  # <-- parse error");
  }

  @Test
  public void evalErrorInExportsHandledGracefully() throws Exception {
    assertBuiltinsFailure(
        "Failed to load builtins sources: initialization of module 'exports.bzl' (internal) failed",
        //
        "exported_toplevels = {}",
        "exported_rules = {}",
        "exported_to_java = {}",
        "1 // 0  # <-- dynamic error");
  }

  @Test
  public void builtinsBzlCannotAccessNative() throws Exception {
    assertBuiltinsFailure(
        "compilation of module 'exports.bzl' (internal) failed",
        //
        "native.overridable_rule",
        "exported_toplevels = {}",
        "exported_rules = {}",
        "exported_to_java = {}");
    assertContainsEvent("name 'native' is not defined");
  }

  @Test
  public void builtinsBzlCannotAccessRuleSpecificSymbol() throws Exception {
    assertBuiltinsFailure(
        "compilation of module 'exports.bzl' (internal) failed",
        //
        "overridable_symbol",
        "exported_toplevels = {}",
        "exported_rules = {}",
        "exported_to_java = {}");
    assertContainsEvent("name 'overridable_symbol' is not defined");
  }

  @Test
  public void builtinsBzlCanAccessBuiltinsInternalModule() throws Exception {
    EvaluationResult<StarlarkBuiltinsValue> result =
        evalBuiltins(
            "print(_builtins)",
            "",
            "exported_toplevels = {}",
            "exported_rules = {}",
            "exported_to_java = {}");
    assertThatEvaluationResult(result).hasNoError();
    assertContainsEvent("<_builtins module>");
  }

  @Test
  public void regularBzlCannotAccessBuiltinsInternalModule() throws Exception {
    scratch.file(
        "pkg/BUILD", //
        "load(':dummy.bzl', 'dummy_symbol')");
    scratch.file(
        "pkg/dummy.bzl", //
        "_builtins",
        "dummy_symbol = None");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//pkg:BUILD");
    assertContainsEvent("name '_builtins' is not defined");
  }
}
