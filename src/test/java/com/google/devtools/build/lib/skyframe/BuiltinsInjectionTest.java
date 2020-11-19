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
import static com.google.common.truth.Truth.assertWithMessage;

import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.analysis.util.MockRule;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for Starlark builtin injection.
 *
 * <p>Essentially these are integration tests between {@link StarlarkBuiltinsFunction} and {@link
 * BzlLoadFunction}.
 */
@RunWith(JUnit4.class)
public class BuiltinsInjectionTest extends BuildViewTestCase {

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

  @Before
  public void setUp() throws Exception {
    setBuildLanguageOptions("--experimental_builtins_bzl_path=tools/builtins_staging");
  }

  /**
   * Writes an exports.bzl file with the given content, in the builtins location.
   *
   * <p>See {@link StarlarkBuiltinsFunction#EXPORTS_ENTRYPOINT} for the significance of exports.bzl.
   */
  private void writeExportsBzl(String... lines) throws Exception {
    scratch.file("tools/builtins_staging/exports.bzl", lines);
  }

  /**
   * Writes a pkg/dummy.bzl file that prints a marker phrase when it finishes evaluating, and an
   * accompanying BUILD file that loads it.
   */
  private void writePkgBzl(String... lines) throws Exception {
    scratch.file("pkg/BUILD", "load(':dummy.bzl', 'dummy_symbol')");
    scratch.file("pkg/dummy");
    List<String> modifiedLines = new ArrayList<>(Arrays.asList(lines));
    modifiedLines.add("dummy_symbol = None");
    // The marker phrase might not be needed, but I don't entirely trust BuildViewTestCase.
    modifiedLines.add("print('dummy.bzl evaluation completed')");
    scratch.file("pkg/dummy.bzl", modifiedLines.toArray(lines));
  }

  /** Builds {@code //pkg} and asserts success, including that the marker print() event occurs. */
  private void buildAndAssertSuccess() throws Exception {
    Object result = getConfiguredTarget("//pkg:BUILD");
    assertContainsEvent("dummy.bzl evaluation completed");
    // On error, getConfiguredTarget sometimes returns null without emitting events; see b/26382502.
    // Though in that case it seems unlikely the above assertion would've passed.
    assertThat(result).isNotNull();
  }

  /** Builds {@code //pkg:dummy} and asserts on the absence of the marker print() event. */
  private void buildAndAssertFailure() throws Exception {
    reporter.removeHandler(failFastHandler);
    Object result = getConfiguredTarget("//pkg:BUILD");
    assertDoesNotContainEvent("dummy.bzl evaluation completed");
    assertWithMessage("Loading of //pkg succeeded unexpectedly").that(result).isNull();
  }

  @Test
  public void basicFunctionality() throws Exception {
    writeExportsBzl(
        "exported_toplevels = {'overridable_symbol': 'new_value'}",
        "exported_rules = {'overridable_rule': 'new_rule'}",
        "exported_to_java = {}");
    writePkgBzl(
        "print('overridable_symbol :: ' + str(overridable_symbol))",
        "print('overridable_rule :: ' + str(native.overridable_rule))");

    buildAndAssertSuccess();
    assertContainsEvent("overridable_symbol :: new_value");
    assertContainsEvent("overridable_rule :: new_rule");
  }

  @Test
  public void builtinsCanLoadFromBuiltins() throws Exception {
    // Define a few files that we can load with different kinds of label syntax. In each case,
    // access the `_internal` symbol to demonstrate that we're being loaded as a builtins bzl.
    scratch.file(
        "tools/builtins_staging/absolute.bzl", //
        "_internal",
        "a = 'A'");
    scratch.file(
        "tools/builtins_staging/repo_relative.bzl", //
        "_internal",
        "b = 'B'");
    scratch.file(
        "tools/builtins_staging/subdir/pkg_relative1.bzl", //
        // Do a relative load within a load, to show it's relative to the (pseudo) package, i.e. the
        // root, and not relative to the file. That is, we specify 'subdir/pkg_relative2.bzl', not
        // just 'pkg_relative2.bzl'.
        "load('subdir/pkg_relative2.bzl', 'c2')",
        "_internal",
        "c = c2");
    scratch.file(
        "tools/builtins_staging/subdir/pkg_relative2.bzl", //
        "_internal",
        "c2 = 'C'");

    // Also create a file in the main repo whose package path coincides with a file in the builtins
    // pseudo-repo, to show that we get the right one.
    scratch.file("BUILD");
    scratch.file("repo_relative.bzl");

    writeExportsBzl(
        "load('@_builtins//:absolute.bzl', 'a')",
        "load('//:repo_relative.bzl', 'b')", // default repo is @_builtins, not main repo
        "load('subdir/pkg_relative1.bzl', 'c')", // relative to (pseudo) package, which is repo root
        "exported_toplevels = {'overridable_symbol': a + b + c}",
        "exported_rules = {}",
        "exported_to_java = {}");
    writePkgBzl("print('overridable_symbol :: ' + str(overridable_symbol))");

    buildAndAssertSuccess();
    assertContainsEvent("overridable_symbol :: ABC");
  }

  @Test
  public void otherBzlsCannotLoadFromBuiltins() throws Exception {
    writeExportsBzl(
        "exported_toplevels = {}", //
        "exported_rules = {}",
        "exported_to_java = {}");
    writePkgBzl("load('@_builtins//:exports.bzl', 'exported_toplevels')");

    buildAndAssertFailure();
    assertContainsEvent("The repository '@_builtins' could not be resolved");
  }

  @Test
  public void builtinsCannotLoadFromNonBuiltins() throws Exception {
    scratch.file("BUILD");
    scratch.file(
        "a_user_written.bzl", //
        "toplevels = {'overridable_symbol': 'new_value'}");
    writeExportsBzl(
        // Use @// syntax to specify the main repo. Otherwise, the load would be relative to the
        // @_builtins pseudo-repo.
        "load('@//:a_user_written.bzl', 'toplevels')",
        "exported_toplevels = toplevels",
        "exported_rules = {}",
        "exported_to_java = {}");
    writePkgBzl();

    buildAndAssertFailure();
    assertContainsEvent(
        "in load statement: .bzl files in @_builtins cannot load from outside of @_builtins");
  }

  @Test
  public void builtinsCannotLoadWithMisplacedColon() throws Exception {
    scratch.file(
        "tools/builtins_staging/subdir/helper.bzl", //
        "toplevels = {'overridable_symbol': 'new_value'}");
    writeExportsBzl(
        "load('//subdir:helper.bzl', 'toplevels')", // Should've been loaded as //:subdir/helper.bzl
        "exported_toplevels = toplevels",
        "exported_rules = {}",
        "exported_to_java = {}");
    writePkgBzl();

    buildAndAssertFailure();
    assertContainsEvent("@_builtins cannot have subpackages");
  }

  @Test
  public void errorInEvaluatingBuiltinsDependency() throws Exception {
    // Test case with a Starlark error in the @_builtins pseudo-repo itself.
    scratch.file(
        "tools/builtins_staging/helper.bzl", //
        "toplevels = {'overridable_symbol': 1//0}  # <-- dynamic error");
    writeExportsBzl(
        "load('@_builtins//:helper.bzl', 'toplevels')",
        "exported_toplevels = toplevels",
        "exported_rules = {}",
        "exported_to_java = {}");
    writePkgBzl();

    buildAndAssertFailure();
    assertContainsEvent(
        "File \"/workspace/tools/builtins_staging/helper.bzl\", line 1, column 37, in <toplevel>");
    assertContainsEvent("Error: integer division by zero");

    // We assert only the parts of the message before and after the module name, since the module
    // identified by the message depends on whether or not the test environment has a prelude file.
    Event ev = assertContainsEvent("Internal error while loading Starlark builtins");
    assertThat(ev.getMessage())
        .contains(
            "Failed to load builtins sources: "
                + "in /workspace/tools/builtins_staging/exports.bzl: "
                + "Extension file 'helper.bzl' (internal) has errors");
  }

  @Test
  public void errorInProcessingExports() throws Exception {
    // Test case with an error in the symbols exported by exports.bzl, but no actual Starlark errors
    // in the builtins files themselves.
    writeExportsBzl(
        "exported_toplevels = None", // should be dict
        "exported_rules = {}",
        "exported_to_java = {}");
    writePkgBzl();

    buildAndAssertFailure();

    // We assert only the parts of the message before and after the module name, since the module
    // identified by the message depends on whether or not the test environment has a prelude file.
    Event ev = assertContainsEvent("Internal error while loading Starlark builtins");
    assertThat(ev.getMessage())
        .contains(
            "Failed to apply declared builtins: "
                + "got NoneType for 'exported_toplevels dict', want dict");
  }

  // TODO(#11437): Remove once disabling is not allowed.
  @Test
  public void injectionDisabledByFlag() throws Exception {
    writeExportsBzl(
        "exported_toplevels = {'overridable_symbol': 'new_value'}",
        "exported_rules = {'overridable_rule': 'new_rule'}",
        "exported_to_java = {}");
    writePkgBzl(
        "print('overridable_symbol :: ' + str(overridable_symbol))",
        "print('overridable_rule :: ' + str(native.overridable_rule))");
    setBuildLanguageOptions("--experimental_builtins_bzl_path=");

    buildAndAssertSuccess();
    assertContainsEvent("overridable_symbol :: original_value");
    assertContainsEvent("overridable_rule :: <built-in rule overridable_rule>");
  }

  // TODO(#11437): Remove once disabling is not allowed.
  @Test
  public void exportsBzlMayBeInErrorWhenInjectionIsDisabled() throws Exception {
    writeExportsBzl( //
        "PARSE ERROR");
    writePkgBzl(
        "print('overridable_symbol :: ' + str(overridable_symbol))",
        "print('overridable_rule :: ' + str(native.overridable_rule))");
    setBuildLanguageOptions("--experimental_builtins_bzl_path=");

    buildAndAssertSuccess();
    assertContainsEvent("overridable_symbol :: original_value");
    assertContainsEvent("overridable_rule :: <built-in rule overridable_rule>");
  }

  // TODO(#11954): Once WORKSPACE- and BUILD-loaded bzls use the exact same environments, we'll want
  // to apply injection to both. This is for uniformity, not because we actually care about builtins
  // injection for WORKSPACE bzls. In the meantime, assert the status quo: WORKSPACE bzls do not use
  // injection.
  @Test
  public void workspaceBzlDoesNotUseInjection() throws Exception {
    writeExportsBzl(
        "exported_toplevels = {'overridable_symbol': 'new_value'}",
        "exported_rules = {'overridable_rule': 'new_rule'}",
        "exported_to_java = {}");
    writePkgBzl();
    scratch.appendFile(
        "WORKSPACE", //
        "load(':foo.bzl', 'dummy_symbol')",
        "print(dummy_symbol)");
    scratch.file("BUILD");
    scratch.file(
        "foo.bzl",
        "dummy_symbol = None",
        "print('overridable_symbol :: ' + str(overridable_symbol))");

    buildAndAssertSuccess();
    // Builtins for WORKSPACE bzls are populated essentially the same as for BUILD bzls, except that
    // injection doesn't apply.
    assertContainsEvent("overridable_symbol :: original_value");
    // We don't assert that the rule isn't injected because the workspace native object doesn't
    // contain our original mock rule. We can test this for WORKSPACE files at the top-level though.
  }

  // TODO(#11437): Add tests of the _internal symbol's usage within builtins bzls.

  // TODO(#11437): Add test cases for BUILD file injection, and WORKSPACE file non-injection, when
  // we add injection support to PackageFunction.

  /**
   * Tests for injection, under inlining of {@link BzlLoadFunction}.
   *
   * <p>See {@link BzlLoadFunction#computeInline} for an explanation of inlining.
   */
  @RunWith(JUnit4.class)
  public static class BuiltinsInjectionTestWithInlining extends BuiltinsInjectionTest {

    @Override
    protected boolean usesInliningBzlLoadFunction() {
      return true;
    }
  }
}
