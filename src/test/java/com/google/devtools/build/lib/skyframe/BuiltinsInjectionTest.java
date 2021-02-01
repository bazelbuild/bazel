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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.AnalysisEnvironment;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.actions.FileWriteAction;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.analysis.util.MockRule;
import com.google.devtools.build.lib.analysis.util.MockRuleDefaults;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.FlagGuardedValue;
import net.starlark.java.eval.Mutability;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkList;
import net.starlark.java.eval.StarlarkThread;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for Starlark builtin injection.
 *
 * <p>Essentially these are integration tests between {@link StarlarkBuiltinsFunction}, {@link
 * BzlLoadFunction}, and the rest of package loading.
 */
@RunWith(JUnit4.class)
public class BuiltinsInjectionTest extends BuildViewTestCase {

  /** A simple dummy rule that doesn't do anything. */
  private static final MockRule OVERRIDABLE_RULE = () -> MockRule.define("overridable_rule");

  /**
   * A dummy native rule that reads a value from {@code @_builtins}.
   *
   * <p>It looks up the symbol "builtins_defined_symbol" in exported_to_java, and prints its value
   * to the event handler.
   */
  private static final MockRule SANDWICH_RULE =
      () -> MockRule.factory(SandwichFactory.class).define("sandwich_rule");

  // Must be public due to reflective construction of rule factories.
  /** Factory for SANDWICH_RULE. (Javadoc'd to pacify linter.) */
  public static class SandwichFactory extends MockRuleDefaults.DefaultConfiguredTargetFactory {
    @Override
    public ConfiguredTarget create(RuleContext ruleContext)
        throws InterruptedException, RuleErrorException, ActionConflictException {
      AnalysisEnvironment env = ruleContext.getAnalysisEnvironment();

      Object value = env.getStarlarkDefinedBuiltins().get("builtins_defined_symbol");

      env.getEventHandler().handle(Event.info("builtins_defined_symbol :: " + value.toString()));
      return super.create(ruleContext);
    }
  }

  /**
   * A dummy native rule that runs {@code @_builtins}-defined code.
   *
   * <p>It looks up the function listed as "builtins_defined_logic" in exported_to_java, and calls
   * it twice on an initially empty list. It prints both return values and the final value of the
   * list.
   */
  private static final MockRule SANDWICH_LOGIC_RULE =
      () -> MockRule.factory(SandwichLogicFactory.class).define("sandwich_logic_rule");

  // Must be public due to reflective construction of rule factories.
  /** Factory for SANDWICH_LOGIC_RULE. (Javadoc'd to pacify linter.) */
  public static class SandwichLogicFactory extends MockRuleDefaults.DefaultConfiguredTargetFactory {
    @Override
    public ConfiguredTarget create(RuleContext ruleContext)
        throws InterruptedException, RuleErrorException, ActionConflictException {
      AnalysisEnvironment env = ruleContext.getAnalysisEnvironment();
      StarlarkThread thread = ruleContext.getStarlarkThread();
      Mutability mu = thread.mutability();

      Object func = env.getStarlarkDefinedBuiltins().get("builtins_defined_logic");
      Object arg = StarlarkList.newList(mu);
      Object return1;
      Object return2;
      try {
        return1 =
            Starlark.call(
                thread, func, /*args=*/ ImmutableList.of(arg), /*kwargs=*/ ImmutableMap.of());
        return2 =
            Starlark.call(
                thread, func, /*args=*/ ImmutableList.of(arg), /*kwargs=*/ ImmutableMap.of());
      } catch (EvalException e) {
        throw new AssertionError("Failure during Starlark evaluation", e);
      }

      EventHandler handler = env.getEventHandler();
      handler.handle(Event.info("builtins_defined_logic call 1 :: " + return1.toString()));
      handler.handle(Event.info("builtins_defined_logic call 2 :: " + return2.toString()));
      handler.handle(Event.info("final list value :: " + arg.toString()));
      return super.create(ruleContext);
    }
  }

  /**
   * A dummy native rule that passes a Starlark rule context ({@code ctx}) object to
   * {@code @_builtins}-defined code.
   *
   * <p>It looks up "builtins_rule_impl_helper" in exported_to_java, and calls it with {@code ctx}
   * as its sole arg. The rule has a "content" string attribute and "out" output label attribute.
   * The Starlark helper function is responsible for registering an action to generate the output.
   */
  private static final MockRule SANDWICH_CTX_RULE =
      () ->
          MockRule.factory(SandwichCtxFactory.class)
              .define(
                  "sandwich_ctx_rule",
                  Attribute.attr("content", Type.STRING),
                  Attribute.attr("out", BuildType.OUTPUT));

  // Must be public due to reflective construction of rule factories.
  /** Factory for SANDWICH_CTX_RULE. (Javadoc'd to pacify linter.) */
  public static class SandwichCtxFactory extends MockRuleDefaults.DefaultConfiguredTargetFactory {
    @Override
    public ConfiguredTarget create(RuleContext ruleContext)
        throws InterruptedException, RuleErrorException, ActionConflictException {
      AnalysisEnvironment env = ruleContext.getAnalysisEnvironment();
      StarlarkThread thread = ruleContext.getStarlarkThread();
      ruleContext.initStarlarkRuleContext();

      Object func = env.getStarlarkDefinedBuiltins().get("builtins_rule_impl_helper");
      try {
        Starlark.call(
            thread,
            func,
            /*args=*/ ImmutableList.of(ruleContext.getStarlarkRuleContext()),
            /*kwargs=*/ ImmutableMap.of());
      } catch (EvalException e) {
        throw new AssertionError("Failure during Starlark evaluation", e);
      }

      // Don't dispatch to super.create(), which would attempt to register an action to produce
      // "out".
      return new RuleConfiguredTargetBuilder(ruleContext)
          .setFilesToBuild(
              NestedSetBuilder.wrap(Order.STABLE_ORDER, ruleContext.getOutputArtifacts()))
          .setRunfilesSupport(null, null)
          .add(RunfilesProvider.class, RunfilesProvider.simple(Runfiles.EMPTY))
          .build();
    }
  }

  @Override
  protected Iterable<String> getDefaultsForConfiguration() {
    // Override BuildViewTestCase's behavior of setting all sorts of extra options that don't exist
    // on our minimal rule class provider.
    // We do need the host platform. Set it to something trivial.
    return ImmutableList.of("--host_platform=//minimal_buildenv/platforms:default_host");
  }

  @Override
  protected void initializeMockClient() throws IOException {
    // Don't let the AnalysisMock sneak in any WORKSPACE file content, which may depend on
    // repository rules that our minimal rule class provider doesn't have.
    analysisMock.setupMockClient(mockToolsConfig, ImmutableList.of());
    // Provide a trivial platform definition.
    mockToolsConfig.create(
        "minimal_buildenv/platforms/BUILD", //
        "platform(name = 'default_host')");
  }

  @Override
  protected ConfiguredRuleClassProvider createRuleClassProvider() {
    // Set up a bare-bones ConfiguredRuleClassProvider. Aside from being minimalistic, this heads
    // off the possibility that we somehow grow an implicit dependency on production builtins code,
    // which would break since we're overwriting --experimental_builtins_bzl_path.
    ConfiguredRuleClassProvider.Builder builder = new ConfiguredRuleClassProvider.Builder();
    TestRuleClassProvider.addMinimalRules(builder);
    // Add some mock symbols to override.
    builder
        .addRuleDefinition(OVERRIDABLE_RULE)
        .addRuleDefinition(SANDWICH_RULE)
        .addRuleDefinition(SANDWICH_LOGIC_RULE)
        .addRuleDefinition(SANDWICH_CTX_RULE)
        .addStarlarkAccessibleTopLevels("overridable_symbol", "original_value")
        .addStarlarkAccessibleTopLevels(
            "flag_guarded_symbol",
            // For this mock symbol, we reuse the same flag that guards the production
            // _builtins_dummy symbol.
            FlagGuardedValue.onlyWhenExperimentalFlagIsTrue(
                BuildLanguageOptions.EXPERIMENTAL_BUILTINS_DUMMY, "original value"));
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
                + "at /workspace/tools/builtins_staging/exports.bzl:1:6: "
                + "initialization of module 'helper.bzl' (internal) failed");
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

  @Test
  public void overriddenSymbolsAreStillFlagGuarded() throws Exception {
    // Implementation note: This works not because of any special handling in the builtins logic,
    // but rather because flag guarding is implemented at name resolution time, before builtins
    // injection is applied.
    writeExportsBzl(
        "exported_toplevels = {'flag_guarded_symbol': 'overridden value'}",
        "exported_rules = {}",
        "exported_to_java = {}");
    writePkgBzl("flag_guarded_symbol");

    buildAndAssertFailure();
    assertContainsEvent("flag_guarded_symbol is experimental");
  }

  // TODO(#11437): Once we allow access to native symbols via _internal, verify that flag guarding
  // works correctly within builtins.

  @Test
  public void nativeRulesCanUseSymbolsFromBuiltins() throws Exception {
    writeExportsBzl(
        "exported_toplevels = {}",
        "exported_rules = {}",
        "exported_to_java = {'builtins_defined_symbol': 'value_from_builtins'}");
    scratch.file(
        "pkg/BUILD", //
        "sandwich_rule(name = 'sandwich')");

    getConfiguredTarget("//pkg:sandwich");
    assertContainsEvent("builtins_defined_symbol :: value_from_builtins");
  }

  // TODO(#11437): Verify whether this works for native-defined aspects as well.

  @Test
  public void nativeRulesCanCallFunctionsDefinedInBuiltins() throws Exception {
    writeExportsBzl(
        // The driver rule calls this helper twice with a list.
        "def func(arg):",
        "  print('got arg %s' % arg)",
        "  arg.append('blah')",
        "  return len(arg)",
        "exported_toplevels = {}",
        "exported_rules = {}",
        "exported_to_java = {'builtins_defined_logic': func}");
    scratch.file(
        "pkg/BUILD", //
        "sandwich_logic_rule(name = 'sandwich_logic')");

    getConfiguredTarget("//pkg:sandwich_logic");
    assertContainsEvent("got arg []");
    assertContainsEvent("builtins_defined_logic call 1 :: 1");
    assertContainsEvent("got arg [\"blah\"]");
    assertContainsEvent("builtins_defined_logic call 2 :: 2");
    assertContainsEvent("final list value :: [\"blah\", \"blah\"]");
  }

  @Test
  public void nativeRulesCanPassCtxToBuiltinsDefinedHelpers() throws Exception {
    writeExportsBzl(
        "def impl_helper(ctx):",
        "  ctx.actions.write(output=ctx.outputs.out, content=ctx.attr.content)",
        "exported_toplevels = {}",
        "exported_rules = {}",
        "exported_to_java = {'builtins_rule_impl_helper': impl_helper}");
    scratch.file(
        "pkg/BUILD", //
        "sandwich_ctx_rule(name = 'sandwich_ctx', content='foo', out='bar.txt')");

    ConfiguredTarget target = getConfiguredTarget("//pkg:sandwich_ctx");
    Artifact output = getBinArtifact("bar.txt", target);
    ActionAnalysisMetadata action = getGeneratingAction(output);
    assertThat(action).isInstanceOf(FileWriteAction.class);
    assertThat(((FileWriteAction) action).getFileContents()).isEqualTo("foo");
  }

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
