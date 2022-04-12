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

package com.google.devtools.build.lib.packages;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.analysis.util.MockRule;
import com.google.devtools.build.lib.packages.BazelStarlarkEnvironment.InjectionException;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import net.starlark.java.eval.Structure;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for PackageFactory's management of the predeclared Starlark symbols. */
@RunWith(JUnit4.class)
public final class BazelStarlarkEnvironmentTest extends BuildViewTestCase {

  private static final MockRule OVERRIDABLE_RULE = () -> MockRule.define("overridable_rule");

  @Override
  protected ConfiguredRuleClassProvider createRuleClassProvider() {
    // Add a fake rule and top-level symbol to override.
    ConfiguredRuleClassProvider.Builder builder =
        new ConfiguredRuleClassProvider.Builder()
            // While reading, feel free to mentally substitute overridable_rule -> cc_library and
            // overridable_symbol -> CcInfo.
            .addRuleDefinition(OVERRIDABLE_RULE)
            .addStarlarkAccessibleTopLevels("overridable_symbol", "original_value")
            .addStarlarkAccessibleTopLevels("another_overridable_symbol", "another_original_value");
    TestRuleClassProvider.addStandardRules(builder);
    return builder.build();
  }

  // TODO(#11954): We want BUILD- and WORKSPACE-loaded bzl files to have the exact same environment.
  // In the meantime these two tests help avoid regressions.

  // This property is important for BzlCompileFunction, which relies on the symbol names in the env
  // matching even if the symbols themselves differ.
  @Test
  public void buildAndWorkspaceBzlEnvsDeclareSameNames() throws Exception {
    BazelStarlarkEnvironment starlarkEnv = pkgFactory.getBazelStarlarkEnvironment();
    Set<String> buildBzlNames = starlarkEnv.getUninjectedBuildBzlEnv().keySet();
    Set<String> workspaceBzlNames = starlarkEnv.getWorkspaceBzlEnv().keySet();
    assertThat(buildBzlNames).isEqualTo(workspaceBzlNames);
  }

  @Test
  public void buildAndWorkspaceBzlEnvsAreSameExceptForNative() throws Exception {
    BazelStarlarkEnvironment starlarkEnv = pkgFactory.getBazelStarlarkEnvironment();
    Map<String, Object> buildBzlEnv = new HashMap<>();
    buildBzlEnv.putAll(starlarkEnv.getUninjectedBuildBzlEnv());
    buildBzlEnv.remove("native");
    Map<String, Object> workspaceBzlEnv = new HashMap<>();
    workspaceBzlEnv.putAll(starlarkEnv.getWorkspaceBzlEnv());
    workspaceBzlEnv.remove("native");
    assertThat(buildBzlEnv).isEqualTo(workspaceBzlEnv);
  }

  @Test
  public void builtinsBzlEnv() throws Exception {
    ImmutableMap<String, Object> env = pkgFactory.getBazelStarlarkEnvironment().getBuiltinsBzlEnv();
    // Can see general toplevel symbols.
    assertThat(env).containsKey("rule");
    // Cannot see rule-specific toplevel symbols.
    assertThat(env).doesNotContainKey("overridable_symbol");
    // Has the special builtins-internal module.
    assertThat(env).containsKey("_builtins");
  }

  /**
   * Asserts that injection for a BUILD-loaded .bzl file fails, using the given maps and expecting
   * the given error substring. The overrides list is empty.
   */
  private void assertBuildBzlInjectionFailure(
      Map<String, Object> exportedToplevels, Map<String, Object> exportedRules, String message) {
    BazelStarlarkEnvironment starlarkEnv = pkgFactory.getBazelStarlarkEnvironment();
    InjectionException ex =
        assertThrows(
            InjectionException.class,
            () ->
                starlarkEnv.createBuildBzlEnvUsingInjection(
                    exportedToplevels, exportedRules, /*overridesList=*/ ImmutableList.of()));
    assertThat(ex).hasMessageThat().contains(message);
  }

  /**
   * Asserts that injection for a BUILD file fails, using the given map and expecting the given
   * error substring. The overrides list is empty.
   */
  private void assertBuildInjectionFailure(Map<String, Object> exportedRules, String message) {
    BazelStarlarkEnvironment starlarkEnv = pkgFactory.getBazelStarlarkEnvironment();
    InjectionException ex =
        assertThrows(
            InjectionException.class,
            () ->
                starlarkEnv.createBuildEnvUsingInjection(
                    exportedRules, /*overridesList=*/ ImmutableList.of()));
    assertThat(ex).hasMessageThat().contains(message);
  }

  @Test
  public void buildBzlInjection() throws Exception {
    BazelStarlarkEnvironment starlarkEnv = pkgFactory.getBazelStarlarkEnvironment();
    Map<String, Object> env =
        starlarkEnv.createBuildBzlEnvUsingInjection(
            ImmutableMap.of("overridable_symbol", "new_value"),
            ImmutableMap.of("overridable_rule", "new_rule"),
            /*overridesList=*/ ImmutableList.of());
    assertThat(env).containsEntry("overridable_symbol", "new_value");
    assertThat(((Structure) env.get("native")).getValue("overridable_rule")).isEqualTo("new_rule");
  }

  @Test
  public void buildInjection() throws Exception {
    BazelStarlarkEnvironment starlarkEnv = pkgFactory.getBazelStarlarkEnvironment();
    Map<String, Object> env =
        starlarkEnv.createBuildEnvUsingInjection(
            ImmutableMap.of("overridable_rule", "new_rule"), /*overridesList=*/ ImmutableList.of());
    assertThat(env).containsEntry("overridable_rule", "new_rule");
  }

  @Test
  public void injectedNameMustOverrideExistingName_toplevel() throws Exception {
    assertBuildBzlInjectionFailure(
        ImmutableMap.of("brand_new_toplevel", "foo"),
        ImmutableMap.of(),
        "Injected top-level symbol 'brand_new_toplevel' must override an existing one by that"
            + " name");
  }

  @Test
  public void injectedNameMustOverrideExistingName_rule() throws Exception {
    assertBuildBzlInjectionFailure(
        ImmutableMap.of(),
        ImmutableMap.of("brand_new_rule", "foo"),
        "Injected rule 'brand_new_rule' must override an existing one by that name");
    assertBuildInjectionFailure(
        ImmutableMap.of("brand_new_rule", "foo"),
        "Injected rule 'brand_new_rule' must override an existing one by that name");
  }

  @Test
  public void cannotInjectGeneralSymbol_toplevel() {
    assertBuildBzlInjectionFailure(
        ImmutableMap.of("provider", "new_builtin"),
        ImmutableMap.of(),
        "Cannot override 'provider' with an injected top-level symbol");
  }

  @Test
  public void cannotInjectGeneralSymbol_nativeField() {
    // (Native field for bzl files, toplevel for BUILD files.)
    assertBuildBzlInjectionFailure(
        ImmutableMap.of(),
        ImmutableMap.of("glob", "new_builtin"),
        "Cannot override 'glob' with an injected rule");
    assertBuildInjectionFailure(
        ImmutableMap.of("glob", "new_builtin"), "Cannot override 'glob' with an injected rule");
  }

  @Test
  public void cannotInjectGeneralSymbol_nativeModuleItself() {
    assertBuildBzlInjectionFailure(
        ImmutableMap.of("native", "new_builtin"),
        ImmutableMap.of(),
        "Cannot override 'native' with an injected top-level symbol");
  }

  @Test
  public void cannotInjectGeneralSymbol_universe() {
    assertBuildBzlInjectionFailure(
        ImmutableMap.of("len", "new_builtin"),
        ImmutableMap.of(),
        "Cannot override 'len' with an injected top-level symbol");
    assertBuildInjectionFailure(
        ImmutableMap.of("len", "new_builtin"), "Cannot override 'len' with an injected rule");
  }

  @Test
  public void injectionStatus_respectsDefault() throws Exception {
    BazelStarlarkEnvironment starlarkEnv = pkgFactory.getBazelStarlarkEnvironment();
    Map<String, Object> env =
        starlarkEnv.createBuildBzlEnvUsingInjection(
            ImmutableMap.of(
                "+overridable_symbol",
                "new_value",
                "-another_overridable_symbol",
                "another_new_value"),
            ImmutableMap.of("-overridable_rule", "new_rule"),
            /*overridesList=*/ ImmutableList.of());
    assertThat(env).containsEntry("overridable_symbol", "new_value");
    assertThat(env).containsEntry("another_overridable_symbol", "another_original_value");
    // Match the original rule's toString since the actual specific object is not easily accessible.
    Object overridableRuleValue = ((Structure) env.get("native")).getValue("overridable_rule");
    assertThat(overridableRuleValue.toString()).contains("overridable_rule");
  }

  @Test
  public void injectionStatus_canBeOverridden() throws Exception {
    BazelStarlarkEnvironment starlarkEnv = pkgFactory.getBazelStarlarkEnvironment();
    Map<String, Object> env =
        starlarkEnv.createBuildBzlEnvUsingInjection(
            ImmutableMap.of(
                "+overridable_symbol",
                "new_value",
                "-another_overridable_symbol",
                "another_new_value"),
            ImmutableMap.of("-overridable_rule", "new_rule"),
            ImmutableList.of(
                "-overridable_symbol", "+another_overridable_symbol", "+overridable_rule"));
    assertThat(env).containsEntry("overridable_symbol", "original_value");
    assertThat(env).containsEntry("another_overridable_symbol", "another_new_value");
    assertThat(((Structure) env.get("native")).getValue("overridable_rule")).isEqualTo("new_rule");
  }

  @Test
  public void injectionStatus_cannotBeOverriddenForUnprefixedKeys() throws Exception {
    BazelStarlarkEnvironment starlarkEnv = pkgFactory.getBazelStarlarkEnvironment();
    Map<String, Object> env =
        starlarkEnv.createBuildBzlEnvUsingInjection(
            ImmutableMap.of(
                "overridable_symbol",
                "new_value",
                "another_overridable_symbol",
                "another_new_value"),
            ImmutableMap.of(),
            ImmutableList.of("+overridable_symbol", "-another_overridable_symbol"));
    // Both the + and - are no-ops since the keys aren't prefixed.
    assertThat(env).containsEntry("overridable_symbol", "new_value");
    assertThat(env).containsEntry("another_overridable_symbol", "another_new_value");
  }

  @Test
  public void injectionStatus_overridingUnknownKeysIsNoop() throws Exception {
    BazelStarlarkEnvironment starlarkEnv = pkgFactory.getBazelStarlarkEnvironment();
    Map<String, Object> env =
        starlarkEnv.createBuildBzlEnvUsingInjection(
            ImmutableMap.of("-overridable_symbol", "new_value"),
            ImmutableMap.of(),
            ImmutableList.of("+overridable_symbol", "+unknown_symbol", "-another_unknown_symbol"));
    // Both the + and - are no-ops since the keys aren't prefixed.
    assertThat(env).containsEntry("overridable_symbol", "new_value");
  }

  @Test
  public void injectionStatus_lastOverrideTakesPrecedence() throws Exception {
    BazelStarlarkEnvironment starlarkEnv = pkgFactory.getBazelStarlarkEnvironment();
    Map<String, Object> env =
        starlarkEnv.createBuildBzlEnvUsingInjection(
            ImmutableMap.of("-overridable_symbol", "new_value"),
            ImmutableMap.of(),
            ImmutableList.of("+overridable_symbol", "-overridable_symbol"));
    // Both the + and - are no-ops since the keys aren't prefixed.
    assertThat(env).containsEntry("overridable_symbol", "original_value");
  }

  @Test
  public void injectionStatus_invalidOverrideItem_empty() throws Exception {
    BazelStarlarkEnvironment starlarkEnv = pkgFactory.getBazelStarlarkEnvironment();
    InjectionException ex =
        assertThrows(
            InjectionException.class,
            () ->
                starlarkEnv.createBuildBzlEnvUsingInjection(
                    ImmutableMap.of(), ImmutableMap.of(), ImmutableList.of("")));
    assertThat(ex).hasMessageThat().contains("Invalid injection override item: ''");
  }

  @Test
  public void injectionStatus_invalidOverrideItem_unprefixed() throws Exception {
    BazelStarlarkEnvironment starlarkEnv = pkgFactory.getBazelStarlarkEnvironment();
    InjectionException ex =
        assertThrows(
            InjectionException.class,
            () ->
                starlarkEnv.createBuildBzlEnvUsingInjection(
                    ImmutableMap.of(), ImmutableMap.of(), ImmutableList.of("foo")));
    assertThat(ex).hasMessageThat().contains("Invalid injection override item: 'foo'");
  }

  @Test
  public void injectionStatus_appliesToBuildFiles() throws Exception {
    BazelStarlarkEnvironment starlarkEnv = pkgFactory.getBazelStarlarkEnvironment();
    Map<String, Object> env =
        starlarkEnv.createBuildEnvUsingInjection(
            ImmutableMap.of("+overridable_rule", "new_rule"),
            ImmutableList.of("-overridable_rule"));
    // Match the original rule's toString since the actual specific object is not easily accessible.
    assertThat(env.get("overridable_rule").toString()).contains("overridable_rule");
  }
}
