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

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.analysis.util.MockRule;
import com.google.devtools.build.lib.packages.PackageFactory.InjectionException;
import com.google.devtools.build.lib.syntax.ClassObject;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
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
            .addStarlarkAccessibleTopLevels("overridable_symbol", "original_value");
    TestRuleClassProvider.addStandardRules(builder);
    return builder.build();
  }

  // TODO(#11954): We want BUILD- and WORKSPACE-loaded bzl files to have the exact same environment.
  // In the meantime these two tests help avoid regressions.

  // This property is important for ASTFileLookupFunction, which relies on the symbol names in the
  // env matching even if the symbols themselves differ.
  @Test
  public void buildAndWorkspaceBzlEnvsDeclareSameNames() throws Exception {
    Set<String> buildBzlNames = pkgFactory.getUninjectedBuildBzlEnv().keySet();
    Set<String> workspaceBzlNames = pkgFactory.getWorkspaceBzlEnv().keySet();
    assertThat(buildBzlNames).isEqualTo(workspaceBzlNames);
  }

  @Test
  public void buildAndWorkspaceBzlEnvsAreSameExceptForNative() throws Exception {
    Map<String, Object> buildBzlEnv = new HashMap<>();
    buildBzlEnv.putAll(pkgFactory.getUninjectedBuildBzlEnv());
    buildBzlEnv.remove("native");
    Map<String, Object> workspaceBzlEnv = new HashMap<>();
    workspaceBzlEnv.putAll(pkgFactory.getWorkspaceBzlEnv());
    workspaceBzlEnv.remove("native");
    assertThat(buildBzlEnv).isEqualTo(workspaceBzlEnv);
  }

  @Test
  public void injection() throws Exception {
    Map<String, Object> env =
        pkgFactory.createBuildBzlEnvUsingInjection(
            ImmutableMap.of("overridable_symbol", "new_value"),
            ImmutableMap.of("overridable_rule", "new_rule"));
    assertThat(env).containsEntry("overridable_symbol", "new_value");
    assertThat(((ClassObject) env.get("native")).getValue("overridable_rule"))
        .isEqualTo("new_rule");
  }

  /** Asserts that injection with the given maps fails with the given error substring. */
  private void assertInjectionFailure(
      ImmutableMap<String, Object> injectedToplevels,
      ImmutableMap<String, Object> injectedRules,
      String message) {
    InjectionException ex =
        assertThrows(
            InjectionException.class,
            () -> pkgFactory.createBuildBzlEnvUsingInjection(injectedToplevels, injectedRules));
    assertThat(ex).hasMessageThat().contains(message);
  }

  @Test
  public void injectedNameMustOverrideExistingName_toplevelSymbol() throws Exception {
    assertInjectionFailure(
        ImmutableMap.of("brand_new_toplevel", "foo"),
        ImmutableMap.of(),
        "Injected top-level symbol 'brand_new_toplevel' must override an existing symbol by"
            + " that name");
  }

  @Test
  public void injectedNameMustOverrideExistingName_nativeField() throws Exception {
    assertInjectionFailure(
        ImmutableMap.of(),
        ImmutableMap.of("brand_new_field", "foo"),
        "Injected native module field 'brand_new_field' must override an existing symbol by "
            + "that name");
  }

  @Test
  public void cannotInjectGenericNonRuleSpecificSymbol_toplevelSymbol() {
    assertInjectionFailure(
        ImmutableMap.of("provider", "new_builtin"),
        ImmutableMap.of(),
        "Cannot override top-level builtin 'provider' with an injected value");
  }

  @Test
  public void cannotInjectGenericNonRuleSpecificSymbol_nativeField() {
    assertInjectionFailure(
        ImmutableMap.of(),
        ImmutableMap.of("glob", "new_builtin"),
        "Cannot override native module field 'glob' with an injected value");
  }

  @Test
  public void cannotInjectGenericNonRuleSpecificSymbol_nativeModuleItself() {
    assertInjectionFailure(
        ImmutableMap.of("native", "new_builtin"),
        ImmutableMap.of(),
        "Cannot override top-level builtin 'native' with an injected value");
  }

  @Test
  public void cannotInjectGenericNonRuleSpecificSymbol_universeSymbol() {
    assertInjectionFailure(
        ImmutableMap.of("len", "new_builtin"),
        ImmutableMap.of(),
        "Cannot override top-level builtin 'len' with an injected value");
  }
}
