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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedSet;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.analysis.config.Scope;
import com.google.devtools.build.lib.analysis.starlark.StarlarkBuildSettingsDetailsValue;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skyframe.config.BaselineOptionsFunction;
import com.google.devtools.build.lib.skyframe.util.SkyframeExecutorTestUtils;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import com.google.testing.junit.testparameterinjector.TestParameters;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Tests for the scope-related parts of {@link StarlarkBuildSettingsDetailsFunction}. */
@RunWith(TestParameterInjector.class)
public final class StarlarkBuildSettingsDetailsFunctionTest extends BuildViewTestCase {

  private static final Scope.ScopeType PROJECT = new Scope.ScopeType(Scope.ScopeType.PROJECT);
  private static final Scope.ScopeType UNIVERSAL = new Scope.ScopeType(Scope.ScopeType.UNIVERSAL);
  private static final Scope.ScopeType DEFAULT = new Scope.ScopeType(Scope.ScopeType.DEFAULT);

  @Before
  public void doBeforeEachTest() throws Exception {
    // inject Precomputed.BASELINE_CONFIGURATION
    AnalysisMock analysisMock = AnalysisMock.get();
    ConfiguredRuleClassProvider ruleClassProvider = analysisMock.createRuleClassProvider();
    ImmutableSortedSet<Class<? extends FragmentOptions>> buildOptionClasses =
        ruleClassProvider.getFragmentRegistry().getOptionsClasses();

    SequencedSkyframeExecutor skyframeExecutor = getSkyframeExecutor();
    BuildOptions defaultBuildOptions =
        BuildOptions.getDefaultBuildOptionsForFragments(buildOptionClasses).clone();
    skyframeExecutor.injectExtraPrecomputedValues(
        new ImmutableList.Builder<PrecomputedValue.Injected>()
            .add(
                PrecomputedValue.injected(
                    BaselineOptionsFunction.BASELINE_CONFIGURATION, defaultBuildOptions))
            .addAll(analysisMock.getPrecomputedValues())
            .build());

    scratch.file("test/BUILD");
    writeProjectSclDefinition("test/project_proto.scl");
    scratch.file(
        "test_flags/build_setting.bzl",
        """
        bool_flag = rule(
            implementation = lambda ctx: [],
            build_setting = config.bool(flag = True),
            attrs = {
                "scope": attr.string(default = "universal"),
                "on_leave_scope": attr.bool(default = False),
            },
        )
        """);
  }

  @Test
  @TestParameters({
    "{scope: 'universal', expectFail: false}",
    "{scope: 'target', expectFail: false}",
    "{scope: 'project', expectFail: false}",
    "{scope: 'badvalue', expectFail: true}",
    "{scope: 'default', expectFail: true}", // Valid internal value but can't be set by users.
  })
  public void validScopeAttributeValues(String scope, boolean expectFail) throws Exception {
    scratch.file(
        "test_flags/BUILD",
        """
        load("//test_flags:build_setting.bzl", "bool_flag")
        bool_flag(
            name = "foo",
            build_setting_default = False,
            scope = "%s",
        )
        """
            .formatted(scope));

    if (!expectFail) {
      assertThat(createBuildOptions("--//test_flags:foo=True")).isNotNull();
    } else {
      var exception =
          assertThrows(
              InvalidConfigurationException.class,
              () -> createBuildOptions("--//test_flags:foo=True"));
      assertThat(exception).hasMessageThat().contains("Invalid \"scope\" attribute value");
    }
  }

  @Test
  public void resolvesScopesOfAllBuildSettings() throws Exception {
    scratch.file(
        "test_flags/BUILD",
        """
        load("//test_flags:build_setting.bzl", "bool_flag")
        bool_flag(
            name = "foo",
            build_setting_default = False,
            scope = "project",
        )
        bool_flag(
            name = "bar",
            build_setting_default = False,
        )
        bool_flag(
            name = "baz",
            build_setting_default = False,
            scope = "target",
            on_leave_scope = True,
        )
        alias(
            name = "foo_alias",
            actual = ":foo",
        )
        """);
    scratch.file(
        "test_flags/PROJECT.scl",
        """
        load("//test:project_proto.scl", "project_pb2")
        project = project_pb2.Project.create(
            project_directories = ["//my_project"],
        )
        """);
    Label foo = Label.parseCanonical("//test_flags:foo");
    Label bar = Label.parseCanonical("//test_flags:bar");
    Label baz = Label.parseCanonical("//test_flags:baz");
    Label fooAlias = Label.parseCanonical("//test_flags:foo_alias");

    StarlarkBuildSettingsDetailsValue details =
        executeFunction(
            StarlarkBuildSettingsDetailsValue.keyForBuildSettings(
                ImmutableSet.of(fooAlias, bar, baz), ImmutableSet.of()));

    assertThat(details.buildSettings()).containsExactly(fooAlias, bar, baz);
    assertThat(details.buildSettingToScopeType())
        .containsExactly(foo, PROJECT, bar, UNIVERSAL, baz, new Scope.ScopeType("target"));
    assertThat(details.buildSettingToOnLeaveScopeValue()).containsExactly(baz, true);
    assertThat(details.projectScopes())
        .containsExactly(
            foo, new Scope(PROJECT, new Scope.ScopeDefinition(ImmutableSet.of("//my_project"))));
    assertThat(details.hasProjectScopedBuildSettings()).isTrue();
    // Aliases resolve to the actual build setting's scope.
    assertThat(details.projectScopeOf(fooAlias)).isEqualTo(details.projectScopes().get(foo));
    assertThat(details.projectScopeOf(bar)).isNull();
    assertThat(details.covers(ImmutableSet.of(bar, baz))).isTrue();
    assertThat(details.covers(ImmutableSet.of(foo))).isFalse();
  }

  @Test
  public void projectScopedBuildSettingWithoutProjectFile_hasNoScopeDefinition() throws Exception {
    scratch.file(
        "test_flags/BUILD",
        """
        load("//test_flags:build_setting.bzl", "bool_flag")
        bool_flag(
            name = "foo",
            build_setting_default = False,
            scope = "project",
        )
        """);
    Label foo = Label.parseCanonical("//test_flags:foo");

    StarlarkBuildSettingsDetailsValue details =
        executeFunction(
            StarlarkBuildSettingsDetailsValue.keyForBuildSettings(
                ImmutableSet.of(foo), ImmutableSet.of()));

    assertThat(details.projectScopes()).containsExactly(foo, new Scope(PROJECT, null));
  }

  @Test
  public void buildSettingWithoutScopeAttribute_hasDefaultScope() throws Exception {
    scratch.file(
        "test_flags/plain.bzl",
        """
        plain_flag = rule(
            implementation = lambda ctx: [],
            build_setting = config.bool(flag = True),
        )
        """);
    scratch.file(
        "test_flags/BUILD",
        """
        load("//test_flags:plain.bzl", "plain_flag")
        plain_flag(
            name = "foo",
            build_setting_default = False,
        )
        """);
    Label foo = Label.parseCanonical("//test_flags:foo");

    StarlarkBuildSettingsDetailsValue details =
        executeFunction(
            StarlarkBuildSettingsDetailsValue.keyForBuildSettings(
                ImmutableSet.of(foo), ImmutableSet.of()));

    assertThat(details.buildSettingToScopeType()).containsExactly(foo, DEFAULT);
    assertThat(details.projectScopes()).isEmpty();
    assertThat(details.hasProjectScopedBuildSettings()).isFalse();
  }

  private StarlarkBuildSettingsDetailsValue executeFunction(
      StarlarkBuildSettingsDetailsValue.Key key) throws Exception {
    SkyframeExecutor skyframeExecutor = getSkyframeExecutor();
    EvaluationResult<StarlarkBuildSettingsDetailsValue> result =
        SkyframeExecutorTestUtils.evaluate(skyframeExecutor, key, /* keepGoing= */ false, reporter);
    if (result.hasError()) {
      throw result.getError(key).getException();
    }
    return result.get(key);
  }
}
