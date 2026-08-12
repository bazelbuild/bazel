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
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedSet;
import com.google.common.truth.Correspondence;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.analysis.config.Scope;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skyframe.config.BaselineOptionsFunction;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import com.google.devtools.build.lib.skyframe.util.SkyframeExecutorTestUtils;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.InMemoryNodeEntry;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import com.google.testing.junit.testparameterinjector.TestParameters;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Tests for {@link BuildOptionsScopeFunction}. */
@RunWith(TestParameterInjector.class)
public final class BuildOptionsScopeFunctionTest extends BuildViewTestCase {

  @Before
  public void doBeforeEachTest() {
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
        "test_flags/build_setting.bzl",
        """
        bool_flag = rule(
            implementation = lambda ctx: [],
            build_setting = config.bool(flag = True),
            attrs = {
                "scope": attr.string(default = "universal"),
            },
        )
        """);
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
  @Ignore("TODO(b/359622692): turns this back on in a follow up CL")
  public void buildOptionsScopesFunction_returnsCorrectScope() throws Exception {
    scratch.file(
        "test_flags/build_setting.bzl",
        """
        bool_flag = rule(
            implementation = lambda ctx: [],
            build_setting = config.bool(flag = True),
            attrs = {
                "scope": attr.string(default = "universal"),
            },
        )
        """);
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
        """);

    scratch.file(
        "test_flags/PROJECT.scl",
        """
        active_directories = {
          "default": [
              "//my_project/"
          ]
        }
        """);
    setBuildLanguageOptions("--experimental_enable_scl_dialect=true");
    BuildOptions buildOptions =
        createBuildOptions("--//test_flags:foo=True", "--//test_flags:bar=True");

    BuildOptionsScopeValue.Key key =
        BuildOptionsScopeValue.Key.create(buildOptions.getStarlarkOptions().keySet());

    BuildOptionsScopeValue buildOptionsScopeValue = executeFunction(key);

    // verify that the Scope is fully resolved for the project-scoped //test_flags:foo and that the
    // universally scoped //test_flags:bar is not reported as scoped
    assertThat(buildOptionsScopeValue.projectScopes())
        .isEqualTo(
            ImmutableMap.of(
                Label.parseCanonical("//test_flags:foo"),
                new Scope(
                    new Scope.ScopeType(Scope.ScopeType.PROJECT),
                    new Scope.ScopeDefinition(ImmutableSet.of("//my_project/")))));
  }

  @Test
  public void buildOptionsScopesFunction_doesNotErrorOut_whenNoProjectFile() throws Exception {
    scratch.file(
        "test_flags/build_setting.bzl",
        """
        bool_flag = rule(
            implementation = lambda ctx: [],
            build_setting = config.bool(flag = True),
            attrs = {
                "scope": attr.string(default = "universal"),
            },
        )
        """);
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

    setBuildLanguageOptions("--experimental_enable_scl_dialect=true");
    BuildOptions buildOptionsWithoutScopes = createBuildOptions("--//test_flags:foo=True");
    BuildOptionsScopeValue.Key key =
        BuildOptionsScopeValue.Key.create(buildOptionsWithoutScopes.getStarlarkOptions().keySet());

    BuildOptionsScopeValue buildOptionsScopeValue = executeFunction(key);
    assertThat(buildOptionsScopeValue.projectScopes())
        .isEqualTo(
            ImmutableMap.of(
                Label.parseCanonical("//test_flags:foo"),
                new Scope(new Scope.ScopeType(Scope.ScopeType.PROJECT), null)));
  }

  // BuildConfigurationValue carries a BuildOptionsScopeValue and is serialized for analysis
  // caching, so scopes have to round-trip through the codec registry.
  @Test
  public void codec() throws Exception {
    Label flag = Label.parseCanonical("//test_flags:foo");
    new SerializationTester(
            BuildOptionsScopeValue.EMPTY,
            new BuildOptionsScopeValue(ImmutableSet.of(flag), ImmutableMap.of()),
            new BuildOptionsScopeValue(
                ImmutableSet.of(flag),
                ImmutableMap.of(
                    flag, new Scope(new Scope.ScopeType(Scope.ScopeType.PROJECT), null))),
            new BuildOptionsScopeValue(
                ImmutableSet.of(flag),
                ImmutableMap.of(
                    flag,
                    new Scope(
                        new Scope.ScopeType(Scope.ScopeType.PROJECT),
                        new Scope.ScopeDefinition(ImmutableSet.of("//my_project/"))))))
        .runTests();
  }

  // Scopes are a property of the flags a configuration sets, so they are resolved once per
  // configuration and read off BuildConfigurationValue. Resolving them per configured target
  // instead would make every configured target a reverse dep of this one node, which is what the
  // analysis phase CPU regression in this area came down to.
  @Test
  public void scopesAreResolvedPerConfiguration_notPerConfiguredTarget() throws Exception {
    scratch.file(
        "test_flags/build_setting.bzl",
        """
        bool_flag = rule(
            implementation = lambda ctx: [],
            build_setting = config.bool(flag = True),
            attrs = {
                "scope": attr.string(default = "universal"),
            },
        )
        """);
    scratch.file(
        "test_flags/BUILD",
        """
        load("//test_flags:build_setting.bzl", "bool_flag")
        bool_flag(
            name = "foo",
            build_setting_default = False,
        )
        """);
    scratch.file(
        "pkg/BUILD",
        """
        filegroup(name = "top", srcs = [":middle"])
        filegroup(name = "middle", srcs = [":bottom"])
        filegroup(name = "bottom")
        """);
    useConfiguration("--//test_flags:foo=True");

    assertThat(getConfiguredTarget("//pkg:top")).isNotNull();

    InMemoryNodeEntry entry =
        getSkyframeExecutor()
            .getEvaluator()
            .getInMemoryGraph()
            .getIfPresent(
                BuildOptionsScopeValue.Key.create(
                    ImmutableSet.of(Label.parseCanonical("//test_flags:foo"))));
    assertThat(entry).isNotNull();
    assertThat(entry.getReverseDepsForDoneEntry())
        .comparingElementsUsing(
            Correspondence.<SkyKey, Class<?>>transforming(SkyKey::getClass, "has class"))
        .doesNotContain(ConfiguredTargetKey.class);
  }

  private BuildOptionsScopeValue executeFunction(BuildOptionsScopeValue.Key key) throws Exception {
    SkyframeExecutor skyframeExecutor = getSkyframeExecutor();
    EvaluationResult<BuildOptionsScopeValue> result =
        SkyframeExecutorTestUtils.evaluate(skyframeExecutor, key, /* keepGoing= */ false, reporter);
    if (result.hasError()) {
      throw result.getError(key).getException();
    }
    return result.get(key);
  }
}
