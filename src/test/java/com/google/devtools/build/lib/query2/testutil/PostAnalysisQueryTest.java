// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.query2.testutil;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static com.google.devtools.build.lib.testutil.TestConstants.PLATFORM_LABEL;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.BuildOptionsView;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.config.transitions.PatchTransition;
import com.google.devtools.build.lib.analysis.util.DummyTestFragment.DummyTestOptions;
import com.google.devtools.build.lib.analysis.util.MockRule;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.query2.PostAnalysisQueryEnvironment;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryFunction;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.Setting;
import com.google.devtools.build.lib.query2.engine.QueryExpression;
import com.google.devtools.build.lib.query2.engine.QueryParser;
import com.google.devtools.build.lib.server.FailureDetails.ConfigurableQuery.Code;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.util.FileTypeSet;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.Set;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

/** Tests for {@link PostAnalysisQueryEnvironment}. */
public abstract class PostAnalysisQueryTest<T> extends AbstractQueryTest<T> {

  // Also filter out platform dependencies.
  @Override
  protected String getDependencyCorrection() {
    return " - deps(" + PLATFORM_LABEL + ")";
  }

  static final String DEFAULT_UNIVERSE = "DEFAULT_UNIVERSE";

  @Before
  public final void disableOrderedResults() {
    helper.setOrderedResults(false);
  }

  @Before
  public final void setMockToolsConfig() {
    this.mockToolsConfig = getHelper().getMockToolsConfig();
  }

  /**
   * In production, cquery constructs the universe by parsing targets from the query expression and
   * building them at the top level. If this is not viable (e.g. component functions) or not desired
   * (e.g. somepath(//foo-built-in-target, //bar-built-in-host), the user must specify the
   * --universe_scope flag. Enforce the same behavior in this test by initializing universe scope to
   * an invalid target expression.
   */
  @Override
  protected String getDefaultUniverseScope() {
    return DEFAULT_UNIVERSE;
  }

  protected PostAnalysisQueryHelper<T> getHelper() {
    return (PostAnalysisQueryHelper<T>) helper;
  }

  /**
   * At the end of each eval, reset the universe scope to the default if the test doesn't use a
   * single universe scope.
   */
  @Override
  protected Set<T> eval(String query) throws Exception {
    maybeParseUniverseScope(query);
    Set<T> queryResult = super.eval(query);
    if (!getHelper().isWholeTestUniverse()) {
      helper.setUniverseScope(getDefaultUniverseScope());
    }
    return queryResult;
  }

  @Override
  protected EvalThrowsResult evalThrows(String query, boolean unconditionallyThrows)
      throws Exception {
    maybeParseUniverseScope(query);
    EvalThrowsResult queryResult = super.evalThrows(query, unconditionallyThrows);
    if (!getHelper().isWholeTestUniverse()) {
      helper.setUniverseScope(getDefaultUniverseScope());
    }
    return queryResult;
  }

  // Parse the universe if the universe has not been set manually through the helper.
  private void maybeParseUniverseScope(String query) throws Exception {
    if (!getHelper()
        .getUniverseScopeAsStringList()
        .equals(Collections.singletonList(getDefaultUniverseScope()))) {
      return;
    }
    QueryExpression expression = QueryParser.parse(query, getDefaultFunctions());
    Set<String> targetPatternSet = new LinkedHashSet<>();
    expression.collectTargetPatterns(targetPatternSet);
    if (!targetPatternSet.isEmpty()) {
      StringBuilder universeScope = new StringBuilder();
      for (String target : targetPatternSet) {
        universeScope.append(target).append(",");
      }
      helper.setUniverseScope(universeScope.toString());
    }
  }

  protected abstract HashMap<String, QueryFunction> getDefaultFunctions();

  protected abstract BuildConfiguration getConfiguration(T target);

  @Override
  protected boolean testConfigurableAttributes() {
    // ConfiguredTargetQuery knows the actual configuration, so it doesn't falsely overapproximate.
    return false;
  }

  @After
  public void cleanUpHelper() {
    getHelper().cleanUp();
    helper = null;
  }

  @Override
  @Test
  public void testTargetLiteralWithMissingTargets() throws Exception {
    getHelper().turnOffFailFast();
    super.testTargetLiteralWithMissingTargets();
  }

  @Override
  @Test
  public void testBadTargetLiterals() throws Exception {
    getHelper().turnOffFailFast();
    // Post-analysis query test infrastructure clobbers certain detailed failures.
    runBadTargetLiteralsTest(/*checkDetailedCode=*/ false);
  }

  @SuppressWarnings("TruthIncompatibleType")
  @Override
  @Test
  public void testNoImplicitDeps() throws Exception {
    MockRule ruleWithImplicitDeps =
        () ->
            MockRule.define(
                "implicit_deps_rule",
                attr("explicit", LABEL).allowedFileTypes(FileTypeSet.ANY_FILE),
                attr("explicit_with_default", LABEL)
                    .value(Label.parseAbsoluteUnchecked("//test:explicit_with_default"))
                    .allowedFileTypes(FileTypeSet.ANY_FILE),
                attr("$implicit", LABEL).value(Label.parseAbsoluteUnchecked("//test:implicit")),
                attr(":latebound", LABEL)
                    .value(
                        Attribute.LateBoundDefault.fromConstantForTesting(
                            Label.parseAbsoluteUnchecked("//test:latebound"))));
    helper.useRuleClassProvider(setRuleClassProviders(ruleWithImplicitDeps).build());

    writeFile(
        "test/BUILD",
        "implicit_deps_rule(",
        "    name = 'my_rule',",
        "    explicit = ':explicit',",
        "    explicit_with_default = ':explicit_with_default',",
        ")",
        "cc_library(name = 'explicit')",
        "cc_library(name = 'explicit_with_default')",
        "cc_library(name = 'implicit')",
        "cc_library(name = 'latebound')");

    final String implicits = "//test:implicit + //test:latebound";
    final String explicits = "//test:my_rule + //test:explicit + //test:explicit_with_default";

    // Check for implicit dependencies (late bound attributes, implicit attributes, platforms)
    assertThat(evalToListOfStrings("deps(//test:my_rule)"))
        .containsAtLeastElementsIn(
            evalToListOfStrings(explicits + " + " + implicits + " + " + PLATFORM_LABEL));

    helper.setQuerySettings(Setting.NO_IMPLICIT_DEPS);
    assertThat(evalToListOfStrings("deps(//test:my_rule)"))
        .containsAtLeastElementsIn(evalToListOfStrings(explicits));
    assertThat(evalToListOfStrings("deps(//test:my_rule)"))
        .containsNoneIn(evalToListOfStrings(implicits));
  }

  @Test
  public void testNoImplicitDeps_toolchains() throws Exception {
    MockRule ruleWithImplicitDeps =
        () ->
            MockRule.define(
                "implicit_toolchain_deps_rule",
                (builder, env) ->
                    builder.addRequiredToolchains(
                        Label.parseAbsoluteUnchecked("//test:toolchain_type")));
    helper.useRuleClassProvider(setRuleClassProviders(ruleWithImplicitDeps).build());

    writeFile(
        "test/toolchain.bzl",
        "def _impl(ctx):",
        "  toolchain = platform_common.ToolchainInfo()",
        "  return [toolchain]",
        "test_toolchain = rule(",
        "    implementation = _impl,",
        ")");
    writeFile(
        "test/BUILD",
        "load(':toolchain.bzl', 'test_toolchain')",
        "implicit_toolchain_deps_rule(",
        "    name = 'my_rule',",
        ")",
        "toolchain_type(name = 'toolchain_type')",
        "toolchain(",
        "    name = 'toolchain',",
        "    toolchain_type = ':toolchain_type',",
        "    toolchain = ':toolchain_impl',",
        ")",
        "test_toolchain(name = 'toolchain_impl')");
    ((PostAnalysisQueryHelper<T>) helper).useConfiguration("--extra_toolchains=//test:toolchain");

    String implicits = "//test:toolchain_impl";
    String explicits = "//test:my_rule";

    // Check for implicit toolchain dependencies
    assertThat(evalToListOfStrings("deps(//test:my_rule)"))
        .containsAtLeastElementsIn(
            evalToListOfStrings(explicits + " + " + implicits + " + " + PLATFORM_LABEL));

    helper.setQuerySettings(Setting.NO_IMPLICIT_DEPS);
    ImmutableList<String> filteredDeps = evalToListOfStrings("deps(//test:my_rule)");
    assertThat(filteredDeps).containsAtLeastElementsIn(evalToListOfStrings(explicits));
    assertThat(filteredDeps).containsNoneIn(evalToListOfStrings(implicits));
  }

  @Test
  public void testNoImplicitDeps_starlark_toolchains() throws Exception {
    writeFile(
        "test/toolchain.bzl",
        "def _impl(ctx):",
        "  toolchain = platform_common.ToolchainInfo()",
        "  return [toolchain]",
        "test_toolchain = rule(",
        "    implementation = _impl,",
        ")");
    writeFile(
        "test/rule.bzl",
        "def _impl(ctx):",
        "  return []",
        "implicit_toolchain_deps_rule = rule(",
        "    implementation = _impl,",
        "    toolchains = ['//test:toolchain_type']",
        ")");
    writeFile(
        "test/BUILD",
        "load(':toolchain.bzl', 'test_toolchain')",
        "load(':rule.bzl', 'implicit_toolchain_deps_rule')",
        "implicit_toolchain_deps_rule(",
        "    name = 'my_rule',",
        ")",
        "toolchain_type(name = 'toolchain_type')",
        "toolchain(",
        "    name = 'toolchain',",
        "    toolchain_type = ':toolchain_type',",
        "    toolchain = ':toolchain_impl',",
        ")",
        "test_toolchain(name = 'toolchain_impl')");
    ((PostAnalysisQueryHelper<T>) helper).useConfiguration("--extra_toolchains=//test:toolchain");

    String implicits = "//test:toolchain_impl";
    String explicits = "//test:my_rule";

    // Check for implicit toolchain dependencies
    assertThat(evalToListOfStrings("deps(//test:my_rule)"))
        .containsAtLeastElementsIn(evalToListOfStrings(explicits + " + " + implicits));

    helper.setQuerySettings(Setting.NO_IMPLICIT_DEPS);
    ImmutableList<String> filteredDeps = evalToListOfStrings("deps(//test:my_rule)");
    assertThat(filteredDeps).containsAtLeastElementsIn(evalToListOfStrings(explicits));
    assertThat(filteredDeps).containsNoneIn(evalToListOfStrings(implicits));
  }

  @Test
  public void testNoImplicitDeps_cc_toolchains() throws Exception {
    writeFile(
        "test/toolchain/toolchain_config.bzl",
        "def _impl(ctx):",
        "    return cc_common.create_cc_toolchain_config_info(",
        "                ctx = ctx,",
        "                toolchain_identifier = 'mock-llvm-toolchain-k8',",
        "                host_system_name = 'mock-system-name-for-k8',",
        "                target_system_name = 'mock-target-system-name-for-k8',",
        "                target_cpu = 'k8',",
        "                target_libc = 'mock-libc-for-k8',",
        "                compiler = 'mock-compiler-for-k8',",
        "                abi_libc_version = 'mock-abi-libc-for-k8',",
        "                abi_version = 'mock-abi-version-for-k8')",
        "cc_toolchain_config = rule(",
        "    implementation = _impl,",
        "    attrs = {},",
        "    provides = [CcToolchainConfigInfo],",
        ")");
    writeFile(
        "test/toolchain/BUILD",
        "load(':toolchain_config.bzl', 'cc_toolchain_config')",
        "cc_toolchain_config(name = 'some-cc-toolchain-config')",
        "filegroup(name = 'nothing', srcs = [])",
        "cc_toolchain(",
        "    name = 'some_cc_toolchain_impl',",
        "    all_files = ':nothing',",
        "    as_files = ':nothing',",
        "    compiler_files = ':nothing',",
        "    dwp_files = ':nothing',",
        "    linker_files = ':nothing',",
        "    objcopy_files = ':nothing',",
        "    strip_files = ':nothing',",
        "    toolchain_config = ':some-cc-toolchain-config',",
        ")",
        "toolchain(",
        "    name = 'some_cc_toolchain',",
        "    toolchain = ':some_cc_toolchain_impl',",
        "    toolchain_type = '" + TestConstants.TOOLS_REPOSITORY + "//tools/cpp:toolchain_type',",
        ")");
    writeFile(
        "test/BUILD", "cc_library(", "    name = 'my_rule',", "    srcs = ['whatever.cpp'],", ")");
    ((PostAnalysisQueryHelper<T>) helper)
        .useConfiguration("--extra_toolchains=//test/toolchain:some_cc_toolchain");

    String implicits = "//test/toolchain:some_cc_toolchain_impl";
    String explicits = "//test:my_rule";

    // Check for implicit toolchain dependencies
    assertThat(evalToListOfStrings("deps(//test:my_rule)"))
        .containsAtLeastElementsIn(
            evalToListOfStrings(explicits + " + " + implicits + " + " + PLATFORM_LABEL));

    helper.setQuerySettings(Setting.NO_IMPLICIT_DEPS);
    ImmutableList<String> filteredDeps = evalToListOfStrings("deps(//test:my_rule)");
    assertThat(filteredDeps).containsAtLeastElementsIn(evalToListOfStrings(explicits));
    assertThat(filteredDeps).containsNoneIn(evalToListOfStrings(implicits));
  }

  // Regression test for b/148550864
  @Test
  public void testNoImplicitDeps_platformDeps() throws Exception {
    MockRule simpleRule = () -> MockRule.define("simple_rule");
    helper.useRuleClassProvider(setRuleClassProviders(simpleRule).build());

    writeFile(
        "test/BUILD",
        "simple_rule(name = 'my_rule')",
        "platform(name = 'host_platform')",
        "platform(name = 'execution_platform')");

    ((PostAnalysisQueryHelper<T>) helper)
        .useConfiguration(
            "--host_platform=//test:host_platform",
            "--extra_execution_platforms=//test:execution_platform");

    // Check for platform dependencies
    assertThat(evalToListOfStrings("deps(//test:my_rule)"))
        .containsAtLeastElementsIn(
            evalToListOfStrings("//test:execution_platform + //test:host_platform"));
    helper.setQuerySettings(Setting.NO_IMPLICIT_DEPS);
    assertThat(evalToListOfStrings("deps(//test:my_rule)")).containsExactly("//test:my_rule");
  }

  @Test
  public void testNoImplicitDeps_computedDefault() throws Exception {
    MockRule computedDefaultRule =
        () ->
            MockRule.define(
                "computed_default_rule",
                attr("conspiracy", Type.STRING).value("space jam was a documentary"),
                attr("dep", LABEL)
                    .allowedFileTypes(FileTypeSet.ANY_FILE)
                    .value(
                        new Attribute.ComputedDefault("conspiracy") {
                          @Override
                          public Object getDefault(AttributeMap rule) {
                            return rule.get("conspiracy", Type.STRING)
                                    .equals("space jam was a documentary")
                                ? Label.parseAbsoluteUnchecked("//test:foo")
                                : null;
                          }
                        }));

    helper.useRuleClassProvider(setRuleClassProviders(computedDefaultRule).build());

    writeFile("test/BUILD", "cc_library(name = 'foo')", "computed_default_rule(name = 'my_rule')");

    String target = "//test:my_rule";

    assertThat(evalToListOfStrings("deps(" + target + ")")).contains("//test:foo");
    helper.setQuerySettings(Setting.NO_IMPLICIT_DEPS);
    assertThat(eval("deps(" + target + ")")).isEqualTo(eval(target));
  }

  @Override
  @Test
  public void testLet() throws Exception {
    getHelper().setWholeTestUniverseScope("//a,//b,//c,//d");
    super.testLet();
  }

  @Override
  @Test
  public void testSet() throws Exception {
    getHelper().setWholeTestUniverseScope("//a:*,//b:*,//c:*,//d:*");
    super.testSet();
  }

  /** PatchTransition on --foo */
  public static class FooPatchTransition implements PatchTransition {
    String toOption;
    String name;

    public FooPatchTransition(String toOption, String name) {
      this.toOption = toOption;
      this.name = name;
    }

    public FooPatchTransition(String toOption) {
      this(toOption, "FooPatchTransition");
    }

    @Override
    public String getName() {
      return this.name;
    }

    @Override
    public ImmutableSet<Class<? extends FragmentOptions>> requiresOptionFragments() {
      return ImmutableSet.of(DummyTestOptions.class);
    }

    @Override
    public BuildOptions patch(BuildOptionsView options, EventHandler eventHandler) {
      BuildOptionsView result = options.clone();
      result.get(DummyTestOptions.class).foo = toOption;
      return result.underlying();
    }
  }

  @Test
  public void testMultipleTopLevelConfigurations() throws Exception {
    MockRule transitionedRule =
        () ->
            MockRule.define(
                "transitioned_rule",
                (builder, env) -> builder.cfg(new FooPatchTransition("SET BY PATCH")).build());

    MockRule untransitionedRule = () -> MockRule.define("untransitioned_rule");

    helper.useRuleClassProvider(
        setRuleClassProviders(transitionedRule, untransitionedRule).build());

    writeFile(
        "test/BUILD",
        "transitioned_rule(name = 'transitioned_rule')",
        "untransitioned_rule(name = 'untransitioned_rule')");

    Set<T> result = eval("//test:transitioned_rule+//test:untransitioned_rule");

    assertThat(result).hasSize(2);

    Iterator<T> resultIterator = result.iterator();
    assertThat(getConfiguration(resultIterator.next()))
        .isNotEqualTo(getConfiguration(resultIterator.next()));
  }

  @Test
  public abstract void testMultipleTopLevelConfigurations_nullConfigs() throws Exception;

  @Test
  public void testMultipleTopLevelConfigurations_multipleConfigsPrefersTopLevel() throws Exception {
    MockRule ruleWithTransitionAndDep =
        () ->
            MockRule.define(
                "rule_with_transition_and_dep",
                (builder, env) ->
                    builder
                        .cfg(new FooPatchTransition("SET BY PATCH"))
                        .addAttribute(
                            attr("dep", LABEL).allowedFileTypes(FileTypeSet.ANY_FILE).build())
                        .build());

    MockRule simpleRule = () -> MockRule.define("simple_rule");

    helper.useRuleClassProvider(
        setRuleClassProviders(ruleWithTransitionAndDep, simpleRule).build());

    writeFile(
        "test/BUILD",
        "rule_with_transition_and_dep(name = 'top-level', dep = ':dep')",
        "simple_rule(name = 'dep')");

    helper.setUniverseScope("//test:*");

    assertThat(getConfiguration(Iterables.getOnlyElement(eval("//test:dep"))))
        .isNotEqualTo(getConfiguration(Iterables.getOnlyElement(eval("//test:top-level"))));
  }

  private void writeSimpleTarget() throws Exception {
    MockRule simpleRule =
        () ->
            MockRule.define(
                "simple_rule", attr("dep", LABEL).allowedFileTypes(FileTypeSet.ANY_FILE));
    helper.useRuleClassProvider(setRuleClassProviders(simpleRule).build());

    writeFile("test/BUILD", "simple_rule(name = 'target')");
  }

  @Test
  public void testVisibleFunctionDoesNotWork() throws Exception {
    writeSimpleTarget();
    EvalThrowsResult result = evalThrows("visible(//test:target, //test:*)", true);
    assertThat(result.getMessage()).isEqualTo("visible() is not supported on configured targets");
    assertConfigurableQueryCode(result.getFailureDetail(), Code.VISIBLE_FUNCTION_NOT_SUPPORTED);
  }

  @Test
  public void testSiblingsFunctionDoesNotWork() throws Exception {
    writeSimpleTarget();
    EvalThrowsResult result = evalThrows("siblings(//test:target)", true);
    assertThat(result.getMessage()).isEqualTo("siblings() not supported for post analysis queries");
    assertConfigurableQueryCode(result.getFailureDetail(), Code.SIBLINGS_FUNCTION_NOT_SUPPORTED);
  }

  @Test
  public void testBuildfilesFunctionDoesNotWork() throws Exception {
    writeSimpleTarget();
    EvalThrowsResult result = evalThrows("buildfiles(//test:target)", true);
    assertThat(result.getMessage())
        .isEqualTo("buildfiles() doesn't make sense for the configured target graph");
    assertConfigurableQueryCode(result.getFailureDetail(), Code.BUILDFILES_FUNCTION_NOT_SUPPORTED);
  }

  // LabelListAttr not currently supported.
  @Override
  public void testLabelsOperator() {}

  // Wants to get the query environment without evaluation -- not worth it.
  @Override
  @Test
  public void testEqualityOfOrderedThreadSafeImmutableSet() {}

  @Override
  public void testDefaultCopts() {}

  @Override
  public void testHdrsCheck() {}

  @Override
  public void testFilesetPackageDeps() {}

  @Override
  public void testRegressionBug1686119() {}

  // The actual crosstool-related targets depended on are not the nominal crosstool label the test
  // expects.

  // "Extended rules" don't play nicely with actual analysis.
  @Override
  public void testNoDepsOnAspectAttributeWhenAspectMissing() {}

  @Override
  public void testNoDepsOnAspectAttributeWithNoImpicitDeps() {}

  @Override
  public void testHaveDepsOnAspectsAttributes() {}

  // Can't handle loading-phase errors.
  @Override
  public void testStrictTestSuiteWithFile() {}

  @Override
  public void testTestsOperatorReportsMissingTargets() {}

  @Override
  public void testCycleInStarlark() {}

  @Override
  public void testCycleInStarlarkParentDir() {}

  @Override
  public void testCycleInSubpackage() {}

  @Override
  public void testRegression1309697() {}

  // Can't handle cycles.
  @Override
  public void testDotDotDotWithCycle() {}

  @Override
  public void testDotDotDotWithUnrelatedCycle() {}

  // ...
  @Override
  public void testQueryTimeLoadingTargetsBelowNonPackageDirectory() {}

  @Override
  public void testQueryTimeLoadingOfTargetsBelowPackageHappyPath() {}

  @Override
  public void testQueryTimeLoadingTargetsBelowMissingPackage() {}

  // These tests clear the universe, getting rid of mock tools that are needed for analysis. Disable
  // at least for now. Other than testSlashSlashDotDotDot, they're only testing visibility anyway.

  @Override
  public void testSlashSlashDotDotDot() {}

  @Override
  public void testVisible_default_private() {}

  @Override
  public void testVisible_default_public() {}

  @Override
  public void testPackageGroupAllBeneath() {}

  @Override
  public void testVisible_java_javatests() {}

  @Override
  public void testVisible_java_javatests_different_package() {}

  @Override
  public void testVisible_javatests_java() {}

  @Override
  public void testVisible_package_group() {}

  @Override
  public void testVisible_package_group_include() {}

  @Override
  public void testVisible_package_group_invisible() {}

  @Override
  public void testVisible_private_same_package() {}

  @Override
  public void testVisible_simple_different_subpackages() {}

  @Override
  public void testVisible_simple_package() {}

  @Override
  public void testVisible_simple_private() {}

  @Override
  public void testVisible_simple_public() {}

  @Override
  public void testVisible_simple_subpackages() {}

  // test_suite rules aren't supported, since they're not configured targets.

  @Override
  public void testTestsOperatorFiltersByNegativeTag() {}

  @Override
  public void testTestsOperatorCrossesPackages() {}

  @Override
  public void testTestsOperatorHandlesCyclesGracefully() {}

  @Override
  public void testTestSuiteInTestsAttributeAndViceVersa() {}

  @Override
  public void testAmbiguousAllResolvesToTestSuiteNamedAll() {}

  @Override
  public void testTestSuiteWithFile() {}

  @Override
  public void testTestsOperatorFiltersByTagSizeAndEnv() {}

  @Override
  public void testTestsOperatorExpandsTestsAndExcludesNonTests() {}

  // buildfiles() operator.
  @Override
  public void testBuildFiles() {}

  @Override
  public void testBuildFilesDoesNotReturnVisibilityOfBUILD() {}

  @Override
  public void testBuildFilesDoesNotReturnVisibilityOfRule() {}

  @Override
  public void testBuildfilesOfBuildfiles() {}

  @Override
  public void testBuildfilesWithDuplicates() {}

  @Override
  public void testTargetsFromBuildfilesAndRealTargets() {}

  // siblings() operator.

  @Override
  public void testSiblings_duplicatePackages() {}

  @Override
  public void testSiblings_samePackageRdeps() {}

  @Override
  public void testSiblings_matchesTargetNamedAll() {}

  @Override
  public void testSiblings_simple() {}

  @Override
  public void testSiblings_withBuildfiles() {}

  // same_pkg_direct_rdeps() operator.

  @Override
  public void testSamePackageRdeps_simple() throws Exception {}

  @Override
  public void testSamePackageRdeps_duplicate() throws Exception {}

  @Override
  public void testSamePackageRdeps_two() throws Exception {}

  @Override
  public void testSamePackageRdeps_twoPackages() throws Exception {}

  @Override
  public void testSamePackageRdeps_crissCross() throws Exception {}

  // We eagerly load all packages, so can't test that we don't load one.
  @Override
  @Test
  public void testWildcardsDontLoadUnnecessaryPackages() {}

  // Query needs a graph.
  @Override
  @Test
  public void testGraphOrderOfWildcards() {}

  // Visibility is checked in the analysis phase, so the post-analysis query done in this unit test
  // would never occur because the visibility error would occur first.
  @Override
  @Test
  public void testVisibleWithNonPackageGroupVisibility() throws Exception {}

  // Visibility is checked in the analysis phase, so the post-analysis query done in this unit test
  // would never occur because the visibility error would occur first.
  @Override
  @Test
  public void testVisibleWithPackageGroupWithNonPackageGroupIncludes() throws Exception {}

  // We don't support --nodep_deps=false.
  @Override
  @Test
  public void testNodepDeps_false() throws Exception {}

  // package_group instances have a null configuration and are filtered out by --host_deps=false.
  @Override
  @Test
  public void testDefaultVisibilityReturnedInDeps_nonEmptyDependencyFilter() throws Exception {}

  protected static void assertConfigurableQueryCode(FailureDetail failureDetail, Code code) {
    assertThat(failureDetail.getConfigurableQuery().getCode()).isEqualTo(code);
  }
}
