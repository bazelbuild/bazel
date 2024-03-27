// Copyright 2022 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.common.truth.Correspondence;
import com.google.devtools.build.lib.analysis.test.AnalysisFailure;
import com.google.devtools.build.lib.analysis.test.AnalysisFailureInfo;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.analysis.util.MockRule;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import javax.annotation.Nullable;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkSemantics;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

/**
 * Tests verifying analysis failure propagation via {@link AnalysisFailureInfo} when {@code
 * --allow_analysis_failures=true}.
 */
@RunWith(TestParameterInjector.class)
public final class AnalysisFailureInfoTest extends BuildViewTestCase {

  @Before
  public void setUp() throws Exception {
    useConfiguration("--allow_analysis_failures=true");
  }

  @Test
  public void analysisFailureInfoStarlarkApi() throws Exception {
    Label label = Label.create("test", "test");
    AnalysisFailure failure = AnalysisFailure.create(label, "ErrorMessage");
    assertThat(getattr(failure, "label")).isSameInstanceAs(label);
    assertThat(getattr(failure, "message")).isEqualTo("ErrorMessage");

    AnalysisFailureInfo info = AnalysisFailureInfo.forAnalysisFailures(ImmutableList.of(failure));
    // info.causes.to_list()[0] == failure
    NestedSet<AnalysisFailure> causes =
        Depset.cast(getattr(info, "causes"), AnalysisFailure.class, "causes");
    assertThat(causes.toList().get(0)).isSameInstanceAs(failure);
  }

  private static Object getattr(Object x, String name) throws Exception {
    return Starlark.getattr(/*mu=*/ null, StarlarkSemantics.DEFAULT, x, name, null);
  }

  /** Regression test for b/154007057 (rule name) and b/186685477 (output file). */
  @Test
  public void nativeRuleExpanderFailure(
      @TestParameter({"//test:bad_variable", "//test:bad_variable.out"}) String targetToRequest)
      throws Exception {
    scratch.file(
        "test/BUILD",
        """
        genrule(
            name = "bad_variable",
            outs = ["bad_variable.out"],
            cmd = "cp $< $@",  # Error to use $< with no srcs
        )
        """);

    ConfiguredTarget target = getConfiguredTarget(targetToRequest);
    AnalysisFailureInfo info =
        (AnalysisFailureInfo) target.get(AnalysisFailureInfo.STARLARK_CONSTRUCTOR.getKey());
    AnalysisFailure failure = info.getCauses().getSet(AnalysisFailure.class).toList().get(0);
    assertThat(failure.getMessage()).contains("variable '$<' : no input file");
    assertThat(failure.getLabel()).isEqualTo(Label.parseCanonicalUnchecked("//test:bad_variable"));
  }

  /** Regression test for b/154007057. */
  @Test
  public void nativeRuleConfiguredTargetFactoryCreateReturningNull() throws Exception {
    scratch.file(
        "test/BUILD",
        """
        native_rule_with_failing_configured_target_factory(
            name = "bad_factory",
        )
        """);

    ConfiguredTarget target = getConfiguredTarget("//test:bad_factory");
    AnalysisFailureInfo info =
        (AnalysisFailureInfo) target.get(AnalysisFailureInfo.STARLARK_CONSTRUCTOR.getKey());
    AnalysisFailure failure = info.getCauses().getSet(AnalysisFailure.class).toList().get(0);
    assertThat(failure.getMessage()).contains("FailingRuleConfiguredTargetFactory.create() fails");
    assertThat(failure.getLabel()).isEqualTo(Label.parseCanonicalUnchecked("//test:bad_factory"));
  }

  /** Dummy factory whose {@code create()} method always returns {@code null}. */
  public static final class FailingRuleConfiguredTargetFactory
      implements RuleConfiguredTargetFactory {
    @Override
    @Nullable
    public ConfiguredTarget create(RuleContext ruleContext) {
      ruleContext.ruleError("FailingRuleConfiguredTargetFactory.create() fails");
      return null;
    }
  }

  @Test
  public void analysisTestNotReturningAnalysisTestResultInfo_cannotPropagate() throws Exception {
    scratch.file(
        "test/BUILD", //
        "providerless_analysis_test(name = 'providerless')");

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test:providerless");
    assertContainsEvent(
        "Error while collecting analysis-phase failure information for '//test:providerless': rules"
            + " with analysis_test=true must return an instance of AnalysisTestResultInfo");
  }

  /** Regression test for b/233890545 */
  @Test
  public void analysisTestExpectingFailureDependedOnByAnalysisTest_cannotPropagate()
      throws Exception {
    useConfiguration("--allow_analysis_failures=false");
    scratch.file(
        "test/extension.bzl",
        """
        def bad_rule_impl(ctx):
            fail("Bad rule fails")

        bad_rule = rule(
            implementation = bad_rule_impl,
            attrs = {"dep": attr.label()},
        )

        def analysis_test_impl(ctx):
            return [AnalysisTestResultInfo(success = False, message = "Expect failure")]

        _transition = analysis_test_transition(
            settings = {"//command_line_option:allow_analysis_failures": "True"},
        )

        analysis_test = rule(
            implementation = analysis_test_impl,
            analysis_test = True,
            attrs = {"dep": attr.label(cfg = _transition)},
        )
        """);

    scratch.file(
        "test/BUILD",
        """
        load("//test:extension.bzl", "analysis_test", "bad_rule")

        analysis_test(
            name = "outer",
            dep = ":inner",
        )

        analysis_test(
            name = "inner",
            dep = ":tested_by_inner",
        )

        bad_rule(name = "tested_by_inner")
        """);

    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//test:outer");
    assertContainsEvent(
        "Error while collecting analysis-phase failure information for '//test:inner':"
            + " analysis_test rule '//test:inner' cannot be transitively depended on by another"
            + " analysis test rule");
  }

  @Override
  protected ConfiguredRuleClassProvider createRuleClassProvider() {
    ConfiguredRuleClassProvider.Builder builder =
        new ConfiguredRuleClassProvider.Builder()
            .addRuleDefinition(
                ((MockRule)
                    () ->
                        MockRule.factory(FailingRuleConfiguredTargetFactory.class)
                            .define("native_rule_with_failing_configured_target_factory")))
            .addRuleDefinition(
                (MockRule)
                    () ->
                        MockRule.ancestor(
                                BaseRuleClasses.TestBaseRule.class,
                                BaseRuleClasses.NativeBuildRule.class)
                            .type(RuleClassType.TEST)
                            .define(
                                "providerless_analysis_test",
                                (ruleClassBuilder, env) -> ruleClassBuilder.setIsAnalysisTest()));
    TestRuleClassProvider.addStandardRules(builder);
    return builder.build();
  }

  private static final Correspondence<AnalysisFailure, AnalysisFailure>
      analysisFailureCorrespondence =
          Correspondence.from(
              (actual, expected) ->
                  actual.getLabel().equals(expected.getLabel())
                      && actual.getMessage().contains(expected.getMessage()),
              "is equivalent to");

  @Test
  public void starlarkRuleFailure() throws Exception {
    scratch.file(
        "test/extension.bzl",
        """
        def custom_rule_impl(ctx):
            fail("This Is My Failure Message")

        custom_rule = rule(implementation = custom_rule_impl)
        """);

    scratch.file(
        "test/BUILD",
        """
        load("//test:extension.bzl", "custom_rule")

        custom_rule(name = "r")
        """);

    ConfiguredTarget target = getConfiguredTarget("//test:r");
    AnalysisFailureInfo info =
        (AnalysisFailureInfo) target.get(AnalysisFailureInfo.STARLARK_CONSTRUCTOR.getKey());
    AnalysisFailure failure = info.getCauses().getSet(AnalysisFailure.class).toList().get(0);
    assertThat(failure.getMessage()).contains("This Is My Failure Message");
    assertThat(failure.getLabel()).isEqualTo(Label.parseCanonicalUnchecked("//test:r"));
  }

  @Test
  public void starlarkRuleFailure_forTest() throws Exception {
    scratch.file(
        "test/extension.bzl",
        """
        def custom_rule_impl(ctx):
            fail("This Is My Failure Message")

        custom_test = rule(
            implementation = custom_rule_impl,
            test = True,
        )
        """);

    scratch.file(
        "test/BUILD",
        """
        load("//test:extension.bzl", "custom_test")

        custom_test(name = "r")
        """);

    ConfiguredTarget target = getConfiguredTarget("//test:r");
    AnalysisFailureInfo info =
        (AnalysisFailureInfo) target.get(AnalysisFailureInfo.STARLARK_CONSTRUCTOR.getKey());
    AnalysisFailure failure = info.getCauses().getSet(AnalysisFailure.class).toList().get(0);
    assertThat(failure.getMessage()).contains("This Is My Failure Message");
    assertThat(failure.getLabel()).isEqualTo(Label.parseCanonicalUnchecked("//test:r"));
  }

  @Test
  public void starlarkRuleFailure_withOutput() throws Exception {
    scratch.file(
        "test/extension.bzl",
        """
        def custom_rule_impl(ctx):
            fail("This Is My Failure Message")

        custom_rule = rule(
            implementation = custom_rule_impl,
            outputs = {"my_output": "%{name}.txt"},
        )
        """);

    scratch.file(
        "test/BUILD",
        """
        load("//test:extension.bzl", "custom_rule")

        custom_rule(name = "r")
        """);

    ConfiguredTarget target = getConfiguredTarget("//test:r");
    AnalysisFailureInfo info =
        (AnalysisFailureInfo) target.get(AnalysisFailureInfo.STARLARK_CONSTRUCTOR.getKey());
    AnalysisFailure failure = info.getCauses().getSet(AnalysisFailure.class).toList().get(0);
    assertThat(failure.getMessage()).contains("This Is My Failure Message");
    assertThat(failure.getLabel()).isEqualTo(Label.parseCanonicalUnchecked("//test:r"));
  }

  @Test
  public void transitiveStarlarkRuleFailure() throws Exception {
    scratch.file(
        "test/extension.bzl",
        """
        def custom_rule_impl(ctx):
            fail("This Is My Failure Message")

        custom_rule = rule(implementation = custom_rule_impl)

        def depending_rule_impl(ctx):
            return []

        depending_rule = rule(
            implementation = depending_rule_impl,
            attrs = {"deps": attr.label_list()},
        )
        """);

    scratch.file(
        "test/BUILD",
        """
        load("//test:extension.bzl", "custom_rule", "depending_rule")

        custom_rule(name = "one")

        custom_rule(name = "two")

        depending_rule(
            name = "failures_are_direct_deps",
            deps = [
                ":one",
                ":two",
            ],
        )

        depending_rule(
            name = "failures_are_indirect_deps",
            deps = [":failures_are_direct_deps"],
        )
        """);

    ConfiguredTarget target = getConfiguredTarget("//test:failures_are_indirect_deps");
    AnalysisFailureInfo info =
        (AnalysisFailureInfo) target.get(AnalysisFailureInfo.STARLARK_CONSTRUCTOR.getKey());

    AnalysisFailure expectedOne =
        AnalysisFailure.create(
            Label.parseCanonicalUnchecked("//test:one"), "This Is My Failure Message");
    AnalysisFailure expectedTwo =
        AnalysisFailure.create(
            Label.parseCanonicalUnchecked("//test:two"), "This Is My Failure Message");

    assertThat(info.getCausesNestedSet().toList())
        .comparingElementsUsing(analysisFailureCorrespondence)
        .containsExactly(expectedOne, expectedTwo);
  }

  @Test
  public void starlarkAspectFailure() throws Exception {
    scratch.file(
        "test/extension.bzl",
        """
        def custom_aspect_impl(target, ctx):
            fail("This Is My Aspect Failure Message")

        custom_aspect = aspect(implementation = custom_aspect_impl, attr_aspects = ["deps"])

        def custom_rule_impl(ctx):
            return []

        custom_rule = rule(
            implementation = custom_rule_impl,
            attrs = {"deps": attr.label_list(aspects = [custom_aspect])},
        )
        """);
    scratch.file(
        "test/BUILD",
        """
        load("//test:extension.bzl", "custom_rule")

        custom_rule(name = "one")

        custom_rule(
            name = "two",
            deps = [":one"],
        )
        """);

    ConfiguredTarget target = getConfiguredTarget("//test:two");
    AnalysisFailureInfo info =
        (AnalysisFailureInfo) target.get(AnalysisFailureInfo.STARLARK_CONSTRUCTOR.getKey());
    AnalysisFailure expectedOne =
        AnalysisFailure.create(
            Label.parseCanonicalUnchecked("//test:one"), "This Is My Aspect Failure Message");

    assertThat(info.getCausesNestedSet().toList())
        .comparingElementsUsing(analysisFailureCorrespondence)
        .containsExactly(expectedOne);
  }

  @Test
  public void transitiveStarlarkAspectFailure() throws Exception {
    scratch.file(
        "test/extension.bzl",
        """
        def custom_aspect_impl(target, ctx):
            if hasattr(ctx.rule.attr, "kaboom") and ctx.rule.attr.kaboom:
                fail("This Is My Aspect Failure Message")
            return []

        custom_aspect = aspect(implementation = custom_aspect_impl, attr_aspects = ["deps"])

        def custom_rule_impl(ctx):
            return []

        custom_rule = rule(
            implementation = custom_rule_impl,
            attrs = {
                "deps": attr.label_list(aspects = [custom_aspect]),
                "kaboom": attr.bool(),
            },
        )
        """);
    scratch.file(
        "test/BUILD",
        """
        load("//test:extension.bzl", "custom_rule")

        custom_rule(
            name = "one",
            kaboom = True,
        )

        custom_rule(
            name = "two",
            deps = [":one"],
        )

        custom_rule(
            name = "three",
            deps = [":two"],
        )
        """);

    ConfiguredTarget target = getConfiguredTarget("//test:three");
    AnalysisFailureInfo info =
        (AnalysisFailureInfo) target.get(AnalysisFailureInfo.STARLARK_CONSTRUCTOR.getKey());
    AnalysisFailure expectedOne =
        AnalysisFailure.create(
            Label.parseCanonicalUnchecked("//test:one"), "This Is My Aspect Failure Message");

    assertThat(info.getCausesNestedSet().toList())
        .comparingElementsUsing(analysisFailureCorrespondence)
        .containsExactly(expectedOne);
  }

  @Test
  public void starlarkAspectAndRuleFailure_analysisFailureInfoPropagatesOnlyFromRuleFailure()
      throws Exception {
    scratch.file(
        "test/extension.bzl",
        """
        def custom_aspect_impl(target, ctx):
            fail("This Is My Aspect Failure Message")

        custom_aspect = aspect(implementation = custom_aspect_impl, attr_aspects = ["deps"])

        def custom_rule_impl(ctx):
            fail("This Is My Rule Failure Message")

        custom_rule = rule(
            implementation = custom_rule_impl,
            attrs = {"deps": attr.label_list(aspects = [custom_aspect])},
        )
        """);
    scratch.file(
        "test/BUILD",
        """
        load("//test:extension.bzl", "custom_rule")

        custom_rule(name = "one")

        custom_rule(
            name = "two",
            deps = [":one"],
        )
        """);

    ConfiguredTarget target = getConfiguredTarget("//test:two");
    AnalysisFailureInfo info =
        (AnalysisFailureInfo) target.get(AnalysisFailureInfo.STARLARK_CONSTRUCTOR.getKey());
    AnalysisFailure expectedRuleFailure =
        AnalysisFailure.create(
            Label.parseCanonicalUnchecked("//test:one"), "This Is My Rule Failure Message");

    assertThat(info.getCausesNestedSet().toList())
        .comparingElementsUsing(analysisFailureCorrespondence)
        .containsExactly(expectedRuleFailure);
  }

  @Test
  public void starlarkAspectWithAdvertisedProvidersFailure_analysisFailurePropagates()
      throws Exception {
    scratch.file(
        "test/extension.bzl",
        """
        MyInfo = provider()

        def custom_aspect_impl(target, ctx):
            fail("Aspect Failure")

        custom_aspect = aspect(implementation = custom_aspect_impl, provides = [MyInfo])

        def custom_rule_impl(ctx):
            pass

        custom_rule = rule(
            implementation = custom_rule_impl,
            attrs = {"deps": attr.label_list(aspects = [custom_aspect])},
        )
        """);
    scratch.file(
        "test/BUILD",
        """
        load("//test:extension.bzl", "custom_rule")

        custom_rule(name = "one")

        custom_rule(
            name = "two",
            deps = [":one"],
        )
        """);

    ConfiguredTarget target = getConfiguredTarget("//test:two");
    AnalysisFailureInfo info =
        (AnalysisFailureInfo) target.get(AnalysisFailureInfo.STARLARK_CONSTRUCTOR.getKey());
    AnalysisFailure expectedRuleFailure =
        AnalysisFailure.create(Label.parseCanonicalUnchecked("//test:one"), "Aspect Failure");

    assertThat(info.getCausesNestedSet().toList())
        .comparingElementsUsing(analysisFailureCorrespondence)
        .containsExactly(expectedRuleFailure);
  }
}
