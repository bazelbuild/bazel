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

package com.google.devtools.build.lib.analysis;

import static com.google.common.truth.Truth.assertWithMessage;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import com.google.testing.junit.testparameterinjector.TestParameters;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

/**
 * Tests for the how the visibility system works with respect to symbolic macros, i.e. the
 * Macro-Aware Visibility design.
 *
 * <p>This does *not* include tests of how the {@code visibility} attribute's value gets determined
 * and threaded through macros.
 */
@RunWith(TestParameterInjector.class)
public final class MacroVisibilityTest extends BuildViewTestCase {

  @Before
  public void setUp() throws Exception {
    setBuildLanguageOptions("--experimental_enable_first_class_macros");
  }

  @Override
  protected String getDefaultVisibility() {
    // We're testing visibility. Avoid having to litter our test cases with `visibility=` attribute
    // declarations, by using the same behavior seen in production.
    return "private";
  }

  /**
   * Returns a substring expected to be in the error event reported for a visibility violation from
   * {@code consumer} (expressed as a label string) to {@code dependency} (expressed as a label
   * string, or for {@code alias} targets, the string returned by {@link
   * AliasProvider#describeTargetWithAliases}).
   */
  private String visibilityErrorMessage(
      String consumer, String dependency, boolean dependencyIsAlias) {
    if (!dependencyIsAlias) {
      dependency = String.format("target '%s'", dependency);
    }
    return String.format(
        "Visibility error:\n%s is not visible from\ntarget '%s'", dependency, consumer);
  }

  /**
   * Convenience wrapper for {@link #visibilityErrorMessage} for when {@code dependency} is not an
   * {@code alias}.
   */
  private String visibilityErrorMessage(String consumer, String dependency) {
    return visibilityErrorMessage(consumer, dependency, /* dependencyIsAlias= */ false);
  }

  /**
   * Requests the evaluation of the configured target identified by the label {@code consumer}, and
   * asserts that the target analyzes successfully and has a dependency on the target identified by
   * the label {@code dependency}.
   *
   * <p>Does not work when {@code dependency} is an {@code alias} target.
   */
  private void assertVisibilityPermits(String consumer, String dependency) throws Exception {
    ConfiguredTarget consumerTarget = getConfiguredTarget(consumer);
    // If the dependency is in fact *not* visible, consumerTarget will be null. So we assert on the
    // absence of the visibility error event first, to deliver a better error message than just
    // saying consumerTarget was null. (We can't just attach a nicer message to the null check
    // because consumerTarget could be null due to other unexpected errors.)
    //
    // This nicer handling is moot if the failFastHandler is present, in which case we'll see the
    // visibility error event in a traceback.
    assertDoesNotContainEvent(visibilityErrorMessage(consumer, dependency));
    assertWithMessage(
            String.format(
                "%s had an error (other than a visibility violation on %s)", consumer, dependency))
        .that(consumerTarget)
        .isNotNull();
    assertWithMessage(String.format("%s does not have a dependency on %s", consumer, dependency))
        .that(getDirectPrerequisite(consumerTarget, dependency))
        .isNotNull();
  }

  /**
   * Requests the evaluation of the configured target identified by the label {@code consumer}, and
   * asserts that the target is in error due to a visibility violation on the target identified by
   * the label {@code dependency}.
   *
   * <p>Removes the {@code failFastHandler}.
   */
  private void assertVisibilityDisallows(
      String consumer, String dependency, boolean dependencyIsAlias) throws Exception {
    reporter.removeHandler(failFastHandler);
    ConfiguredTarget consumerTarget = getConfiguredTarget(consumer);
    assertWithMessage(
            String.format(
                "%s unexpectedly did not have an error (no visibility violation due to %s)",
                consumer, dependency))
        .that(consumerTarget)
        .isNull();
    assertContainsEvent(visibilityErrorMessage(consumer, dependency, dependencyIsAlias));
  }

  /**
   * Convenience wrapper for {@link #assertVisibilityDisallows} for when {@code dependency} is not
   * an {@code alias}.
   */
  private void assertVisibilityDisallows(String consumer, String dependency) throws Exception {
    assertVisibilityDisallows(consumer, dependency, /* dependencyIsAlias= */ false);
  }

  /**
   * Creates definition of {@code //rules:simple_rule.bzl%simple_rule}, a rule that has a label
   * attribute {@code dep} and implicit output {@code <NAME>.bin}.
   */
  private void defineSimpleRule() throws Exception {
    scratch.file("rules/BUILD");
    scratch.file(
        "rules/simple_rule.bzl",
        """
        def _impl(ctx):
            ctx.actions.write(ctx.outputs.out, "")

        simple_rule = rule(
            implementation = _impl,
            attrs = {"dep": attr.label(mandatory=False, allow_files=True)},
            outputs = {"out": "%{name}.bin"},
        )
        """);
  }

  /**
   * Creates a package {@code //common} defining several {@code simple_rule} targets with the given
   * names and visibility declarations.
   *
   * <p>The argument list must have even length, and contains pairs of (unquoted) target names and
   * Starlark expression strings that evaluate to visibility lists (e.g. pass the string {@code
   * "[]"} for no visibility).
   */
  private void defineCommonPackageWithSimpleTargets(String... targetNamesAndVisibilities)
      throws Exception {
    Preconditions.checkArgument(targetNamesAndVisibilities.length % 2 == 0);
    StringBuilder s = new StringBuilder();
    s.append(
        """
        load("//rules:simple_rule.bzl", "simple_rule")
        """);
    for (int i = 0; i < targetNamesAndVisibilities.length; i += 2) {
      s.append(
          String.format(
              """
              simple_rule(
                  name = "%s",
                  visibility = %s,
              )
              """,
              targetNamesAndVisibilities[i], targetNamesAndVisibilities[i + 1]));
    }
    scratch.file("common/BUILD", s.toString());
  }

  @Test
  public void buildFileAccessToMacroTargetsIsControlled() throws Exception {
    defineSimpleRule();
    scratch.file("lib/BUILD");
    scratch.file(
        "lib/macro.bzl",
        """
        load("//rules:simple_rule.bzl", "simple_rule")

        def _impl(name, visibility):
            simple_rule(
                name = name + "_exported",
                visibility = ["//pkg:__pkg__"],
            )
            simple_rule(
                name = name + "_internal",
            )
            native.exports_files([name + "_exported_input"], visibility = ["//pkg:__pkg__"])
            native.exports_files([name + "_internal_input"], visibility = ["//visibility:private"])

        my_macro = macro(implementation=_impl)
        """);
    scratch.file(
        "pkg/BUILD",
        """
        load("//rules:simple_rule.bzl", "simple_rule")
        load("//lib:macro.bzl", "my_macro")

        my_macro(name = "foo")

        simple_rule(name = "consumes_exported_ruletarget", dep = ":foo_exported")
        simple_rule(name = "consumes_exported_output", dep = ":foo_exported.bin")
        simple_rule(name = "consumes_exported_input", dep = ":foo_exported_input")
        simple_rule(name = "consumes_internal_ruletarget", dep = ":foo_internal")
        simple_rule(name = "consumes_internal_output", dep = ":foo_internal.bin")
        simple_rule(name = "consumes_internal_input", dep = ":foo_internal_input")
        """);

    assertVisibilityPermits("//pkg:consumes_exported_ruletarget", "//pkg:foo_exported");
    assertVisibilityPermits("//pkg:consumes_exported_output", "//pkg:foo_exported.bin");
    assertVisibilityPermits("//pkg:consumes_exported_input", "//pkg:foo_exported_input");
    assertVisibilityDisallows("//pkg:consumes_internal_ruletarget", "//pkg:foo_internal");
    assertVisibilityDisallows("//pkg:consumes_internal_output", "//pkg:foo_internal.bin");
    assertVisibilityDisallows("//pkg:consumes_internal_input", "//pkg:foo_internal_input");
  }

  @Test
  public void macroAccessToBuildFileTargetsIsControlled() throws Exception {
    defineSimpleRule();
    scratch.file("lib/BUILD");
    scratch.file(
        "lib/macro.bzl",
        """
        load("//rules:simple_rule.bzl", "simple_rule")

        def _impl(name, visibility):
            simple_rule(name = name + "_consumes_exported_ruletarget", dep = "//pkg:exported")
            simple_rule(name = name + "_consumes_exported_output", dep = "//pkg:exported.bin")
            simple_rule(name = name + "_consumes_exported_input", dep = "//pkg:exported_input")
            simple_rule(name = name + "_consumes_internal_ruletarget", dep = "//pkg:internal")
            simple_rule(name = name + "_consumes_internal_output", dep = "//pkg:internal.bin")
            simple_rule(name = name + "_consumes_internal_input", dep = "//pkg:internal_input")

        my_macro = macro(implementation=_impl)
        """);
    scratch.file(
        "pkg/BUILD",
        """
        load("//rules:simple_rule.bzl", "simple_rule")
        load("//lib:macro.bzl", "my_macro")

        my_macro(name = "foo")

        simple_rule(
            name = "exported",
            visibility = ["//lib:__pkg__"],
        )
        simple_rule(
            name = "internal",
        )
        exports_files(["exported_input"], visibility = ["//lib:__pkg__"])
        exports_files(["internal_input"], visibility = ["//visibility:private"])
        """);

    assertVisibilityPermits("//pkg:foo_consumes_exported_ruletarget", "//pkg:exported");
    assertVisibilityPermits("//pkg:foo_consumes_exported_output", "//pkg:exported.bin");
    assertVisibilityPermits("//pkg:foo_consumes_exported_input", "//pkg:exported_input");
    assertVisibilityDisallows("//pkg:foo_consumes_internal_ruletarget", "//pkg:internal");
    assertVisibilityDisallows("//pkg:foo_consumes_internal_output", "//pkg:internal.bin");
    assertVisibilityDisallows("//pkg:foo_consumes_internal_input", "//pkg:internal_input");
  }

  @Test
  public void macroAccessToOtherMacrosInSameBuildFileIsControlled() throws Exception {
    defineSimpleRule();
    scratch.file("producer/BUILD");
    scratch.file(
        "producer/macro.bzl",
        """
        load("//rules:simple_rule.bzl", "simple_rule")

        def _impl(name, visibility):
            simple_rule(
                name = name + "_internal",
            )
            simple_rule(
                name = name + "_visible_to_pkg",
                visibility = ["//pkg:__pkg__"],
            )
            simple_rule(
                name = name + "_visible_to_consumer",
                visibility = ["//consumer:__pkg__"],
            )

        producer_macro = macro(implementation=_impl)
        """);
    scratch.file("consumer/BUILD");
    scratch.file(
        "consumer/macro.bzl",
        """
        load("//rules:simple_rule.bzl", "simple_rule")

        def _impl(name, visibility):
            simple_rule(
                name = name + "_consumes_internal",
                dep = "//pkg:foo_internal",
            )
            simple_rule(
                name = name + "_consumes_visible_to_pkg",
                dep = "//pkg:foo_visible_to_pkg",
            )
            simple_rule(
                name = name + "_consumes_visible_to_consumer",
                dep = "//pkg:foo_visible_to_consumer",
            )

        consumer_macro = macro(implementation=_impl)
        """);
    scratch.file(
        "pkg/BUILD",
        """
        load("//producer:macro.bzl", "producer_macro")
        load("//consumer:macro.bzl", "consumer_macro")

        producer_macro(name = "foo")
        consumer_macro(name = "bar")
        """);

    assertVisibilityDisallows("//pkg:bar_consumes_internal", "//pkg:foo_internal");
    assertVisibilityDisallows("//pkg:bar_consumes_visible_to_pkg", "//pkg:foo_visible_to_pkg");
    assertVisibilityPermits(
        "//pkg:bar_consumes_visible_to_consumer", "//pkg:foo_visible_to_consumer");
  }

  @Test
  public void siblingsInSameMacroCanSeeEachOther() throws Exception {
    defineSimpleRule();
    scratch.file("lib/BUILD");
    scratch.file(
        "lib/macro.bzl",
        """
        load("//rules:simple_rule.bzl", "simple_rule")

        def _impl(name, visibility):
            simple_rule(
                name = name + "_ruletarget",
            )
            simple_rule(
                name = name + "_consumes_ruletarget",
                dep = name + "_ruletarget",
            )
            native.exports_files([name + "_input"], visibility = ["//visibility:private"])
            simple_rule(
                name = name + "_consumes_input",
                dep = name + "_input",
            )

        my_macro = macro(implementation=_impl)
        """);
    scratch.file(
        "pkg/BUILD",
        """
        load("//rules:simple_rule.bzl", "simple_rule")
        load("//lib:macro.bzl", "my_macro")

        my_macro(name = "foo")

        simple_rule(
            name = "control",
            dep = ":foo_ruletarget",
        )
        """);

    assertVisibilityPermits("//pkg:foo_consumes_ruletarget", "//pkg:foo_ruletarget");
    assertVisibilityPermits("//pkg:foo_consumes_input", "//pkg:foo_input");
    assertVisibilityDisallows("//pkg:control", "//pkg:foo_ruletarget");
  }

  @Test
  public void buildFileTargetIsVisibleToTargetsDefinedByMacroFromSamePackage() throws Exception {
    // This test is needed because the declaration location of a target is only materialized in its
    // visibility attribute when the target is declared inside a symbolic macro. We need to
    // confirm that we're still allowing the edge from a target declared by a macro whose code is in
    // //pkg, to a dependency declared in //pkg:BUILD, despite the dependency not materializing
    // //pkg in its own visibility.
    defineSimpleRule();
    scratch.file(
        "pkg/macro.bzl",
        """
        load("//rules:simple_rule.bzl", "simple_rule")

        def _impl(name, visibility):
            simple_rule(name = name + "_macro_target", dep = "//pkg:build_target")

        my_macro = macro(implementation=_impl)
        """);
    scratch.file(
        "other_pkg/BUILD",
        """
        load("//pkg:macro.bzl", "my_macro")

        my_macro(name = "foo")
        """);
    scratch.file(
        "pkg/BUILD",
        """
        load("//rules:simple_rule.bzl", "simple_rule")

        simple_rule(name = "build_target")
        """);

    // Allowed because my_macro's code was defined in //pkg, even though it was instantiated in
    // //other_pkg.
    assertVisibilityPermits("//other_pkg:foo_macro_target", "//pkg:build_target");
  }

  @Test
  public void macroDefinitionLocationIsDeterminedByExportAssignment() throws Exception {
    // Consider a target declared by a helper func in //A, which is called from an impl func in //B,
    // which is used in a macro() definition in //C, which is loaded and exported by //D, and called
    // by a legacy macro defined in //E, itself called in a BUILD file in //F. Then the declaration
    // location of the target is considered to be //D, where the export occurs, rather than any of
    // those other packages.
    //
    // This test defines six targets in //pkg, visible to each of //A through //F. The helper
    // function in //A declares six targets that try to consume each of the respective targets from
    // //pkg. Only the one that consumes the target visible to //D is allowed to succeed.
    defineSimpleRule();
    scratch.file("A/BUILD");
    scratch.file(
        "A/helper.bzl",
        """
        load("//rules:simple_rule.bzl", "simple_rule")
        def helper(name):
            for c in ["A", "B", "C", "D", "E", "F"]:
              simple_rule(
                  name = name + "_wants_vis_to_%s" % c,
                  dep = "//pkg:vis_to_%s" % c,
              )
        """);
    scratch.file("B/BUILD");
    scratch.file(
        "B/impl.bzl",
        """
        load("//A:helper.bzl", "helper")
        def impl(name, visibility):
            helper(name)
        """);
    scratch.file("C/BUILD");
    scratch.file(
        "C/metamacro.bzl",
        """
        load("//B:impl.bzl", "impl")
        def metamacro():
            return macro(implementation = impl)
        """);
    scratch.file("D/BUILD");
    scratch.file(
        "D/my_macro.bzl",
        """
        load("//C:metamacro.bzl", "metamacro")
        my_macro = metamacro()
        """);
    scratch.file("E/BUILD");
    scratch.file(
        "E/legacy_macro.bzl",
        """
        load("//D:my_macro.bzl", "my_macro")
        def legacy_macro(name):
            my_macro(name = name)
        """);
    scratch.file(
        "F/BUILD",
        """
        load("//E:legacy_macro.bzl", "legacy_macro")

        legacy_macro(name = "consumer")
        """);
    scratch.file(
        "pkg/BUILD",
        """
        load("//rules:simple_rule.bzl", "simple_rule")

        [
            simple_rule(
                name = "vis_to_%s" % c,
                visibility = ["//%s:__pkg__" % c])
            for c in ["A", "B", "C", "D", "E", "F"]
        ]
        """);

    assertVisibilityDisallows("//F:consumer_wants_vis_to_A", "//pkg:vis_to_A");
    assertVisibilityDisallows("//F:consumer_wants_vis_to_B", "//pkg:vis_to_B");
    assertVisibilityDisallows("//F:consumer_wants_vis_to_C", "//pkg:vis_to_C");
    assertVisibilityPermits("//F:consumer_wants_vis_to_D", "//pkg:vis_to_D");
    assertVisibilityDisallows("//F:consumer_wants_vis_to_E", "//pkg:vis_to_E");
    assertVisibilityDisallows("//F:consumer_wants_vis_to_F", "//pkg:vis_to_F");
  }

  @Test
  public void locationOfTargetDeclarationIsInnermostMacro() throws Exception {
    // The //common package has a target visible to an outer macro and a target visible to an inner
    // macro. In //pkg, the outer macro is instantiated as "foo" and the inner as "foo_inner". We
    // check each combination of foo and foo_inner trying to access the two common targets.
    defineSimpleRule();
    defineCommonPackageWithSimpleTargets(
        "vis_to_inner", "['//inner:__pkg__']", "vis_to_outer", "['//outer:__pkg__']");
    scratch.file("inner/BUILD");
    scratch.file(
        "inner/macro.bzl",
        """
        load("//rules:simple_rule.bzl", "simple_rule")

        def _impl(name, visibility):
            simple_rule(
                name = name + "_wants_vis_to_inner",
                dep = "//common:vis_to_inner",
            )
            simple_rule(
                name = name + "_wants_vis_to_outer",
                dep = "//common:vis_to_outer",
            )

        inner_macro = macro(implementation=_impl)
        """);
    scratch.file("outer/BUILD");
    scratch.file(
        "outer/macro.bzl",
        """
        load("//rules:simple_rule.bzl", "simple_rule")
        load("//inner:macro.bzl", "inner_macro")

        def _impl(name, visibility):
            inner_macro(name = name + "_inner")
            simple_rule(
                name = name + "_wants_vis_to_inner",
                dep = "//common:vis_to_inner",
            )
            simple_rule(
                name = name + "_wants_vis_to_outer",
                dep = "//common:vis_to_outer",
            )

        outer_macro = macro(implementation=_impl)
        """);
    scratch.file(
        "pkg/BUILD",
        """
        load("//outer:macro.bzl", "outer_macro")

        outer_macro(name = "foo")
        """);

    assertVisibilityPermits("//pkg:foo_inner_wants_vis_to_inner", "//common:vis_to_inner");
    assertVisibilityDisallows("//pkg:foo_inner_wants_vis_to_outer", "//common:vis_to_outer");
    assertVisibilityDisallows("//pkg:foo_wants_vis_to_inner", "//common:vis_to_inner");
    assertVisibilityPermits("//pkg:foo_wants_vis_to_outer", "//common:vis_to_outer");
  }

  // This is the simplest version of the caller delegating visibility privileges to a macro.
  // Subsequent test cases don't bother to test opting out of select() promotion by setting
  // configurable=False.
  @Test
  @TestParameters({"{depIsConfigurable: false}", "{depIsConfigurable: true}"})
  public void buildFileCanDelegateVisibilityPrivilegesToMacro(boolean depIsConfigurable)
      throws Exception {
    defineSimpleRule();
    defineCommonPackageWithSimpleTargets("vis_to_pkg", "['//pkg:__pkg__']");
    scratch.file("macros/BUILD");
    scratch.file(
        "macros/simple_rule_wrapper.bzl",
        String.format(
            """
            load("//rules:simple_rule.bzl", "simple_rule")

            def _impl(name, visibility, dep):
                simple_rule(
                    name = name,
                    dep = dep,
                )

            simple_rule_wrapper = macro(
                implementation = _impl,
                attrs = {"dep": attr.label(configurable=%s)},
            )
            """,
            depIsConfigurable ? "True" : "False"));
    scratch.file(
        "pkg/BUILD",
        """
        load("//macros:simple_rule_wrapper.bzl", "simple_rule_wrapper")

        simple_rule_wrapper(
            name = "foo",
            dep = "//common:vis_to_pkg",
        )
        """);

    // Allowed, even though //pkg:foo's declaration location is not //pkg, because the macro that
    // //pkg:foo is declared in is itself declared in //pkg and passes vis_to_pkg in as an
    // attribute.
    assertVisibilityPermits("//pkg:foo", "//common:vis_to_pkg");
  }

  /**
   * Defines a macro {@code name} whose body calls symbol {@code wraps} to instantiate its main
   * target or submacro, passing along {@code attrExpr}.
   *
   * <p>The new macro is declared in {@code //<name>:macro.bzl}. It takes an optional {@code dep}
   * attribute.
   *
   * @param name the macro being introduced, e.g. "my_macro", which would be loaded from {@code
   *     //my_macro:macro.bzl}
   * @param wraps the rule or macro to be called by this macro, in {@code bzlLabel%symbol} form,
   *     e.g. "//some_pkg:defs.bzl%my_rule"
   * @param attrExpr an attribute argument expression for the call site of {@code wraps}, e.g.
   *     "some_attr = dep"
   */
  private void defineWrappingMacro(String name, String wraps, String attrExpr) throws Exception {
    int i = wraps.indexOf("%");
    Preconditions.checkArgument(i != -1);
    String bzlToLoad = wraps.substring(0, i);
    String symbolName = wraps.substring(i + 1);

    scratch.file(String.format("%s/BUILD", name));
    scratch.file(
        String.format("%s/macro.bzl", name),
        String.format(
            """
            load("%2$s", "%3$s")

            def _impl(name, visibility, dep):
                %3$s(
                    name = name,
                    %4$s,
                )

            %1$s = macro(
                implementation = _impl,
                attrs = {"dep": attr.label(mandatory=False)},
            )
            """,
            name, bzlToLoad, symbolName, attrExpr));
  }

  /**
   * Defines a macro {@code name} whose body calls {@code wraps} to instantiate its main target or
   * submacro, forwarding along the {@code dep} attribute (if supplied) unchanged.
   *
   * <p>The new macro is declared in {@code //<name>:macro.bzl}.
   *
   * @param name the macro being introduced, e.g. "my_macro", which would be loaded from {@code
   *     //my_macro:macro.bzl}
   * @param wraps the rule or macro to be called by this macro, in {@code bzlLabel%symbol} form,
   *     e.g. "//some_pkg:defs.bzl%my_rule"
   */
  private void defineWrappingMacroWithSameDep(String name, String wraps) throws Exception {
    defineWrappingMacro(name, wraps, "dep = dep");
  }

  /**
   * Defines a macro {@code name} whose body calls {@code wraps} to instantiate its main target or
   * submacro, passing {@code hardcodedDep} as the value of the {@code dep} attribute.
   *
   * <p>The new macro is declared in {@code //<name>:macro.bzl}. It takes an optional {@code dep}
   * attribute itself, which gets ignored.
   *
   * @param name the macro being introduced, e.g. "my_macro", which would be loaded from {@code
   *     //my_macro:macro.bzl}
   * @param wraps the rule or macro to be called by this macro, in {@code bzlLabel%symbol} form,
   *     e.g. "//some_pkg:defs.bzl%my_rule"
   * @param hardcodedDep a label string to pass as {@code dep = <hardcodedDep>} at the call site of
   *     {@code wraps}, e.g. "//some_pkg:my_target"
   */
  private void defineWrappingMacroWithHardcodedDep(String name, String wraps, String hardcodedDep)
      throws Exception {
    defineWrappingMacro(name, wraps, String.format("dep = \"%s\"", hardcodedDep));
  }

  @Test
  public void outerMacroCanDelegateVisibilityPrivilegesToSubmacro() throws Exception {
    defineSimpleRule();
    defineCommonPackageWithSimpleTargets("vis_to_outer", "['//outer:__pkg__']");
    defineWrappingMacroWithSameDep("inner", "//rules:simple_rule.bzl%simple_rule");
    defineWrappingMacroWithHardcodedDep(
        "outer", "//inner:macro.bzl%inner", "//common:vis_to_outer");
    scratch.file(
        "pkg/BUILD",
        """
        load("//outer:macro.bzl", "outer")

        outer(name = "foo")
        """);

    // Allowed by same reasoning as buildFileCanDelegateVisibilityPrivilegesToMacro(), but with the
    // privilege originating from an outer macro rather than from the BUILD file.
    assertVisibilityPermits("//pkg:foo", "//common:vis_to_outer");
  }

  @Test
  public void delegationCanBeTransitiveThroughAnIntermediaryMacro() throws Exception {
    defineSimpleRule();
    defineCommonPackageWithSimpleTargets("vis_to_pkg", "['//pkg:__pkg__']");
    defineWrappingMacroWithSameDep("inner", "//rules:simple_rule.bzl%simple_rule");
    defineWrappingMacroWithSameDep("outer", "//inner:macro.bzl%inner");
    scratch.file(
        "pkg/BUILD",
        """
        load("//outer:macro.bzl", "outer")

        outer(
            name = "foo",
            dep = "//common:vis_to_pkg",
        )
        """);

    // Allowed because the package delegates its privilege to the outer macro, which in turn
    // delegates its privilege to the inner macro.
    assertVisibilityPermits("//pkg:foo", "//common:vis_to_pkg");
  }

  @Test
  public void callerPermissionIsCheckedEvenIfCalleeHasItsOwnPermission() throws Exception {
    defineSimpleRule();
    defineCommonPackageWithSimpleTargets("vis_to_macro", "['//macros:__pkg__']");
    defineWrappingMacroWithSameDep("my_macro", "//rules:simple_rule.bzl%simple_rule");
    scratch.file(
        "pkg/BUILD",
        """
        load("//my_macro:macro.bzl", "my_macro")

        my_macro(
            name = "foo",
            dep = "//common:vis_to_macro",
        )
        """);

    // Even though the macro has visibility on the dep, we still fail because the caller tried to
    // pass it in and the caller has no such permission.
    assertVisibilityDisallows("//pkg:foo", "//common:vis_to_macro");
  }

  @Test
  public void noDelegationIfCallerCannotSeeDep() throws Exception {
    defineSimpleRule();
    defineCommonPackageWithSimpleTargets("not_visible", "[]");
    defineWrappingMacroWithSameDep("my_macro", "//rules:simple_rule.bzl%simple_rule");
    scratch.file(
        "pkg/BUILD",
        """
        load("//my_macro:macro.bzl", "my_macro")

        my_macro(
            name = "foo",
            dep = "//common:not_visible",
        )
        """);

    // Passed in from caller, but caller had no permission to delegate.
    assertVisibilityDisallows("//pkg:foo", "//common:not_visible");
  }

  @Test
  public void noDelegationIfCallerDoesNotPassInTarget() throws Exception {
    defineSimpleRule();
    defineCommonPackageWithSimpleTargets(
        // Read: "v2[P|M]" -> "visible to [package|macro]"
        "v2P_hardcoded",
        "['//pkg:__pkg__']",
        "v2P_implicitdep",
        "['//pkg:__pkg__']",
        "v2M_hardcoded",
        "['//macros:__pkg__']",
        "v2M_implicitdep",
        "['//macros:__pkg__']");
    scratch.file("macros/BUILD");
    scratch.file(
        "macros/my_macro.bzl",
        """
        load("//rules:simple_rule.bzl", "simple_rule")

        def _impl(name, visibility, _v2P_implicitdep, _v2M_implicitdep):
            simple_rule(
                name = name + "_consumes_v2P_hardcoded",
                dep = "//common:v2P_hardcoded",
            )
            simple_rule(
                name = name + "_consumes_v2P_implicitdep",
                dep = _v2P_implicitdep,
            )
            simple_rule(
                name = name + "_consumes_v2M_hardcoded",
                dep = "//common:v2M_hardcoded",
            )
            simple_rule(
                name = name + "_consumes_v2M_implicitdep",
                dep = _v2M_implicitdep,
            )

        my_macro = macro(
            implementation = _impl,
            attrs = {
                "_v2P_implicitdep": attr.label(
                    default="//common:v2P_implicitdep"),
                "_v2M_implicitdep": attr.label(
                    default="//common:v2M_implicitdep"),
            },
        )
        """);
    scratch.file(
        "pkg/BUILD",
        """
        load("//macros:my_macro.bzl", "my_macro")

        my_macro(name = "foo")
        """);

    // Macro can't see it. Caller didn't pass it in, so caller's permission is not delegated to the
    // macro.
    assertVisibilityDisallows("//pkg:foo_consumes_v2P_hardcoded", "//common:v2P_hardcoded");
    // Having an implicit dep doesn't count as the caller passing it in.
    assertVisibilityDisallows("//pkg:foo_consumes_v2P_implicitdep", "//common:v2P_implicitdep");

    // Macro can see it. Caller didn't pass it in, so it's fine that the caller can't see it.
    assertVisibilityPermits("//pkg:foo_consumes_v2M_hardcoded", "//common:v2M_hardcoded");
    // Having an implicit dep doesn't count as the caller passing it in.
    assertVisibilityPermits("//pkg:foo_consumes_v2M_implicitdep", "//common:v2M_implicitdep");
  }

  @Test
  public void delegationCannotSkipOverAGap() throws Exception {
    // We have a chain of macro calls A -> B -> C -> D, with a target in D consuming a dep passed in
    // by A. But the dep is not actually threaded all the way through; C hardcodes it, instead of
    // taking it as an argument from B.
    //
    // In this case, when validating the dep's visibility w.r.t. D, we don't consider A at all, even
    // though both A and D participate in passing or receiving this dep, respectively. In
    // particular, 1) we don't get the benefit of A's privileges passed on to us, and 2) if A is
    // violating the dep's visibility, we don't actually enforce that check (because macro
    // visibility violations are only discovered when actually relying on them for visibility
    // delegation).

    defineSimpleRule();
    defineCommonPackageWithSimpleTargets(
        "some_dep", "['//A:__pkg__']", "irrelevant", "['//B:__pkg__']");
    // D calls simple_rule with args unchanged.
    defineWrappingMacroWithSameDep("D", "//rules:simple_rule.bzl%simple_rule");
    // C calls D with hardcoded dep = "//common:some_dep".
    defineWrappingMacroWithHardcodedDep("C", "//D:macro.bzl%D", "//common:some_dep");
    // B calls C with a different hardcoded dep = "//common:irrelevant"
    defineWrappingMacroWithHardcodedDep("B", "//C:macro.bzl%C", "//common:irrelevant");
    // A calls B with hardcoded dep = "//common:some_dep".
    defineWrappingMacroWithHardcodedDep("A", "//B:macro.bzl%B", "//common:some_dep");
    scratch.file(
        "pkg/BUILD",
        """
        load("//A:macro.bzl", "A")

        A(name = "foo")
        """);

    // No privileges delegated through gap.
    assertVisibilityDisallows("//pkg:foo", "//common:some_dep");

    // Let's retry it with the visibility rewritten to allow C, but not A.
    scratch.deleteFile("common/BUILD");
    defineCommonPackageWithSimpleTargets("some_dep", "['//C:__pkg__']");
    invalidatePackages(); // to pick up the changes to common/BUILD
    eventCollector.clear(); // to avoid the assertion matching events emitted above

    // Now D has its own privileges, and we *don't* discover that A is illegally referring to it.
    assertVisibilityPermits("//pkg:foo", "//common:some_dep");
  }

  @Test
  public void delegationDoesNotFollowAliases() throws Exception {
    // The whole visibility system considers alias targets to have distinct permissions from the
    // underlying target they refer to. So check that these are separate entities for the purpose
    // of a caller delegating permission to a macro.
    defineSimpleRule();
    scratch.file(
        "common/BUILD",
        """
        load("//rules:simple_rule.bzl", "simple_rule")

        simple_rule(
            name = "vis_to_pkg",
            visibility = ["//pkg:__pkg__"],
        )

        alias(
            name = "aliasof_vis_to_pkg",
            actual = ":vis_to_pkg",
        )
        """);
    defineWrappingMacroWithHardcodedDep(
        "my_macro", "//rules:simple_rule.bzl%simple_rule", "//common:vis_to_pkg");
    scratch.file(
        "pkg/BUILD",
        """
        load("//my_macro:macro.bzl", "my_macro")

        my_macro(
            name = "foo",
            dep = "//common:aliasof_vis_to_pkg",
        )
        """);

    // Delegation doesn't work, needed the actual but passed the alias.
    assertVisibilityDisallows("//pkg:foo", "//common:vis_to_pkg");

    // Try it the other way around.
    scratch.deleteFile("my_macro/BUILD");
    scratch.deleteFile("my_macro/macro.bzl");
    defineWrappingMacroWithHardcodedDep(
        "my_macro", "//rules:simple_rule.bzl%simple_rule", "//common:aliasof_vis_to_pkg");
    scratch.overwriteFile(
        "pkg/BUILD",
        """
        load("//my_macro:macro.bzl", "my_macro")

        my_macro(
            name = "foo",
            dep = "//common:vis_to_pkg",
        )
        """);
    invalidatePackages(); // to pick up the changes to common/BUILD
    eventCollector.clear(); // to avoid the assertion matching events emitted above

    // Delegation doesn't work, needed the alias but passed the actual.
    assertVisibilityDisallows(
        "//pkg:foo",
        "alias '//common:aliasof_vis_to_pkg' referring to target '//common:vis_to_pkg'",
        /* dependencyIsAlias= */ true);
  }

  @Test
  public void packageDefaultVisibilityDoesNotPropagateToInsideMacro() throws Exception {
    defineSimpleRule();
    defineWrappingMacroWithSameDep("my_macro", "//rules:simple_rule.bzl%simple_rule");
    defineWrappingMacroWithSameDep("inner", "//rules:simple_rule.bzl%simple_rule");
    defineWrappingMacroWithSameDep("outer", "//inner:macro.bzl%inner");
    scratch.file(
        "lib/BUILD",
        """
        load("//rules:simple_rule.bzl", "simple_rule")
        load("//my_macro:macro.bzl", "my_macro")
        load("//outer:macro.bzl", "outer")

        package(default_visibility=["//pkg:__pkg__"])

        simple_rule(name = "foo")

        my_macro(name = "bar")

        outer(name = "baz")
        """);
    scratch.file(
        "pkg/BUILD",
        """
        load("//rules:simple_rule.bzl", "simple_rule")

        simple_rule(
            name = "consumes_foo",
            dep = "//lib:foo",
        )

        simple_rule(
            name = "consumes_bar",
            dep = "//lib:bar",
        )

        simple_rule(
            name = "consumes_baz",
            dep = "//lib:baz",
        )
        """);

    // Not in a macro, default_visibility applies.
    assertVisibilityPermits("//pkg:consumes_foo", "//lib:foo");
    // In a macro, default_visibility does not propagate.
    assertVisibilityDisallows("//pkg:consumes_bar", "//lib:bar");
    // In more than one macro, default_visibility still does not propagate.
    assertVisibilityDisallows("//pkg:consumes_baz", "//lib:baz");
  }

  /*
   * TODO: #19922 - Tests cases to add:
   *
   * ---- Propagating target usages from parent macro to child ----
   *
   * - When checking a parent macro to see if the label occurs, only normal dep attributes are
   *   considered, not nodep labels like visibility (that's a funny test case).
   *
   * ---- Implicit deps ----
   *
   * - If a rule's implicit dep is defined in a macro, the check of the rule's def loc against the
   *   dep takes into account the macro's location. In particular, a rule can see implicit deps
   *   defined by macros whose defs are in the same package as the rule's def (when the macro is
   *   called from a totally different package), even if the dep is otherwise private.
   *
   * - If a macro has an implicit dep, that dep's visibility is checked against the macro def's
   *   location, not its instantiation location. So pass even when the instance doesn't have
   *   permission. And fail if the macro def location doesn't have permission, even if the instance
   *   does have permission (either because of its declaration location or because its caller
   *   received permission delegated by its own caller via an explicit attr value).
   *
   * ---- Visibility attr usage ----
   *
   * - Visibility attr is passed and contains the call site's package.
   *
   * - Exporting via visibility = visibility works, including transitively.
   *
   * - Passing visibility to a macro does not force that visibility upon the macro's internal
   *   targets that don't declare a visibility.
   *
   * - Can compose public and private visibilities with other visibilities via concatenation.
   *   Visibility attr is normalized. (Unclear whether to apply normalization to targets defined
   *   outside symbolic macros.)
   *
   * ---- default_visibility ----
   *
   * - default_visibility affects the visibility of a top-level macro that does not set
   *   visibility=..., and does not affect a top-level macro that does set visibility=...
   *
   * ---- Accounting for CommonPrerequisiteValidator#isSameLogicalPackage() ----
   *
   * - Targets of a macro defined in //javatests/foo can see targets defined in //java/foo, even
   *   when the macro is instantiated in //bar. (Rationale: We just look at the consumer's location,
   *   irrespective of whether that location comes from being in a macro or a BUILD file.)
   *
   * - Targets in //javatest/foo cannot automatically see unexported targets created by a symbolic
   *   macro defined in //java/foo, even when that macro is also instantiated in //java/foo.
   *   (Rationale: Additional visibilities granted by same-logical-package apply only at the
   *   top-level, similar to default_visibility. Macros can still propagate the additional
   *   visibility down to exported targets.)
   *
   * - A target exported from a macro instantiated in //java/foo can be seen in //javatests/foo,
   *   even when the macro is defined in //bar. (Rationale: Exported targets behave similarly to
   *   declaring the target in the BUILD file.)
   */

  // TODO: #19922 - Consider any other edge cases regarding exotic dependencies and other
  // PrerequisiteValidator code paths, e.g. implicit deps, toolchain resolution, etc.

  // TODO: #19922 - Consider correctness for alias targets. Probably just means auditing anywhere we
  // use a PackageIdentifier in visibility logic without going through
  // AliasProvider.getDependencyLabel().
}
