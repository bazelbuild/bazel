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

import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for the how the visibility system works with respect to symbolic macros. */
@RunWith(JUnit4.class)
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
   * {@code consumer} to {@code dependency} (expressed as label strings).
   */
  private String visibilityErrorMessage(String consumer, String dependency) {
    return String.format(
        "Visibility error:\ntarget '%s' is not visible from\ntarget '%s'", dependency, consumer);
  }

  /**
   * Requests the evaluation of the configured target identified by the label {@code consumer}, and
   * asserts that the target analyzes successfully and has a dependency on the target identified by
   * the label {@code dependency}.
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
  private void assertVisibilityDisallows(String consumer, String dependency) throws Exception {
    reporter.removeHandler(failFastHandler);
    ConfiguredTarget consumerTarget = getConfiguredTarget(consumer);
    assertWithMessage(
            String.format(
                "%s unexpectedly did not have an error (no visibility violation due to %s)",
                consumer, dependency))
        .that(consumerTarget)
        .isNull();
    assertContainsEvent(visibilityErrorMessage(consumer, dependency));
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

  @Test
  public void buildFileAccessToMacroTargetsIsControlled() throws Exception {
    defineSimpleRule();
    scratch.file("lib/BUILD");
    scratch.file(
        "lib/macro.bzl",
        """
        load("//rules:simple_rule.bzl", "simple_rule")

        def _impl(name):
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

        def _impl(name):
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

        def _impl(name):
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

        def _impl(name):
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

        def _impl(name):
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

        def _impl(name):
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
        def impl(name):
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
    scratch.file(
        "common/BUILD",
        """
        load("//rules:simple_rule.bzl", "simple_rule")

        simple_rule(
            name = "vis_to_inner",
            visibility = ["//inner:__pkg__"],
        )

        simple_rule(
            name = "vis_to_outer",
            visibility = ["//outer:__pkg__"],
        )
        """);
    scratch.file("inner/BUILD");
    scratch.file(
        "inner/macro.bzl",
        """
        load("//rules:simple_rule.bzl", "simple_rule")

        def _impl(name):
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

        def _impl(name):
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

  /*
   * TODO: #19922 - Tests cases to add:
   *
   * ---- Propagating target usages from parent macro to child ----
   *
   * - An inner macro can see a target it doesn't have permission on, if the parent macro has
   *   permission and the parent passes the label into the inner macro as an explicit attribute of
   *   the inner macro.
   *
   * - This doesn't work if the parent macro has permission but doesn't pass the label in. Implicit
   *   deps of the inner macro do not qualify as the outer macro passing it in.
   *
   * - This doesn't work if the parent macro passes the label in as an explicit attribute, but
   *   doesn't itself have permission.
   *
   * - It's an error if the parent passes the label in but does not have permission, even if the
   *   inner macro independently has its own permission.
   *
   * - Permission can be passed recursively through multiple levels, but not through a gap (e.g.
   *   middle macro does not have permission or does not thread it through).
   *
   * - But if there is a gap, and if the inner macro properly has permission and does not get passed
   *   the label from the middle, then the original usage by the outer macro (which is independent /
   *   inconsequential to the usage by the target in the inner macro) is not validated and is
   *   allowed to pass -- erroneously, we tolerate it.
   *
   * - Alias targets aren't followed when checking whether a label occurs in the parent macro. This
   *   is in line with how we look at the alias target's visibility, not the underlying, and anyway
   *   a macro wouldn't follow the reference when forwarding it along.
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
   *   does.
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
   * - default_visibility does not propagate to inside any symbolic macro, to either macros or
   *   targets.
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
