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

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.analysis.util.AnalysisTestCase;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.Rule;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for dormant dependencies. */
@RunWith(JUnit4.class)
public class DormantDependencyTest extends AnalysisTestCase {
  @Before
  public void enableDormantDeps() throws Exception {
    useConfiguration("--experimental_dormant_deps");
  }

  @Test
  public void testDormantLabelDisabledWithoutExperimentalFlag() throws Exception {
    useConfiguration("--noexperimental_dormant_deps");

    scratch.file(
        "dormant/dormant.bzl",
        """
        def _r_impl(ctx):
          fail("should not be called")

        r = rule(
          implementation = _r_impl,
          attrs = {
            "dormant": attr.dormant_label(),
          })""");

    scratch.file(
        "dormant/BUILD",
        """
        load(":dormant.bzl", "r")
        r(name="r")""");

    reporter.removeHandler(failFastHandler);
    assertThrows(TargetParsingException.class, () -> update("//dormant:r"));
    assertContainsEvent("no field or method 'dormant_label'");
  }

  @Test
  public void testDormantAttribute() throws Exception {
    scratch.file(
        "dormant/dormant.bzl",
        """
        def _r_impl(ctx):
          print("dormant label is " + str(ctx.attr.dormant.label()))
          print("dormant label list is " + str(ctx.attr.dormant_list[0].label()))
          return [DefaultInfo()]

        r = rule(
          implementation = _r_impl,
          dependency_resolution_rule = True,
          attrs = {
            "dormant": attr.dormant_label(),
            "dormant_list": attr.dormant_label_list(),
          })""");

    scratch.file(
        "dormant/BUILD",
        """
        load(":dormant.bzl", "r")

        filegroup(name="a")
        filegroup(name="b")
        r(name="r", dormant=":a", dormant_list=[":b"])""");

    update("//dormant:r");
    assertContainsEvent("dormant label is @@//dormant:a");
    assertContainsEvent("dormant label list is @@//dormant:b");
  }

  @Test
  public void testDormantAttributeComputedDefaultsFail() throws Exception {
    scratch.file(
        "dormant/dormant.bzl",
        """
        def _r_impl(ctx):
          fail("should not happen")

        def computed_default():
          fail("should not happen")

        r = rule(
          implementation = _r_impl,
          dependency_resolution_rule = True,
          attrs = {
            "dormant": attr.dormant_label(default=computed_default),
          })""");

    scratch.file(
        "dormant/BUILD",
        """
        load(":dormant.bzl", "r")
        r(name="r")""");

    reporter.removeHandler(failFastHandler);
    assertThrows(TargetParsingException.class, () -> update("//dormant:r"));
    assertContainsEvent("got value of type 'function'");
  }

  @Test
  public void testDormantAttributeDefaultValues() throws Exception {
    scratch.file(
        "dormant/dormant.bzl",
        """
        def _r_impl(ctx):
          print("dormant label is " + str(ctx.attr.dormant.label()))
          print("dormant label list is " + str(ctx.attr.dormant_list[0].label()))
          return [DefaultInfo()]

        r = rule(
          implementation = _r_impl,
          dependency_resolution_rule = True,
          attrs = {
            "dormant": attr.dormant_label(default="//dormant:a"),
            "dormant_list": attr.dormant_label_list(default=["//dormant:b"]),
          })""");

    scratch.file(
        "dormant/BUILD",
        """
        load(":dormant.bzl", "r")

        filegroup(name="a")
        filegroup(name="b")
        r(name="r")""");

    update("//dormant:r");
    assertContainsEvent("dormant label is @@//dormant:a");
    assertContainsEvent("dormant label list is @@//dormant:b");
  }

  @Test
  public void testExistenceOfMaterializerParameter() throws Exception {
    scratch.file(
        "dormant/dormant.bzl",
        """
        def _r_impl(ctx):
          return [DefaultInfo()]

        def _label_materializer(*args, **kwargs):
          return None

        def _list_materializer(*args, **kwargs):
          return []

        r = rule(
          implementation = _r_impl,
          attrs = {
            "_materialized": attr.label(materializer=_label_materializer),
            "_materialized_list": attr.label_list(materializer=_list_materializer),
          })""");

    scratch.file(
        "dormant/BUILD",
        """
        load(":dormant.bzl", "r")
        r(name="r")""");

    update("//dormant:r");
  }

  @Test
  public void testMaterializedOnNonHiddenAttribute() throws Exception {
    scratch.file(
        "dormant/dormant.bzl",
        """
        def _r_impl(ctx):
          return [DefaultInfo()]

        def _label_materializer(*args, **kwargs):
          return None

        r = rule(
          implementation = _r_impl,
          attrs = {
            "materialized": attr.label(materializer=_label_materializer),
          })""");

    scratch.file(
        "dormant/BUILD",
        """
        load(":dormant.bzl", "r")
        r(name="r")""");

    reporter.removeHandler(failFastHandler);
    assertThrows(TargetParsingException.class, () -> update("//dormant:r"));
    assertContainsEvent("attribute must be private");
  }

  @Test
  public void testMaterializerAndDefaultAreIncompatible() throws Exception {
    scratch.file(
        "dormant/dormant.bzl",
        """
        def _r_impl(ctx):
          return [DefaultInfo()]

        def _label_materializer(*args, **kwargs):
          return None

        def _list_materializer(*args, **kwargs):
          return []

        r = rule(
          implementation = _r_impl,
          attrs = {
            "_materialized": attr.label(
                materializer=_label_materializer,
                default=Label("//dormant:default")),
          })""");

    scratch.file(
        "dormant/BUILD",
        """
        load(":dormant.bzl", "r")
        r(name="r")""");

    reporter.removeHandler(failFastHandler);
    assertThrows(TargetParsingException.class, () -> update("//dormant:r"));
    assertContainsEvent("parameters are incompatible");
  }

  @Test
  public void testMaterializerAndMandatoryAreIncompatible() throws Exception {
    scratch.file(
        "dormant/dormant.bzl",
        """
        def _r_impl(ctx):
          return [DefaultInfo()]

        def _label_materializer(*args, **kwargs):
          return None

        def _list_materializer(*args, **kwargs):
          return []

        r = rule(
          implementation = _r_impl,
          attrs = {
            "_materialized": attr.label(materializer=_label_materializer, mandatory=True),
          })""");

    scratch.file(
        "dormant/BUILD",
        """
        load(":dormant.bzl", "r")
        r(name="r")""");

    reporter.removeHandler(failFastHandler);
    assertThrows(TargetParsingException.class, () -> update("//dormant:r"));
    assertContainsEvent("parameters are incompatible");
  }

  @Test
  public void testMaterializerAndConfigurableAreIncompatible() throws Exception {
    scratch.file(
        "dormant/dormant.bzl",
        """
        def _r_impl(ctx):
          return [DefaultInfo()]

        def _label_materializer(*args, **kwargs):
          return None

        def _list_materializer(*args, **kwargs):
          return []

        r = rule(
          implementation = _r_impl,
          attrs = {
            "_materialized": attr.label(materializer=_label_materializer, configurable=True),
          })""");

    scratch.file(
        "dormant/BUILD",
        """
        load(":dormant.bzl", "r")
        r(name="r")""");

    reporter.removeHandler(failFastHandler);
    assertThrows(TargetParsingException.class, () -> update("//dormant:r"));
    assertContainsEvent("parameters are incompatible");
  }

  @Test
  public void testMaterializerOnAspectNotAllowed() throws Exception {
    scratch.file(
        "a/a.bzl",
        """
        def _r_impl(ctx):
          fail("rule implementation should not be called")

        def _a_impl(target, ctx):
          fail("aspect implementation should not be called")

        def _materializer(ctx):
           fail("materializer should not be called")

        a = aspect(
          implementation = _a_impl,
          attrs = { "_materialized": attr.label_list(materializer=_materializer)})

        r = rule(
          implementation = _r_impl,
          attrs = { "dep": attr.label(aspects=[a])})
        """);

    scratch.file(
        "a/BUILD",
        """
        load(":a.bzl", "r")

        filegroup(name="f")
        r(name="r", dep=":f")
        """);

    reporter.removeHandler(failFastHandler);
    assertThrows(TargetParsingException.class, () -> update("//a:r"));
    assertContainsEvent("has a materializer, which is not allowed on aspects");
  }

  @Test
  public void testAttributesOfDependencyResolutionRulesCannotBeMarkedOtherwise() throws Exception {
    scratch.file(
        "a/a.bzl",
        """
        def _a_impl(ctx):
          fail("rule implementation should not be called")

        a = rule(
          implementation = _a_impl,
          attrs = {"dep": attr.label(for_dependency_resolution = False)},
          dependency_resolution_rule = True)
        """);

    scratch.file(
        "a/BUILD",
        """
        load("//a:a.bzl", "a")
        a(name="x")""");

    reporter.removeHandler(failFastHandler);
    assertThrows(TargetParsingException.class, () -> update("//a:x"));
    assertContainsEvent("explicitly marked as not for dependency resolution");
  }

  @Test
  public void testAttributesOfDependencyResolutionRulesAreNonconfigurable() throws Exception {
    scratch.file(
        "a/a.bzl",
        """
        def _a_impl(ctx):
          return [DefaultInfo()]

        a = rule(
          implementation = _a_impl,
          attrs = {"dep": attr.label()},
          dependency_resolution_rule = True)
        """);

    scratch.file("a/BUILD");
    scratch.file(
        "x/BUILD",
        """
        load("//a:a.bzl", "a")
        a(name="x")""");

    scratch.file(
        "y/BUILD",
        """
        load("//a:a.bzl", "a")
        config_setting(name = "cs", values = {"define": "cs"})
        a(name="y", dep=select({":cs": ":y1", "//conditions:default": ":y2"}))
        """);

    update("//x");
    Rule xRule = (Rule) getConfiguredTargetAndTarget("//x").getTargetForTesting();
    Attribute depAttribute = xRule.getRuleClassObject().getAttributeByName("dep");
    assertThat(depAttribute.isConfigurable()).isFalse();
    assertThat(depAttribute.isForDependencyResolution()).isTrue();

    reporter.removeHandler(failFastHandler);
    assertThrows(TargetParsingException.class, () -> update("//y"));
    assertContainsEvent("attribute \"dep\" is not configurable");
  }

  @Test
  public void testRuleMustBeMarkedAsDormant() throws Exception {
    scratch.file(
        "a/a.bzl",
        """
        def _a_impl(ctx):
          return []

        a = rule(
          implementation = _a_impl,
          attrs = {"dep": attr.dormant_label()})
        """);

    scratch.file(
        "a/BUILD",
        """
        load(":a.bzl", "a")
        a(name="a")
        """);

    reporter.removeHandler(failFastHandler);
    assertThrows(TargetParsingException.class, () -> update("//a:a"));
    assertContainsEvent("Has dormant attributes ('dep')");
  }

  @Test
  public void testNoDormantDepsOnAspects() throws Exception {
    scratch.file(
        "a/a.bzl",
        """
        def _impl(*args):
          fail("should not be called")

        a = aspect(
          implementation = _impl,
          attrs = { "_dormant": attr.dormant_label(default="//x:x")})

        r = rule(
          implementation = _impl,
          attrs = { "dep": attr.label(aspects=[a]) })
        """);

    scratch.file(
        "a/BUILD",
        """
        load(":a.bzl", "r")
        r(name="a")
        """);

    reporter.removeHandler(failFastHandler);
    assertThrows(TargetParsingException.class, () -> update("//a"));
    assertContainsEvent("'_dormant' has a dormant label type");
  }

  @Test
  public void testNoAspectsOnDependencyResolutionRules() throws Exception {
    scratch.file(
        "a/a.bzl",
        """
        def _impl(*args):
          fail("should not be called")

        a = aspect(implementation = _impl)

        r = rule(
          implementation = _impl,
          dependency_resolution_rule = True,
          attrs = {"dep": attr.label_list(aspects=[a])},
        )
        """);

    scratch.file(
        "a/BUILD",
        """
        load(":a.bzl", "r")
        r(name="r")
        """);

    reporter.removeHandler(failFastHandler);
    assertThrows(TargetParsingException.class, () -> update("//a:r"));
    assertContainsEvent("cannot propagate aspects");
  }

  @Test
  public void testNoToolchainsOnDependencyResolutionRules() throws Exception {
    scratch.file(
        "a/a.bzl",
        """
        def _impl(*args):
          fail("should not be called")

        a = aspect(implementation = _impl)

        r = rule(
          implementation = _impl,
          dependency_resolution_rule = True,
          attrs = {"dep": attr.label_list()},
          toolchains = ["//a:nonexistent_toolchain"],
        )
        """);

    scratch.file(
        "a/BUILD",
        """
        load(":a.bzl", "r")
        r(name="r")
        """);

    reporter.removeHandler(failFastHandler);
    assertThrows(TargetParsingException.class, () -> update("//a:r"));
    assertContainsEvent("cannot depend on toolchains");
  }

  @Test
  public void testErrorOnUnmarkedRuleInAttributeAvailableInMaterializers() throws Exception {
    scratch.file(
        "a/dormant.bzl",
        """
        def _component_impl(ctx):
          fail("should not be called")

        component = rule(
          implementation = _component_impl,
          dependency_resolution_rule = True,
          attrs = {
              "impl": attr.label(),
          })""");

    scratch.file(
        "a/BUILD",
        """
        load(":dormant.bzl", "component")
        component(name="c", impl=":bad")
        filegroup(name="bad")
        """);

    reporter.removeHandler(failFastHandler);
    assertThrows(ViewCreationFailedException.class, () -> update("//a:c"));
    assertContainsEvent(
        "marked as available in materializers but prerequisite filegroup rule '//a:bad' isn't");
  }

  @Test
  public void testMarkedRulesCannotHaveParents() throws Exception {
    scratch.file(
        "a/dormant.bzl",
        """
        def _impl(ctx):
          fail("rule implementation should not be called")


        p = rule(
          implementation = _impl,
          attrs = {})

        r = rule(
          implementation = _impl,
          dependency_resolution_rule = True,
          parent = p,
          attrs = {})""");

    scratch.file(
        "a/BUILD",
        """
        load(":dormant.bzl", "r")
        r(name="r")
        """);

    reporter.removeHandler(failFastHandler);
    assertThrows(TargetParsingException.class, () -> update("//a:r"));
    assertContainsEvent("cannot have a parent");
  }

  @Test
  public void testMarkedRulesCannotCreateActions() throws Exception {
    scratch.file(
        "a/dormant.bzl",
        """
        def _r_impl(ctx):
          a = ctx.actions.declare_file("f")
          ctx.actions.write(a, "foo")
          return [DefaultInfo(files=depset([a]))]

        r = rule(
          implementation = _r_impl,
          dependency_resolution_rule = True,
          attrs = {})""");

    scratch.file(
        "a/BUILD",
        """
        load(":dormant.bzl", "r")
        r(name="r")
        """);

    reporter.removeHandler(failFastHandler);
    assertThrows(ViewCreationFailedException.class, () -> update("//a:r"));
    assertContainsEvent("shouldn't have actions");
  }
}
