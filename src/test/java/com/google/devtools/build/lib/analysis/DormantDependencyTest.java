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

import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.analysis.util.AnalysisTestCase;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
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
}
