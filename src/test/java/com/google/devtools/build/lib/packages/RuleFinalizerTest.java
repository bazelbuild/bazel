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

package com.google.devtools.build.lib.packages;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import javax.annotation.Nullable;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests the execution of symbolic macro implementations. */
@RunWith(JUnit4.class)
public final class RuleFinalizerTest extends BuildViewTestCase {

  @Before
  public void setUp() throws Exception {
    setBuildLanguageOptions("--experimental_enable_first_class_macros");
  }

  /**
   * Returns a package by the given name (no leading "//"), or null upon {@link
   * NoSuchPackageException}.
   */
  @CanIgnoreReturnValue
  @Nullable
  private Package getPackage(String pkgName) throws InterruptedException, NoSuchPackageException {
    return getPackageManager().getPackage(reporter, PackageIdentifier.createInMainRepo(pkgName));
  }

  private void assertPackageNotInError(@Nullable Package pkg) {
    assertThat(pkg).isNotNull();
    assertThat(pkg.containsErrors()).isFalse();
  }

  private void assertGetPackageFailsWithEvent(String pkgName, String msg) throws Exception {
    reporter.removeHandler(failFastHandler);
    Package pkg = getPackage(pkgName);
    assertThat(pkg).isNotNull();
    assertThat(pkg.containsErrors()).isTrue();
    assertContainsEvent(msg);
  }

  @Test
  public void basicFunctionality() throws Exception {
    scratch.file(
        "pkg/foo.bzl",
        """
        def _impl(name, targets_of_interest):
            for r in native.existing_rules().values():
                if r["name"] in [t.name for t in targets_of_interest]:
                    genrule_name = name + "_" + r["name"] + "_finalize"
                    native.genrule(
                        name = genrule_name,
                        srcs = [r["name"]],
                        outs = [genrule_name + ".txt"],
                        cmd = "... > $@",
                    )

        my_finalizer = macro(
            implementation = _impl,
            finalizer = True,
            attrs = {"targets_of_interest": attr.label_list(configurable = False)},
        )
        """);
    scratch.file(
        "pkg/BUILD",
        """
        load(":foo.bzl", "my_finalizer")
        cc_library(name = "foo")
        my_finalizer(name = "abc", targets_of_interest = [":foo"])
        """);

    Package pkg = getPackage("pkg");
    assertPackageNotInError(pkg);
    assertThat(pkg.getTargets().keySet())
        .containsAtLeast("abc_foo_finalize", "abc_foo_finalize.txt");
  }

  @Test
  public void finalizer_canCallFinalizer() throws Exception {
    scratch.file(
        "pkg/foo.bzl",
        """
        def _impl_inner(name):
            for r in native.existing_rules().values():
                if r["name"] == "foo":
                    genrule_name = name + "_" + r["name"] + "_finalize"
                    native.genrule(
                        name = genrule_name,
                        srcs = [r["name"]],
                        outs = [genrule_name + ".txt"],
                        cmd = "... > $@",
                    )

        my_finalizer_inner = macro(implementation = _impl_inner, finalizer = True)

        def _impl_outer(name):
            my_finalizer_inner(name = name + "_inner")

        my_finalizer_outer = macro(implementation = _impl_outer, finalizer = True)
        """);
    scratch.file(
        "pkg/BUILD",
        """
        load(":foo.bzl", "my_finalizer_outer")
        cc_library(name = "foo")
        my_finalizer_outer(name = "abc")
        """);

    Package pkg = getPackage("pkg");
    assertPackageNotInError(pkg);
    assertThat(pkg.getTargets()).containsKey("abc_inner_foo_finalize");
  }

  @Test
  public void finalizer_canCallNonFinalizerMacro() throws Exception {
    scratch.file(
        "pkg/foo.bzl",
        """
        def _impl_macro(name, deps):
            native.genrule(
                name = name,
                srcs = deps,
                outs = [name + ".txt"],
                cmd = "... > $@",
            )

        my_macro = macro(implementation = _impl_macro, attrs = {"deps": attr.label_list()})

        def _impl_finalizer(name):
            for r in native.existing_rules().values():
                if r["name"] == "foo":
                    my_macro(name=name + "_" + r["name"] + "_finalize", deps = [r["name"]])

        my_finalizer = macro(implementation = _impl_finalizer, finalizer = True)
        """);
    scratch.file(
        "pkg/BUILD",
        """
        load(":foo.bzl", "my_finalizer")
        cc_library(name = "foo")
        my_finalizer(name = "abc")
        """);

    Package pkg = getPackage("pkg");
    assertPackageNotInError(pkg);
    assertThat(pkg.getTargets().keySet())
        .containsAtLeast("abc_foo_finalize", "abc_foo_finalize.txt");
  }

  @Test
  public void nonFinalizerMacro_cannotCallFinalizer() throws Exception {
    scratch.file(
        "pkg/foo.bzl",
        """
        def _impl_finalizer(name):
            for r in native.existing_rules().values():
                if r["name"] == "foo":
                    genrule_name = name + "_" + r["name"] + "_finalize"
                    native.genrule(
                        name = genrule_name,
                        srcs = [r["name"]],
                        outs = [genrule_name + ".txt"],
                        cmd = "... > $@",
                    )

        my_finalizer = macro(implementation = _impl_finalizer, finalizer = True)

        def _impl_macro(name):
            my_finalizer(name = name + "_inner")

        my_macro = macro(implementation = _impl_macro)
        """);
    scratch.file(
        "pkg/BUILD",
        """
        load(":foo.bzl", "my_macro")
        my_macro(name = "abc")
        """);

    assertGetPackageFailsWithEvent(
        "pkg", "Cannot instantiate a rule finalizer within a non-finalizer symbolic macro");
  }

  @Test
  public void finalizer_nativeExistingRules_seesOnlyNonFinalizerTargets_inAllLexicalPositions()
      throws Exception {
    scratch.file(
        "pkg/foo.bzl",
        """
        def _impl_macro(name):
            native.cc_library(name = name + "_inner_lib")

        my_macro = macro(implementation = _impl_macro)

        def _impl_inner_finalizer(name):
            native.cc_library(name = name + "_inner_lib")
            for r in native.existing_rules().values():
                print("saw " + r["name"])

        inner_finalizer = macro(implementation = _impl_inner_finalizer, finalizer = True)

        def _impl_finalizer(name):
            native.cc_library(name = name + "_inner_lib")
            my_macro(name = name + "_inner_macro")
            inner_finalizer(name = name + "_inner_finalizer")
            for r in native.existing_rules().values():
                print("saw " + r["name"])

        my_finalizer = macro(implementation = _impl_finalizer, finalizer = True)
        """);
    scratch.file(
        "pkg/BUILD",
        """
        load(":foo.bzl", "my_finalizer", "my_macro")
        cc_library(name = "top_level_lexically_before_finalizer")
        my_macro(name = "macro_lexically_before_finalizer")
        my_finalizer(name = "finalizer")
        my_finalizer(name = "other_finalizer")
        cc_library(name = "top_level_lexically_after_finalizer")
        my_macro(name = "macro_lexically_after_finalizer")
        """);

    Package pkg = getPackage("pkg");
    assertPackageNotInError(pkg);
    assertContainsEventWithFrequency("saw top_level_lexically_before_finalizer", 4);
    assertContainsEventWithFrequency("saw macro_lexically_before_finalizer_inner_lib", 4);
    assertContainsEventWithFrequency("saw top_level_lexically_after_finalizer", 4);
    assertContainsEventWithFrequency("saw macro_lexically_after_finalizer_inner_lib", 4);
    assertDoesNotContainEvent("saw finalizer_inner_lib");
    assertDoesNotContainEvent("saw finalizer_inner_macro_inner_lib");
    assertDoesNotContainEvent("saw finalizer_inner_finalizer_inner_lib");
    assertDoesNotContainEvent("saw other_finalizer_inner_lib");
    assertDoesNotContainEvent("saw other_finalizer_inner_macro_inner_lib");
    assertDoesNotContainEvent("saw other_finalizer_inner_finalizer_inner_lib");
  }
}
