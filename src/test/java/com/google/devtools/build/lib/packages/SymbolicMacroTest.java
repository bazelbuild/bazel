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
public final class SymbolicMacroTest extends BuildViewTestCase {

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
  private Package getPackage(String pkgName) throws InterruptedException {
    try {
      return getPackageManager().getPackage(reporter, PackageIdentifier.createInMainRepo(pkgName));
    } catch (NoSuchPackageException unused) {
      return null;
    }
  }

  private void assertPackageNotInError(@Nullable Package pkg) {
    assertThat(pkg).isNotNull();
    assertThat(pkg.containsErrors()).isFalse();
  }

  @Test
  public void implementationIsInvokedWithNameParam() throws Exception {
    scratch.file(
        "pkg/foo.bzl",
        """
        def _impl(name):
            print("my_macro called with name = %s" % name)
        my_macro = macro(implementation=_impl)
        """);
    scratch.file(
        "pkg/BUILD",
        """
        load(":foo.bzl", "my_macro")
        my_macro(name="abc")
        """);

    Package pkg = getPackage("pkg");
    assertPackageNotInError(pkg);
    assertContainsEvent("called with name = abc");
  }

  @Test
  public void implementationFailsDueToBadSignature() throws Exception {
    scratch.file(
        "pkg/foo.bzl",
        """
        def _impl():
            pass
        my_macro = macro(implementation=_impl)
        """);
    scratch.file(
        "pkg/BUILD",
        """
        load(":foo.bzl", "my_macro")
        my_macro(name="abc")
        """);

    reporter.removeHandler(failFastHandler);
    Package pkg = getPackage("pkg");
    assertThat(pkg).isNotNull();
    assertThat(pkg.containsErrors()).isTrue();
    assertContainsEvent("_impl() got unexpected keyword argument: name");
  }

  @Test
  public void macroCanDeclareTargets() throws Exception {
    scratch.file(
        "pkg/foo.bzl",
        """
        def _impl(name):
            native.cc_library(name = name + "$lib")
        my_macro = macro(implementation=_impl)
        """);
    scratch.file(
        "pkg/BUILD",
        """
        load(":foo.bzl", "my_macro")
        my_macro(name="abc")
        """);

    Package pkg = getPackage("pkg");
    assertPackageNotInError(pkg);
    assertThat(pkg.getTargets()).containsKey("abc$lib");
  }

  @Test
  public void macroCanDeclareSubmacros() throws Exception {
    scratch.file(
        "pkg/foo.bzl",
        """
        def _inner_impl(name):
            native.cc_library(name = name + "$lib")
        inner_macro = macro(implementation=_inner_impl)
        def _impl(name):
            inner_macro(name = name + "$inner")
        my_macro = macro(implementation=_impl)
        """);
    scratch.file(
        "pkg/BUILD",
        """
        load(":foo.bzl", "my_macro")
        my_macro(name="abc")
        """);

    Package pkg = getPackage("pkg");
    assertPackageNotInError(pkg);
    assertThat(pkg.getTargets()).containsKey("abc$inner$lib");
  }

  // TODO: #19922 - Invert this, symbolic macros shouldn't be able to call glob().
  @Test
  public void macroCanCallGlob() throws Exception {
    scratch.file("pkg/foo.txt");
    scratch.file(
        "pkg/foo.bzl",
        """
        def _impl(name):
            print("Glob result: %s" % native.glob(["foo*"]))
        my_macro = macro(implementation=_impl)
        """);
    scratch.file(
        "pkg/BUILD",
        """
        load(":foo.bzl", "my_macro")
        my_macro(name="abc")
        """);

    Package pkg = getPackage("pkg");
    assertPackageNotInError(pkg);
    assertContainsEvent("Glob result: [\"foo.bzl\", \"foo.txt\"]");
  }

  // TODO: #19922 - Invert this, symbolic macros shouldn't be able to call existing_rules().
  @Test
  public void macroCanCallExistingRules() throws Exception {
    scratch.file(
        "pkg/foo.bzl",
        """
        def _impl(name):
            native.cc_binary(name = name + "$lib")
            print("existing_rules() keys: %s" % native.existing_rules().keys())
        my_macro = macro(implementation=_impl)
        """);
    scratch.file(
        "pkg/BUILD",
        """
        load(":foo.bzl", "my_macro")
        cc_library(name = "outer_target")
        my_macro(name="abc")
        """);

    Package pkg = getPackage("pkg");
    assertPackageNotInError(pkg);
    assertContainsEvent("existing_rules() keys: [\"outer_target\", \"abc$lib\"]");
  }

  // TODO: #19922 - This behavior is necessary to preserve compatibility with use cases for
  // native.existing_rules(), but it's a blocker for making symbolic macro evaluation lazy.
  @Test
  public void macroDeclaredTargetsAreVisibleToExistingRules() throws Exception {
    scratch.file(
        "pkg/foo.bzl",
        """
        def _impl(name):
            native.cc_binary(name = name + "$lib")
        my_macro = macro(implementation=_impl)
        def query():
            print("existing_rules() keys: %s" % native.existing_rules().keys())
        """);
    scratch.file(
        "pkg/BUILD",
        """
        load(":foo.bzl", "my_macro", "query")
        cc_library(name = "outer_target")
        my_macro(name="abc")
        query()
        """);

    Package pkg = getPackage("pkg");
    assertPackageNotInError(pkg);
    assertContainsEvent("existing_rules() keys: [\"outer_target\", \"abc$lib\"]");
  }

  // TODO: #19922 - Add more test cases for interaction between macros and environment_group,
  // package_group, implicit/explicit input files, and the package() function. But all of these
  // behaviors are about to change (from allowed to prohibited).
}
