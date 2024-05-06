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

  /**
   * Convenience method for asserting that a package evaluates in error and produces an event
   * containing the given substring.
   *
   * <p>Note that this is not suitable for errors that occur during top-level .bzl evaluation (i.e.,
   * triggered by load() rather than during BUILD evaluation), since our test framework fails to
   * produce a result in that case (b/26382502).
   */
  private void assertGetPackageFailsWithEvent(String pkgName, String msg) throws Exception {
    reporter.removeHandler(failFastHandler);
    Package pkg = getPackage(pkgName);
    assertThat(pkg).isNotNull();
    assertThat(pkg.containsErrors()).isTrue();
    assertContainsEvent(msg);
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

    assertGetPackageFailsWithEvent("pkg", "_impl() got unexpected keyword argument: name");
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
    // TODO(#19922): change naming convention to not use "$""
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

  /**
   * Implementation of a test that ensures a given API cannot be called from inside a symbolic
   * macro.
   */
  private void doCannotCallApiTest(String apiName, String usageLine) throws Exception {
    scratch.file(
        "pkg/foo.bzl",
        String.format(
            """
            def _impl(name):
                %s
            my_macro = macro(implementation=_impl)
            """,
            usageLine));
    scratch.file(
        "pkg/BUILD",
        """
        load(":foo.bzl", "my_macro")
        my_macro(name="abc")
        """);

    assertGetPackageFailsWithEvent(
        "pkg",
        String.format(
            "%s can only be used while evaluating a BUILD file or legacy macro", apiName));
  }

  @Test
  public void macroCannotCallPackage() throws Exception {
    doCannotCallApiTest(
        "package()", "native.package(default_visibility = ['//visibility:public'])");
  }

  @Test
  public void macroCannotCallGlob() throws Exception {
    doCannotCallApiTest("glob()", "native.glob(['foo*'])");
  }

  @Test
  public void macroCannotCallSubpackages() throws Exception {
    doCannotCallApiTest("subpackages()", "native.subpackages(include = ['*'])");
  }

  @Test
  public void macroCannotCallExistingRule() throws Exception {
    doCannotCallApiTest("existing_rule()", "native.existing_rule('foo')");
  }

  @Test
  public void macroCannotCallExistingRules() throws Exception {
    doCannotCallApiTest("existing_rules()", "native.existing_rules()");
  }

  // There are other symbols that must not be called from within symbolic macros, but we don't test
  // them because they can't be obtained from a symbolic macro implementation anyway, since they are
  // not under `native` (at least, for BUILD-loaded .bzl files) and because symbolic macros can't
  // take arbitrary parameter types from their caller. These untested symbols include:
  //
  //  - For BUILD threads: licenses(), environment_group()
  //  - For WORKSPACE threads: workspace(), register_toolchains(), register_execution_platforms(),
  //    bind(), and repository rules.
  //
  // Starlark-defined repository rules might technically be callable but we skip over that edge
  // case here.

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

  @Test
  public void defaultAttrValue_isUsedWhenNotOverridden() throws Exception {
    scratch.file(
        "pkg/foo.bzl",
        """
        def _impl(name, xyz):
            print("xyz is %s" % xyz)
        my_macro = macro(
            implementation=_impl,
            attrs = {
              "xyz": attr.string(default="DEFAULT", configurable=False)
            },
        )
        """);
    scratch.file(
        "pkg/BUILD",
        """
        load(":foo.bzl", "my_macro")
        my_macro(name="abc")
        """);

    Package pkg = getPackage("pkg");
    assertPackageNotInError(pkg);
    assertContainsEvent("xyz is DEFAULT");
  }

  @Test
  public void defaultAttrValue_canBeOverridden() throws Exception {
    scratch.file(
        "pkg/foo.bzl",
        """
        def _impl(name, xyz):
            print("xyz is %s" % xyz)
        my_macro = macro(
            implementation=_impl,
            attrs = {
              "xyz": attr.string(default="DEFAULT", configurable=False)
            },
        )
        """);
    scratch.file(
        "pkg/BUILD",
        """
        load(":foo.bzl", "my_macro")
        my_macro(
            name = "abc",
            xyz = "OVERRIDDEN",
        )
        """);

    Package pkg = getPackage("pkg");
    assertPackageNotInError(pkg);
    assertContainsEvent("xyz is OVERRIDDEN");
  }

  @Test
  public void defaultAttrValue_isUsed_whenAttrIsImplicit() throws Exception {
    scratch.file(
        "pkg/foo.bzl",
        """
        def _impl(name, _xyz):
            print("xyz is %s" % _xyz)
        my_macro = macro(
            implementation=_impl,
            attrs = {
              "_xyz": attr.string(default="IMPLICIT", configurable=False)
            },
        )
        """);
    scratch.file(
        "pkg/BUILD",
        """
        load(":foo.bzl", "my_macro")
        my_macro(name="abc")
        """);

    Package pkg = getPackage("pkg");
    assertPackageNotInError(pkg);
    assertContainsEvent("xyz is IMPLICIT");
  }

  @Test
  public void noneAttrValue_doesNotOverrideDefault() throws Exception {
    scratch.file(
        "pkg/foo.bzl",
        """
        def _impl(name, xyz):
            print("xyz is %s" % xyz)
        my_macro = macro(
            implementation=_impl,
            attrs = {
              "xyz": attr.string(default="DEFAULT", configurable=False)
            },
        )
        """);
    scratch.file(
        "pkg/BUILD",
        """
        load(":foo.bzl", "my_macro")
        my_macro(
            name = "abc",
            xyz = None,
        )
        """);

    Package pkg = getPackage("pkg");
    assertPackageNotInError(pkg);
    assertContainsEvent("xyz is DEFAULT");
  }

  @Test
  public void noneAttrValue_doesNotSatisfyMandatoryRequirement() throws Exception {
    setBuildLanguageOptions("--experimental_enable_first_class_macros");

    scratch.file(
        "pkg/foo.bzl",
        """
        def _impl(name):
            pass
        my_macro = macro(
            implementation = _impl,
            attrs = {
                "xyz": attr.string(mandatory=True),
            },
        )
        """);
    scratch.file(
        "pkg/BUILD",
        """
        load(":foo.bzl", "my_macro")
        my_macro(
            name = "abc",
            xyz = None,
        )
        """);

    assertGetPackageFailsWithEvent(
        "pkg", "missing value for mandatory attribute 'xyz' in 'my_macro' macro");
  }

  @Test
  public void noneAttrValue_disallowedWhenAttrDoesNotExist() throws Exception {
    setBuildLanguageOptions("--experimental_enable_first_class_macros");

    scratch.file(
        "pkg/foo.bzl",
        """
        def _impl(name):
            pass
        my_macro = macro(
            implementation = _impl,
            attrs = {
                "xzz": attr.string(doc="This attr is public"),
            },
        )
        """);
    scratch.file(
        "pkg/BUILD",
        """
        load(":foo.bzl", "my_macro")
        my_macro(
            name = "abc",
            xyz = None,
        )
        """);

    assertGetPackageFailsWithEvent(
        "pkg", "no such attribute 'xyz' in 'my_macro' macro (did you mean 'xzz'?)");
  }

  @Test
  public void stringAttrsAreConvertedToLabelsAndInRightContext() throws Exception {
    scratch.file("lib/BUILD");
    scratch.file(
        "lib/foo.bzl",
        """
        def _impl(name, xyz, _xyz):
            print("xyz is %s" % xyz)
            print("_xyz is %s" % _xyz)
        my_macro = macro(
            implementation=_impl,
            attrs = {
              "xyz": attr.label(configurable = False),
              "_xyz": attr.label(default=":BUILD", configurable=False)
            },
        )
        """);
    scratch.file(
        "pkg/BUILD",
        """
        load("//lib:foo.bzl", "my_macro")
        my_macro(
            name = "abc",
            xyz = ":BUILD",  # Should be parsed relative to //pkg, not //lib
        )
        """);

    Package pkg = getPackage("pkg");
    assertPackageNotInError(pkg);
    assertContainsEvent("xyz is @@//pkg:BUILD");
    assertContainsEvent("_xyz is @@//lib:BUILD");
  }

  @Test
  public void cannotMutateAttrValues() throws Exception {
    scratch.file(
        "pkg/foo.bzl",
        """
        def _impl(name, xyz):
            xyz.append(4)
        my_macro = macro(
            implementation=_impl,
            attrs = {
              "xyz": attr.int_list(configurable=False),
            },
        )
        """);
    scratch.file(
        "pkg/BUILD",
        """
        load(":foo.bzl", "my_macro")
        my_macro(
            name = "abc",
            xyz = [1, 2, 3],
        )
        """);

    assertGetPackageFailsWithEvent("pkg", "Error in append: trying to mutate a frozen list value");
  }

  @Test
  public void macroCanDefineMainTargetOfSameName() throws Exception {
    scratch.file(
        "pkg/foo.bzl",
        """
        def _impl(name):
            native.cc_library(
                name = name,
            )
        my_macro = macro(implementation=_impl)
        """);
    scratch.file(
        "pkg/BUILD",
        """
        load(":foo.bzl", "my_macro")
        my_macro(
            name = "abc",
        )
        """);

    Package pkg = getPackage("pkg");
    assertPackageNotInError(pkg);
    assertThat(pkg.getTargets()).containsKey("abc");
    assertThat(pkg.getMacros()).containsKey("abc");
  }

  // TODO: #19922 - Add more test cases for implicit/explicit input files

  @Test
  public void attrsAllowSelectsByDefault() throws Exception {
    scratch.file("lib/BUILD");
    scratch.file(
        "pkg/foo.bzl",
        """
        def _impl(name, xyz):
            print("xyz is %s" % xyz)
        my_macro = macro(
            implementation=_impl,
            attrs = {
              "xyz": attr.string(),
            },
        )
        """);
    scratch.file(
        "pkg/BUILD",
        """
        load(":foo.bzl", "my_macro")
        my_macro(
            name = "abc",
            xyz = select({"//some:condition": ":target1", "//some:other_condition": ":target2"}),
        )
        """);

    Package pkg = getPackage("pkg");
    assertPackageNotInError(pkg);
    assertContainsEvent(
        "xyz is select({Label(\"//some:condition\"): \":target1\","
            + " Label(\"//some:other_condition\"): \":target2\"})");
  }

  @Test
  public void configurableAttrValuesArePromotedToSelects() throws Exception {
    scratch.file("lib/BUILD");
    scratch.file(
        "pkg/foo.bzl",
        """
def _impl(name, configurable_xyz, nonconfigurable_xyz):
    print("configurable_xyz is '%s' (type %s)" % (str(configurable_xyz), type(configurable_xyz)))
    print(
        "nonconfigurable_xyz is '%s' (type %s)" % (str(
            nonconfigurable_xyz), type(nonconfigurable_xyz)))

my_macro = macro(
    implementation=_impl,
    attrs = {
      "configurable_xyz": attr.string(),
      "nonconfigurable_xyz": attr.string(configurable=False),
    },
)
""");
    scratch.file(
        "pkg/BUILD",
        """
        load(":foo.bzl", "my_macro")
        my_macro(
            name = "abc",
            configurable_xyz = "configurable",
            nonconfigurable_xyz = "nonconfigurable",
        )
        """);

    Package pkg = getPackage("pkg");
    assertPackageNotInError(pkg);
    assertContainsEvent(
        "configurable_xyz is 'select({\"//conditions:default\": \"configurable\"})' (type select)");
    assertContainsEvent("nonconfigurable_xyz is 'nonconfigurable' (type string)");
  }

  @Test
  public void nonconfigurableAttrValuesProhibitSelects() throws Exception {
    scratch.file("lib/BUILD");
    scratch.file(
        "pkg/foo.bzl",
        """
        def _impl(name, xyz):
            print("xyz is %s" % xyz)
        my_macro = macro(
            implementation=_impl,
            attrs = {
              "xyz": attr.string(configurable=False),
            },
        )
        """);
    scratch.file(
        "pkg/BUILD",
        """
        load(":foo.bzl", "my_macro")
        my_macro(
            name = "abc",
            xyz = select({"//some:condition": ":target1", "//some:other_condition": ":target2"}),
        )
        """);

    assertGetPackageFailsWithEvent("pkg", "attribute \"xyz\" is not configurable");
  }

  // TODO(b/331193690): Prevent selects from being evaluated as bools
  @Test
  public void selectableAttrCanBeEvaluatedAsBool() throws Exception {
    scratch.file("lib/BUILD");
    scratch.file(
        "pkg/foo.bzl",
        """
def _impl(name, xyz):
    # Allowed for now when xyz is a select().
    # In the future, we'll ban implicit conversion and only allow
    # if there's an explicit bool(xyz).
    if xyz:
      print ("xyz evaluates to True")
    else:
      print("xyz evaluates to False")



my_macro = macro(
    implementation=_impl,
    attrs = {
      "xyz": attr.string(),
    },
)
""");
    scratch.file(
        "pkg/BUILD",
        """
        load(":foo.bzl", "my_macro")
        my_macro(
            name = "abc",
            xyz = select({"//conditions:default" :"False"}),
        )
        """);

    Package pkg = getPackage("pkg");
    assertPackageNotInError(pkg);
    assertContainsEvent("xyz evaluates to True");
    assertDoesNotContainEvent("xyz evaluates to False");
  }
}
