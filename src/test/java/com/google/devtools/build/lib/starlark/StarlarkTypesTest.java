// Copyright 2025 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.starlark;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for Starlark types. */
@RunWith(JUnit4.class)
public class StarlarkTypesTest extends BuildViewTestCase {

  @Test
  public void experimentalStarlarkTypes_on_allowsTypeAnnotations() throws Exception {
    setBuildLanguageOptions(
        "--experimental_starlark_type_syntax",
        "--experimental_starlark_types_allowed_paths=//test");
    scratch.file(
        "test/foo.bzl",
        """
        def f(a: int):
          pass\
        """);
    scratch.file("test/BUILD", "load(':foo.bzl', 'f')");

    getTarget("//test:BUILD");

    assertNoEvents();
  }

  @Test
  public void experimentalStarlarkTypes_off_disallowsTypeAnnotations() throws Exception {
    setBuildLanguageOptions(
        "--noexperimental_starlark_type_syntax",
        "--experimental_starlark_types_allowed_paths=//test");
    scratch.file(
        "test/foo.bzl",
        """
        def f(a: int):
          pass\
        """);
    scratch.file("test/BUILD", "load(':foo.bzl', 'f')");

    AssertionError e = assertThrows(AssertionError.class, () -> getTarget("//test:BUILD"));
    assertThat(e)
        .hasMessageThat()
        .contains(
            """
            syntax error at ':': type annotations are disallowed. Enable them with \
            --experimental_starlark_type_syntax and/or --experimental_starlark_types_allowed_paths.\
            """);
  }

  @Test
  public void starlarkTypesAllowedPath_notOnPath_disallowsTypeAnnotations() throws Exception {
    setBuildLanguageOptions(
        "--experimental_starlark_type_syntax",
        "--experimental_starlark_types_allowed_paths=//main");
    scratch.file(
        "test/foo.bzl",
        """
        def f(a: int):
          pass\
        """);
    scratch.file("test/BUILD", "load(':foo.bzl', 'f')");

    AssertionError e = assertThrows(AssertionError.class, () -> getTarget("//test:BUILD"));
    assertThat(e)
        .hasMessageThat()
        .contains(
            """
            syntax error at ':': type annotations are disallowed. Enable them with \
            --experimental_starlark_type_syntax and/or --experimental_starlark_types_allowed_paths.\
            """);
  }

  @Test
  public void starlarkTypesAllowedPath_externalPath_allowsTypeAnnotations() throws Exception {
    setBuildLanguageOptions(
        "--experimental_starlark_type_syntax",
        "--experimental_starlark_types_allowed_paths=@@r+//test");
    scratch.overwriteFile(
        "MODULE.bazel", "bazel_dep(name='r')", "local_path_override(module_name='r', path='/r')");
    scratch.file("/r/MODULE.bazel", "module(name='r')");
    scratch.file(
        "/r/test/foo.bzl",
        """
        def f(a: int):
          pass\
        """);
    scratch.file("/r/test/BUILD", "load(':foo.bzl', 'f')");

    // Required since we have a new MODULE.bazel file.
    invalidatePackages(true);
    getTarget("@@r+//test:BUILD");

    assertNoEvents();
  }

  @Test
  public void typeResolverDoesNotRunByDefault() throws Exception {
    // If the type resolver were running, it'd complain about the redefinition of g.
    //
    // We need a type annotation in f()'s signature to ensure its body is considered typed code.
    //
    // TODO: #27728 - Replace this test setup with a simple erroneous re-annotation of an previously
    // defined var, once validation of that is moved into the resolver.
    setBuildLanguageOptions("--experimental_starlark_type_syntax");
    scratch.file(
        "test/foo.bzl",
        """
        def f() -> None:
            def g():
                pass
            def g():
                pass
        """);
    scratch.file(
        "test/BUILD",
        """
        load(":foo.bzl", "f")
        """);

    getTarget("//test:BUILD");
    assertNoEvents();
  }

  @Test
  public void typeResolverDoesRunWithTypeCheckingFlag() throws Exception {
    setBuildLanguageOptions(
        "--experimental_starlark_type_syntax", "--experimental_starlark_type_checking");
    scratch.file(
        "test/foo.bzl",
        """
        def f() -> None:
            def g():
                pass
            def g():
                pass
        """);
    scratch.file(
        "test/BUILD",
        """
        load(":foo.bzl", "f")
        """);

    // TODO: #27728 -- Replace this exception/crash with an error on the StarlarkFile that we can
    // assert on as an ordinary Event.
    Exception ex = assertThrows(RuntimeException.class, () -> getTarget("//test:BUILD"));
    assertThat(ex)
        .hasCauseThat()
        .hasMessageThat()
        .contains(
            "Expected type of binding local[0] g @ /workspace/test/foo.bzl:2:9 to be null but was"
                + " Callable[[], Any]");
  }

  @Test
  public void dynamicTypeCheckingDoesNotRunByDefault() throws Exception {
    setBuildLanguageOptions("--experimental_starlark_type_syntax");
    scratch.file(
        "test/foo.bzl",
        """
        def f(x: int):
            pass
        """);
    scratch.file(
        "test/BUILD",
        """
        load(":foo.bzl", "f")
        f("abc")
        """);

    getTarget("//test:BUILD");
    assertNoEvents();
  }

  @Test
  public void dynamicTypeCheckingDoesRunWithTypeCheckingFlag() throws Exception {
    setBuildLanguageOptions(
        "--experimental_starlark_type_syntax", "--experimental_starlark_type_checking");
    scratch.file(
        "test/foo.bzl",
        """
        def f(x: int):
            pass
        """);
    scratch.file(
        "test/BUILD",
        """
        load(":foo.bzl", "f")
        f("abc")
        """);

    reporter.removeHandler(failFastHandler);
    getTarget("//test:BUILD");
    assertContainsEvent("in call to f(), parameter 'x' got value of type 'str', want 'int'");
  }
}
