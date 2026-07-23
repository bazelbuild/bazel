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

import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.starlarkbuildapi.core.StructApi;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for Starlark types. */
@RunWith(JUnit4.class)
public class StarlarkTypesTest extends BuildViewTestCase {

  @StarlarkBuiltin(name = "TestStructApiImpl")
  private static final class TestStructApiImpl implements StructApi {
    private final int answer;

    private TestStructApiImpl(int answer) {
      this.answer = answer;
    }

    @StarlarkMethod(name = "answer", doc = "A field", structField = true)
    public int answer() {
      return answer;
    }

    @StarlarkMethod(
        name = "plus",
        doc = "Not a field",
        parameters = {
          @Param(name = "other"),
        })
    public TestStructApiImpl plus(TestStructApiImpl other) {
      return new TestStructApiImpl(this.answer + other.answer);
    }

    @StarlarkMethod(
        name = "plus_or",
        doc = "Not a field",
        parameters = {
          @Param(name = "other"),
        })
    public StructApi abstractPlus(StructApi other) {
      if (other instanceof TestStructApiImpl otherTestStruct) {
        return new TestStructApiImpl(this.answer + otherTestStruct.answer);
      } else {
        return other;
      }
    }
  }

  @Override
  protected ConfiguredRuleClassProvider createRuleClassProvider() {
    ConfiguredRuleClassProvider.Builder builder = new ConfiguredRuleClassProvider.Builder();
    TestRuleClassProvider.addStandardRules(builder);
    builder.addBzlToplevel("test_struct_api_impl", new TestStructApiImpl(42));
    return builder.build();
  }

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

    checkLoadingPhaseError("//test:BUILD", "syntax error at ':': type annotations are disallowed");
    assertContainsEvent(
        "Type annotations syntax can be enabled with --experimental_starlark_type_syntax and/or"
            + " --experimental_starlark_types_allowed_paths.");
  }

  @Test
  public void experimentalStarlarkTypes_prohibitedInSclRegardlessOfFlag() throws Exception {
    setBuildLanguageOptions("--experimental_starlark_type_syntax");
    scratch.file(
        "test/foo.scl",
        """
        def f(a: int):
          pass\
        """);
    scratch.file("test/BUILD", "load(':foo.scl', 'f')");

    checkLoadingPhaseError("//test:BUILD", "syntax error at ':': type annotations are disallowed");
    assertContainsEvent("Type annotations are not permitted in .scl files.");
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

    checkLoadingPhaseError("//test:BUILD", "syntax error at ':': type annotations are disallowed");
    assertContainsEvent(
        "Type annotations syntax can be enabled with --experimental_starlark_type_syntax and/or"
            + " --experimental_starlark_types_allowed_paths.");
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
    // If the type resolver were running, it'd complain about the var annotation after x has already
    // been assigned to.
    setBuildLanguageOptions("--experimental_starlark_type_syntax");
    scratch.file(
        "test/foo.bzl",
        """
        def f():
            x = 1
            x : int
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
  public void typeResolverDoesRunWithDynamicTypeCheckingFlag() throws Exception {
    setBuildLanguageOptions(
        "--experimental_starlark_type_syntax", "--experimental_starlark_dynamic_type_checking");
    scratch.file(
        "test/foo.bzl",
        """
        def f():
            x = 1
            x : int
        """);
    scratch.file(
        "test/BUILD",
        """
        load(":foo.bzl", "f")
        """);

    checkLoadingPhaseError(
        "//test:BUILD", "type annotation on 'x' may only appear at its declaration");
  }

  @Test
  public void staticTypeCheckingDoesNotRunByDefault() throws Exception {
    setBuildLanguageOptions("--experimental_starlark_type_syntax");
    scratch.file(
        "test/foo.bzl",
        """
        x: int = "a"
        """);
    scratch.file(
        "test/BUILD",
        """
        load(":foo.bzl", "x")
        """);

    getTarget("//test:BUILD");
    assertNoEvents();
  }

  @Test
  public void staticTypeCheckingDoesRunWithStaticTypeCheckingFlag() throws Exception {
    setBuildLanguageOptions(
        "--experimental_starlark_type_syntax", "--experimental_starlark_static_type_checking");
    scratch.file(
        "test/foo.bzl",
        """
        x: int = "a"
        """);
    scratch.file(
        "test/BUILD",
        """
        load(":foo.bzl", "x")
        """);

    checkLoadingPhaseError("//test:BUILD", "cannot assign type 'str' to 'x' of type 'int'");
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
  public void dynamicTypeCheckingDoesRunWithDynamicTypeCheckingFlag() throws Exception {
    setBuildLanguageOptions(
        "--experimental_starlark_type_syntax", "--experimental_starlark_dynamic_type_checking");
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

  @Test
  public void structConstructor_typedAsReturningAnyStruct() throws Exception {
    setBuildLanguageOptions(
        "--experimental_starlark_type_syntax", "--experimental_starlark_static_type_checking");

    scratch.file(
        "good/good.bzl",
        """
        def f(s: struct[{"x": int}]):
            return s.x + 1

        good = f(struct(x = 1))
        """);
    scratch.file("good/BUILD", "load('good.bzl', 'good')");
    getConfiguredTarget("//good:BUILD");
    assertNoEvents();

    scratch.file("bad/bad.bzl", "bad: int = struct(x = 1)");
    scratch.file("bad/BUILD", "load('bad.bzl', 'bad')");
    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//bad:BUILD");
    assertContainsEvent("cannot assign type 'struct' to 'bad' of type 'int'");
  }

  @Test
  public void structValue_typeNarrowedOnExport() throws Exception {
    setBuildLanguageOptions(
        "--experimental_starlark_type_syntax", "--experimental_starlark_static_type_checking");

    scratch.file("lib/lib.bzl", "value = struct(x = 1, y = 2)");
    scratch.file("lib/BUILD", "load('lib.bzl', 'value')");

    scratch.file(
        "good/good.bzl",
        """
        load("//lib:lib.bzl", "value")
        good: struct[{"x": int}] = value
        """);
    scratch.file("good/BUILD", "load('good.bzl', 'good')");
    getConfiguredTarget("//good:BUILD");
    assertNoEvents();

    scratch.file(
        "bad/bad.bzl",
        """
        load("//lib:lib.bzl", "value")
        bad: struct[{"y": float}] = value
        """);
    scratch.file("bad/BUILD", "load('bad.bzl', 'bad')");
    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//bad:BUILD");
    assertContainsEvent(
        "cannot assign type 'struct[{\"x\": int, \"y\": int}]' to 'bad' of type 'struct[{\"y\":"
            + " float}]'");
  }

  @Test
  public void structApiImplementations_assignableToStructType() throws Exception {
    setBuildLanguageOptions(
        "--experimental_starlark_type_syntax", "--experimental_starlark_static_type_checking");

    scratch.file(
        "good/good.bzl",
        """
        good: struct[{"answer": int}] = test_struct_api_impl.plus(test_struct_api_impl)
        """);
    scratch.file("good/BUILD", "load('good.bzl', 'good')");
    getConfiguredTarget("//good:BUILD");
    assertNoEvents();

    scratch.file(
        "bad/bad.bzl",
        """
        bad: struct[{"answer": float}] = test_struct_api_impl.plus(test_struct_api_impl)
        """);
    scratch.file("bad/BUILD", "load('bad.bzl', 'bad')");
    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//bad:BUILD");
    assertContainsEvent(
        "cannot assign type 'TestStructApiImpl' to 'bad' of type 'struct[{\"answer\": float}]'");
  }

  @Test
  public void structApiItself_isAnyStruct() throws Exception {
    setBuildLanguageOptions(
        "--experimental_starlark_type_syntax", "--experimental_starlark_static_type_checking");

    scratch.file(
        "good/good.bzl",
        """
        arg: struct[{"foo": int}] = struct(foo = 1)
        good: struct[{"bar": float}] = test_struct_api_impl.plus_or(arg)
        """);
    scratch.file("good/BUILD", "load('good.bzl', 'good')");
    getConfiguredTarget("//good:BUILD");
    assertNoEvents();

    scratch.file(
        "bad/bad.bzl",
        """
        bad: int = test_struct_api_impl.plus_or(struct(baz = "baz"))
        """);
    scratch.file("bad/BUILD", "load('bad.bzl', 'bad')");
    reporter.removeHandler(failFastHandler);
    getConfiguredTarget("//bad:BUILD");
    assertContainsEvent("cannot assign type 'struct' to 'bad' of type 'int'");
  }

  private void assertTypeConstructorUsable(String typeExpr, String valueExpr) throws Exception {
    setBuildLanguageOptions(
        "--experimental_starlark_type_syntax", "--experimental_starlark_static_type_checking");

    scratch.file("lib/lib.bzl", "value = " + valueExpr);
    scratch.file("lib/BUILD");

    scratch.file(
        "good/good.bzl",
        String.format(
            """
            load("//lib:lib.bzl", "value")
            good: %s = value
            """,
            typeExpr));
    scratch.file("good/BUILD", "load('good.bzl', 'good')");
    assertThat(getConfiguredTarget("//good:BUILD")).isNotNull();
    assertNoEvents();

    scratch.file(
        "bad/bad.bzl",
        """
        load("//lib:lib.bzl", "value")
        bad: None = value
        """);
    scratch.file("bad/BUILD", "load('bad.bzl', 'bad')");
    checkLoadingPhaseError(
        "//bad:BUILD", String.format("cannot assign type '%s' to 'bad' of type 'None'", typeExpr));
  }

  @Test
  public void autogeneratedTypeConstructors_usable() throws Exception {
    assertTypeConstructorUsable("Label", "Label('//foo:bar')");
  }
}
