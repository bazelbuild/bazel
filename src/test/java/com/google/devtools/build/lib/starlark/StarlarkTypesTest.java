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
import static java.util.stream.Collectors.joining;

import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.packages.SelectorValue;
import com.google.devtools.build.lib.starlarkbuildapi.core.StructApi;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import com.google.testing.junit.testparameterinjector.TestParameters;
import java.util.function.Predicate;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.syntax.StarlarkType;
import net.starlark.java.syntax.TokenKind;
import net.starlark.java.syntax.Types;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Tests for Starlark types. */
@RunWith(TestParameterInjector.class)
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
    // TODO: #30499 - we force a type error to reveal the type of the RHS of the assignment; replace
    // with `reveal_type` when we have it.
    checkLoadingPhaseError(
        "//bad:BUILD", String.format("cannot assign type '%s' to 'bad' of type 'None'", typeExpr));
  }

  @Test
  public void autogeneratedTypeConstructors_usable() throws Exception {
    assertTypeConstructorUsable("Label", "Label('//foo:bar')");
  }

  @Test
  public void select_basicUsage() throws Exception {
    setBuildLanguageOptions(
        "--experimental_starlark_type_syntax", "--experimental_starlark_static_type_checking");

    scratch.file(
        "lib/select.bzl",
        """
        select_of_int = select({"//cfg": 1, "//conditions:default": 0})
        select_of_str = select({"//cfg": "foo", "//conditions:default": "bar"})
        select_of_empty_list = select({"//cfg": [], "//conditions:default": []})
        select_of_str_list = select({"//cfg": ["a", "b"], "//conditions:default": ["c"]})
        select_of_potentially_empty_label_list = select({"//cfg": [Label("//foo:a")], "//conditions:default": []})
        select_of_potentially_empty_dict = select({"//cfg": {"a": Label("//foo:a")}, "//conditions:default": {}})
        select_of_label_or_none = select({"//cfg": Label("//foo:a"), "//conditions:default": None})
        """);
    scratch.file("lib/BUILD");

    scratch.file(
        "good/good.bzl",
        """
        load(
            "//lib:select.bzl",
            "select_of_int",
            "select_of_str",
            "select_of_empty_list",
            "select_of_str_list",
            "select_of_potentially_empty_label_list",
            "select_of_potentially_empty_dict",
            "select_of_label_or_none",
        )
        good: select[int] = select_of_int + select_of_int
        good2: select[str] = select_of_str + "hello"
        good3: select[list[str]] = ["hello"] + select_of_empty_list + select_of_str_list + []
        good4: select[list[Label]] = [] + select_of_potentially_empty_label_list + [Label("//x")]
        good5: select[dict[str, Label]] = select_of_potentially_empty_dict | {"b": Label("//foo:b")}
        good6: select[Label] = select_of_label_or_none
        """);
    scratch.file("good/BUILD", "load('good.bzl', 'good')");
    getConfiguredTarget("//good:BUILD");
    assertNoEvents();

    scratch.file(
        "bad/bad.bzl",
        """
        load("//lib:select.bzl", "select_of_str")
        bad: str = "foo" + select_of_str
        """);
    scratch.file("bad/BUILD", "load('bad.bzl', 'bad')");
    // TODO: #30499 - we force a type error to reveal the type of the RHS of the assignment; replace
    // with `reveal_type` when we have it.
    checkLoadingPhaseError(
        "//bad:BUILD", "cannot assign type 'select[str]' to 'bad' of type 'str'");
  }

  @Test
  @TestParameters({
    "{x: select_of_int, op: '+', y: 1.5, result: 'select[float]'}",
    "{x: '\"hello\"', op: '+', y: select_of_str, result: 'select[str]'}",
    "{x: select_of_list_of_str, op: '+', y: select_of_list_of_label, result:"
        + " 'select[list[str|Label]]'}",
    "{x: select_of_dict_of_str, op: '|', y: '{\"y\": Label(\"//foo\")}', "
        + "result: 'select[dict[str, str|Label]]'}",
  })
  public void select_validBinaryOperator(String x, String op, String y, String result)
      throws Exception {
    setBuildLanguageOptions(
        "--experimental_starlark_type_syntax", "--experimental_starlark_static_type_checking");
    scratch.file(
        "lib/lib.bzl",
        """
        select_of_int = select({"//cfg": 1, "//conditions:default": 0})
        select_of_str = select({"//cfg": "foo", "//conditions:default": "bar"})
        select_of_list_of_str = select({"//cfg": ["foo"], "//conditions:default": ["bar"]})
        select_of_list_of_label = select({"//cfg": [Label("//a")], "//conditions:default": [Label("//b")]})
        select_of_dict_of_str = select({"//cfg": {"a": "A"}, "//conditions:default": None})
        """);
    scratch.file("lib/BUILD");
    scratch.file(
        "bad/bad.bzl",
        String.format(
            """
            load("//lib:lib.bzl", %s)
            bad: None = %s %s %s
            """,
            ImmutableList.of(x, y).stream()
                .filter(n -> n.startsWith("select_of_"))
                .map(s -> String.format("'%s'", s))
                .collect(joining(", ")),
            x,
            op,
            y));
    scratch.file("bad/BUILD", "load('bad.bzl', 'bad')");
    // TODO: #30499 - we force a type error to reveal the type of the RHS of the assignment; replace
    // with `reveal_type` when we have it.
    checkLoadingPhaseError(
        "//bad:BUILD", String.format("cannot assign type '%s' to 'bad' of type 'None'", result));
  }

  @Test
  @TestParameters({
    "{x: 'select_of_int', op: '+', y: 'select_of_str', "
        + "error: \"'+' cannot be applied to types 'select[int]' and 'select[str]'\"}",
    "{x: 'select_of_str', op: '+', y: 'select_of_list_of_str', "
        + "error: \"'+' cannot be applied to types 'select[str]' and 'select[list[str]]'\"}",
    "{x: 'select_of_list_of_str', op: '|', y: 'select_of_dict', error: \"'|' cannot be applied to"
        + " types 'select[list[str]]' and 'select[dict[str, int]]'\"}"
  })
  public void select_invalidBinaryOperator(String x, String op, String y, String error)
      throws Exception {
    setBuildLanguageOptions(
        "--experimental_starlark_type_syntax", "--experimental_starlark_static_type_checking");
    scratch.file(
        "lib/lib.bzl",
        """
        select_of_int = select({"//cfg": 1, "//conditions:default": 0})
        select_of_str = select({"//cfg": "foo", "//conditions:default": "bar"})
        select_of_list_of_str = select({"//cfg": ["foo"], "//conditions:default": ["bar"]})
        select_of_dict = select({"//cfg": {"a": 1}, "//conditions:default": None})
        """);
    scratch.file("lib/BUILD");
    scratch.file(
        "bad/bad.bzl",
        String.format(
            """
            load("//lib:lib.bzl", "%s", "%s")
            bad = %s %s %s
            _: None = None  # enable type syntax
            """,
            x, y, x, op, y));
    scratch.file("bad/BUILD", "load('bad.bzl', 'bad')");
    checkLoadingPhaseError("//bad:BUILD", error);
  }

  /** Fake type for testing; returns itself when added to the given type on the given side. */
  private static final class AdditiveType extends StarlarkType {
    private final String name;
    private final Predicate<StarlarkType> addsTo;
    private final StarlarkType result;
    private final boolean thisLeft;

    AdditiveType(
        String name, Predicate<StarlarkType> addsTo, StarlarkType result, boolean thisLeft) {
      this.name = name;
      this.addsTo = addsTo;
      this.result = result;
      this.thisLeft = thisLeft;
    }

    @Override
    public String toString() {
      return name;
    }

    @Override
    public StarlarkType inferBinaryOperator(
        TokenKind operator, StarlarkType that, boolean thisLeft) {
      if (operator == TokenKind.PLUS && addsTo.test(that) && thisLeft == this.thisLeft) {
        return result;
      }
      return null;
    }
  }

  /**
   * Verifies that {@link StarlarkType#inferBinaryOperator} produces the given result when called
   * with selector types of the given arguments, or with the selector type of one argument and a
   * plain type of the other.
   */
  private void assetSelectBinaryOperator(
      StarlarkType lhsArgToSelec,
      TokenKind operator,
      StarlarkType rhsArgToSelect,
      StarlarkType expectedArgToSelect) {
    assertThat(
            StarlarkType.inferBinaryOperator(
                SelectorValue.Type.of(lhsArgToSelec),
                operator,
                SelectorValue.Type.of(rhsArgToSelect)))
        .isEqualTo(SelectorValue.Type.of(expectedArgToSelect));
    assertThat(
            StarlarkType.inferBinaryOperator(
                SelectorValue.Type.of(lhsArgToSelec), operator, rhsArgToSelect))
        .isEqualTo(SelectorValue.Type.of(expectedArgToSelect));
    assertThat(
            StarlarkType.inferBinaryOperator(
                lhsArgToSelec, operator, SelectorValue.Type.of(rhsArgToSelect)))
        .isEqualTo(SelectorValue.Type.of(expectedArgToSelect));
  }

  @Test
  public void select_inferBinaryOperator() throws Exception {
    // Basic cases
    assertThat(
            SelectorValue.Type.of(Types.INT).inferBinaryOperator(TokenKind.PLUS, Types.INT, true))
        .isEqualTo(SelectorValue.Type.of(Types.INT));
    assertThat(
            SelectorValue.Type.of(Types.FLOAT)
                .inferBinaryOperator(TokenKind.PLUS, Types.INT, false))
        .isEqualTo(SelectorValue.Type.of(Types.FLOAT));
    assertThat(
            SelectorValue.Type.of(Types.list(Types.INT))
                .inferBinaryOperator(TokenKind.PLUS, Types.list(Types.STR), true))
        .isEqualTo(SelectorValue.Type.of(Types.list(Types.union(Types.INT, Types.STR))));

    // Distinct types which add to str on the left, producing int and float respectively.
    AdditiveType l1 = new AdditiveType("l1", Predicates.equalTo(Types.STR), Types.INT, true);
    AdditiveType l2 = new AdditiveType("l2", Predicates.equalTo(Types.STR), Types.FLOAT, true);
    StarlarkType l1or2 = Types.union(l1, l2);
    assetSelectBinaryOperator(l1or2, TokenKind.PLUS, Types.STR, Types.NUMERIC);

    // Type which adds to l1 or l2 on the right, producing bool.
    AdditiveType r1 = new AdditiveType("r1", t -> t.equals(l1) || t.equals(l2), Types.BOOL, false);
    StarlarkType r1orStr = Types.union(r1, Types.STR);
    StarlarkType boolOrNumeric = Types.union(Types.BOOL, Types.NUMERIC);
    assetSelectBinaryOperator(l1or2, TokenKind.PLUS, r1orStr, boolOrNumeric);
  }

  @Test
  public void select_staticAndDynamicBinaryOperatorConsistency() throws Exception {
    setBuildLanguageOptions(
        "--experimental_starlark_type_syntax", "--experimental_starlark_static_type_checking");
  }
}
