// Copyright 2025 The Bazel Authors. All Rights Reserved.
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

package net.starlark.java.syntax;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;
import static net.starlark.java.syntax.TestUtils.assertContainsError;
import static org.junit.Assert.assertThrows;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ObjectArrays;
import net.starlark.java.syntax.Resolver.Module;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class TypeCheckerTest {

  private final FileOptions.Builder options =
      FileOptions.builder()
          .allowTypeSyntax(true)
          .resolveTypeSyntax(true)
          // This lets us construct simpler test cases without wrapper `def` statements.
          .allowToplevelRebinding(true);

  private Module module = TestUtils.Module.withUniversalTypes();

  /**
   * Throws {@link AssertionError} if a file has errors, with an exception message that includes
   * {@code what} and the errors.
   */
  private void assertNoErrors(String what, StarlarkFile file) {
    if (!file.ok()) {
      throw new AssertionError(
          String.format("Unexpected errors: %s:\n%s", what, Joiner.on("\n").join(file.errors())));
    }
  }

  /**
   * Parses, resolve, and type-resolves a file, without typechecking it.
   *
   * <p>Returns a file without errors or else asserts failure.
   */
  private StarlarkFile prepareFile(String... lines) throws Exception {
    ParserInput input = ParserInput.fromLines(lines);
    StarlarkFile file = StarlarkFile.parse(input, options.build());
    assertNoErrors("parsing", file);
    Resolver.resolveFile(file, module);
    assertNoErrors("resolving", file);
    TypeTagger.tagFile(file, module);
    assertNoErrors("type-resolving", file);
    return file;
  }

  /**
   * Statically typechecks a program.
   *
   * <p>Asserts that steps before typechecking succeeded, but the typechecking itself may fail. The
   * resulting errors are available in the returned {@code StarlarkFile}.
   */
  private StarlarkFile typecheckFilePossiblyFailing(String... lines) throws Exception {
    StarlarkFile file = prepareFile(lines);
    TypeChecker.checkFile(file);
    return file;
  }

  /** As in {@link #typecheckFilePossiblyFailing} but asserts that even type checking succeeded. */
  private StarlarkFile assertValid(String... lines) throws Exception {
    StarlarkFile file = typecheckFilePossiblyFailing(lines);
    assertThat(file.ok()).isTrue();
    return file;
  }

  /** Asserts that type checking fails with at least the specified error. */
  private void assertInvalid(String expectedError, String... lines) throws Exception {
    StarlarkFile file = typecheckFilePossiblyFailing(lines);
    assertWithMessage("type checking suceeded unexpectedly").that(file.ok()).isFalse();
    assertContainsError(file.errors(), expectedError);
  }

  /**
   * Returns the inferred type of an expression, given zero or more {@code var} declarations for
   * identifiers appearing within the expression.
   */
  private StarlarkType inferTypeGivenDecls(String expr, String... decls) throws Exception {
    StarlarkFile file = prepareFile(ObjectArrays.concat(decls, expr));
    var resolvedExpr = ((ExpressionStatement) file.getStatements().getLast()).getExpression();
    return TypeChecker.inferTypeOf(resolvedExpr);
  }

  /**
   * Asserts that the inferred type of an expression is equal to the expected type, given zero or
   * more {@code var} declarations for identifiers appearing within the expression.
   */
  private void assertTypeGivenDecls(String expr, StarlarkType expected, String... decls)
      throws Exception {
    StarlarkType actual = inferTypeGivenDecls(expr, decls);
    assertThat(actual).isEqualTo(expected);
  }

  @Test
  public void staticTypeCheckingFlagRequirements() {
    var builder =
        FileOptions.builder()
            .staticTypeChecking(true)
            .resolveTypeSyntax(false)
            .tolerateInvalidTypeExpressions(false);
    assertThat(assertThrows(IllegalArgumentException.class, builder::build))
        .hasMessageThat()
        .contains("staticTypeChecking requires that resolveTypeSyntax is set");

    builder =
        FileOptions.builder()
            .staticTypeChecking(true)
            .resolveTypeSyntax(true)
            .tolerateInvalidTypeExpressions(true);
    assertThat(assertThrows(IllegalArgumentException.class, builder::build))
        .hasMessageThat()
        .contains("staticTypeChecking requires that tolerateInvalidTypeExpressions is not set");
  }

  @Test
  public void infer_identifier() throws Exception {
    assertTypeGivenDecls("x", Types.INT, "x: int");
  }

  // TODO: #27370 - The real behavior we want is that an unannotated variable has an inferred type
  // if it is a non-parameter local variable in typed code, and Any type otherwise.
  @Test
  public void unannotatedVarIsAnyType() throws Exception {
    assertTypeGivenDecls("x", Types.ANY, "x = 'ignored'");
  }

  @Test
  public void infer_literals() throws Exception {
    assertTypeGivenDecls("'abc'", Types.STR);
    assertTypeGivenDecls("123", Types.INT);
    assertTypeGivenDecls("1.0", Types.FLOAT);
  }

  // TODO: #27728 - We should add a test that the types of universals, and in particular the
  // keyword-like symbols `None`, `True`, and `False`, are appropriately inferred to have types
  // None, bool, and bool respectively. This test would have to live in the eval package, since the
  // universal environment is not available to the syntax/ package.

  @Test
  public void assignment_simple() throws Exception {
    assertValid(
        """
        n: int = 123
        """);

    assertInvalid(
        ":1:1: cannot assign type 'str' to 'n' of type 'int'",
        """
        n: int = "abc"
        """);
  }

  @Test
  public void canTolerateIrrelevantStatementTypes() throws Exception {
    assertValid(
        """
        load("...", "B")
        type A = B
        B  # expression statement
        """);
    // TODO: #28037 - Check break/continue, once we support for and def statements
  }

  /** A dummy type having a single field 'f' of type int. */
  private static final class FooType extends StarlarkType {
    @Override
    public StarlarkType getField(String name) {
      return name.equals("f") ? Types.INT : null;
    }

    @Override
    public boolean equals(Object obj) {
      return obj != null && obj.getClass().equals(this.getClass());
    }

    @Override
    public int hashCode() {
      return this.getClass().hashCode();
    }

    @Override
    public String toString() {
      return "Foo";
    }
  }

  private final Module fooModule =
      TestUtils.Module.withUniversalTypesAnd("Foo", Types.wrapType("Foo", new FooType()));

  @Test
  public void infer_dot() throws Exception {
    module = fooModule;

    assertTypeGivenDecls("o.f", Types.INT, "o: Foo");
    assertTypeGivenDecls("o.f", Types.ANY, "o: Any");

    assertInvalid(
        ":2:2: 'n' of type 'int' does not have field 'f'",
        """
        n: int
        n.f
        """);
    assertInvalid(
        ":2:2: 'o' of type 'Foo' does not have field 'g'",
        """
        o: Foo
        o.g
        """);
  }

  @Test
  public void assignment_dot() throws Exception {
    module = fooModule;

    assertValid(
        """
        o1: Foo
        o1.f = 123

        o2: Any
        o2.f = 123
        """);

    assertInvalid(
        ":2:2: 's' of type 'str' does not have field 'f'",
        """
        s: str
        s.f = 123
        """);

    assertInvalid(
        ":2:1: cannot assign type 'str' to 'o.f' of type 'int'",
        """
        o: Foo
        o.f = 'abc'
        """);
    assertInvalid(
        ":2:2: 'o' of type 'Foo' does not have field 'g'",
        """
        o: Foo
        o.g = 123
        """);
  }

  @Test
  public void infer_index_nonIndexable() throws Exception {
    assertInvalid(
        ":2:2: cannot index 'n' of type 'int'",
        """
        n: int
        n["abc"]
        """);

    // Any doesn't save us from doing a bad operation on a non-Any type.
    assertInvalid(
        ":3:2: cannot index 'n' of type 'int'",
        """
        n: int
        a: Any
        n[a]
        """);
  }

  @Test
  public void assignment_index_nonIndexable() throws Exception {
    assertInvalid(
        ":2:2: cannot index 'n' of type 'int'",
        """
        n: int
        n["abc"] = 123
        """);
  }

  @Test
  public void infer_index_any() throws Exception {
    assertTypeGivenDecls("a[123]", Types.ANY, "a: Any");
  }

  @Test
  public void assign_index_any() throws Exception {
    assertValid(
        """
        a: Any
        a["abc"] = 123
        """);
  }

  @Test
  public void infer_index_dict() throws Exception {
    // Exact key type match.
    assertTypeGivenDecls("d['abc']", Types.INT, "d: dict[str, int]");
    // Match based on subtyping.
    assertTypeGivenDecls("d[s]", Types.INT, "d: dict[object, int]; s: str");
    // Bypass key type constraint using Any.
    assertTypeGivenDecls("d[a]", Types.INT, "d: dict[str, int]; a: Any");

    assertInvalid(
        ":2:2: 'd' of type 'dict[str, int]' requires key type 'str', but got 'int'",
        """
        d: dict[str, int]
        d[123]
        """);
  }

  @Test
  public void assignment_index_dict() throws Exception {
    assertValid(
        """
        # Exact match.
        d1: dict[str, int]
        d1["abc"] = 123

        # Subtyping match.
        d2: dict[object, int]
        d2["abc"] = 123

        # Any match.
        a: Any
        d1["abc"] = a
        """);

    assertInvalid(
        """
        :2:1: cannot assign type 'str' to 'd["abc"]' of type 'int'\
        """,
        """
        d: dict[str, int]
        d["abc"] = "abc"
        """);

    // This failure is through the infer() code path, also exercised in the test case above.
    assertInvalid(
        """
        :2:2: 'd' of type 'dict[str, int]' requires key type 'str', but got 'int'\
        """,
        """
        d: dict[str, int]
        d[123] = 123
        """);
  }

  @Test
  public void infer_index_list() throws Exception {
    assertTypeGivenDecls("arr[123]", Types.STR, "arr: list[str]");

    assertTypeGivenDecls("arr[a]", Types.STR, "arr: list[str]; a: Any");

    assertInvalid(
        ":2:4: 'arr' of type 'list[str]' must be indexed by an integer, but got 'str'",
        """
        arr: list[str]
        arr["abc"]
        """);
  }

  @Test
  public void assignment_index_list() throws Exception {
    assertValid(
        """
        # Normal case.
        arr: list[str]
        arr[123] = "abc"

        # Any as index.
        a: Any
        arr[a] = "abc"

        # Any as value.
        arr[123] = a
        """);

    assertInvalid(
        """
        :2:1: cannot assign type 'int' to 'arr[123]' of type 'str'\
        """,
        """
        arr: list[str]
        arr[123] = 456
        """);

    // This failure is through the infer() code path, also exercised in the test case above.
    assertInvalid(
        """
        :2:4: 'arr' of type 'list[str]' must be indexed by an integer, but got 'str'\
        """,
        """
        arr: list[str]
        arr["abc"] = "xyz"
        """);
  }

  @Test
  public void infer_index_str() throws Exception {
    assertTypeGivenDecls("s[123]", Types.STR, "s: str");

    assertTypeGivenDecls("s[a]", Types.STR, "s: str; a: Any");

    assertInvalid(
        ":2:2: 's' of type 'str' must be indexed by an integer, but got 'str'",
        """
        s: str
        s["abc"]
        """);
  }

  @Test
  public void assignment_index_str() throws Exception {
    // Strings are immutable, so any assignment to an index expression of a string will fail
    // dynamically. But it's not currently a static error, if the types are correct.
    // TODO: #28037 - Fail static type checking on assignments to immutable values.
    assertValid(
        """
        # Normal case.
        s: str
        s[123] = "abc"

        # Any as index.
        a: Any
        s[a] = "abc"

        # Any as value.
        s[123] = a
        """);

    assertInvalid(
        """
        :2:1: cannot assign type 'int' to 's[123]' of type 'str'\
        """,
        """
        s: str
        s[123] = 456
        """);

    // This failure is through the infer() code path, also exercised in the test case above.
    assertInvalid(
        """
        :2:2: 's' of type 'str' must be indexed by an integer, but got 'str'\
        """,
        """
        s: str
        s["abc"] = "xyz"
        """);
  }

  @Test
  public void infer_index_tuple() throws Exception {
    // Statically knowable index in-range.
    assertTypeGivenDecls("t[1]", Types.STR, "t: tuple[int, str, bool]");

    // Index can't be statically determined.
    StarlarkType unionType = Types.union(Types.INT, Types.STR, Types.BOOL);
    assertTypeGivenDecls("t[n]", unionType, "t: tuple[int, str, bool]; n: int");
    assertTypeGivenDecls("t[a]", unionType, "t: tuple[int, str, bool]; a: Any");
    // TODO: #28037 - Add negative indices here, once we support unary expressions.

    // Bad index type.
    assertInvalid(
        ":2:2: 't' of type 'tuple[int, str, bool]' must be indexed by an integer, but got 'str'",
        """
        t: tuple[int, str, bool]
        t["abc"]
        """);

    // Statically knowable index out-of-range.
    assertInvalid(
        ":2:2: 't' of type 'tuple[int, str, bool]' is indexed by integer 3, which is out-of-range",
        """
        t: tuple[int, str, bool]
        t[3]
        """);
  }

  @Test
  public void assignment_index_tuple() throws Exception {
    // Tuple mutation is illegal, but not currently a static error if there's no type mismatch.
    // TODO: #28037 - Fail static type checking on assignments to immutable values.
    assertValid(
        """
        # Normal case.
        t: tuple[int, str, bool]
        t[1] = "abc"

        # Any as index.
        # This is a particularly nonsensical assignment that nonetheless passes the checker.
        a: Any
        u: int | str | bool
        t[a] = u

        # Any as value.
        t[1] = a
        """);

    assertInvalid(
        """
        :2:1: cannot assign type 'int' to 't[1]' of type 'str'\
        """,
        """
        t: tuple[int, str, bool]
        t[1] = 123
        """);
  }

  @Test
  public void infer_dict() throws Exception {
    // Empty case.
    assertTypeGivenDecls("{}", Types.dict(Types.NEVER, Types.NEVER));

    // Homogeneous case.
    assertTypeGivenDecls("{'a': 1, 'b': 2}", Types.dict(Types.STR, Types.INT));

    // Heterogeneous case.
    StarlarkType unionType = Types.union(Types.STR, Types.INT);
    assertTypeGivenDecls("{'a': 'abc', 1: 123}", Types.dict(unionType, unionType));
  }

  @Test
  public void infer_list() throws Exception {
    // Empty case.
    assertTypeGivenDecls("[]", Types.list(Types.NEVER));

    // Homogeneous case.
    assertTypeGivenDecls("[1, 2, 3]", Types.list(Types.INT));

    // Heterogeneous case.
    StarlarkType unionType = Types.union(Types.INT, Types.STR);
    assertTypeGivenDecls("[1, 'a']", Types.list(unionType));
  }

  @Test
  public void infer_tuple() throws Exception {
    // Empty case.
    assertTypeGivenDecls("()", Types.tuple(ImmutableList.of()));

    // Homogeneous case.
    assertTypeGivenDecls(
        "(1, 2, 3)", Types.tuple(ImmutableList.of(Types.INT, Types.INT, Types.INT)));

    // Heterogeneous case.
    assertTypeGivenDecls("(1, 'a')", Types.tuple(ImmutableList.of(Types.INT, Types.STR)));
  }

  @Test
  public void infer_unary_operator() throws Exception {
    // NOT is always boolean.
    assertTypeGivenDecls("not x", Types.BOOL, "x: bool");
    assertTypeGivenDecls("not x", Types.BOOL, "x: Any");
    assertTypeGivenDecls("not x", Types.BOOL, "x: list[int] | str");

    // The remaining unary operators preserve the type of their operand.
    assertTypeGivenDecls("-i", Types.INT, "i: int");
    assertTypeGivenDecls("-42", Types.INT);
    assertTypeGivenDecls("-x", Types.FLOAT, "x: float");
    assertTypeGivenDecls("-99.9", Types.FLOAT);
    assertTypeGivenDecls("-x", Types.INT, "x: int");
    assertTypeGivenDecls("-x", Types.ANY, "x: Any");
    assertTypeGivenDecls("-x", Types.NUMERIC, "x: int | float");

    assertTypeGivenDecls("+i", Types.INT, "i: int");
    assertTypeGivenDecls("+42", Types.INT);
    assertTypeGivenDecls("+x", Types.FLOAT, "x: float");
    assertTypeGivenDecls("+99.9", Types.FLOAT);
    assertTypeGivenDecls("+x", Types.ANY, "x: Any");
    assertTypeGivenDecls("+x", Types.NUMERIC, "x: int | float");

    assertTypeGivenDecls("~i", Types.INT, "i: int");
    assertTypeGivenDecls("~1", Types.INT);
    assertTypeGivenDecls("~x", Types.ANY, "x: Any");

    // Unsupported operations.
    assertInvalid(":1:1: operator '-' cannot be applied to type 'str'", "-'hello'");
    assertInvalid(":1:1: operator '+' cannot be applied to type 'str'", "+'hello'");
    assertInvalid(":1:1: operator '~' cannot be applied to type 'str'", "~'hello'");
    assertInvalid(":1:15: operator '-' cannot be applied to type 'str|int'", "x: str | int; -x");
  }

  @Test
  public void infer_and_or() throws Exception {
    assertTypeGivenDecls("x and y", Types.BOOL, "x: int; y: str");
    assertTypeGivenDecls("x or y", Types.BOOL, "x: int; y: str");
    assertTypeGivenDecls("x and y", Types.BOOL, "x: int | float; y: str | bool");
    assertTypeGivenDecls("x or y", Types.BOOL, "x: list[int]; y: list[str]");
  }

  @Test
  public void infer_equality() throws Exception {
    assertTypeGivenDecls("x == y", Types.BOOL, "x: int; y: str");
    assertTypeGivenDecls("x != y", Types.BOOL, "x: int; y: str");
    assertTypeGivenDecls("x == y", Types.BOOL, "x: int | float; y: str | bool");
    assertTypeGivenDecls("x != y", Types.BOOL, "x: int | float; y: str | bool");
    assertTypeGivenDecls("x == y", Types.BOOL, "x: int; y: Any");
    assertTypeGivenDecls("x != y", Types.BOOL, "x: Any; y: str");
    assertTypeGivenDecls("x == y", Types.BOOL, "x: Any; y: Any");
    assertTypeGivenDecls("x != y", Types.BOOL, "x: Any; y: Any");
  }

  @Test
  public void infer_comparison() throws Exception {
    assertTypeGivenDecls("x < y", Types.BOOL, "x: int; y: float");
    assertTypeGivenDecls("x >= y", Types.BOOL, "x: bool; y: bool");
    assertTypeGivenDecls("x <= y", Types.BOOL, "x: str; y: str");

    // Any inference
    assertTypeGivenDecls("x < y", Types.BOOL, "x: Any; y: Any");
    assertTypeGivenDecls("x >= y", Types.BOOL, "x: Any; y: int");
    assertTypeGivenDecls("x <= y", Types.BOOL, "x: str; y: Any");

    // Unions
    assertTypeGivenDecls("x < y", Types.BOOL, "x: int | float; y: float");
    assertTypeGivenDecls("x >= y", Types.BOOL, "x: int; y: int | float");
    assertTypeGivenDecls("x >= y", Types.BOOL, "x: Any; y: int | list[str]");
    assertTypeGivenDecls("x < y", Types.BOOL, "x: int | str; y: Any");

    // Compound types
    assertTypeGivenDecls("(1, 2) >= (3, 4)", Types.BOOL);
    assertTypeGivenDecls("x < y", Types.BOOL, "x: list[int]; y: list[int|float]");
    assertTypeGivenDecls("x <= y", Types.BOOL, "x: list[int|float]; y: list[float|int]");
    assertTypeGivenDecls("x > y", Types.BOOL, "x: tuple[str, int]; y: tuple[str]");
    assertTypeGivenDecls(
        "x <= y", Types.BOOL, "x: list[tuple[str, int]]; y: list[tuple[Any, float]]");
    // Lists of Never are always comparable to other lists
    assertTypeGivenDecls("[] < [1]", Types.BOOL);
    assertTypeGivenDecls("['a'] >= []", Types.BOOL);
    assertTypeGivenDecls("[] > []", Types.BOOL);

    // unsupported operations
    assertInvalid(":1:5: operator '<' cannot be applied to types 'str' and 'int'", "'0' < 1");
    assertInvalid(
        ":1:14: operator '>' cannot be applied to types 'float' and 'bool'", "x: bool; 0.0 > x");
    assertInvalid(
        ":1:10: operator '>=' cannot be applied to types 'dict[str, int]' and 'dict[str, int]'",
        "{'a': 1} >= {'b': 2}");
    assertInvalid(
        "operator '<' cannot be applied to types 'dict[str, int]' and 'Any'",
        "x: dict[str, int]; y: Any; x < y");
    assertInvalid(
        "operator '>=' cannot be applied to types 'Any' and 'dict[str, int]'",
        "x: Any; y: dict[str, int]; x >= y");
    // because lhs str is incomparable to rhs int (and vice versa)
    assertInvalid(
        "operator '<' cannot be applied to types 'int|str' and 'int|str'",
        "x: int | str; y: int | str; x < y");
    // Incomparable compound types
    assertInvalid(
        "operator '<' cannot be applied to types 'list[int|str]' and 'list[str]'",
        "x: list[int|str]; y: list[str]; x < y");
    assertInvalid(
        "operator '>=' cannot be applied to types 'tuple[int, str]' and 'tuple[str, int]'",
        "x: tuple[int, str]; y: tuple[str, int]; x >= y");
    assertInvalid(
        "operator '>=' cannot be applied to types 'list[tuple[str, int]]' and 'list[tuple[bool,"
            + " Any]]'",
        "x: list[tuple[str, int]]; y: list[tuple[bool, Any]]; x >= y");
  }

  @Test
  public void infer_plus_binary_operator() throws Exception {
    // numeric addition
    assertTypeGivenDecls("x + y", Types.INT, "x: int; y: int");
    assertTypeGivenDecls("x + y", Types.FLOAT, "x: int; y: float");
    assertTypeGivenDecls("x + y", Types.FLOAT, "x: float; y: int | float");

    // concatenation
    assertTypeGivenDecls("'hello' + 'world'", Types.STR);
    assertTypeGivenDecls("[] + []", Types.list(Types.NEVER));
    assertTypeGivenDecls("[] + [1]", Types.list(Types.INT));
    assertTypeGivenDecls("['hello'] + []", Types.list(Types.STR));
    assertTypeGivenDecls(
        "[1, 2.0] + [3, '4']", Types.list(Types.union(Types.INT, Types.FLOAT, Types.STR)));
    assertTypeGivenDecls(
        "x + y",
        Types.tuple(ImmutableList.of(Types.INT, Types.FLOAT, Types.INT, Types.STR)),
        "x: tuple[int, float]; y: tuple[int, str]");

    // Any inference
    assertTypeGivenDecls("x + y", Types.ANY, "x: Any; y: Any");
    // TODO: #28037 - the following cases can be tightened to int | float
    assertTypeGivenDecls("x + y", Types.ANY, "x: Any; y: int");
    assertTypeGivenDecls("x + y", Types.ANY, "x: int; y: Any");
    // TODO: #28037 - the following cases can be tightened to float
    assertTypeGivenDecls("x + y", Types.ANY, "x: Any; y: float");
    assertTypeGivenDecls("x + y", Types.ANY, "x: float; y: Any");
    // TODO: #28037 - the following cases can be tightened to str
    assertTypeGivenDecls("x + y", Types.ANY, "x: Any; y: str");
    assertTypeGivenDecls("x + y", Types.ANY, "x: str; y: Any");
    // TODO: #28037 - the following cases can be tightened to list[str]
    assertTypeGivenDecls("x + y", Types.ANY, "x: Any; y: list[str]");
    assertTypeGivenDecls("x + y", Types.ANY, "x: list[int]; y: Any");
    // TODO: #28037 - the following cases can be tightened to "tuple of indeterminable shape".
    assertTypeGivenDecls("x + y", Types.ANY, "x: Any; y: tuple[str]");
    assertTypeGivenDecls("x + y", Types.ANY, "x: tuple[int, int]; y: Any");
    // TODO: #28037 - the following can be tightened to int | float | str.
    assertTypeGivenDecls("x + y", Types.ANY, "x: Any; y: int | str");
    // TODO: #28037 - the following can be tightened to list[Any] | float
    assertTypeGivenDecls("x + y", Types.ANY, "x: list[int] | float; y: Any");
    // TODO: #28037 - the following cases should fail
    assertTypeGivenDecls("x + y", Types.ANY, "x: Any; y: bool");
    assertTypeGivenDecls("x + y", Types.ANY, "x: bool; y: Any");

    // unsupported operations
    assertInvalid(":1:9: operator '+' cannot be applied to types 'str' and 'int'", "'hello' + 1");
    assertInvalid(
        "operator '+' cannot be applied to types 'int|str' and 'str'", "x: int|str; y: str; x + y");
  }

  @Test
  public void infer_pipe_binary_operator() throws Exception {
    assertTypeGivenDecls("x | y", Types.INT, "x: int; y: int");
    assertTypeGivenDecls(
        "x | y",
        Types.dict(Types.union(Types.STR, Types.INT), Types.union(Types.BOOL, Types.FLOAT)),
        "x: dict[str, bool]; y: dict[int, float]");
    assertTypeGivenDecls(
        "x | y", Types.set(Types.union(Types.INT, Types.STR)), "x: set[int]; y: set[str]");
    // TODO: #28037 - add a test for a union with a set[Never] once we can construct empty sets in
    // test machinery.

    // Any inference
    assertTypeGivenDecls("x | y", Types.ANY, "x: Any; y: Any");
    // TODO: #28037 - the following cases can be tightened to int
    assertTypeGivenDecls("x | y", Types.ANY, "x: Any; y: int");
    assertTypeGivenDecls("x | y", Types.ANY, "x: int; y: Any");
    // TODO: #28037 - the following cases can be tightened to dict[Any, Any]
    assertTypeGivenDecls("x | y", Types.ANY, "x: Any; y: dict[str, int]");
    assertTypeGivenDecls("x | y", Types.ANY, "x: dict[str, int]; y: Any");
    // TODO: #28037 - the following cases can be tightened to set[Any]
    assertTypeGivenDecls("x | y", Types.ANY, "x: Any; y: set[int]");
    assertTypeGivenDecls("x | y", Types.ANY, "x: set[str]; y: Any");
    // TODO: #28037 - the following can be tightened to int | set[Any]
    assertTypeGivenDecls("x | y", Types.ANY, "x: Any; y: int | set[str]");
    // TODO: #28037 - the following can be tightened to dict[Any, Any] | set[Any]
    assertTypeGivenDecls("x | y", Types.ANY, "x: Any; y: dict[str, str] | set[str]");
    // TODO: #28037 - the following cases should fail
    assertTypeGivenDecls("x | y", Types.ANY, "x: Any; y: int | bool");
    assertTypeGivenDecls("x | y", Types.ANY, "x: int | bool; y: Any");

    // unsupported operations
    assertInvalid(":1:3: operator '|' cannot be applied to types 'int' and 'float'", "1 | 2.0");
    assertInvalid(
        "operator '|' cannot be applied to types 'int|set[int]' and 'int|set[int]'",
        "x: int|set[int]; y: int|set[int]; x | y");
  }

  @Test
  public void infer_ampersand_binary_operator() throws Exception {
    assertTypeGivenDecls("x & y", Types.INT, "x: int; y: int");
    // TODO: #28037 - tighter inference for set intersections.
    assertTypeGivenDecls(
        "x & y", Types.set(Types.union(Types.STR, Types.INT)), "x: set[str|int]; y: set[str|bool]");

    // Any inference
    assertTypeGivenDecls("x & y", Types.ANY, "x: Any; y: Any");
    // TODO: #28037 - the following cases can be tightened to int
    assertTypeGivenDecls("x & y", Types.ANY, "x: Any; y: int");
    assertTypeGivenDecls("x & y", Types.ANY, "x: int; y: Any");
    // TODO: #28037 - the following cases can be tightened to set[Any]
    assertTypeGivenDecls("x & y", Types.ANY, "x: Any; y: set[int]");
    assertTypeGivenDecls("x & y", Types.ANY, "x: set[str]; y: Any");

    // unsupported operations
    assertInvalid(":1:3: operator '&' cannot be applied to types 'int' and 'float'", "1 & 2.0");
    assertInvalid(
        "operator '&' cannot be applied to types 'int' and 'set[int]'",
        "x: int; y: set[int]; x & y");
  }

  @Test
  public void infer_caret_binary_operator() throws Exception {
    assertTypeGivenDecls("x ^ y", Types.INT, "x: int; y: int");
    assertTypeGivenDecls(
        "x ^ y",
        Types.set(Types.union(Types.STR, Types.INT, Types.BOOL)),
        "x: set[str|int]; y: set[str|bool]");

    // Any inference
    assertTypeGivenDecls("x ^ y", Types.ANY, "x: Any; y: Any");
    // TODO: #28037 - the following can be tightened to int
    assertTypeGivenDecls("x ^ y", Types.ANY, "x: Any; y: int");
    assertTypeGivenDecls("x ^ y", Types.ANY, "x: int; y: Any");
    // TODO: #28037 - the following can be tightened to set[Any]
    assertTypeGivenDecls("x ^ y", Types.ANY, "x: Any; y: set[int]");
    assertTypeGivenDecls("x ^ y", Types.ANY, "x: set[str]; y: Any");

    // unsupported operations
    assertInvalid(":1:3: operator '^' cannot be applied to types 'int' and 'float'", "1 ^ 2.0");
    assertInvalid(
        "operator '^' cannot be applied to types 'int' and 'set[int]'",
        "x: int; y: set[int]; x ^ y");
  }

  @Test
  public void infer_bitshift_binary_operators() throws Exception {
    assertTypeGivenDecls("x << y", Types.INT, "x: int; y: int");
    assertTypeGivenDecls("x >> y", Types.INT, "x: int; y: int");

    // Any inference
    assertTypeGivenDecls("x << y", Types.ANY, "x: Any; y: Any");
    // TODO: #28037 - can be tightened to int
    assertTypeGivenDecls("x >> y", Types.ANY, "x: Any; y: int");
    assertTypeGivenDecls("x << y", Types.ANY, "x: int; y: Any");
    // TODO: #28037 - the following should fail
    assertTypeGivenDecls("x << y", Types.ANY, "x: Any; y: bool");
    assertTypeGivenDecls("x >> y", Types.ANY, "x: bool; y: Any");

    // unsupported operations
    assertInvalid(":1:3: operator '<<' cannot be applied to types 'int' and 'float'", "1 << 2.0");
    assertInvalid(
        "operator '>>' cannot be applied to types 'bool' and 'int'", "x: bool; y: int; x >> y");
  }

  @Test
  public void infer_minus_binary_operator() throws Exception {
    assertTypeGivenDecls("x - y", Types.INT, "x: int; y: int");
    assertTypeGivenDecls("x - y", Types.FLOAT, "x: int; y: float");
    assertTypeGivenDecls("x - y", Types.FLOAT, "x: float; y: int | float");
    assertTypeGivenDecls(
        "x - y", Types.set(Types.union(Types.STR, Types.INT)), "x: set[str|int]; y: set[str|bool]");

    // Any inference
    assertTypeGivenDecls("x - y", Types.ANY, "x: Any; y: Any");
    // TODO: #28037 - the following cases can be tightened to int | float
    assertTypeGivenDecls("x - y", Types.ANY, "x: Any; y: int");
    assertTypeGivenDecls("x - y", Types.ANY, "x: int; y: Any");
    // TODO: #28037 - the following cases can be tightened to float
    assertTypeGivenDecls("x - y", Types.ANY, "x: Any; y: float");
    assertTypeGivenDecls("x - y", Types.ANY, "x: float; y: Any");
    // TODO: #28037 - the following cases can be tightened to set[Any]
    assertTypeGivenDecls("x - y", Types.ANY, "x: Any; y: set[int]");
    assertTypeGivenDecls("x - y", Types.ANY, "x: set[str]; y: Any");

    // unsupported operations
    assertInvalid(":1:5: operator '-' cannot be applied to types 'str' and 'int'", "'2' - 1");
    assertInvalid(
        "operator '-' cannot be applied to types 'int' and 'set[int]'",
        "x: int; y: set[int]; x - y");
  }

  @Test
  public void infer_star_binary_operator() throws Exception {
    // numeric multiplication
    assertTypeGivenDecls("x * y", Types.INT, "x: int; y: int");
    assertTypeGivenDecls("x * y", Types.FLOAT, "x: int; y: float");
    assertTypeGivenDecls("x * y", Types.FLOAT, "x: float; y: int | float");

    // string repetition
    assertTypeGivenDecls("'hello' * 2", Types.STR);
    assertTypeGivenDecls("2 * 'bye'", Types.STR);

    // list repetition
    assertTypeGivenDecls("[1, 2.0] * 2", Types.list(Types.union(Types.INT, Types.FLOAT)));
    assertTypeGivenDecls("2 * [1, 2.0]", Types.list(Types.union(Types.INT, Types.FLOAT)));
    // preserve list type even when the returned list is size 0
    assertTypeGivenDecls("[1, 2.0] * 0", Types.list(Types.union(Types.INT, Types.FLOAT)));
    assertTypeGivenDecls("0 * [1, 2.0]", Types.list(Types.union(Types.INT, Types.FLOAT)));
    assertTypeGivenDecls("x * y", Types.list(Types.INT), "x: int; y: list[int]");

    // tuple repetition
    assertTypeGivenDecls(
        "x * 2",
        Types.tuple(ImmutableList.of(Types.INT, Types.FLOAT, Types.INT, Types.FLOAT)),
        "x: tuple[int, float]");
    assertTypeGivenDecls(
        "2 * x",
        Types.tuple(ImmutableList.of(Types.INT, Types.FLOAT, Types.INT, Types.FLOAT)),
        "x: tuple[int, float]");
    assertTypeGivenDecls("x * 0", Types.tuple(ImmutableList.of()), "x: tuple[int, float]");
    assertTypeGivenDecls("0 * x", Types.tuple(ImmutableList.of()), "x: tuple[int, float]");
    // TODO: #27370 - the following case could be tightened to empty tuples.
    assertTypeGivenDecls("x * -1", Types.ANY, "x: tuple[int, float]");
    assertTypeGivenDecls("-1 * x", Types.ANY, "x: tuple[int, float]");
    // TODO: #28037 - the following case can be tightened to "tuple of indeterminable shape".
    assertTypeGivenDecls("x * y", Types.ANY, "x: int; y: tuple[int]");

    // Any inference
    assertTypeGivenDecls("x * y", Types.ANY, "x: Any; y: Any");
    // The next 2 cases are tricky - that `Any` could be numeric, str, list, or tuple!
    // TODO: #28037 - can be tightened to "int | float | str | list | tuple of any shape"
    assertTypeGivenDecls("x * y", Types.ANY, "x: Any; y: int");
    assertTypeGivenDecls("x * y", Types.ANY, "x: int; y: Any");
    // TODO: #28037 - the following cases can be tightened to float
    assertTypeGivenDecls("x * y", Types.ANY, "x: Any; y: float");
    assertTypeGivenDecls("x * y", Types.ANY, "x: float; y: Any");
    // TODO: #28037 - the following cases can be tightened to str
    assertTypeGivenDecls("x * y", Types.ANY, "x: Any; y: str");
    assertTypeGivenDecls("x * y", Types.ANY, "x: str; y: Any");
    // TODO: #28037 - can be tightened to list[str]
    assertTypeGivenDecls("x * y", Types.ANY, "x: Any; y: list[str]");
    // TODO: #28037 - can be tightened to list[int]
    assertTypeGivenDecls("x * y", Types.ANY, "x: list[int]; y: Any");
    // TODO: #28037 - the following cases can be tightened to "tuple of indeterminable shape".
    assertTypeGivenDecls("x * y", Types.ANY, "x: Any; y: tuple[str]");
    assertTypeGivenDecls("x * y", Types.ANY, "x: tuple[int, int]; y: Any");
    // TODO: #28037 - can be tightened to float | str
    assertTypeGivenDecls("x * y", Types.ANY, "x: Any; y: float|str");
    // TODO: #28037 - can be tightened to list[str] | str
    assertTypeGivenDecls("x * y", Types.ANY, "x: list[str]|str; y: Any");
    // TODO: #28037 - the following cases should fail
    assertTypeGivenDecls("x * y", Types.ANY, "x: Any; y: bool");
    assertTypeGivenDecls("x * y", Types.ANY, "x: bool; y: Any");

    // unsupported operations
    assertInvalid(
        ":1:9: operator '*' cannot be applied to types 'str' and 'float'", "'hello' * 1.0");
    assertInvalid(
        "operator '*' cannot be applied to types 'bool' and 'int'", "x: bool; y: int; x * y");
  }

  @Test
  public void infer_slash_binary_operator() throws Exception {
    assertTypeGivenDecls("x / y", Types.FLOAT, "x: int; y: int");
    assertTypeGivenDecls("x // y", Types.INT, "x: int; y: int");
    assertTypeGivenDecls("x / y", Types.FLOAT, "x: int; y: float");
    assertTypeGivenDecls("x // y", Types.FLOAT, "x: int; y: float");
    assertTypeGivenDecls("x / y", Types.FLOAT, "x: float; y: int | float");
    assertTypeGivenDecls("x // y", Types.FLOAT, "x: float; y: int | float");

    // Any inference
    assertTypeGivenDecls("x / y", Types.ANY, "x: Any; y: Any");
    assertTypeGivenDecls("x // y", Types.ANY, "x: Any; y: Any");
    // TODO: #28037 - can be tightened to float
    assertTypeGivenDecls("x / y", Types.ANY, "x: Any; y: int");
    // TODO: #28037 - can be tightened to int | float
    assertTypeGivenDecls("x // y", Types.ANY, "x: Any; y: int");
    // TODO: #28037 - can be tightened to float
    assertTypeGivenDecls("x / y", Types.ANY, "x: int; y: Any");
    // TODO: #28037 - can be tightened to int | float
    assertTypeGivenDecls("x // y", Types.ANY, "x: int; y: Any");
    // TODO: #28037 - the following can be tightened to float
    assertTypeGivenDecls("x / y", Types.ANY, "x: Any; y: float");
    assertTypeGivenDecls("x // y", Types.ANY, "x: Any; y: float");
    assertTypeGivenDecls("x / y", Types.ANY, "x: float; y: Any");
    assertTypeGivenDecls("x // y", Types.ANY, "x: float; y: Any");

    // unsupported operations
    assertInvalid("operator '/' cannot be applied to types 'int' and 'str'", "1 / '2'");
    assertInvalid("operator '//' cannot be applied to types 'str' and 'float'", "'2' // 3.0");
  }

  @Test
  public void infer_percent_binary_operator() throws Exception {
    // numeric modulo
    assertTypeGivenDecls("x % y", Types.INT, "x: int; y: int");
    assertTypeGivenDecls("x % y", Types.FLOAT, "x: int; y: float");
    assertTypeGivenDecls("x % y", Types.FLOAT, "x: float; y: int | float");
    // string substitution
    assertTypeGivenDecls("'hello %s' % 'world'", Types.STR);
    assertTypeGivenDecls("'hello %s %s' % (' ', 'world')", Types.STR);
    assertTypeGivenDecls("'the answer is %s' % x", Types.STR, "x: int");

    // Any inference
    // TODO: #28037 - can be tightened to int | float | str
    assertTypeGivenDecls("x % y", Types.ANY, "x: Any; y: Any");
    assertTypeGivenDecls("x % y", Types.ANY, "x: Any; y: int");
    // TODO: #28037 - can be tightened to int | float
    assertTypeGivenDecls("x % y", Types.ANY, "x: int; y: Any");
    // TODO: #28037 - can be tightened to float | str
    assertTypeGivenDecls("x % y", Types.ANY, "x: Any; y: float");
    // TODO: #28037 - can be tightened to float
    assertTypeGivenDecls("x % y", Types.ANY, "x: float; y: Any");
    // TODO: #28037 - can be tightened to str
    assertTypeGivenDecls("x % y", Types.ANY, "x: Any; y: str");
    assertTypeGivenDecls("x % y", Types.STR, "x: str; y: Any");

    // unsupported operations
    assertInvalid("operator '%' cannot be applied to types 'float' and 'str'", "1.0 % 'hello'");
  }

  @Test
  public void infer_in_binary_operator() throws Exception {
    // in Any
    assertTypeGivenDecls("x in y", Types.BOOL, "x: Any; y: Any");
    assertTypeGivenDecls("x not in y", Types.BOOL, "x: Any; y: Any");
    assertTypeGivenDecls("x in y", Types.BOOL, "x: bool; y: Any");
    assertTypeGivenDecls("x not in y", Types.BOOL, "x: list[int]; y: Any");

    // in str
    assertTypeGivenDecls("x in y", Types.BOOL, "x: str; y: str");
    assertTypeGivenDecls("x not in y", Types.BOOL, "x: str; y: str");
    assertTypeGivenDecls("x in y", Types.BOOL, "x: Any; y: str");
    assertTypeGivenDecls("x not in y", Types.BOOL, "x: Any; y: str");

    // in collections (type of lhs doesn't need to match collection's type)
    assertTypeGivenDecls("x in y", Types.BOOL, "x: str; y: list[bool]");
    assertTypeGivenDecls("x not in y", Types.BOOL, "x: str; y: tuple[str]");
    assertTypeGivenDecls("x in y", Types.BOOL, "x: str; y: dict[str, int]");
    assertTypeGivenDecls("x not in y", Types.BOOL, "x: str; y: set[int]");
    assertTypeGivenDecls("x in y", Types.BOOL, "x: bool; y: Any");
    assertTypeGivenDecls("x not in y", Types.BOOL, "x: list[int]; y: Any");
    assertTypeGivenDecls("x in y", Types.BOOL, "x: Any; y: list[int|float]");
    assertTypeGivenDecls("x not in y", Types.BOOL, "x: Any; y: set[str]");

    // unsupported operations
    assertInvalid("operator 'in' cannot be applied to types 'Any' and 'int'", "x: Any; x in 42");
    assertInvalid(
        "operator 'not in' cannot be applied to types 'list[str]' and 'str'",
        "['e'] not in 'hello'");
  }
}
