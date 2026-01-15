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

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ObjectArrays;
import net.starlark.java.syntax.Resolver.Module;
import net.starlark.java.types.StarlarkType;
import net.starlark.java.types.Types;
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
    Module module = TestUtils.moduleWithUniversalTypes();
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
    TypeChecker.check(file);
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

  @Test
  public void infer_dot() throws Exception {
    // TODO: #27370 - Make this more interesting when we support struct types.

    assertTypeGivenDecls("o.f", Types.ANY, "o: Any");

    assertInvalid(
        ":2:2: 'n' of type 'int' does not have field 'f'",
        """
        n: int
        n.f
        """);
  }

  @Test
  public void assignment_dot() throws Exception {
    assertValid(
        """
        o: Any
        o.f = 123
        """);

    assertInvalid(
        ":2:2: 's' of type 'str' does not have field 'f'",
        """
        s: str
        s.f = 123
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
    StarlarkType numeric = Types.union(Types.INT, Types.FLOAT);

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
    assertTypeGivenDecls("-x", numeric, "x: int | float");

    assertTypeGivenDecls("+i", Types.INT, "i: int");
    assertTypeGivenDecls("+42", Types.INT);
    assertTypeGivenDecls("+x", Types.FLOAT, "x: float");
    assertTypeGivenDecls("+99.9", Types.FLOAT);
    assertTypeGivenDecls("+x", Types.ANY, "x: Any");
    assertTypeGivenDecls("+x", numeric, "x: int | float");

    assertTypeGivenDecls("~i", Types.INT, "i: int");
    assertTypeGivenDecls("~1", Types.INT);
    assertTypeGivenDecls("~x", Types.ANY, "x: Any");

    // Unsupported operations.
    assertInvalid(":1:1: operator '-' cannot be applied to type 'str'", "-'hello'");
    assertInvalid(":1:1: operator '+' cannot be applied to type 'str'", "+'hello'");
    assertInvalid(":1:1: operator '~' cannot be applied to type 'str'", "~'hello'");
    assertInvalid(":1:15: operator '-' cannot be applied to type 'str|int'", "x: str | int; -x");
  }
}
