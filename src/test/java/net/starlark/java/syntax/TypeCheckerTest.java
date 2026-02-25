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
    assertThat(file.errors()).isEmpty();
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

    assertInvalid(
        ":2:1: cannot assign type 'str' to 'n' of type 'int'",
        """
        n: int
        n = "abc"
        """);
  }

  @Test
  public void infer_assignment() throws Exception {
    assertValid(
        """
        n = 123
        n + 456
        x: bool  # ensure toplevel code is type-checked
        """);

    assertValid(
        """
        n = 123
        n + "456"  # not a static type error in untyped code
        """);

    assertInvalid(
        "operator '+' cannot be applied to types 'int' and 'str'",
        """
        n = 123
        n + "456"
        x: bool  # ensure toplevel code is type-checked
        """);

    assertInvalid(
        "cannot assign type 'str' to 'n' of type 'int'",
        """
        n = 123
        n = "456"  # subsequent assignments do not change the type
        x: bool    # ensure toplevel code is type-checked
        """);

    // TODO: #28037 - in mypy, this is an error (attempt to use a variable of unknown type, since
    // the assignment is lexically below first use). We should treat it the same.
    assertValid(
        """
        def f() -> None:  # ensure function is type-checked
            for i in [0, 1]:
                if i == 1:
                    n + "456"
                else:
                    n = 123
        """);

    // TODO: #28037 - infer LHSs with multiple identifier targets
    assertInvalid(
        "UNSUPPORTED: cannot typecheck assignment statements with multiple targets on the LHS",
        """
        x: list[int] = [0]
        x[0], y, z = 1, 2, "3"
        y + z
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
    assertTypeGivenDecls("t[-1]", Types.BOOL, "t: tuple[int, str, bool]");
    // Index into unknown-length homogeneous tuples.
    assertTypeGivenDecls("t[1]", Types.INT, "t: tuple[int, ...]");
    assertTypeGivenDecls("t[n]", Types.INT, "t: tuple[int, ...]; n: Any");

    // Index can't be statically determined.
    StarlarkType unionType = Types.union(Types.INT, Types.STR, Types.BOOL);
    assertTypeGivenDecls("t[n]", unionType, "t: tuple[int, str, bool]; n: int");
    assertTypeGivenDecls("t[a]", unionType, "t: tuple[int, str, bool]; a: Any");

    // Bad index type.
    assertInvalid(
        ":2:2: 't' of type 'tuple[int, str, bool]' must be indexed by an integer, but got 'str'",
        """
        t: tuple[int, str, bool]
        t["abc"]
        """);
    assertInvalid(
        ":2:2: 't' of type 'tuple[str, ...]' must be indexed by an integer, but got 'str'",
        """
        t: tuple[str, ...]
        t["abc"]
        """);

    // Statically knowable index out-of-range.
    assertInvalid(
        ":2:2: 't' of type 'tuple[int, str, bool]' is indexed by integer 3, which is out-of-range",
        """
        t: tuple[int, str, bool]
        t[3]
        """);
    // Statically knowable index out-of-range.
    assertInvalid(
        ":2:2: 't' of type 'tuple[int, str, bool]' is indexed by integer -4, which is out-of-range",
        """
        t: tuple[int, str, bool]
        t[-4]
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
        # Negative index
        t[-3] = 42

        t2: tuple[int|str, ...]
        t2[0] = 0
        t2[42] = "42"

        # Any as index.
        # This is a particularly nonsensical assignment that nonetheless passes the checker.
        a: Any
        u: int | str | bool
        t[a] = u

        # Any as value.
        t[1] = a
        t2[1000] = a
        """);

    assertInvalid(
        """
        :2:1: cannot assign type 'int' to 't[1]' of type 'str'\
        """,
        """
        t: tuple[int, str, bool]
        t[1] = 123
        """);

    assertInvalid(
        ":3:1: cannot assign type 'int|bool' to 't[1]' of type 'int|str'",
        """
        t: tuple[int | str, ...]
        x: int | bool
        t[1] = x
        """);
  }

  @Test
  public void infer_slice() throws Exception {
    assertTypeGivenDecls("x[1:2]", Types.STR, "x: str");
    assertTypeGivenDecls("x[1:]", Types.list(Types.INT), "x: list[int]");
    assertTypeGivenDecls(
        "x[y:z:w]",
        Types.union(Types.sequence(Types.STR), Types.ANY),
        "x: Sequence[str] | Any; y: Any; z: Any; w: Any");

    // Invalid operand type
    assertInvalid(
        "invalid slice operand 'x' of type 'int', expected Sequence or str", "x: int; x[:2:-1]");

    // Invalid index types
    assertInvalid("got 'str' for start index, want int", "x: str; [][x:]");
    assertInvalid("got 'Any|bool' for stop index, want int", "y: Any | bool; [][:y:]");
    assertInvalid("got 'float' for slice step, want int", "z: float; [][::z]");

    // Invalid step
    assertInvalid("slice step cannot be zero", "x: list; x[::0]");
  }

  @Test
  public void infer_slice_tuple_indices() throws Exception {
    assertTypeGivenDecls(
        "x[0:4:1]", Types.tuple(Types.INT, Types.STR, Types.BOOL), "x: tuple[int, str, bool]");
    assertTypeGivenDecls(
        "x[:]", Types.tuple(Types.INT, Types.STR, Types.BOOL), "x: tuple[int, str, bool]");
    assertTypeGivenDecls("x[1:3]", Types.tuple(Types.STR, Types.BOOL), "x: tuple[int, str, bool]");
    assertTypeGivenDecls(
        "x[-9999:2]", Types.tuple(Types.INT, Types.STR), "x: tuple[int, str, bool]");
    assertTypeGivenDecls(
        "x[1:9999]", Types.tuple(Types.STR, Types.BOOL), "x: tuple[int, str, bool]");
    assertTypeGivenDecls(
        "x[-3::2]", Types.tuple(Types.INT, Types.BOOL), "x: tuple[int, str, bool]");
    assertTypeGivenDecls(
        "x[::-1]", Types.tuple(Types.BOOL, Types.STR, Types.INT), "x: tuple[int, str, bool]");
    assertTypeGivenDecls(
        "x[-1:-4:-2]", Types.tuple(Types.BOOL, Types.INT), "x: tuple[int, str, bool]");

    assertTypeGivenDecls(
        "x[0:99:9]",
        Types.homogeneousTuple(Types.union(Types.INT, Types.STR)),
        "x: tuple[int | str, ...]");

    assertTypeGivenDecls(
        "x[y:]",
        Types.homogeneousTuple(Types.union(Types.INT, Types.STR, Types.BOOL)),
        "x: tuple[int, str, bool]; y: int");
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
    assertTypeGivenDecls("()", Types.tuple());

    // Fixed-length with homogeneous elements.
    assertTypeGivenDecls("(1, 2, 3)", Types.tuple(Types.INT, Types.INT, Types.INT));

    // Fixed-length with heterogeneous elements.
    assertTypeGivenDecls("(1, 'a')", Types.tuple(Types.INT, Types.STR));
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
    assertInvalid(":2:1: operator '-' cannot be applied to type 'str'", "x: str", "-x");
    assertInvalid(":2:1: operator '+' cannot be applied to type 'str'", "x: str", "+x");
    assertInvalid(":2:1: operator '~' cannot be applied to type 'str'", "x: str", "~x");
    assertInvalid(":2:1: operator '-' cannot be applied to type 'str|int'", "x: str | int", "-x");
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
    assertTypeGivenDecls("t1 <= t2", Types.BOOL, "t1: tuple[int, float]; t2: tuple[int, ...]");
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
    assertInvalid(
        "operator '<' cannot be applied to types 'str' and 'int'", "x: str; y: int; x < y");
    assertInvalid("operator '>' cannot be applied to types 'float' and 'bool'", "x: bool; 0.0 > x");
    assertInvalid(
        "operator '>=' cannot be applied to types 'dict[str, int]' and 'dict[str, int]'",
        "x: str; y: str; {x: 1} >= {y: 2}");
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
        "operator '>=' cannot be applied to types 'tuple[int, str]' and 'tuple[int|str, ...]'",
        "x: tuple[int, str]; y: tuple[int|str, ...]; x >= y");
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
        "[1, 2.0] + [3, 'four']", Types.list(Types.union(Types.INT, Types.FLOAT, Types.STR)));
    assertTypeGivenDecls(
        "x + y",
        Types.tuple(Types.INT, Types.FLOAT, Types.INT, Types.STR),
        "x: tuple[int, float]; y: tuple[int, str]");
    assertTypeGivenDecls(
        "x + y",
        Types.homogeneousTuple(Types.union(Types.INT, Types.FLOAT, Types.BOOL)),
        "x: tuple[int, float]; y: tuple[bool, ...]");
    assertTypeGivenDecls(
        "x + y", Types.homogeneousTuple(Types.BOOL), "x: tuple[]; y: tuple[bool, ...]");
    assertTypeGivenDecls(
        "x + y",
        Types.homogeneousTuple(Types.union(Types.INT, Types.BOOL)),
        "x: tuple[int, ...]; y: tuple[bool, ...]");
    assertTypeGivenDecls(
        "x + y", Types.homogeneousTuple(Types.INT), "x: tuple[int, ...]; y: tuple[]");

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
    assertInvalid("operator '+' cannot be applied to types 'str' and 'int'", "x: str; x + 1");
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
    assertInvalid("operator '|' cannot be applied to types 'int' and 'float'", "x: int; x | 2.0");
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
    assertInvalid("operator '&' cannot be applied to types 'int' and 'float'", "x: int; x & 2.0");
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
    assertInvalid(
        "operator '^' cannot be applied to types 'int' and 'float'", "x: int; y: float; x ^ y");
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
    assertInvalid("operator '<<' cannot be applied to types 'int' and 'float'", "x: int; x << 2.0");
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
    assertInvalid("operator '-' cannot be applied to types 'str' and 'int'", "x: str; x - 1");
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
        Types.tuple(Types.INT, Types.FLOAT, Types.INT, Types.FLOAT),
        "x: tuple[int, float]");
    assertTypeGivenDecls(
        "2 * x",
        Types.tuple(Types.INT, Types.FLOAT, Types.INT, Types.FLOAT),
        "x: tuple[int, float]");
    assertTypeGivenDecls("x * 2", Types.homogeneousTuple(Types.INT), "x: tuple[int, ...]");
    assertTypeGivenDecls("2 * x", Types.homogeneousTuple(Types.INT), "x: tuple[int, ...]");
    assertTypeGivenDecls("x * 0", Types.tuple(), "x: tuple[int, float]");
    assertTypeGivenDecls("0 * x", Types.tuple(), "x: tuple[int, float]");
    assertTypeGivenDecls("x * 0", Types.tuple(), "x: tuple[int, ...]");
    assertTypeGivenDecls("0 * x", Types.tuple(), "x: tuple[int, ...]");
    assertTypeGivenDecls("x * -1", Types.tuple(), "x: tuple[str]");
    assertTypeGivenDecls("-1 * x", Types.tuple(), "x: tuple[str]");
    assertTypeGivenDecls("x * -1", Types.tuple(), "x: tuple[str, ...]");
    assertTypeGivenDecls("-1 * x", Types.tuple(), "x: tuple[str, ...]");
    assertTypeGivenDecls("x * y", Types.homogeneousTuple(Types.INT), "x: int; y: tuple[int]");
    assertTypeGivenDecls("x * y", Types.homogeneousTuple(Types.INT), "x: int; y: tuple[int, ...]");

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
    // TODO: #28037 - can be tightened to tuple[str, ...]
    assertTypeGivenDecls("x * y", Types.ANY, "x: Any; y: tuple[str]");
    // TODO: #28037 - can be tightened to tuple[int, ...]
    assertTypeGivenDecls("x * y", Types.ANY, "x: tuple[int, int]; y: Any");
    // TODO: #28037 - can be tightened to float | str
    assertTypeGivenDecls("x * y", Types.ANY, "x: Any; y: float|str");
    // TODO: #28037 - can be tightened to list[str] | str
    assertTypeGivenDecls("x * y", Types.ANY, "x: list[str]|str; y: Any");
    // TODO: #28037 - the following cases should fail
    assertTypeGivenDecls("x * y", Types.ANY, "x: Any; y: bool");
    assertTypeGivenDecls("x * y", Types.ANY, "x: bool; y: Any");

    // unsupported operations
    assertInvalid("operator '*' cannot be applied to types 'str' and 'float'", "x: str; x * 1.0");
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
    assertInvalid(
        "operator '/' cannot be applied to types 'int' and 'str'", "x: int; y: str; x / y");
    assertInvalid(
        "operator '//' cannot be applied to types 'str' and 'float'", "x: str; y: float; x // y");
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
    assertInvalid(
        "operator '%' cannot be applied to types 'float' and 'str'", "x: float; x % 'hello'");
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
        "x: str; ['e'] not in x");
  }

  @Test
  public void infer_conditional() throws Exception {
    assertTypeGivenDecls("x if cond else y", Types.INT, "cond: bool; x: int; y: int");
    assertTypeGivenDecls("x if cond else y", Types.NUMERIC, "cond: bool; x: int; y: float");

    // Any handling; the following test cases assume no Any-simplification in unions.
    assertTypeGivenDecls(
        "x if cond else y", Types.union(Types.INT, Types.ANY), "cond: bool; x: int; y: Any");
    assertTypeGivenDecls(
        "x if cond else y", Types.union(Types.ANY, Types.FLOAT), "cond: bool; x: Any; y: float");
    assertTypeGivenDecls("x if cond else y", Types.ANY, "cond: bool; x: Any; y: Any");

    // Condition's type does not matter.
    assertTypeGivenDecls("x if cond else y", Types.INT, "cond: float; x: int; y: int");
    assertTypeGivenDecls("x if cond else y", Types.INT, "cond: Any; x: int; y: int");
    assertTypeGivenDecls(
        "x if cond else y", Types.INT, "def cond() -> int: return 42", "x: int; y: int");
  }

  @Test
  public void infer_call() throws Exception {
    assertTypeGivenDecls(
        "f(1, y = 'y')",
        Types.INT,
        """
        def f(x: int, y: str, z: int = 42) -> int:
            return 0
        """);

    // Function type unions and Any handling
    assertTypeGivenDecls("f(42)", Types.ANY, "f: Any");
    assertTypeGivenDecls(
        "(f if 1 else g)(42)",
        Types.union(Types.INT, Types.STR),
        """
        def f(x: int) -> int:
            return x
        def g(y: int) -> str:
            return "hello"
        """);
    assertTypeGivenDecls(
        "(f if 1 else g)(42)",
        Types.union(Types.INT, Types.ANY),
        """
        def f(x: int) -> int:
            return x
        g: Any
        """);

    // Omitted return type is Any
    assertTypeGivenDecls(
        "f(42)",
        Types.ANY,
        """
        def f(x: int):
            return x
        """);

    // Parameter type unions and Any handling
    assertTypeGivenDecls(
        "f(1, y = 'y')",
        Types.INT,
        """
        def f(x: int | str, y) -> int:
            return 0
        """);

    // Argument type unions and Any handling
    assertTypeGivenDecls(
        "f(X, y = Y)",
        Types.INT,
        """
        def f(x: int | float | str, y: str, z: int = 42) -> int:
            return 0
        X: int | str
        Y: Any
        """);

    // Cannot call a non-callable
    assertInvalid(
        ":2:1: 'f' is not callable; got type 'int'",
        """
        f: int
        f(42)
        """);
    assertInvalid(
        "'f if 1 else g' is not callable; got type 'Callable[[int], int]|int'",
        """
        def f(x: int) -> int:
            return x
        g: int
        (f if 1 else g)(42)
        """);
  }

  @Test
  public void infer_call_bad_arguments() throws Exception {
    // Wrong argument types
    assertInvalid(
        "in call to 'f()', parameter 'y' got value of type 'str', want 'int'",
        """
        def f(x: Any, y: int) -> int:
            return 0
        f(123, "hello")
        """);
    // Too many positionals
    assertInvalid(
        "'f()' accepts no more than 2 positional arguments but got 3",
        """
        def f(x: int, y: int) -> int:
            return 0
        f(1, 2, 3)
        """);
    // Unexpected arguments
    assertInvalid(
        "'f()' got unexpected keyword argument: mispelled (did you mean 'misspelled'?)",
        """
        def f(x: int, misspelled: int) -> int:
            return 0
        f(x = 1, mispelled = 2)
        """);
    // Missing required arguments
    assertInvalid(
        "'f()' missing 1 required argument: y",
        """
        def f(x: int, y: int) -> int:
            return 0
        f(42)
        """);
    assertInvalid(
        "'f()' missing 2 required arguments: y, z",
        """
        def f(x: int, y: int, z) -> int:
            return 0
        f(42)
        """);
    assertInvalid(
        "'f()' missing 2 required arguments: y, z",
        """
        def f(x: int, y: int, *, z) -> int:
            return 0
        f(42)
        """);
  }

  @Test
  public void infer_call_varargs() throws Exception {
    assertTypeGivenDecls(
        "f(1, *args)",
        Types.INT,
        "args: list[str]",
        """
        def f(x: int, y: str, *args) -> int:
            return 0
        """);
    // Complex types
    assertTypeGivenDecls(
        "f(1, *args)",
        Types.INT,
        """
        def f(x: int, *args: str|float) -> int:
            return 0
        args: list[str] | tuple[float]
        """);
    // Caller varargs satisfy missing positional arguments
    assertTypeGivenDecls(
        "f(1, *args)",
        Types.INT,
        """
        def f(x: int, y: str, z: str) -> int:
            return 0
        args: list[str]
        """);
    // Callable varargs absorb residual positional arguments
    assertTypeGivenDecls(
        "f(1, 'two', 3.0)",
        Types.INT,
        """
        def f(x: int, *args: str|float) -> int:
            return 0
        """);
    // Wrong shape
    assertInvalid(
        "argument after * must be a sequence, not 'str'",
        """
        def f(*args) -> int:
            return 0
        args: str
        f(*args)
        """);
    assertInvalid(
        "argument after * must be a sequence, not 'str|list[str]'",
        """
        def f(*args) -> int:
            return 0
        args: str | list[str]
        f(*args)
        """);
    // Wrong element type
    assertInvalid(
        "in call to 'f()', elements of argument after * must be 'float', not 'str|float'",
        """
        def f(*args: float) -> int:
            return 0
        args: list[str] | list[float]
        f(*args)
        """);
    // Wrong type of residual positional arguments
    assertInvalid(
        "in call to 'f()', residual positional arguments must be 'str|float', not 'int'",
        """
        def f(x: int, *args: str|float) -> int:
            return 0
        f(1, 2, 3)
        """);
  }

  @Test
  public void infer_call_kwargs() throws Exception {
    assertTypeGivenDecls(
        "f(1, **kwargs)",
        Types.INT,
        """
        def f(x: int, y: float, **kwargs) -> int:
            return 0
        kwargs: dict[str, float]
        """);
    // Complex types
    assertTypeGivenDecls(
        "f(1, **kwargs)",
        Types.INT,
        """
        def f(x: int, **kwargs: str|float) -> int:
            return 0
        kwargs: dict[str, str] | dict[str, float]
        """);
    // Caller kwargs satisfy missing keyword arguments
    assertTypeGivenDecls(
        "f(1, **kwargs)",
        Types.INT,
        """
        def f(x: int, y: str, *, z: str) -> int:
            return 0
        kwargs: dict[str, str]
        """);
    // Callable kwargs absorb residual keyword arguments
    assertTypeGivenDecls(
        "f(1, y='two', z=3.0)",
        Types.INT,
        """
        def f(x: int, **kwargs: str|float) -> int:
            return 0
        """);
    // Wrong shape
    assertInvalid(
        "argument after ** must be a dict with string keys, not 'list[Any]'",
        """
        def f(**kwargs) -> int:
            return 0
        kwargs: list
        f(**kwargs)
        """);
    assertInvalid(
        "argument after ** must be a dict with string keys, not 'dict[Any, Any]|list[Any]'",
        """
        def f(**kwargs) -> int:
            return 0
        kwargs: dict | list
        f(**kwargs)
        """);
    // Wrong element type
    assertInvalid(
        "in call to 'f()', values of argument after ** must be 'float', not 'str|float'",
        """
        def f(**kwargs: float) -> int:
            return 0
        kwargs: dict[str, str] | dict[str, float]
        f(**kwargs)
        """);
    // Wrong type of residual keyword arguments
    assertInvalid(
        "in call to 'f()', residual keyword arguments must be 'str|float', not 'int'",
        """
        def f(x: int, **kwargs: str|float) -> int:
            return 0
        f(x=1, y=2, z=3)
        """);
  }

  @Test
  public void def_argument_defaults() throws Exception {
    assertValid("def f(x: int = 42, y: str= '', z = {}): pass");
    String invalid = "def f(x: int = 42.0, y: str = 43, z = []): pass";
    assertInvalid("f(): parameter 'x' has default value of type 'float', declares 'int'", invalid);
    assertInvalid("f(): parameter 'y' has default value of type 'int', declares 'str'", invalid);
  }

  @Test
  public void def_return_type() throws Exception {
    assertValid("def f(): pass");
    assertValid("def f(): return 42");
    assertValid("def f() -> int: return 42");
    assertValid("def f() -> None: pass");
    assertValid("def f() -> None: return");
    assertValid(
        """
        def f() -> int|None:
            if 2 + 2 == 4:
                return 42
        """);
    assertValid(
        """
        def f() -> int|float|str:
            if 2 + 2 == 4:
                return 42
            elif 2.0 + 2.0 == 4.0:
                return 42.0
            else:
                return 'abc'
        """);

    assertInvalid(
        ":2:5: f() declares return type 'int' but may exit without an explicit 'return'",
        """
        def f() -> int:
            if 2 + 2 == 4:
                return 42
        """);
    assertInvalid(
        ":3:16: f() declares return type 'None' but may return 'int'",
        """
        def f() -> None:
            if 2 + 2 == 4:
                return 42
        """);
  }

  @Test
  public void def_body_checked_iff_function_uses_type_syntax() throws Exception {
    assertInvalid(
        "operator '+' cannot be applied to types 'int' and 'str'",
        """
        X: int = 42
        def typed() -> int:
            return X + "abc"
        """);
    assertValid(
        """
        X: int = 42
        def untyped():
            # error ignored by static type checker because function is untyped
            return X + "abc"
        """);
    assertInvalid(
        "operator '+' cannot be applied to types 'int' and 'str'",
        """
        def typed(x):
            # type syntax in nested lambdas causes outer function to be type-checked.
            return (lambda x: cast(int, x))(x) + "abc"
        """);
    assertValid(
        """
        X: int = 42
        def untyped():
            # type syntax in nested defs does not affect outer function
            def get_int() -> int:
                return X
            # error ignored by static type checker because outer function is untyped
            return get_int() + "abc"
        """);

    assertInvalid(
        ":5:18: operator '%' cannot be applied to types 'int' and 'str'",
        """
        X: int = 42
        def untyped():
            def get_int() -> int:
                # type syntax in nested typed defs is checked even if the outer def is untyped
                return X % "abc"
            return get_int() + "def"
        """);
  }

  @Test
  public void infer_cast() throws Exception {
    assertTypeGivenDecls("cast(int, x)", Types.INT, "x: Any");
    // cast expression allows casting to the wrong type
    assertTypeGivenDecls(
        "cast(list[int] | bool, 42)", Types.union(Types.list(Types.INT), Types.BOOL));
    // cast expression always checks that its second argument is well-typed
    assertInvalid(
        "operator '+' cannot be applied to types 'int' and 'str'", "cast(int, 1 + 'two')");
  }

  @Test
  public void infer_lambda() throws Exception {
    // no inference on the type of a lamda's argument
    assertTypeGivenDecls("(lambda x: x + y)(42)", Types.ANY, "y: int");
    // ... but a cast in the body allows inferring the return type
    assertTypeGivenDecls("(lambda x: cast(int, x) + 1)(42)", Types.INT);
  }

  @Test
  public void if_statement() throws Exception {
    // condition
    assertInvalid(
        "operator '+' cannot be applied to types 'float' and 'str'",
        """
        def _wrapper() -> None:
            if 12.3 + '45.6' > 0:
                pass
        """);
    // then body
    assertInvalid(
        "operator '+' cannot be applied to types 'int' and 'str'",
        """
        def _wrapper() -> None:
            if 1 == 2:
                123 + '456'
        """);
    // else body
    assertInvalid(
        "operator '+' cannot be applied to types 'str' and 'int'",
        """
        def _wrapper() -> None:
            if 1 == 2:
                pass
            else:
                '123' + 456
        """);
  }

  @Test
  public void if_statement_in_untyped_code() throws Exception {
    // In untyped code, don't type-check the condition or non-def statements in then/else blocks ...
    assertValid(
        """
        def _untyped_wrapper():
            if 1 + "two":   # type error ignored in untyped code
                3 + "four"  # type error ignored in untyped code
            else:
                5 + "six"   # type error ignored in untyped code
        """);
    // ... but do recurse into inner typed defs in then/else blocks
    assertInvalid(
        ":4:20: typed() declares return type 'int' but may return 'str'",
        """
        def _untyped_wrapper():
            if 1 + "two":         # type error ignored in untyped code
                def typed() -> int:
                    return "abc"  # type error checked in typed innner def
        """);
    assertInvalid(
        ":6:20: typed() declares return type 'int' but may return 'float'",
        """
        def _untyped_wrapper():
            if 1 + "two":        # type error ignored in untyped code
                pass
            else:
                def typed() -> int:
                    return 3.14  # type error checked in typed innner def
        """);
  }

  @Test
  public void for_statement_operand() throws Exception {
    assertValid(
        """
        def _wrapper() -> None:
            for x in [1, 2, 3]:
                pass
        """);
    assertValid(
        """
        def _wrapper() -> None:
            for x in (1, 2, 3):
                pass
        """);
    assertValid(
        """
        def _wrapper() -> None:
            for x in {'a': 'b', 'c': 'd'}:
                pass
        """);
    assertValid(
        """
        y: Any
        def _wrapper() -> None:
            for x in y:
                pass
        """);
    assertValid(
        """
        y: Any | list[int]
        def f(x: int) -> None: pass  # to verify type of x
        def _wrapper() -> None:
            for x in y:
                f(x)
        """);

    assertInvalid(
        "'for' loop operand must be an iterable, got 'int'",
        """
        def _wrapper() -> None:
            for x in 42:
                pass
        """);

    // TODO: #28037 - Support multi-argument vars and var indexing in for statements.
    assertInvalid(
        "UNSUPPORTED: cannot typecheck assignment statements with multiple targets on the LHS",
        """
        def _wrapper() -> None:
            for x, y in [(1, 2)]:
                pass
        """);
  }

  @Test
  public void for_statement_operand_with_previously_typed_vars() throws Exception {
    assertValid(
        """
        def _wrapper() -> None:
            x: int
            for x in [1, 2, 3]:
                pass

            for x in (1, 2, 3):
                pass

            y: str
            for y in {'a': 'b', 'c': 'd'}:
                pass

            z: Any
            for z in [1, "two", 3.14, None]:
                pass
        """);
    assertInvalid(
        ":3:9: cannot assign type 'int|str' to 'x' of type 'int'",
        """
        def _wrapper() -> None:
            x: int
            for x in [1, "two"]:
                pass
        """);

    // TODO: #28037 - Support multi-argument vars and var indexing in for statements.
    assertInvalid(
        "UNSUPPORTED: cannot typecheck assignment statements with multiple targets on the LHS",
        """
        def _wrapper() -> None:
            x: int
            y: str
            for x, y in [(1, "two")]:
                pass
        """);
  }

  @Test
  public void for_statement_body() throws Exception {
    assertValid(
        """
        def _wrapper() -> None:
            for x in [1, 2, 3]:
                x + 1
        """);

    assertInvalid(
        "operator '+' cannot be applied to types 'str' and 'int'",
        """
        def _wrapper() -> None:
            for x in ['a', 'b', 'c']:
                x + 1
        """);
  }

  @Test
  public void for_statement_in_untyped_code() throws Exception {
    // In untyped code, don't type-check the operand or non-def statements in body ...
    assertValid(
        """
        def _untyped_wrapper():
            for x in (1, "two", 3.14):  # type error ignored in untyped code
                x / "bad"               # type error ignored in untyped code
        """);
    // ... but do recurse into inner typed defs in body
    assertInvalid(
        ":4:20: typed() declares return type 'int' but may return 'str'",
        """
        def _untyped_wrapper():
            for x in (1, "two", 3.14):  # type error ignored in untyped code
                def typed() -> int:
                    return "abc"        # type error checked in typed innner def
        """);
  }
}
