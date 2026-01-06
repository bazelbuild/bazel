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
import static net.starlark.java.syntax.LexerTest.assertContainsError;

import com.google.common.base.Joiner;
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
    Module module = Resolver.moduleWithPredeclared();
    Resolver.resolveFile(file, module);
    assertNoErrors("resolving", file);
    TypeResolver.annotateFile(file, module);
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
}
