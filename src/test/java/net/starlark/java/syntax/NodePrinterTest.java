// Copyright 2017 The Bazel Authors. All Rights Reserved.
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

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests {@link Node#toString} and {@code NodePrinter}. */
@RunWith(JUnit4.class)
public final class NodePrinterTest {
  private FileOptions fileOptions = FileOptions.DEFAULT;

  private void setFileOptions(FileOptions fileOptions) {
    this.fileOptions = fileOptions;
  }

  private StarlarkFile parseFile(String... lines) throws SyntaxError.Exception {
    ParserInput input = ParserInput.fromLines(lines);
    StarlarkFile file = StarlarkFile.parse(input, fileOptions);
    if (!file.ok()) {
      throw new SyntaxError.Exception(file.errors());
    }
    return file;
  }

  private Statement parseStatement(String... lines) throws SyntaxError.Exception {
    return parseFile(lines).getStatements().get(0);
  }

  private Expression parseExpression(String... lines) throws SyntaxError.Exception {
    return Expression.parse(ParserInput.fromLines(lines), fileOptions);
  }

  /**
   * Asserts that the given node's pretty print at a given indent level matches the given string.
   */
  private static void assertPrettyMatches(Node node, int indentLevel, String expected) {
    StringBuilder buf = new StringBuilder();
    new NodePrinter(buf, indentLevel).printNode(node);
    assertThat(buf.toString()).isEqualTo(expected);
  }

  /** Asserts that the given node's pretty print with no indent matches the given string. */
  private static void assertPrettyMatches(Node node, String expected) {
    assertPrettyMatches(node, 0, expected);
  }

  /** Asserts that the given node's pretty print with one indent matches the given string. */
  private static void assertIndentedPrettyMatches(Node node, String expected) {
    assertPrettyMatches(node, 1, expected);
  }

  /** Asserts that the given node's {@code toString} matches the given string. */
  private static void assertTostringMatches(Node node, String expected) {
    assertThat(node.toString()).isEqualTo(expected);
  }

  /**
   * Parses the given string as an expression, and asserts that its pretty print matches the given
   * string.
   */
  private void assertExprPrettyMatches(String source, String expected)
      throws SyntaxError.Exception {
    Expression node = parseExpression(source);
    assertPrettyMatches(node, expected);
  }

  /**
   * Parses the given string as an expression, and asserts that its {@code toString} matches the
   * given string.
   */
  private void assertExprTostringMatches(String source, String expected)
      throws SyntaxError.Exception {
    Expression node = parseExpression(source);
    assertThat(node.toString()).isEqualTo(expected);
  }

  /**
   * Parses the given string as an expression, and asserts that both its pretty print and {@code
   * toString} return the original string.
   */
  private void assertExprBothRoundTrip(String source) throws SyntaxError.Exception {
    assertExprPrettyMatches(source, source);
    assertExprTostringMatches(source, source);
  }

  /**
   * Parses the given string as a statement, and asserts that its pretty print with one indent
   * matches the given string.
   */
  private void assertStmtIndentedPrettyMatches(String source, String expected)
      throws SyntaxError.Exception {
    Statement node = parseStatement(source);
    assertIndentedPrettyMatches(node, expected);
  }

  /**
   * Parses the given string as an statement, and asserts that its {@code toString} matches the
   * given string.
   */
  private void assertStmtTostringMatches(String source, String expected)
      throws SyntaxError.Exception {
    Statement node = parseStatement(source);
    assertThat(node.toString()).isEqualTo(expected);
  }

  // Expressions.

  @Test
  public void abstractComprehension() throws SyntaxError.Exception {
    // Covers DictComprehension and ListComprehension.
    assertExprBothRoundTrip("[z for y in x if True for z in y]");
    assertExprBothRoundTrip("{z: x for y in x if True for z in y}");
  }

  @Test
  public void binaryOperatorExpression() throws SyntaxError.Exception {
    assertExprPrettyMatches("1 + 2", "(1 + 2)");
    assertExprTostringMatches("1 + 2", "1 + 2");

    assertExprPrettyMatches("1 + (2 * 3)", "(1 + (2 * 3))");
    assertExprTostringMatches("1 + (2 * 3)", "1 + 2 * 3");
  }

  @Test
  public void conditionalExpression() throws SyntaxError.Exception {
    assertExprBothRoundTrip("1 if True else 2");
  }

  @Test
  public void dictExpression() throws SyntaxError.Exception {
    assertExprBothRoundTrip("{1: \"a\", 2: \"b\"}");
  }

  @Test
  public void dotExpression() throws SyntaxError.Exception {
    assertExprBothRoundTrip("o.f");
  }

  @Test
  public void funcallExpression() throws SyntaxError.Exception {
    assertExprBothRoundTrip("f()");
    assertExprBothRoundTrip("f(a)");
    assertExprBothRoundTrip("f(a, b = B, c = C, *d, **e)");
    assertExprBothRoundTrip("o.f()");
    assertExprBothRoundTrip("f(1 + 1)");
  }

  @Test
  public void identifier() throws SyntaxError.Exception {
    assertExprBothRoundTrip("foo");
  }

  @Test
  public void indexExpression() throws SyntaxError.Exception {
    assertExprBothRoundTrip("a[i]");
    assertExprBothRoundTrip("a[(1,)]");
    assertExprPrettyMatches("a[1,2]", "a[(1, 2)]");
    assertExprTostringMatches("a[1,2]", "a[(1, 2)]");
  }

  @Test
  public void integerLiteral() throws SyntaxError.Exception {
    assertExprBothRoundTrip("5");
  }

  @Test
  public void listLiteralShort() throws SyntaxError.Exception {
    assertExprBothRoundTrip("[]");
    assertExprBothRoundTrip("[5]");
    assertExprBothRoundTrip("[5, 6]");
    assertExprBothRoundTrip("()");
    assertExprBothRoundTrip("(5,)");
    assertExprBothRoundTrip("(5, 6)");
  }

  @Test
  public void listLiteralLong() throws SyntaxError.Exception {
    // List literals with enough elements to trigger the abbreviated toString() format.
    assertExprPrettyMatches("[1, 2, 3, 4, 5, 6]", "[1, 2, 3, 4, 5, 6]");
    assertExprTostringMatches("[1, 2, 3, 4, 5, 6]", "[1, 2, 3, 4, +2 more]");

    assertExprPrettyMatches("(1, 2, 3, 4, 5, 6)", "(1, 2, 3, 4, 5, 6)");
    assertExprTostringMatches("(1, 2, 3, 4, 5, 6)", "(1, 2, 3, 4, +2 more)");
  }

  @Test
  public void listLiteralNested() throws SyntaxError.Exception {
    // Make sure that the inner list doesn't get abbreviated when the outer list is printed using
    // prettyPrint().
    assertExprPrettyMatches(
        "[1, 2, 3, [10, 20, 30, 40, 50, 60], 4, 5, 6]",
        "[1, 2, 3, [10, 20, 30, 40, 50, 60], 4, 5, 6]");
    // It doesn't matter as much what toString does.
    assertExprTostringMatches("[1, 2, 3, [10, 20, 30, 40, 50, 60], 4, 5, 6]", "[1, 2, 3, +4 more]");
  }

  @Test
  public void sliceExpression() throws SyntaxError.Exception {
    assertExprBothRoundTrip("a[b:c:d]");
    assertExprBothRoundTrip("a[b:c]");
    assertExprBothRoundTrip("a[b:]");
    assertExprBothRoundTrip("a[:c:d]");
    assertExprBothRoundTrip("a[:c]");
    assertExprBothRoundTrip("a[::d]");
    assertExprBothRoundTrip("a[:]");
  }

  @Test
  public void stringLiteral() throws SyntaxError.Exception {
    assertExprBothRoundTrip("\"foo\"");
    assertExprBothRoundTrip("\"quo\\\"ted\"");
  }

  @Test
  public void unaryOperatorExpression() throws SyntaxError.Exception {
    assertExprPrettyMatches("not True", "not (True)");
    assertExprTostringMatches("not True", "not True");
    assertExprPrettyMatches("-5", "-(5)");
    assertExprTostringMatches("-5", "-5");
  }

  // Statements.

  @Test
  public void assignmentStatement() throws SyntaxError.Exception {
    assertStmtIndentedPrettyMatches("x = y", "  x = y\n");
    assertStmtTostringMatches("x = y", "x = y\n");
  }

  @Test
  public void augmentedAssignmentStatement() throws SyntaxError.Exception {
    assertStmtIndentedPrettyMatches("x += y", "  x += y\n");
    assertStmtTostringMatches("x += y", "x += y\n");
  }

  @Test
  public void expressionStatement() throws SyntaxError.Exception {
    assertStmtIndentedPrettyMatches("5", "  5\n");
    assertStmtTostringMatches("5", "5\n");
  }

  @Test
  public void defStatement() throws SyntaxError.Exception {
    assertStmtIndentedPrettyMatches(
        """
        def f(x):
          print(x)\
        """,
        """
          def f(x):
            print(x)
        """);
    assertStmtTostringMatches(
        """
        def f(x):
          print(x)\
        """,
        "def f(x): ...\n");

    assertStmtIndentedPrettyMatches(
        """
        def f(a, b=B, *c, d=D, **e):
          print(x)\
        """,
        """
          def f(a, b=B, *c, d=D, **e):
            print(x)
        """);
    assertStmtTostringMatches(
        """
        def f(a, b=B, *c, d=D, **e):
          print(x)\
        """,
        "def f(a, b=B, *c, d=D, **e): ...\n");

    assertStmtIndentedPrettyMatches(
        """
        def f():
          pass\
        """,
        """
          def f():
            pass
        """);
    assertStmtTostringMatches(
        """
        def f():
          pass\
        """,
        "def f(): ...\n");
  }

  @Test
  public void defStatementWithTypeAnnotations() throws SyntaxError.Exception {
    setFileOptions(FileOptions.builder().allowTypeSyntax(true).build());
    assertStmtIndentedPrettyMatches(
        """
        def f(x:int):
          print(x)\
        """,
        """
          def f(x: int):
            print(x)
        """);
    assertStmtTostringMatches(
        """
        def f(x:bool):
          print(x)\
        """,
        "def f(x: bool): ...\n");

    assertStmtIndentedPrettyMatches(
        """
        def f()->int:
          print(x)\
        """,
        """
          def f() -> int:
            print(x)
        """);
    assertStmtTostringMatches(
        """
        def f() -> bool:
          print(x)
        """,
        "def f() -> bool: ...\n");
  }

  @Test
  public void typeAnnotations() throws SyntaxError.Exception {
    // TODO(ilist@): replace with parsing type annotations directly (remove `def` from this test)
    setFileOptions(FileOptions.builder().allowTypeSyntax(true).build());
    assertStmtTostringMatches("def f(x:bool): pass", "def f(x: bool): ...\n");
    assertStmtTostringMatches("def f(x:None | bool): pass", "def f(x: None | bool): ...\n");
    assertStmtTostringMatches("def f(x:list[str]): pass", "def f(x: list[str]): ...\n");
    assertStmtTostringMatches("def f(x:dict[str,int]): pass", "def f(x: dict[str, int]): ...\n");
    assertStmtTostringMatches(
        "def f(x:Callable[[str],int]): pass", "def f(x: Callable[[str], int]): ...\n");
    assertStmtTostringMatches(
        "def f(x:Callable[[str|int],int]): pass", "def f(x: Callable[[str | int], int]): ...\n");
    assertStmtTostringMatches(
        "def f(x:TypedDict[{'field1': int}]): pass",
        "def f(x: TypedDict[{\"field1\": int}]): ...\n");
  }

  @Test
  public void typeAliasStatement() throws SyntaxError.Exception {
    setFileOptions(FileOptions.builder().allowTypeSyntax(true).build());
    assertStmtTostringMatches("type my_int=int", "type my_int = ...\n");
    assertStmtIndentedPrettyMatches("type my_int=int", "  type my_int = int\n");
    assertStmtTostringMatches("type X[T,U]=dict[T,U]|list[U]", "type X[T, U] = ...\n");
    assertStmtIndentedPrettyMatches(
        "type X[T,U]=dict[T,U]|list[U]", "  type X[T, U] = dict[T, U] | list[U]\n");
  }

  @Test
  public void flowStatement() throws SyntaxError.Exception {
    assertStmtIndentedPrettyMatches(
        """
        def f():
             pass
             continue
             break\
        """,
        """
          def f():
            pass
            continue
            break
        """);
  }

  @Test
  public void forStatement() throws SyntaxError.Exception {
    assertStmtIndentedPrettyMatches(
        """
        for x in y:
          print(x)\
        """,
        """
          for x in y:
            print(x)
        """);
    assertStmtTostringMatches(
        """
        for x in y:
          print(x)\
        """,
        "for x in y: ...\n");

    assertStmtIndentedPrettyMatches(
        """
        for x in y:
          pass\
        """,
        """
          for x in y:
            pass
        """);
    assertStmtTostringMatches(
        """
        for x in y:
          pass\
        """,
        "for x in y: ...\n");
  }

  @Test
  public void ifStatement() throws SyntaxError.Exception {
    assertStmtIndentedPrettyMatches(
        """
        if True:
          print(x)\
        """,
        """
          if True:
            print(x)
        """);
    assertStmtTostringMatches(
        """
        if True:
          print(x)\
        """,
        "if True: ...\n");

    assertStmtIndentedPrettyMatches(
        """
        if True:
          print(x)
        elif False:
          print(y)
        else:
          print(z)\
        """,
        """
          if True:
            print(x)
          elif False:
            print(y)
          else:
            print(z)
        """);
    assertStmtTostringMatches(
        """
        if True:
          print(x)
        elif False:
          print(y)
        else:
          print(z)\
        """,
        "if True: ...\n");
  }

  @Test
  public void loadStatement() throws SyntaxError.Exception {
    assertStmtIndentedPrettyMatches(
        "load(\"foo.bzl\", a=\"A\", \"B\")", "  load(\"foo.bzl\", a=\"A\", \"B\")\n");
    assertStmtTostringMatches(
        "load(\"foo.bzl\", a=\"A\", \"B\")\n", "load(\"foo.bzl\", a=\"A\", \"B\")\n");
  }

  @Test
  public void returnStatement() throws SyntaxError.Exception {
    assertStmtIndentedPrettyMatches("return \"foo\"", "  return \"foo\"\n");
    assertStmtTostringMatches("return \"foo\"", "return \"foo\"\n");

    assertStmtIndentedPrettyMatches("return None", "  return None\n");
    assertStmtTostringMatches("return None", "return None\n");

    assertStmtIndentedPrettyMatches("return", "  return\n");
    assertStmtTostringMatches("return", "return\n");
  }

  // Miscellaneous.

  @Test
  public void file() throws SyntaxError.Exception {
    Node node = parseFile("print(x)\nprint(y)");
    assertIndentedPrettyMatches(
        node,
        """
          print(x)
          print(y)
        """);
    assertTostringMatches(node, "<StarlarkFile with 2 statements>");
  }

  @Test
  public void comment() throws SyntaxError.Exception {
    ParserInput input =
        ParserInput.fromLines(
            "# foo", //
            "expr # bar");
    Parser.ParseResult r = Parser.parseFile(input, FileOptions.DEFAULT);
    Comment c0 = r.comments.get(0);
    assertIndentedPrettyMatches(c0, "  # foo");
    assertTostringMatches(c0, "# foo");
    Comment c1 = r.comments.get(1);
    assertIndentedPrettyMatches(c1, "  # bar");
    assertTostringMatches(c1, "# bar");
  }

  /* Not tested explicitly because they're covered implicitly by tests for other nodes:
   * - DictExpression.Entry
   * - Argument / Parameter
   * - IfStatements
   */
}
