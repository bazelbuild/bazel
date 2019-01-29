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

package com.google.devtools.build.lib.syntax;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.syntax.Parser.ParsingLevel.LOCAL_LEVEL;
import static com.google.devtools.build.lib.syntax.Parser.ParsingLevel.TOP_LEVEL;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.syntax.util.EvaluationTestCase;
import java.io.IOException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests the {@code toString} and pretty printing methods for {@link ASTNode} subclasses. */
@RunWith(JUnit4.class)
public class ASTPrettyPrintTest extends EvaluationTestCase {

  private String join(String... lines) {
    return Joiner.on("\n").join(lines);
  }

  /**
   * Asserts that the given node's pretty print at a given indent level matches the given string.
   */
  private void assertPrettyMatches(ASTNode node, int indentLevel, String expected) {
    StringBuilder prettyBuilder = new StringBuilder();
    try {
      node.prettyPrint(prettyBuilder, indentLevel);
    } catch (IOException e) {
      // Impossible for StringBuilder.
      throw new AssertionError(e);
    }
    assertThat(prettyBuilder.toString()).isEqualTo(expected);
  }

  /** Asserts that the given node's pretty print with no indent matches the given string. */
  private void assertPrettyMatches(ASTNode node, String expected) {
    assertPrettyMatches(node, 0, expected);
  }

  /** Asserts that the given node's pretty print with one indent matches the given string. */
  private void assertIndentedPrettyMatches(ASTNode node, String expected) {
    assertPrettyMatches(node, 1, expected);
  }

  /** Asserts that the given node's {@code toString} matches the given string. */
  private void assertTostringMatches(ASTNode node, String expected) {
    assertThat(node.toString()).isEqualTo(expected);
  }

  /**
   * Parses the given string as an expression, and asserts that its pretty print matches the given
   * string.
   */
  private void assertExprPrettyMatches(String source, String expected) {
    Expression node = parseExpression(source);
    assertPrettyMatches(node, expected);
  }

  /**
   * Parses the given string as an expression, and asserts that its {@code toString} matches the
   * given string.
   */
  private void assertExprTostringMatches(String source, String expected) {
    Expression node = parseExpression(source);
    assertThat(node.toString()).isEqualTo(expected);
  }

  /**
   * Parses the given string as an expression, and asserts that both its pretty print and {@code
   * toString} return the original string.
   */
  private void assertExprBothRoundTrip(String source) {
    assertExprPrettyMatches(source, source);
    assertExprTostringMatches(source, source);
  }

  /**
   * Parses the given string as a statement, and asserts that its pretty print with one indent
   * matches the given string.
   */
  private void assertStmtIndentedPrettyMatches(
      Parser.ParsingLevel parsingLevel, String source, String expected) {
    Statement node = parseStatement(parsingLevel, source);
    assertIndentedPrettyMatches(node, expected);
  }

  /**
   * Parses the given string as an statement, and asserts that its {@code toString} matches the
   * given string.
   */
  private void assertStmtTostringMatches(
      Parser.ParsingLevel parsingLevel, String source, String expected) {
    Statement node = parseStatement(parsingLevel, source);
    assertThat(node.toString()).isEqualTo(expected);
  }

  // Expressions.

  @Test
  public void abstractComprehension() {
    // Covers DictComprehension and ListComprehension.
    assertExprBothRoundTrip("[z for y in x if True for z in y]");
    assertExprBothRoundTrip("{z: x for y in x if True for z in y}");
  }

  @Test
  public void binaryOperatorExpression() {
    assertExprPrettyMatches("1 + 2", "(1 + 2)");
    assertExprTostringMatches("1 + 2", "1 + 2");

    assertExprPrettyMatches("1 + (2 * 3)", "(1 + (2 * 3))");
    assertExprTostringMatches("1 + (2 * 3)", "1 + 2 * 3");
  }

  @Test
  public void conditionalExpression() {
    assertExprBothRoundTrip("1 if True else 2");
  }

  @Test
  public void dictionaryLiteral() {
    assertExprBothRoundTrip("{1: \"a\", 2: \"b\"}");
  }

  @Test
  public void dotExpression() {
    assertExprBothRoundTrip("o.f");
  }

  @Test
  public void funcallExpression() {
    assertExprBothRoundTrip("f()");
    assertExprBothRoundTrip("f(a)");
    assertExprBothRoundTrip("f(a, b = B, *c, d = D, **e)");
    assertExprBothRoundTrip("o.f()");
  }

  @Test
  public void identifier() {
    assertExprBothRoundTrip("foo");
  }

  @Test
  public void indexExpression() {
    assertExprBothRoundTrip("a[i]");
  }

  @Test
  public void integerLiteral() {
    assertExprBothRoundTrip("5");
  }

  @Test
  public void listLiteralShort() {
    assertExprBothRoundTrip("[]");
    assertExprBothRoundTrip("[5]");
    assertExprBothRoundTrip("[5, 6]");
    assertExprBothRoundTrip("()");
    assertExprBothRoundTrip("(5,)");
    assertExprBothRoundTrip("(5, 6)");
  }

  @Test
  public void listLiteralLong() {
    // List literals with enough elements to trigger the abbreviated toString() format.
    assertExprPrettyMatches("[1, 2, 3, 4, 5, 6]", "[1, 2, 3, 4, 5, 6]");
    assertExprTostringMatches("[1, 2, 3, 4, 5, 6]", "[1, 2, 3, 4, <2 more arguments>]");

    assertExprPrettyMatches("(1, 2, 3, 4, 5, 6)", "(1, 2, 3, 4, 5, 6)");
    assertExprTostringMatches("(1, 2, 3, 4, 5, 6)", "(1, 2, 3, 4, <2 more arguments>)");
  }

  @Test
  public void listLiteralNested() {
    // Make sure that the inner list doesn't get abbreviated when the outer list is printed using
    // prettyPrint().
    assertExprPrettyMatches(
        "[1, 2, 3, [10, 20, 30, 40, 50, 60], 4, 5, 6]",
        "[1, 2, 3, [10, 20, 30, 40, 50, 60], 4, 5, 6]");
    // It doesn't matter as much what toString does. This case demonstrates an apparent bug in how
    // Printer#printList abbreviates the nested contents. We can keep this test around to help
    // monitor changes in the buggy behavior or eventually fix it.
    assertExprTostringMatches(
        "[1, 2, 3, [10, 20, 30, 40, 50, 60], 4, 5, 6]",
        "[1, 2, 3, [10, 20, 30, 40, <2 more argu...<2 more arguments>], <3 more arguments>]");
  }

  @Test
  public void sliceExpression() {
    assertExprBothRoundTrip("a[b:c:d]");
    assertExprBothRoundTrip("a[b:c]");
    assertExprBothRoundTrip("a[b:]");
    assertExprBothRoundTrip("a[:c:d]");
    assertExprBothRoundTrip("a[:c]");
    assertExprBothRoundTrip("a[::d]");
    assertExprBothRoundTrip("a[:]");
  }

  @Test
  public void stringLiteral() {
    assertExprBothRoundTrip("\"foo\"");
    assertExprBothRoundTrip("\"quo\\\"ted\"");
  }

  @Test
  public void unaryOperatorExpression() {
    assertExprPrettyMatches("not True", "not (True)");
    assertExprTostringMatches("not True", "not True");
    assertExprPrettyMatches("-5", "-(5)");
    assertExprTostringMatches("-5", "-5");
  }

  // Statements.

  @Test
  public void assignmentStatement() {
    assertStmtIndentedPrettyMatches(LOCAL_LEVEL, "x = y", "  x = y\n");
    assertStmtTostringMatches(LOCAL_LEVEL, "x = y", "x = y\n");
  }

  @Test
  public void augmentedAssignmentStatement() {
    assertStmtIndentedPrettyMatches(LOCAL_LEVEL, "x += y", "  x += y\n");
    assertStmtTostringMatches(LOCAL_LEVEL, "x += y", "x += y\n");
  }

  @Test
  public void expressionStatement() {
    assertStmtIndentedPrettyMatches(LOCAL_LEVEL, "5", "  5\n");
    assertStmtTostringMatches(LOCAL_LEVEL, "5", "5\n");
  }

  @Test
  public void functionDefStatement() {
    assertStmtIndentedPrettyMatches(
        TOP_LEVEL,
        join("def f(x):",
             "  print(x)"),
        join("  def f(x):",
             "    print(x)",
             ""));
    assertStmtTostringMatches(
        TOP_LEVEL,
        join("def f(x):",
             "  print(x)"),
        "def f(x): ...\n");

    assertStmtIndentedPrettyMatches(
        TOP_LEVEL,
        join("def f(a, b=B, *c, d=D, **e):",
             "  print(x)"),
        join("  def f(a, b=B, *c, d=D, **e):",
             "    print(x)",
             ""));
    assertStmtTostringMatches(
        TOP_LEVEL,
        join("def f(a, b=B, *c, d=D, **e):",
             "  print(x)"),
        "def f(a, b = B, *c, d = D, **e): ...\n");

    assertStmtIndentedPrettyMatches(
        TOP_LEVEL,
        join("def f():",
             "  pass"),
        join("  def f():",
             "    pass",
             ""));
    assertStmtTostringMatches(
        TOP_LEVEL,
        join("def f():",
             "  pass"),
        "def f(): ...\n");
  }

  @Test
  public void flowStatement() {
    // The parser would complain if we tried to construct them from source.
    ASTNode breakNode = new FlowStatement(FlowStatement.Kind.BREAK);
    assertIndentedPrettyMatches(breakNode, "  break\n");
    assertTostringMatches(breakNode, "break\n");

    ASTNode continueNode = new FlowStatement(FlowStatement.Kind.CONTINUE);
    assertIndentedPrettyMatches(continueNode, "  continue\n");
    assertTostringMatches(continueNode, "continue\n");
  }

  @Test
  public void forStatement() {
    assertStmtIndentedPrettyMatches(
        LOCAL_LEVEL,
        join("for x in y:",
             "  print(x)"),
        join("  for x in y:",
             "    print(x)",
             ""));
    assertStmtTostringMatches(
        LOCAL_LEVEL,
        join("for x in y:",
             "  print(x)"),
        "for x in y: ...\n");

    assertStmtIndentedPrettyMatches(
        LOCAL_LEVEL,
        join("for x in y:",
             "  pass"),
        join("  for x in y:",
             "    pass",
             ""));
    assertStmtTostringMatches(
        LOCAL_LEVEL,
        join("for x in y:",
             "  pass"),
        "for x in y: ...\n");
  }

  @Test
  public void ifStatement() {
    assertStmtIndentedPrettyMatches(
        LOCAL_LEVEL,
        join("if True:",
             "  print(x)"),
        join("  if True:",
             "    print(x)",
             ""));
    assertStmtTostringMatches(
        LOCAL_LEVEL,
        join("if True:",
             "  print(x)"),
        "if True: ...\n");

    assertStmtIndentedPrettyMatches(
        LOCAL_LEVEL,
        join("if True:",
             "  print(x)",
             "elif False:",
             "  print(y)",
             "else:",
             "  print(z)"),
        join("  if True:",
             "    print(x)",
             "  elif False:",
             "    print(y)",
             "  else:",
             "    print(z)",
             ""));
    assertStmtTostringMatches(
        LOCAL_LEVEL,
        join("if True:",
            "  print(x)",
            "elif False:",
            "  print(y)",
            "else:",
            "  print(z)"),
        "if True: ...\n");
  }

  @Test
  public void loadStatement() {
    // load("foo.bzl", a="A", "B")
    ASTNode loadStatement =
        new LoadStatement(
            new StringLiteral("foo.bzl"),
            ImmutableList.of(
                new LoadStatement.Binding(Identifier.of("a"), Identifier.of("A")),
                new LoadStatement.Binding(Identifier.of("B"), Identifier.of("B"))));
    assertIndentedPrettyMatches(
        loadStatement,
        "  load(\"foo.bzl\", a=\"A\", \"B\")\n");
    assertTostringMatches(
        loadStatement,
        "load(\"foo.bzl\", a=\"A\", \"B\")\n");
  }

  @Test
  public void returnStatement() {
    assertIndentedPrettyMatches(
        new ReturnStatement(new StringLiteral("foo")),
        "  return \"foo\"\n");
    assertTostringMatches(
        new ReturnStatement(new StringLiteral("foo")),
        "return \"foo\"\n");

    assertIndentedPrettyMatches(new ReturnStatement(Identifier.of("None")), "  return None\n");
    assertTostringMatches(new ReturnStatement(Identifier.of("None")), "return None\n");

    assertIndentedPrettyMatches(new ReturnStatement(null), "  return\n");
    assertTostringMatches(new ReturnStatement(null), "return\n");
  }

  // Miscellaneous.

  @Test
  public void buildFileAST() {
    ASTNode node = parseBuildFileASTWithoutValidation("print(x)\nprint(y)");
    assertIndentedPrettyMatches(
        node,
        join("  print(x)",
             "  print(y)",
             ""));
    assertTostringMatches(
        node,
        "<BuildFileAST with 2 statements>");
  }

  @Test
  public void comment() {
    Comment node = new Comment("foo");
    assertIndentedPrettyMatches(node, "  # foo");
    assertTostringMatches(node, "foo");
  }

  /* Not tested explicitly because they're covered implicitly by tests for other nodes:
   * - LValue
   * - DictionaryEntryLiteral
   * - passed arguments / formal parameters
   * - ConditionalStatements
   */
}
