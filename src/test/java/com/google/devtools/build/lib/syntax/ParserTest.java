// Copyright 2014 Google Inc. All rights reserved.
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
import static com.google.common.truth.Truth.assertWithMessage;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.syntax.DictionaryLiteral.DictionaryEntryLiteral;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.List;

/**
 *  Tests of parser behaviour.
 */
@RunWith(JUnit4.class)
public class ParserTest extends AbstractParserTestCase {

  private static String getText(String text, ASTNode node) {
    return text.substring(node.getLocation().getStartOffset(),
                          node.getLocation().getEndOffset());
  }

  // helper func for testListLiterals:
  private static int getIntElem(DictionaryEntryLiteral entry, boolean key) {
    return ((IntegerLiteral) (key ? entry.getKey() : entry.getValue())).getValue();
  }

  // helper func for testListLiterals:
  private static DictionaryEntryLiteral getElem(DictionaryLiteral list, int index) {
    return list.getEntries().get(index);
  }

  // helper func for testListLiterals:
  private static int getIntElem(ListLiteral list, int index) {
    return ((IntegerLiteral) list.getElements().get(index)).getValue();
  }

  // helper func for testListLiterals:
  private static Expression getElem(ListLiteral list, int index) {
    return list.getElements().get(index);
  }

  // helper func for testing arguments:
  private static Expression getArg(FuncallExpression f, int index) {
    return f.getArguments().get(index).getValue();
  }

  @Test
  public void testPrecedence1() throws Exception {
    BinaryOperatorExpression e =
      (BinaryOperatorExpression) parseExpr("'%sx' % 'foo' + 'bar'");

    assertEquals(Operator.PLUS, e.getOperator());
  }

  @Test
  public void testPrecedence2() throws Exception {
    BinaryOperatorExpression e =
      (BinaryOperatorExpression) parseExpr("('%sx' % 'foo') + 'bar'");
    assertEquals(Operator.PLUS, e.getOperator());
  }

  @Test
  public void testPrecedence3() throws Exception {
    BinaryOperatorExpression e =
      (BinaryOperatorExpression) parseExpr("'%sx' % ('foo' + 'bar')");
    assertEquals(Operator.PERCENT, e.getOperator());
  }

  @Test
  public void testPrecedence4() throws Exception {
    BinaryOperatorExpression e =
        (BinaryOperatorExpression) parseExpr("1 + - (2 - 3)");
    assertEquals(Operator.PLUS, e.getOperator());
  }

  @Test
  public void testUnaryMinusExpr() throws Exception {
    FuncallExpression e = (FuncallExpression) parseExpr("-5");
    FuncallExpression e2 = (FuncallExpression) parseExpr("- 5");

    assertEquals("-", e.getFunction().getName());
    assertEquals("-", e2.getFunction().getName());

    assertThat(e.getArguments()).hasSize(1);
    assertEquals(1, e.getNumPositionalArguments());

    IntegerLiteral arg0 = (IntegerLiteral) e.getArguments().get(0).getValue();
    assertEquals(5, (int) arg0.getValue());
  }

  @Test
  public void testFuncallExpr() throws Exception {
    FuncallExpression e = (FuncallExpression) parseExpr("foo(1, 2, bar=wiz)");

    Ident ident = e.getFunction();
    assertEquals("foo", ident.getName());

    assertThat(e.getArguments()).hasSize(3);
    assertEquals(2, e.getNumPositionalArguments());

    IntegerLiteral arg0 = (IntegerLiteral) e.getArguments().get(0).getValue();
    assertEquals(1, (int) arg0.getValue());

    IntegerLiteral arg1 = (IntegerLiteral) e.getArguments().get(1).getValue();
    assertEquals(2, (int) arg1.getValue());

    Argument.Passed arg2 = e.getArguments().get(2);
    assertEquals("bar", arg2.getName());
    Ident arg2val = (Ident) arg2.getValue();
    assertEquals("wiz", arg2val.getName());
  }

  @Test
  public void testMethCallExpr() throws Exception {
    FuncallExpression e =
      (FuncallExpression) parseExpr("foo.foo(1, 2, bar=wiz)");

    Ident ident = e.getFunction();
    assertEquals("foo", ident.getName());

    assertThat(e.getArguments()).hasSize(3);
    assertEquals(2, e.getNumPositionalArguments());

    IntegerLiteral arg0 = (IntegerLiteral) e.getArguments().get(0).getValue();
    assertEquals(1, (int) arg0.getValue());

    IntegerLiteral arg1 = (IntegerLiteral) e.getArguments().get(1).getValue();
    assertEquals(2, (int) arg1.getValue());

    Argument.Passed arg2 = e.getArguments().get(2);
    assertEquals("bar", arg2.getName());
    Ident arg2val = (Ident) arg2.getValue();
    assertEquals("wiz", arg2val.getName());
  }

  @Test
  public void testChainedMethCallExpr() throws Exception {
    FuncallExpression e =
      (FuncallExpression) parseExpr("foo.replace().split(1)");

    Ident ident = e.getFunction();
    assertEquals("split", ident.getName());

    assertThat(e.getArguments()).hasSize(1);
    assertEquals(1, e.getNumPositionalArguments());

    IntegerLiteral arg0 = (IntegerLiteral) e.getArguments().get(0).getValue();
    assertEquals(1, (int) arg0.getValue());
  }

  @Test
  public void testPropRefExpr() throws Exception {
    DotExpression e = (DotExpression) parseExpr("foo.foo");

    Ident ident = e.getField();
    assertEquals("foo", ident.getName());
  }

  @Test
  public void testStringMethExpr() throws Exception {
    FuncallExpression e = (FuncallExpression) parseExpr("'foo'.foo()");

    Ident ident = e.getFunction();
    assertEquals("foo", ident.getName());

    assertThat(e.getArguments()).isEmpty();
  }

  @Test
  public void testStringLiteralOptimizationValue() throws Exception {
    StringLiteral l = (StringLiteral) parseExpr("'abc' + 'def'");
    assertEquals("abcdef", l.value);
  }

  @Test
  public void testStringLiteralOptimizationToString() throws Exception {
    StringLiteral l = (StringLiteral) parseExpr("'abc' + 'def'");
    assertEquals("'abcdef'", l.toString());
  }

  @Test
  public void testStringLiteralOptimizationLocation() throws Exception {
    StringLiteral l = (StringLiteral) parseExpr("'abc' + 'def'");
    assertEquals(0, l.getLocation().getStartOffset());
    assertEquals(13, l.getLocation().getEndOffset());
  }

  @Test
  public void testStringLiteralOptimizationDifferentQuote() throws Exception {
    assertThat(parseExpr("'abc' + \"def\"")).isInstanceOf(BinaryOperatorExpression.class);
  }

  @Test
  public void testSubstring() throws Exception {
    FuncallExpression e = (FuncallExpression) parseExpr("'FOO.CC'[:].lower()[1:]");
    assertEquals("$slice", e.getFunction().getName());
    assertThat(e.getArguments()).hasSize(2);

    e = (FuncallExpression) parseExpr("'FOO.CC'.lower()[1:].startswith('oo')");
    assertEquals("startswith", e.getFunction().getName());
    assertThat(e.getArguments()).hasSize(1);

    e = (FuncallExpression) parseExpr("'FOO.CC'[1:][:2]");
    assertEquals("$slice", e.getFunction().getName());
    assertThat(e.getArguments()).hasSize(2);
  }

  private void assertLocation(int start, int end, Location location)
      throws Exception {
    int actualStart = location.getStartOffset();
    int actualEnd = location.getEndOffset();

    if (actualStart != start || actualEnd != end) {
      fail("Expected location = [" + start + ", " + end + "), found ["
          + actualStart + ", " + actualEnd + ")");
    }
  }

  @Test
  public void testErrorRecovery() throws Exception {
    syntaxEvents.setFailFast(false);

    String expr = "f(1, [x for foo foo foo], 3)";
    FuncallExpression e = (FuncallExpression) parseExpr(expr);

    syntaxEvents.assertContainsEvent("syntax error at 'foo'");

    // Test that the actual parameters are: (1, $error$, 3):

    Ident ident = e.getFunction();
    assertEquals("f", ident.getName());

    assertThat(e.getArguments()).hasSize(3);
    assertEquals(3, e.getNumPositionalArguments());

    IntegerLiteral arg0 = (IntegerLiteral) e.getArguments().get(0).getValue();
    assertEquals(1, (int) arg0.getValue());

    Argument.Passed arg1 = e.getArguments().get(1);
    Ident arg1val = ((Ident) arg1.getValue());
    assertEquals("$error$", arg1val.getName());

    assertLocation(5, 24, arg1val.getLocation());
    assertEquals("[x for foo foo foo]", expr.substring(5, 24));
    assertEquals(25, arg1val.getLocation().getEndLineAndColumn().getColumn());

    IntegerLiteral arg2 = (IntegerLiteral) e.getArguments().get(2).getValue();
    assertEquals(3, (int) arg2.getValue());
  }

  @Test
  public void testDoesntGetStuck() throws Exception {
    syntaxEvents.setFailFast(false);

    // Make sure the parser does not get stuck when trying
    // to parse an expression containing a syntax error.
    // This usually results in OutOfMemoryError because the
    // parser keeps filling up the error log.
    // We need to make sure that we will always advance
    // in the token stream.
    parseExpr("f(1, ], 3)");
    parseExpr("f(1, ), 3)");
    parseExpr("[ ) for v in 3)");

    syntaxEvents.assertContainsEvent(""); // "" matches any;
                                          // i.e. there were some events
  }

  @Test
  public void testSecondaryLocation() {
    String expr = "f(1 % 2)";
    FuncallExpression call = (FuncallExpression) parseExpr(expr);
    Argument.Passed arg = call.getArguments().get(0);
    assertThat(arg.getLocation().getEndOffset()).isLessThan(call.getLocation().getEndOffset());
  }

  @Test
  public void testPrimaryLocation() {
    String expr = "f(1 + 2)";
    FuncallExpression call = (FuncallExpression) parseExpr(expr);
    Argument.Passed arg = call.getArguments().get(0);
    assertThat(arg.getLocation().getEndOffset()).isLessThan(call.getLocation().getEndOffset());
  }

  @Test
  public void testAssignLocation() {
    String expr = "a = b;c = d\n";
    List<Statement> statements = parseFile(expr);
    Statement statement = statements.get(0);
    assertEquals(5, statement.getLocation().getEndOffset());
  }

  @Test
  public void testAssign() {
    String expr = "list[0] = 5; dict['key'] = value\n";
    List<Statement> statements = parseFile(expr);
    assertThat(statements).hasSize(2);
  }

  @Test
  public void testInvalidAssign() {
    syntaxEvents.setFailFast(false);
    parseExpr("1 + (b = c)");
    syntaxEvents.assertContainsEvent("syntax error");
    syntaxEvents.collector().clear();
  }

  @Test
  public void testAugmentedAssign() throws Exception {
    assertEquals("[x = x + 1\n]", parseFile("x += 1").toString());
  }

  @Test
  public void testPrettyPrintFunctions() throws Exception {
    assertEquals("[x[1:3]\n]", parseFile("x[1:3]").toString());
    assertEquals("[str[42]\n]", parseFile("str[42]").toString());
    assertEquals("[ctx.new_file(['hello'])\n]", parseFile("ctx.new_file('hello')").toString());
    assertEquals("[new_file(['hello'])\n]", parseFile("new_file('hello')").toString());
  }

  @Test
  public void testFuncallLocation() {
    String expr = "a(b);c = d\n";
    List<Statement> statements = parseFile(expr);
    Statement statement = statements.get(0);
    assertEquals(4, statement.getLocation().getEndOffset());
  }

  @Test
  public void testSpecialFuncallLocation() throws Exception {
    List<Statement> statements = parseFile("-x\n");
    assertLocation(0, 3, statements.get(0).getLocation());

    statements = parseFile("arr[15]\n");
    assertLocation(0, 8, statements.get(0).getLocation());

    statements = parseFile("str[1:12]\n");
    assertLocation(0, 10, statements.get(0).getLocation());
  }

  @Test
  public void testListPositions() throws Exception {
    String expr = "[0,f(1),2]";
    ListLiteral list = (ListLiteral) parseExpr(expr);
    assertEquals("[0,f(1),2]", getText(expr, list));
    assertEquals("0",    getText(expr, getElem(list, 0)));
    assertEquals("f(1)", getText(expr, getElem(list, 1)));
    assertEquals("2",    getText(expr, getElem(list, 2)));
  }

  @Test
  public void testDictPositions() throws Exception {
    String expr = "{1:2,2:f(1),3:4}";
    DictionaryLiteral list = (DictionaryLiteral) parseExpr(expr);
    assertEquals("{1:2,2:f(1),3:4}", getText(expr, list));
    assertEquals("1:2",    getText(expr, getElem(list, 0)));
    assertEquals("2:f(1)", getText(expr, getElem(list, 1)));
    assertEquals("3:4",    getText(expr, getElem(list, 2)));
  }

  @Test
  public void testArgumentPositions() throws Exception {
    String stmt = "f(0,g(1,2),2)";
    FuncallExpression f = (FuncallExpression) parseExpr(stmt);
    assertEquals(stmt, getText(stmt, f));
    assertEquals("0",    getText(stmt, getArg(f, 0)));
    assertEquals("g(1,2)", getText(stmt, getArg(f, 1)));
    assertEquals("2",    getText(stmt, getArg(f, 2)));
  }

  @Test
  public void testListLiterals1() throws Exception {
    ListLiteral list = (ListLiteral) parseExpr("[0,1,2]");
    assertFalse(list.isTuple());
    assertThat(list.getElements()).hasSize(3);
    assertFalse(list.isTuple());
    for (int i = 0; i < 3; ++i) {
      assertEquals(i, getIntElem(list, i));
    }
  }

  @Test
  public void testTupleLiterals2() throws Exception {
    ListLiteral tuple = (ListLiteral) parseExpr("(0,1,2)");
    assertTrue(tuple.isTuple());
    assertThat(tuple.getElements()).hasSize(3);
    assertTrue(tuple.isTuple());
    for (int i = 0; i < 3; ++i) {
      assertEquals(i, getIntElem(tuple, i));
    }
  }

  @Test
  public void testTupleLiterals3() throws Exception {
    ListLiteral emptyTuple = (ListLiteral) parseExpr("()");
    assertTrue(emptyTuple.isTuple());
    assertThat(emptyTuple.getElements()).isEmpty();
  }

  @Test
  public void testTupleLiterals4() throws Exception {
    ListLiteral singletonTuple = (ListLiteral) parseExpr("(42,)");
    assertTrue(singletonTuple.isTuple());
    assertThat(singletonTuple.getElements()).hasSize(1);
    assertEquals(42, getIntElem(singletonTuple, 0));
  }

  @Test
  public void testTupleLiterals5() throws Exception {
    IntegerLiteral intLit = (IntegerLiteral) parseExpr("(42)"); // not a tuple!
    assertEquals(42, (int) intLit.getValue());
  }

  @Test
  public void testListLiterals6() throws Exception {
    ListLiteral emptyList = (ListLiteral) parseExpr("[]");
    assertFalse(emptyList.isTuple());
    assertThat(emptyList.getElements()).isEmpty();
  }

  @Test
  public void testListLiterals7() throws Exception {
    ListLiteral singletonList = (ListLiteral) parseExpr("[42,]");
    assertFalse(singletonList.isTuple());
    assertThat(singletonList.getElements()).hasSize(1);
    assertEquals(42, getIntElem(singletonList, 0));
  }

  @Test
  public void testListLiterals8() throws Exception {
    ListLiteral singletonList = (ListLiteral) parseExpr("[42]"); // a singleton
    assertFalse(singletonList.isTuple());
    assertThat(singletonList.getElements()).hasSize(1);
    assertEquals(42, getIntElem(singletonList, 0));
  }

  @Test
  public void testDictionaryLiterals() throws Exception {
    DictionaryLiteral dictionaryList =
      (DictionaryLiteral) parseExpr("{1:42}"); // a singleton dictionary
    assertThat(dictionaryList.getEntries()).hasSize(1);
    DictionaryEntryLiteral tuple = getElem(dictionaryList, 0);
    assertEquals(1, getIntElem(tuple, true));
    assertEquals(42, getIntElem(tuple, false));
  }

  @Test
  public void testDictionaryLiterals1() throws Exception {
    DictionaryLiteral dictionaryList =
      (DictionaryLiteral) parseExpr("{}"); // an empty dictionary
    assertThat(dictionaryList.getEntries()).isEmpty();
  }

  @Test
  public void testDictionaryLiterals2() throws Exception {
    DictionaryLiteral dictionaryList =
      (DictionaryLiteral) parseExpr("{1:42,}"); // a singleton dictionary
    assertThat(dictionaryList.getEntries()).hasSize(1);
    DictionaryEntryLiteral tuple = getElem(dictionaryList, 0);
    assertEquals(1, getIntElem(tuple, true));
    assertEquals(42, getIntElem(tuple, false));
  }

  @Test
  public void testDictionaryLiterals3() throws Exception {
    DictionaryLiteral dictionaryList = (DictionaryLiteral) parseExpr("{1:42,2:43,3:44}");
    assertThat(dictionaryList.getEntries()).hasSize(3);
    for (int i = 0; i < 3; i++) {
      DictionaryEntryLiteral tuple = getElem(dictionaryList, i);
      assertEquals(i + 1, getIntElem(tuple, true));
      assertEquals(i + 42, getIntElem(tuple, false));
    }
  }

  @Test
  public void testListLiterals9() throws Exception {
    ListLiteral singletonList =
      (ListLiteral) parseExpr("[ abi + opt_level + \'/include\' ]");
    assertFalse(singletonList.isTuple());
    assertThat(singletonList.getElements()).hasSize(1);
  }

  @Test
  public void testListComprehensionSyntax() throws Exception {
    syntaxEvents.setFailFast(false);

    parseExpr("[x for");
    syntaxEvents.assertContainsEvent("syntax error at 'newline'");
    syntaxEvents.collector().clear();

    parseExpr("[x for x");
    syntaxEvents.assertContainsEvent("syntax error at 'newline'");
    syntaxEvents.collector().clear();

    parseExpr("[x for x in");
    syntaxEvents.assertContainsEvent("syntax error at 'newline'");
    syntaxEvents.collector().clear();

    parseExpr("[x for x in []");
    syntaxEvents.assertContainsEvent("syntax error at 'newline'");
    syntaxEvents.collector().clear();

    parseExpr("[x for x for y in ['a']]");
    syntaxEvents.assertContainsEvent("syntax error at 'for'");
    syntaxEvents.collector().clear();
  }

  @Test
  public void testListComprehension() throws Exception {
    ListComprehension list =
      (ListComprehension) parseExpr(
          "['foo/%s.java' % x "
          + "for x in []]");
    assertThat(list.getLists()).hasSize(1);

    list = (ListComprehension) parseExpr("['foo/%s.java' % x "
        + "for x in ['bar', 'wiz', 'quux']]");
    assertThat(list.getLists()).hasSize(1);

    list = (ListComprehension) parseExpr("['%s/%s.java' % (x, y) "
        + "for x in ['foo', 'bar'] for y in ['baz', 'wiz', 'quux']]");
    assertThat(list.getLists()).hasSize(2);
  }

  @Test
  public void testParserContainsErrorsIfSyntaxException() throws Exception {
    syntaxEvents.setFailFast(false);
    parseExpr("'foo' %%");
    syntaxEvents.assertContainsEvent("syntax error at '%'");
  }

  @Test
  public void testParserDoesNotContainErrorsIfSuccess() throws Exception {
    parseExpr("'foo'");
  }

  @Test
  public void testParserContainsErrors() throws Exception {
    syntaxEvents.setFailFast(false);
    parseStmt("+");
    syntaxEvents.assertContainsEvent("syntax error at '+'");
  }

  @Test
  public void testSemicolonAndNewline() throws Exception {
    List<Statement> stmts = parseFile(
      "foo='bar'; foo(bar)" + '\n'
      + "" + '\n'
      + "foo='bar'; foo(bar)"
    );
    assertThat(stmts).hasSize(4);
  }

  @Test
  public void testSemicolonAndNewline2() throws Exception {
    syntaxEvents.setFailFast(false);
    List<Statement> stmts = parseFile(
      "foo='foo' error(bar)" + '\n'
      + "" + '\n'
    );
    syntaxEvents.assertContainsEvent("syntax error at 'error'");
    assertThat(stmts).hasSize(2);
  }

  @Test
  public void testExprAsStatement() throws Exception {
    List<Statement> stmts = parseFile(
      "li = []\n"
      + "li.append('a.c')\n"
      + "\"\"\" string comment \"\"\"\n"
      + "foo(bar)"
    );
    assertThat(stmts).hasSize(4);
  }

  @Test
  public void testParseBuildFileWithSingeRule() throws Exception {
    List<Statement> stmts = parseFile(
      "genrule(name = 'foo'," + '\n'
      + "   srcs = ['input.csv']," + '\n'
      + "   outs = [ 'result.txt'," + '\n'
      + "           'result.log']," + '\n'
      + "   cmd = 'touch result.txt result.log')" + '\n'
      );
    assertThat(stmts).hasSize(1);
  }

  @Test
  public void testParseBuildFileWithMultipleRules() throws Exception {
    List<Statement> stmts = parseFile(
      "genrule(name = 'foo'," + '\n'
      + "   srcs = ['input.csv']," + '\n'
      + "   outs = [ 'result.txt'," + '\n'
      + "           'result.log']," + '\n'
      + "   cmd = 'touch result.txt result.log')" + '\n'
      + "" + '\n'
      + "genrule(name = 'bar'," + '\n'
      + "   srcs = ['input.csv']," + '\n'
      + "   outs = [ 'graph.svg']," + '\n'
      + "   cmd = 'touch graph.svg')" + '\n'
      );
    assertThat(stmts).hasSize(2);
  }

  @Test
  public void testParseBuildFileWithComments() throws Exception {
    Parser.ParseResult result = parseFileWithComments(
      "# Test BUILD file" + '\n'
      + "# with multi-line comment" + '\n'
      + "" + '\n'
      + "genrule(name = 'foo'," + '\n'
      + "   srcs = ['input.csv']," + '\n'
      + "   outs = [ 'result.txt'," + '\n'
      + "           'result.log']," + '\n'
      + "   cmd = 'touch result.txt result.log')" + '\n'
      );
    assertThat(result.statements).hasSize(1);
    assertThat(result.comments).hasSize(2);
  }

  @Test
  public void testParseBuildFileWithManyComments() throws Exception {
    Parser.ParseResult result = parseFileWithComments(
      "# 1" + '\n'
      + "# 2" + '\n'
      + "" + '\n'
      + "# 4 " + '\n'
      + "# 5" + '\n'
      + "#" + '\n' // 6 - find empty comment for syntax highlighting
      + "# 7 " + '\n'
      + "# 8" + '\n'
      + "genrule(name = 'foo'," + '\n'
      + "   srcs = ['input.csv']," + '\n'
      + "   # 11" + '\n'
      + "   outs = [ 'result.txt'," + '\n'
      + "           'result.log'], # 13" + '\n'
      + "   cmd = 'touch result.txt result.log')" + '\n'
      + "# 15" + '\n'
      );
    assertThat(result.statements).hasSize(1); // Single genrule
    StringBuilder commentLines = new StringBuilder();
    for (Comment comment : result.comments) {
      // Comments start and end on the same line
      assertEquals(comment.getLocation().getStartLineAndColumn().getLine() + " ends on "
          + comment.getLocation().getEndLineAndColumn().getLine(),
          comment.getLocation().getStartLineAndColumn().getLine(),
          comment.getLocation().getEndLineAndColumn().getLine());
      commentLines.append('(');
      commentLines.append(comment.getLocation().getStartLineAndColumn().getLine());
      commentLines.append(',');
      commentLines.append(comment.getLocation().getStartLineAndColumn().getColumn());
      commentLines.append(") ");
    }
    assertWithMessage("Found: " + commentLines)
        .that(result.comments.size()).isEqualTo(10); // One per '#'
  }

  @Test
  public void testMissingComma() throws Exception {
    syntaxEvents.setFailFast(false);
    // Regression test.
    // Note: missing comma after name='foo'
    parseFile("genrule(name = 'foo'\n"
              + "      srcs = ['in'])");
    syntaxEvents.assertContainsEvent("syntax error at 'srcs'");
  }

  @Test
  public void testDoubleSemicolon() throws Exception {
    syntaxEvents.setFailFast(false);
    // Regression test.
    parseFile("x = 1; ; x = 2;");
    syntaxEvents.assertContainsEvent("syntax error at ';'");
  }

  @Test
  public void testFunctionDefinitionErrorRecovery() throws Exception {
    // Parser skips over entire function definitions, and reports a meaningful
    // error.
    syntaxEvents.setFailFast(false);
    List<Statement> stmts = parseFile(
        "x = 1;\n"
        + "def foo(x, y, **z):\n"
        + "  # a comment\n"
        + "  x = 2\n"
        + "  foo(bar)\n"
        + "  return z\n"
        + "x = 3");
    assertThat(stmts).hasSize(2);
  }

  @Test
  public void testFunctionDefinitionIgnored() throws Exception {
    // Parser skips over entire function definitions without reporting error,
    // when parsePython is set to true.
    List<Statement> stmts = parseFile(
        "x = 1;\n"
        + "def foo(x, y, **z):\n"
        + "  # a comment\n"
        + "  if true:"
        + "    x = 2\n"
        + "  foo(bar)\n"
        + "  return z\n"
        + "x = 3", true /* parsePython */);
    assertThat(stmts).hasSize(2);

    stmts = parseFile(
        "x = 1;\n"
        + "def foo(x, y, **z): return x\n"
        + "x = 3", true /* parsePython */);
    assertThat(stmts).hasSize(2);
  }

  @Test
  public void testMissingBlock() throws Exception {
    syntaxEvents.setFailFast(false);
    List<Statement> stmts = parseFile(
        "x = 1;\n"
        + "def foo(x):\n"
        + "x = 2;\n",
        true /* parsePython */);
    assertThat(stmts).hasSize(2);
    syntaxEvents.assertContainsEvent("expected an indented block");
  }

  @Test
  public void testInvalidDef() throws Exception {
    syntaxEvents.setFailFast(false);
    parseFile(
        "x = 1;\n"
        + "def foo(x)\n"
        + "x = 2;\n",
        true /* parsePython */);
    syntaxEvents.assertContainsEvent("syntax error at 'EOF'");
  }

  @Test
  public void testDefSingleLine() throws Exception {
    List<Statement> statements = parseFileForSkylark(
        "def foo(): x = 1; y = 2\n");
    FunctionDefStatement stmt = (FunctionDefStatement) statements.get(0);
    assertThat(stmt.getStatements()).hasSize(2);
  }

  @Test
  public void testSkipIfBlock() throws Exception {
    // Skip over 'if' blocks, when parsePython is set
    List<Statement> stmts = parseFile(
        "x = 1;\n"
        + "if x == 1:\n"
        + "  foo(x)\n"
        + "else:\n"
        + "  bar(x)\n"
        + "x = 3;\n",
        true /* parsePython */);
    assertThat(stmts).hasSize(2);
  }

  @Test
  public void testSkipIfBlockFail() throws Exception {
    // Do not parse 'if' blocks, when parsePython is not set
    syntaxEvents.setFailFast(false);
    List<Statement> stmts = parseFile(
        "x = 1;\n"
        + "if x == 1:\n"
        + "  x = 2\n"
        + "x = 3;\n",
        false /* no parsePython */);
    assertThat(stmts).hasSize(2);
    syntaxEvents.assertContainsEvent("This Python-style construct is not supported");
  }

  @Test
  public void testForLoopMultipleVariablesFail() throws Exception {
    // For loops with multiple variables are not allowed, when parsePython is not set
    syntaxEvents.setFailFast(false);
    List<Statement> stmts = parseFile(
        "[ i for i, j, k in [(1, 2, 3)] ]\n",
        false /* no parsePython */);
    assertThat(stmts).hasSize(1);
    syntaxEvents.assertContainsEvent("For loops with multiple variables are not yet supported");
  }

  @Test
  public void testForLoopMultipleVariables() throws Exception {
    // For loops with multiple variables is ok, when parsePython is set
    List<Statement> stmts1 = parseFile(
        "[ i for i, j, k in [(1, 2, 3)] ]\n",
        true /* parsePython */);
    assertThat(stmts1).hasSize(1);

    List<Statement> stmts2 = parseFile(
        "[ i for i, j in [(1, 2, 3)] ]\n",
        true /* parsePython */);
    assertThat(stmts2).hasSize(1);

    List<Statement> stmts3 = parseFile(
        "[ i for (i, j, k) in [(1, 2, 3)] ]\n",
        true /* parsePython */);
    assertThat(stmts3).hasSize(1);
  }

  @Test
  public void testForLoopBadSyntax() throws Exception {
    syntaxEvents.setFailFast(false);
    parseFile(
        "[1 for (a, b, c in var]\n",
        false /* no parsePython */);
    syntaxEvents.assertContainsEvent("syntax error");
  }

  @Test
  public void testForLoopBadSyntax2() throws Exception {
    syntaxEvents.setFailFast(false);
    parseFile(
        "[1 for () in var]\n",
        false /* no parsePython */);
    syntaxEvents.assertContainsEvent("syntax error");
  }

  @Test
  public void testFunCallBadSyntax() throws Exception {
    syntaxEvents.setFailFast(false);
    parseFile("f(1,\n");
    syntaxEvents.assertContainsEvent("syntax error");
  }

  @Test
  public void testFunCallBadSyntax2() throws Exception {
    syntaxEvents.setFailFast(false);
    parseFile("f(1, 5, ,)\n");
    syntaxEvents.assertContainsEvent("syntax error");
  }

  @Test
  public void testLoadNoSymbol() throws Exception {
    syntaxEvents.setFailFast(false);
    parseFileForSkylark("load('/foo/bar/file')\n");
    syntaxEvents.assertContainsEvent("syntax error");
  }

  @Test
  public void testLoadOneSymbol() throws Exception {
    List<Statement> statements = parseFileForSkylark(
        "load('/foo/bar/file', 'fun_test')\n");
    LoadStatement stmt = (LoadStatement) statements.get(0);
    assertEquals("/foo/bar/file.bzl", stmt.getImportPath().toString());
    assertThat(stmt.getSymbols()).hasSize(1);
  }

  @Test
  public void testLoadOneSymbolWithTrailingComma() throws Exception {
    List<Statement> statements = parseFileForSkylark(
        "load('/foo/bar/file', 'fun_test',)\n");
    LoadStatement stmt = (LoadStatement) statements.get(0);
    assertEquals("/foo/bar/file.bzl", stmt.getImportPath().toString());
    assertThat(stmt.getSymbols()).hasSize(1);
  }

  @Test
  public void testLoadMultipleSymbols() throws Exception {
    List<Statement> statements = parseFileForSkylark(
        "load('file', 'foo', 'bar')\n");
    LoadStatement stmt = (LoadStatement) statements.get(0);
    assertEquals("file.bzl", stmt.getImportPath().toString());
    assertThat(stmt.getSymbols()).hasSize(2);
  }

  @Test
  public void testLoadSyntaxError() throws Exception {
    syntaxEvents.setFailFast(false);
    parseFileForSkylark("load(non_quoted, 'a')\n");
    syntaxEvents.assertContainsEvent("syntax error");
  }

  @Test
  public void testLoadSyntaxError2() throws Exception {
    syntaxEvents.setFailFast(false);
    parseFileForSkylark("load('non_quoted', a)\n");
    syntaxEvents.assertContainsEvent("syntax error");
  }

  @Test
  public void testLoadNotAtTopLevel() throws Exception {
    syntaxEvents.setFailFast(false);
    parseFileForSkylark("if 1: load(8)\n");
    syntaxEvents.assertContainsEvent("function 'load' does not exist");
  }

  @Test
  public void testParseErrorNotComparison() throws Exception {
    syntaxEvents.setFailFast(false);
    parseFile("2 < not 3");
    syntaxEvents.assertContainsEvent("syntax error at 'not'");
  }

  @Test
  public void testNotWithArithmeticOperatorsBadSyntax() throws Exception {
    syntaxEvents.setFailFast(false);
    parseFile("0 + not 0");
    syntaxEvents.assertContainsEvent("syntax error at 'not'");
  }

  @Test
  public void testOptionalArgBeforeMandatoryArgInFuncDef() throws Exception {
    syntaxEvents.setFailFast(false);
    parseFileForSkylark("def func(a, b = 'a', c):\n  return 0\n");
    syntaxEvents.assertContainsEvent(
        "a mandatory positional parameter must not follow an optional parameter");
  }

  @Test
  public void testKwargBeforePositionalArg() throws Exception {
    syntaxEvents.setFailFast(false);
    parseFileForSkylark(
        "def func(a, b): return a + b\n"
        + "func(**{'b': 1}, 'a')");
    syntaxEvents.assertContainsEvent("unexpected tokens after kwarg");
  }

  @Test
  public void testDuplicateKwarg() throws Exception {
    syntaxEvents.setFailFast(false);
    parseFileForSkylark(
        "def func(a, b): return a + b\n"
        + "func(**{'b': 1}, **{'a': 2})");
    syntaxEvents.assertContainsEvent("unexpected tokens after kwarg");
  }

  @Test
  public void testUnnamedStar() throws Exception {
    syntaxEvents.setFailFast(false);
    parseFileForSkylark(
        "def func(a, b1=2, b2=3, *, c1, c2, d=4): return a + b1 + b2 + c1 + c2 + d\n");
    syntaxEvents.assertContainsEvent("no star, star-star or named-only parameters (for now)");
  }

  @Test
  public void testTopLevelForFails() throws Exception {
    syntaxEvents.setFailFast(false);
    parseFileForSkylark("for i in []: 0\n");
    syntaxEvents.assertContainsEvent(
        "for loops are not allowed on top-level. Put it into a function");
  }

  @Test
  public void testNestedFunctionFails() throws Exception {
    syntaxEvents.setFailFast(false);
    parseFileForSkylark(
          "def func(a):\n"
        + "  def bar(): return 0\n"
        + "  return bar()\n");
    syntaxEvents.assertContainsEvent(
        "nested functions are not allowed. Move the function to top-level");
  }

  @Test
  public void testIncludeFailureSkylark() throws Exception {
    syntaxEvents.setFailFast(false);
    parseFileForSkylark("include('//foo:bar')");
    syntaxEvents.assertContainsEvent("function 'include' does not exist");
  }

  @Test
  public void testIncludeFailure() throws Exception {
    syntaxEvents.setFailFast(false);
    parseFile("include('nonexistent')\n");
    syntaxEvents.assertContainsEvent("Invalid label 'nonexistent'");
  }
}
