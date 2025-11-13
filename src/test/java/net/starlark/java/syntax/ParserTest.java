// Copyright 2014 The Bazel Authors. All rights reserved.
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
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.collect.Sets;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import javax.annotation.Nullable;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Tests of parser. */
@RunWith(TestParameterInjector.class)
public final class ParserTest {

  private final List<SyntaxError> events = new ArrayList<>();
  private boolean failFast = true;
  private FileOptions fileOptions = FileOptions.DEFAULT;

  private SyntaxError assertContainsError(String expectedMessage) {
    return LexerTest.assertContainsError(events, expectedMessage);
  }

  private void setFailFast(boolean failFast) {
    this.failFast = failFast;
  }

  private void setFileOptions(FileOptions fileOptions) {
    this.fileOptions = fileOptions;
  }

  // Joins the lines, parse, and returns an expression.
  private Expression parseExpression(String... lines) throws SyntaxError.Exception {
    ParserInput input = ParserInput.fromLines(lines);
    return Expression.parse(input, fileOptions);
  }

  // Parses the expression, asserts that parsing fails,
  // and returns the first error message.
  private String parseExpressionError(String src) {
    ParserInput input = ParserInput.fromLines(src);
    try {
      Expression.parse(input, fileOptions);
      throw new AssertionError("parseExpressionError() succeeded unexpectedly: " + src);
    } catch (SyntaxError.Exception ex) {
      return ex.errors().get(0).toString();
    }
  }

  // Parses the statement, asserts that parsing fails, and returns the first error message.
  private String parseStatementError(String src) throws SyntaxError.Exception {
    ParserInput input = ParserInput.fromLines(src);
    StarlarkFile file = StarlarkFile.parse(input, fileOptions);
    if (file.ok()) {
      throw new AssertionError("parseStatementError() succeeded unexpectedly: " + src);
    }
    return file.errors().get(0).toString();
  }

  // Joins the lines, parse, and returns a type expression.
  private Expression parseTypeExpression(String... lines) throws SyntaxError.Exception {
    ParserInput input = ParserInput.fromLines(lines);
    return Expression.parseTypeExpression(input, fileOptions);
  }

  // Parses the type expression, asserts that parsing fails, and returns the first error message.
  private String parseTypeExpressionError(String src) {
    ParserInput input = ParserInput.fromLines(src);
    try {
      Expression.parseTypeExpression(input, fileOptions);
      throw new AssertionError("parseTypeExpressionError() succeeded unexpectedly: " + src);
    } catch (SyntaxError.Exception ex) {
      return ex.errors().get(0).toString();
    }
  }

  // Joins the lines, parses, and returns a file.
  // Errors are added to this.events, or thrown if this.failFast;
  private StarlarkFile parseFile(String... lines) throws SyntaxError.Exception {
    ParserInput input = ParserInput.fromLines(lines);
    StarlarkFile file = StarlarkFile.parse(input, fileOptions);
    if (!file.ok()) {
      if (failFast) {
        throw new SyntaxError.Exception(file.errors());
      }
      // TODO(adonovan): return these, and eliminate a stateful field.
      events.addAll(file.errors());
    }
    return file;
  }

  // Joins the lines, parses, and returns the sole statement.
  private Statement parseStatement(String... lines) throws SyntaxError.Exception {
    return Iterables.getOnlyElement(parseStatements(lines));
  }

  // Joins the lines, parses, and returns the statements.
  private ImmutableList<Statement> parseStatements(String... lines) throws SyntaxError.Exception {
    return parseFile(lines).getStatements();
  }

  private static String getText(String text, Node node) {
    return text.substring(node.getStartOffset(), node.getEndOffset());
  }

  private static void assertLocation(int start, int end, Node node) throws Exception {
    int actualStart = node.getStartOffset();
    int actualEnd = node.getEndOffset();

    if (actualStart != start || actualEnd != end) {
      fail("Expected location = [" + start + ", " + end + "), found ["
          + actualStart + ", " + actualEnd + ")");
    }
  }

  // helper func for testListExpressions:
  private static Number getIntElem(DictExpression.Entry entry, boolean key) {
    return ((IntLiteral) (key ? entry.getKey() : entry.getValue())).getValue();
  }

  // helper func for testListExpressions:
  private static Number getIntElem(ListExpression list, int index) {
    return ((IntLiteral) list.getElements().get(index)).getValue();
  }

  // helper func for testListExpressions:
  private static DictExpression.Entry getElem(DictExpression list, int index) {
    return list.getEntries().get(index);
  }

  // helper func for testListExpressions:
  private static Expression getElem(ListExpression list, int index) {
    return list.getElements().get(index);
  }

  // helper func for testing arguments:
  private static Expression getArg(CallExpression f, int index) {
    return f.getArguments().get(index).getValue();
  }

  @Test
  public void testPrecedence1() throws Exception {
    BinaryOperatorExpression e =
      (BinaryOperatorExpression) parseExpression("'%sx' % 'foo' + 'bar'");

    assertThat(e.getOperator()).isEqualTo(TokenKind.PLUS);
  }

  @Test
  public void testPrecedence2() throws Exception {
    BinaryOperatorExpression e =
      (BinaryOperatorExpression) parseExpression("('%sx' % 'foo') + 'bar'");
    assertThat(e.getOperator()).isEqualTo(TokenKind.PLUS);
  }

  @Test
  public void testPrecedence3() throws Exception {
    BinaryOperatorExpression e =
      (BinaryOperatorExpression) parseExpression("'%sx' % ('foo' + 'bar')");
    assertThat(e.getOperator()).isEqualTo(TokenKind.PERCENT);
  }

  @Test
  public void testPrecedence4() throws Exception {
    BinaryOperatorExpression e =
        (BinaryOperatorExpression) parseExpression("1 + - (2 - 3)");
    assertThat(e.getOperator()).isEqualTo(TokenKind.PLUS);
  }

  @Test
  public void testPrecedence5() throws Exception {
    BinaryOperatorExpression e =
        (BinaryOperatorExpression) parseExpression("2 * x | y + 1");
    assertThat(e.getOperator()).isEqualTo(TokenKind.PIPE);
  }

  @Test
  public void testNonAssociativeOperators() throws Exception {
    assertThat(parseExpressionError("0 < 2 < 4"))
        .contains("Operator '<' is not associative with operator '<'");
    assertThat(parseExpressionError("0 == 2 < 4"))
        .contains("Operator '==' is not associative with operator '<'");
    assertThat(parseExpressionError("1 in [1, 2] == True"))
        .contains("Operator 'in' is not associative with operator '=='");
    assertThat(parseExpressionError("1 >= 2 <= 3"))
        .contains("Operator '>=' is not associative with operator '<='");
  }

  @Test
  public void testNonAssociativeOperatorsWithParens() throws Exception {
    parseExpression("(0 < 2) < 4");
    parseExpression("(0 == 2) < 4");
    parseExpression("(1 in [1, 2]) == True");
    parseExpression("1 >= (2 <= 3)");
  }

  @Test
  public void testUnaryMinusExpr() throws Exception {
    UnaryOperatorExpression e = (UnaryOperatorExpression) parseExpression("-5");
    UnaryOperatorExpression e2 = (UnaryOperatorExpression) parseExpression("- 5");

    IntLiteral i = (IntLiteral) e.getX();
    assertThat(i.getValue()).isEqualTo(5);
    IntLiteral i2 = (IntLiteral) e2.getX();
    assertThat(i2.getValue()).isEqualTo(5);
    assertLocation(0, 2, e);
    assertLocation(0, 3, e2);
  }

  @Test
  public void testFuncallExpr() throws Exception {
    CallExpression e = (CallExpression) parseExpression("foo[0](1, 2, bar=wiz)");

    IndexExpression function = (IndexExpression) e.getFunction();
    Identifier functionList = (Identifier) function.getObject();
    assertThat(functionList.getName()).isEqualTo("foo");
    IntLiteral listIndex = (IntLiteral) function.getKey();
    assertThat(listIndex.getValue()).isEqualTo(0);

    assertThat(e.getArguments()).hasSize(3);
    assertThat(e.getNumPositionalArguments()).isEqualTo(2);

    IntLiteral arg0 = (IntLiteral) e.getArguments().get(0).getValue();
    assertThat((int) arg0.getValue()).isEqualTo(1);

    IntLiteral arg1 = (IntLiteral) e.getArguments().get(1).getValue();
    assertThat((int) arg1.getValue()).isEqualTo(2);

    Argument arg2 = e.getArguments().get(2);
    assertThat(arg2.getName()).isEqualTo("bar");
    Identifier arg2val = (Identifier) arg2.getValue();
    assertThat(arg2val.getName()).isEqualTo("wiz");
  }

  @Test
  public void testMethCallExpr() throws Exception {
    CallExpression e = (CallExpression) parseExpression("foo.foo(1, 2, bar=wiz)");

    DotExpression dotExpression = (DotExpression) e.getFunction();
    assertThat(dotExpression.getField().getName()).isEqualTo("foo");

    assertThat(e.getArguments()).hasSize(3);
    assertThat(e.getNumPositionalArguments()).isEqualTo(2);

    IntLiteral arg0 = (IntLiteral) e.getArguments().get(0).getValue();
    assertThat((int) arg0.getValue()).isEqualTo(1);

    IntLiteral arg1 = (IntLiteral) e.getArguments().get(1).getValue();
    assertThat((int) arg1.getValue()).isEqualTo(2);

    Argument arg2 = e.getArguments().get(2);
    assertThat(arg2.getName()).isEqualTo("bar");
    Identifier arg2val = (Identifier) arg2.getValue();
    assertThat(arg2val.getName()).isEqualTo("wiz");
  }

  @Test
  public void testChainedMethCallExpr() throws Exception {
    CallExpression e = (CallExpression) parseExpression("foo.replace().split(1)");

    DotExpression dotExpr = (DotExpression) e.getFunction();
    assertThat(dotExpr.getField().getName()).isEqualTo("split");

    assertThat(e.getArguments()).hasSize(1);
    assertThat(e.getNumPositionalArguments()).isEqualTo(1);

    IntLiteral arg0 = (IntLiteral) e.getArguments().get(0).getValue();
    assertThat((int) arg0.getValue()).isEqualTo(1);
  }

  @Test
  public void testPropRefExpr() throws Exception {
    DotExpression e = (DotExpression) parseExpression("foo.foo");

    Identifier ident = e.getField();
    assertThat(ident.getName()).isEqualTo("foo");
  }

  @Test
  public void testStringMethExpr() throws Exception {
    CallExpression e = (CallExpression) parseExpression("'foo'.foo()");

    DotExpression dotExpression = (DotExpression) e.getFunction();
    assertThat(dotExpression.getField().getName()).isEqualTo("foo");

    assertThat(e.getArguments()).isEmpty();
  }

  @Test
  public void testStringLiteralOptimizationValue() throws Exception {
    StringLiteral l = (StringLiteral) parseExpression("'abc' + 'def'");
    assertThat(l.getValue()).isEqualTo("abcdef");
  }

  @Test
  public void testStringLiteralOptimizationToString() throws Exception {
    StringLiteral l = (StringLiteral) parseExpression("'abc' + 'def'");
    assertThat(l.toString()).isEqualTo("\"abcdef\"");
  }

  @Test
  public void testStringLiteralOptimizationLocation() throws Exception {
    StringLiteral l = (StringLiteral) parseExpression("'abc' + 'def'");
    assertThat(l.getStartOffset()).isEqualTo(0);
    assertThat(l.getEndOffset()).isEqualTo(13);
  }

  @Test
  public void testStringLiteralOptimizationDifferentQuote() throws Exception {
    StringLiteral l = (StringLiteral) parseExpression("'abc' + \"def\"");
    assertThat(l.getStartOffset()).isEqualTo(0);
    assertThat(l.getEndOffset()).isEqualTo(13);
  }

  @Test
  public void testIndex() throws Exception {
    IndexExpression e = (IndexExpression) parseExpression("a[i]");
    assertThat(e.getObject().toString()).isEqualTo("a");
    assertThat(e.getKey().toString()).isEqualTo("i");
    assertLocation(0, 4, e);
  }

  @Test
  public void testSubstring() throws Exception {
    SliceExpression s = (SliceExpression) parseExpression("'FOO.CC'[:].lower()[1:]");
    assertThat(((IntLiteral) s.getStart()).getValue()).isEqualTo(1);

    CallExpression e = (CallExpression) parseExpression("'FOO.CC'.lower()[1:].startswith('oo')");
    DotExpression dotExpression = (DotExpression) e.getFunction();
    assertThat(dotExpression.getField().getName()).isEqualTo("startswith");
    assertThat(e.getArguments()).hasSize(1);

    s = (SliceExpression) parseExpression("'FOO.CC'[1:][:2]");
    assertThat(((IntLiteral) s.getStop()).getValue()).isEqualTo(2);
  }

  @Test
  public void testSlice() throws Exception {
    evalSlice("'0123'[:]", "", "", "");
    evalSlice("'0123'[1:]", 1, "", "");
    evalSlice("'0123'[:3]", "", 3, "");
    evalSlice("'0123'[::]", "", "", "");
    evalSlice("'0123'[1::]", 1, "", "");
    evalSlice("'0123'[:3:]", "", 3, "");
    evalSlice("'0123'[::-1]", "", "", -1);
    evalSlice("'0123'[1:3:]", 1, 3, "");
    evalSlice("'0123'[1::-1]", 1, "", -1);
    evalSlice("'0123'[:3:-1]", "", 3, -1);
    evalSlice("'0123'[1:3:-1]", 1, 3, -1);

    Expression slice = parseExpression("'0123'[1:3:-1]");
    assertLocation(0, 14, slice);
  }

  private void evalSlice(String statement, Object... expectedArgs) throws SyntaxError.Exception {
    SliceExpression e = (SliceExpression) parseExpression(statement);

    // There is no way to evaluate the expression here, so we rely on string comparison.
    String start = e.getStart() == null ? "" : e.getStart().toString();
    String stop = e.getStop() == null ? "" : e.getStop().toString();
    String step = e.getStep() == null ? "" : e.getStep().toString();

    assertThat(start).isEqualTo(expectedArgs[0].toString());
    assertThat(stop).isEqualTo(expectedArgs[1].toString());
    assertThat(step).isEqualTo(expectedArgs[2].toString());
  }

  @Test
  public void testErrorRecovery() throws Exception {
    setFailFast(false);

    // We call parseFile, not parseExpression, as the latter is all-or-nothing.
    String src = "f(1, [x for foo foo foo foo], 3)";
    CallExpression e = (CallExpression) ((ExpressionStatement) parseStatement(src)).getExpression();

    assertContainsError("syntax error at 'foo'");

    // Test that the arguments are (1, '[x for foo foo foo foo]', 3),
    // where the second, errant one is represented as an Identifier.

    Identifier ident = (Identifier) e.getFunction();
    assertThat(ident.getName()).isEqualTo("f");

    assertThat(e.getArguments()).hasSize(3);
    assertThat(e.getNumPositionalArguments()).isEqualTo(3);

    IntLiteral arg0 = (IntLiteral) e.getArguments().get(0).getValue();
    assertThat((int) arg0.getValue()).isEqualTo(1);

    Argument arg1 = e.getArguments().get(1);
    Identifier arg1val = ((Identifier) arg1.getValue());
    assertThat(arg1val.getName()).isEqualTo("[x for foo foo foo foo]");

    assertLocation(5, 28, arg1val);
    assertThat(src.substring(5, 28)).isEqualTo("[x for foo foo foo foo]");
    assertThat(arg1val.getEndLocation().column()).isEqualTo(29);

    IntLiteral arg2 = (IntLiteral) e.getArguments().get(2).getValue();
    assertThat((int) arg2.getValue()).isEqualTo(3);
  }

  @Test
  public void testDoesntGetStuck() throws Exception {
    // Make sure the parser does not get stuck when trying
    // to parse an expression containing a syntax error.
    // This usually results in OutOfMemoryError because the
    // parser keeps filling up the error log.
    // We need to make sure that we will always advance
    // in the token stream.
    parseExpressionError("f(1, ], 3)");
    parseExpressionError("f(1, ), 3)");
    parseExpressionError("[ ) for v in 3)");
  }

  @Test
  public void testPrimaryLocation() throws SyntaxError.Exception {
    String expr = "f(1 + 2)";
    CallExpression call = (CallExpression) parseExpression(expr);
    Argument arg = call.getArguments().get(0);
    assertThat(arg.getEndLocation()).isLessThan(call.getEndLocation());
  }

  @Test
  public void testAssignLocation() throws Exception {
    List<Statement> statements = parseStatements("a = b;c = d\n");
    Statement statement = statements.get(0);
    assertThat(statement.getEndOffset()).isEqualTo(5);
  }

  @Test
  public void testAssignKeyword() throws Exception {
    assertThat(parseExpressionError("with = 4")).contains("keyword 'with' not supported");
  }

  @Test
  public void testBreak() throws Exception {
    assertThat(parseExpressionError("break"))
        .contains("syntax error at 'break': expected expression");
  }

  @Test
  public void testTry() throws Exception {
    assertThat(parseExpressionError("try: 1 + 1"))
        .contains("'try' not supported, all exceptions are fatal");
  }

  @Test
  public void testDel() throws Exception {
    assertThat(parseExpressionError("del d['a']"))
        .contains("'del' not supported, use '.pop()' to delete");
  }

  @Test
  public void testTupleAssign() throws Exception {
    List<Statement> statements = parseStatements("list[0] = 5; dict['key'] = value\n");
    assertThat(statements).hasSize(2);
    assertThat(statements.get(0)).isInstanceOf(AssignmentStatement.class);
    assertThat(statements.get(1)).isInstanceOf(AssignmentStatement.class);
  }

  @Test
  public void testAssign() throws Exception {
    List<Statement> statements = parseStatements("a, b = 5\n");
    assertThat(statements).hasSize(1);
    assertThat(statements.get(0)).isInstanceOf(AssignmentStatement.class);
    AssignmentStatement assign = (AssignmentStatement) statements.get(0);
    assertThat(assign.getLHS()).isInstanceOf(ListExpression.class);
  }

  @Test
  public void testInvalidAssign() throws Exception {
    assertThat(parseExpressionError("1 + (b = c)")).contains("syntax error");
  }

  @Test
  public void testAugmentedAssign() throws Exception {
    assertThat(parseStatements("x += 1").toString()).isEqualTo("[x += 1\n]");
    assertThat(parseStatements("x -= 1").toString()).isEqualTo("[x -= 1\n]");
    assertThat(parseStatements("x *= 1").toString()).isEqualTo("[x *= 1\n]");
    assertThat(parseStatements("x /= 1").toString()).isEqualTo("[x /= 1\n]");
    assertThat(parseStatements("x %= 1").toString()).isEqualTo("[x %= 1\n]");
    assertThat(parseStatements("x |= 1").toString()).isEqualTo("[x |= 1\n]");
    assertThat(parseStatements("x &= 1").toString()).isEqualTo("[x &= 1\n]");
    assertThat(parseStatements("x <<= 1").toString()).isEqualTo("[x <<= 1\n]");
    assertThat(parseStatements("x >>= 1").toString()).isEqualTo("[x >>= 1\n]");
  }

  @Test
  public void testVarAnnotation_basic() throws Exception {
    setFileOptions(FileOptions.builder().allowTypeSyntax(true).build());

    Statement stmt = parseStatement("x : T");
    assertThat(stmt).isInstanceOf(VarStatement.class);
    assertThat(((VarStatement) stmt).getType().toString()).isEqualTo("T");

    stmt = parseStatement("x : T = 123");
    assertThat(stmt).isInstanceOf(AssignmentStatement.class);
    assertThat(((AssignmentStatement) stmt).getType().toString()).isEqualTo("T");
  }

  @Test
  public void testVarAnnotation_requiresTypeSyntax() throws Exception {
    setFileOptions(FileOptions.builder().allowTypeSyntax(false).build());
    assertThat(parseStatementError("x : T"))
        .contains("syntax error at ':': type annotations are disallowed");
    assertThat(parseStatementError("x : T = 123"))
        .contains("syntax error at ':': type annotations are disallowed");
  }

  @Test
  public void testVarAnnotation_takesOneIdentifier() throws Exception {
    setFileOptions(FileOptions.builder().allowTypeSyntax(true).build());

    // Complaint located at colon at column 6.
    String errMessage =
        ":1:6: syntax error at ':': type annotations must have a single identifier on the"
            + " left-hand side";

    assertThat(parseStatementError("x, y : T")).contains(errMessage);
    assertThat(parseStatementError("x, y : T = 123")).contains(errMessage);

    // This is *not* parsed as `x : (T, y)`, even though it might look unambiguous. Doing so would
    // require knowing what the comma is before we know whether we're in a VarStatement or
    // assignment statement. It's also not allowed by Python.
    assertThat(parseStatementError("x : T, y"))
        .contains(":1:6: syntax error at ',': expected newline");
    assertThat(parseStatementError("x : T, y = 123"))
        .contains(":1:6: syntax error at ',': expected newline");

    assertThat(parseStatementError("x[0] : T")).contains(errMessage);
    assertThat(parseStatementError("x[0] : T = 123")).contains(errMessage);

    // Only applicable to assignment, not VarStatement.
    assertThat(parseStatementError("(x : T, y) = 123"))
        // TODO: #27370 - Is there a reasonable way to produce a more informative error message
        // here, e.g. "type annotations are only allowed in assignment statements or variable
        // declarations"?
        .contains(":1:4: syntax error at ':': expected )");
  }

  @Test
  public void testVarAnnotation_notAllowedOnAugmentedAssignment() throws Exception {
    setFileOptions(FileOptions.builder().allowTypeSyntax(true).build());
    assertThat(parseStatementError("x : T += 123"))
        .contains(
            ":1:3: syntax error at ':': type annotations not allowed on augmented assignment"
                + " statements");
  }

  @Test
  public void testVarAnnotation_illegalTypeExpression_allowedWithFlag() throws Exception {
    setFileOptions(
        FileOptions.builder().allowTypeSyntax(true).allowArbitraryTypeExpressions(true).build());
    assertThat(((VarStatement) parseStatement("x : (lambda x: x)")).getType())
        .isInstanceOf(LambdaExpression.class);
    assertThat(((AssignmentStatement) parseStatement("x : (lambda x: x) = 123")).getType())
        .isInstanceOf(LambdaExpression.class);
  }

  @Test
  public void testAssignWithAnnotation_illegalTypeExpression_disallowedWithoutFlag()
      throws Exception {
    setFileOptions(
        FileOptions.builder().allowTypeSyntax(true).allowArbitraryTypeExpressions(false).build());
    assertThat(parseStatementError("x : (lambda x: x)"))
        .contains(":1:5: syntax error at '(': expected a type");
    assertThat(parseStatementError("x : (lambda x: x) = 123"))
        .contains(":1:5: syntax error at '(': expected a type");
  }

  @Test
  public void testPrettyPrintFunctions() throws Exception {
    assertThat(parseStatements("x[1:3]").toString()).isEqualTo("[x[1:3]\n]");
    assertThat(parseStatements("x[1:3:1]").toString()).isEqualTo("[x[1:3:1]\n]");
    assertThat(parseStatements("x[1:3:2]").toString()).isEqualTo("[x[1:3:2]\n]");
    assertThat(parseStatements("x[1::2]").toString()).isEqualTo("[x[1::2]\n]");
    assertThat(parseStatements("x[1:]").toString()).isEqualTo("[x[1:]\n]");
    assertThat(parseStatements("str[42]").toString()).isEqualTo("[str[42]\n]");
    assertThat(parseStatements("ctx.actions.declare_file('hello')").toString())
        .isEqualTo("[ctx.actions.declare_file(\"hello\")\n]");
    assertThat(parseStatements("new_file(\"hello\")").toString())
        .isEqualTo("[new_file(\"hello\")\n]");
  }

  @Test
  public void testEndLineAndColumnIsExclusive() throws Exception {
    // The behavior was 'inclusive' for a couple of years (see CL 170723732),
    // but this was a mistake. Arithmetic on half-open intervals is much simpler.
    AssignmentStatement stmt = (AssignmentStatement) parseStatement("a = b");
    assertThat(stmt.getLHS().getEndLocation().toString()).isEqualTo(":1:2");
  }

  @Test
  public void testFuncallLocation() throws Exception {
    List<Statement> statements = parseStatements("a(b);c = d\n");
    Statement statement = statements.get(0);
    assertThat(statement.getEndOffset()).isEqualTo(4);
  }

  @Test
  public void testListPositions() throws Exception {
    String expr = "[0,f(1),2]";
    assertExpressionLocationCorrect(expr);
    ListExpression list = (ListExpression) parseExpression(expr);
    assertThat(getText(expr, getElem(list, 0))).isEqualTo("0");
    assertThat(getText(expr, getElem(list, 1))).isEqualTo("f(1)");
    assertThat(getText(expr, getElem(list, 2))).isEqualTo("2");
  }

  @Test
  public void testDictPositions() throws Exception {
    String expr = "{1:2,2:f(1),3:4}";
    assertExpressionLocationCorrect(expr);
    DictExpression list = (DictExpression) parseExpression(expr);
    assertThat(getText(expr, getElem(list, 0))).isEqualTo("1:2");
    assertThat(getText(expr, getElem(list, 1))).isEqualTo("2:f(1)");
    assertThat(getText(expr, getElem(list, 2))).isEqualTo("3:4");
  }

  @Test
  public void testArgumentPositions() throws Exception {
    String expr = "f(0,g(1,2),2)";
    assertExpressionLocationCorrect(expr);
    CallExpression f = (CallExpression) parseExpression(expr);
    assertThat(getText(expr, getArg(f, 0))).isEqualTo("0");
    assertThat(getText(expr, getArg(f, 1))).isEqualTo("g(1,2)");
    assertThat(getText(expr, getArg(f, 2))).isEqualTo("2");
  }

  @Test
  public void testSuffixPosition() throws Exception {
    assertExpressionLocationCorrect("'a'.len");
    assertExpressionLocationCorrect("'a'[0]");
    assertExpressionLocationCorrect("'a'[0:1]");
  }

  @Test
  public void testTuplePosition() throws Exception {
    String input = "for a,b in []: pass";
    ForStatement stmt = (ForStatement) parseStatement(input);
    assertThat(getText(input, stmt.getVars())).isEqualTo("a,b");

    input = "for (a,b) in []: pass";
    stmt = (ForStatement) parseStatement(input);
    assertThat(getText(input, stmt.getVars())).isEqualTo("(a,b)");

    assertExpressionLocationCorrect("a, b");
    assertExpressionLocationCorrect("(a, b)");
  }

  @Test
  public void testComprehensionPosition() throws Exception {
    assertExpressionLocationCorrect("[[] for x in []]");
    assertExpressionLocationCorrect("{1: [] for x in []}");
  }

  @Test
  public void testUnaryOperationPosition() throws Exception {
    assertExpressionLocationCorrect("not True");
  }

  @Test
  public void testLoadStatementPosition() throws Exception {
    String input = "load(':foo.bzl', 'bar')";
    LoadStatement stmt = (LoadStatement) parseStatement(input);
    assertThat(getText(input, stmt)).isEqualTo(input);
    // Also try it with another token at the end (newline), which broke the location in the past.
    stmt = (LoadStatement) parseStatement(input + "\n");
    assertThat(getText(input, stmt)).isEqualTo(input);
  }

  @Test
  public void testElif() throws Exception {
    IfStatement ifA =
        (IfStatement)
            parseStatement(
                "if a:", //
                "  pass", "elif b:", "  pass", "else:", "  pass", "");
    IfStatement ifB = (IfStatement) Iterables.getOnlyElement(ifA.getElseBlock());
    assertThat(ifB.isElif()).isTrue();

    ifA =
        (IfStatement)
            parseStatement(
                "if a:", //
                "  pass",
                "else:",
                "  if b:",
                "    pass",
                "  else:",
                "    pass",
                "");
    ifB = (IfStatement) Iterables.getOnlyElement(ifA.getElseBlock());
    assertThat(ifB.isElif()).isFalse();
  }

  @Test
  public void testIfStatementPosition() throws Exception {
    assertStatementLocationCorrect("if True:\n  pass");
    assertStatementLocationCorrect("if True:\n  pass\nelif True:\n  pass");
    assertStatementLocationCorrect("if True:\n  pass\nelse:\n  pass");
  }

  @Test
  public void testForStatementPosition() throws Exception {
    assertStatementLocationCorrect("for x in []:\n  pass");
  }

  @Test
  public void testDefStatementPosition() throws Exception {
    assertStatementLocationCorrect("def foo():\n  pass");
  }

  private void assertStatementLocationCorrect(String stmtStr) throws SyntaxError.Exception {
    Statement stmt = parseStatement(stmtStr);
    assertThat(getText(stmtStr, stmt)).isEqualTo(stmtStr);
    // Also try it with another token at the end (newline), which broke the location in the past.
    stmt = parseStatement(stmtStr + "\n");
    assertThat(getText(stmtStr, stmt)).isEqualTo(stmtStr);
  }

  private void assertExpressionLocationCorrect(String exprStr) throws SyntaxError.Exception {
    Expression expr = parseExpression(exprStr);
    assertThat(getText(exprStr, expr)).isEqualTo(exprStr);
    // Also try it with another token at the end (newline), which broke the location in the past.
    expr = parseExpression(exprStr + "\n");
    assertThat(getText(exprStr, expr)).isEqualTo(exprStr);
  }

  @Test
  public void testForBreakContinuePass() throws Exception {
    List<Statement> file =
        parseStatements(
            "def foo():", //
            "  for i in [1, 2]:",
            "    break",
            "    continue",
            "    pass",
            "    break");
    assertThat(file).hasSize(1);
    List<Statement> body = ((DefStatement) file.get(0)).getBody();
    assertThat(body).hasSize(1);

    List<Statement> loop = ((ForStatement) body.get(0)).getBody();
    assertThat(loop).hasSize(4);

    assertThat(((FlowStatement) loop.get(0)).getFlowKind()).isEqualTo(TokenKind.BREAK);
    assertLocation(34, 39, loop.get(0));

    assertThat(((FlowStatement) loop.get(1)).getFlowKind()).isEqualTo(TokenKind.CONTINUE);
    assertLocation(44, 52, loop.get(1));

    assertThat(((FlowStatement) loop.get(2)).getFlowKind()).isEqualTo(TokenKind.PASS);
    assertLocation(57, 61, loop.get(2));

    assertThat(((FlowStatement) loop.get(3)).getFlowKind()).isEqualTo(TokenKind.BREAK);
    assertLocation(66, 71, loop.get(3));
  }

  @Test
  public void testListExpressions1() throws Exception {
    ListExpression list = (ListExpression) parseExpression("[0,1,2]");
    assertThat(list.isTuple()).isFalse();
    assertThat(list.getElements()).hasSize(3);
    assertThat(list.isTuple()).isFalse();
    for (int i = 0; i < 3; ++i) {
      assertThat(getIntElem(list, i)).isEqualTo(i);
    }
  }

  @Test
  public void testTupleLiterals2() throws Exception {
    ListExpression tuple = (ListExpression) parseExpression("(0,1,2)");
    assertThat(tuple.isTuple()).isTrue();
    assertThat(tuple.getElements()).hasSize(3);
    assertThat(tuple.isTuple()).isTrue();
    for (int i = 0; i < 3; ++i) {
      assertThat(getIntElem(tuple, i)).isEqualTo(i);
    }
  }

  @Test
  public void testTupleWithoutParens() throws Exception {
    ListExpression tuple = (ListExpression) parseExpression("0, 1, 2");
    assertThat(tuple.isTuple()).isTrue();
    assertThat(tuple.getElements()).hasSize(3);
    assertThat(tuple.isTuple()).isTrue();
    for (int i = 0; i < 3; ++i) {
      assertThat(getIntElem(tuple, i)).isEqualTo(i);
    }
  }

  @Test
  public void testTupleWithTrailingComma() throws Exception {
    // Unlike Python, we require parens here.
    assertThat(parseExpressionError("0, 1, 2, 3,")).contains("Trailing comma");
    assertThat(parseExpressionError("1 + 2,")).contains("Trailing comma");

    ListExpression tuple = (ListExpression) parseExpression("(0, 1, 2, 3,)");
    assertThat(tuple.isTuple()).isTrue();
    assertThat(tuple.getElements()).hasSize(4);
    assertThat(tuple.isTuple()).isTrue();
    for (int i = 0; i < 4; ++i) {
      assertThat(getIntElem(tuple, i)).isEqualTo(i);
    }
  }

  @Test
  public void testTupleLiterals3() throws Exception {
    ListExpression emptyTuple = (ListExpression) parseExpression("()");
    assertThat(emptyTuple.isTuple()).isTrue();
    assertThat(emptyTuple.getElements()).isEmpty();
  }

  @Test
  public void testTupleLiterals4() throws Exception {
    ListExpression singletonTuple = (ListExpression) parseExpression("(42,)");
    assertThat(singletonTuple.isTuple()).isTrue();
    assertThat(singletonTuple.getElements()).hasSize(1);
    assertThat(getIntElem(singletonTuple, 0)).isEqualTo(42);
  }

  @Test
  public void testTupleLiterals5() throws Exception {
    IntLiteral intLit = (IntLiteral) parseExpression("(42)"); // not a tuple!
    assertThat((int) intLit.getValue()).isEqualTo(42);
  }

  @Test
  public void testListExpressions6() throws Exception {
    ListExpression emptyList = (ListExpression) parseExpression("[]");
    assertThat(emptyList.isTuple()).isFalse();
    assertThat(emptyList.getElements()).isEmpty();
  }

  @Test
  public void testListExpressions7() throws Exception {
    ListExpression singletonList = (ListExpression) parseExpression("[42,]");
    assertThat(singletonList.isTuple()).isFalse();
    assertThat(singletonList.getElements()).hasSize(1);
    assertThat(getIntElem(singletonList, 0)).isEqualTo(42);
  }

  @Test
  public void testListExpressions8() throws Exception {
    ListExpression singletonList = (ListExpression) parseExpression("[42]"); // a singleton
    assertThat(singletonList.isTuple()).isFalse();
    assertThat(singletonList.getElements()).hasSize(1);
    assertThat(getIntElem(singletonList, 0)).isEqualTo(42);
  }

  @Test
  public void testDictExpressions() throws Exception {
    DictExpression dictionaryList =
        (DictExpression) parseExpression("{1:42}"); // a singleton dictionary
    assertThat(dictionaryList.getEntries()).hasSize(1);
    DictExpression.Entry tuple = getElem(dictionaryList, 0);
    assertThat(getIntElem(tuple, true)).isEqualTo(1);
    assertThat(getIntElem(tuple, false)).isEqualTo(42);
  }

  @Test
  public void testDictExpressions1() throws Exception {
    DictExpression dictionaryList = (DictExpression) parseExpression("{}"); // an empty dictionary
    assertThat(dictionaryList.getEntries()).isEmpty();
  }

  @Test
  public void testDictExpressions2() throws Exception {
    DictExpression dictionaryList =
        (DictExpression) parseExpression("{1:42,}"); // a singleton dictionary
    assertThat(dictionaryList.getEntries()).hasSize(1);
    DictExpression.Entry tuple = getElem(dictionaryList, 0);
    assertThat(getIntElem(tuple, true)).isEqualTo(1);
    assertThat(getIntElem(tuple, false)).isEqualTo(42);
  }

  @Test
  public void testDictExpressions3() throws Exception {
    DictExpression dictionaryList = (DictExpression) parseExpression("{1:42,2:43,3:44}");
    assertThat(dictionaryList.getEntries()).hasSize(3);
    for (int i = 0; i < 3; i++) {
      DictExpression.Entry tuple = getElem(dictionaryList, i);
      assertThat(getIntElem(tuple, true)).isEqualTo(i + 1);
      assertThat(getIntElem(tuple, false)).isEqualTo(i + 42);
    }
  }

  @Test
  public void testListExpressions9() throws Exception {
    ListExpression singletonList =
        (ListExpression) parseExpression("[ abi + opt_level + \'/include\' ]");
    assertThat(singletonList.isTuple()).isFalse();
    assertThat(singletonList.getElements()).hasSize(1);
  }

  @Test
  public void testListComprehensionSyntax() throws Exception {
    assertThat(parseExpressionError("[x for")).contains("syntax error at 'newline'");
    assertThat(parseExpressionError("[x for x")).contains("syntax error at 'newline'");
    assertThat(parseExpressionError("[x for x in")).contains("syntax error at 'newline'");
    assertThat(parseExpressionError("[x for x in []")).contains("syntax error at 'newline'");
    assertThat(parseExpressionError("[x for x for y in ['a']]")).contains("syntax error at 'for'");
    assertThat(parseExpressionError("[x for x for y in 1, 2]")).contains("syntax error at 'for'");
  }

  @Test
  public void testListComprehensionEmptyList() throws Exception {
    List<Comprehension.Clause> clauses =
        ((Comprehension) parseExpression("['foo/%s.java' % x for x in []]")).getClauses();
    assertThat(clauses).hasSize(1);
    Comprehension.For for0 = (Comprehension.For) clauses.get(0);
    assertThat(for0.getIterable().toString()).isEqualTo("[]");
    assertThat(for0.getVars().toString()).isEqualTo("x");
  }

  @Test
  public void testListComprehension() throws Exception {
    List<Comprehension.Clause> clauses =
        ((Comprehension) parseExpression("['foo/%s.java' % x for x in ['bar', 'wiz', 'quux']]"))
            .getClauses();
    assertThat(clauses).hasSize(1);
    Comprehension.For for0 = (Comprehension.For) clauses.get(0);
    assertThat(for0.getVars().toString()).isEqualTo("x");
    assertThat(for0.getIterable()).isInstanceOf(ListExpression.class);
  }

  @Test
  public void testForForListComprehension() throws Exception {
    List<Comprehension.Clause> clauses =
        ((Comprehension)
                parseExpression("['%s/%s.java' % (x, y) for x in ['foo', 'bar'] for y in list]"))
            .getClauses();
    assertThat(clauses).hasSize(2);
    Comprehension.For for0 = (Comprehension.For) clauses.get(0);
    assertThat(for0.getVars().toString()).isEqualTo("x");
    assertThat(for0.getIterable()).isInstanceOf(ListExpression.class);
    Comprehension.For for1 = (Comprehension.For) clauses.get(1);
    assertThat(for1.getVars().toString()).isEqualTo("y");
    assertThat(for1.getIterable()).isInstanceOf(Identifier.class);
  }

  @Test
  public void testParserRecovery() throws Exception {
    setFailFast(false);
    List<Statement> statements =
        parseStatements(
            "def foo():",
            "  a = 2 for 4", // parse error
            "  b = [3, 4]",
            "",
            "d = 4 ada", // parse error
            "",
            "def bar():",
            "  a = [3, 4]",
            "  b = 2 * * 5", // parse error
            "");

    assertContainsError("syntax error at 'for': expected newline");
    assertContainsError("syntax error at 'ada': expected newline");
    assertContainsError("syntax error at '*': expected expression");
    assertThat(events).hasSize(3);
    assertThat(statements).hasSize(3);
  }

  @Test
  public void testParserContainsErrors() throws Exception {
    setFailFast(false);
    parseFile("*");
    assertContainsError("syntax error at '*'");
  }

  @Test
  public void testSemicolonAndNewline() throws Exception {
    List<Statement> stmts =
        parseStatements(
            "foo='bar'; foo(bar)", //
            "",
            "foo='bar'; foo(bar)");
    assertThat(stmts).hasSize(4);
  }

  @Test
  public void testSemicolonAndNewline2() throws Exception {
    setFailFast(false);
    List<Statement> stmts = parseStatements("foo='foo' error(bar)", "", "");
    assertContainsError("syntax error at 'error'");
    assertThat(stmts).hasSize(1);
  }

  @Test
  public void testExprAsStatement() throws Exception {
    List<Statement> stmts =
        parseStatements(
            "li = []", //
            "li.append('a.c')",
            "\"\"\" string comment \"\"\"",
            "foo(bar)");
    assertThat(stmts).hasSize(4);
  }

  @Test
  public void testParseBuildFileWithSingleRule() throws Exception {
    List<Statement> stmts =
        parseStatements(
            "genrule(name = 'foo',", //
            "   srcs = ['input.csv'],",
            "   outs = [ 'result.txt',",
            "           'result.log'],",
            "   cmd = 'touch result.txt result.log')",
            "");
    assertThat(stmts).hasSize(1);
  }

  @Test
  public void testParseBuildFileWithMultipleRules() throws Exception {
    List<Statement> stmts =
        parseStatements(
            "genrule(name = 'foo',", //
            "   srcs = ['input.csv'],",
            "   outs = [ 'result.txt',",
            "           'result.log'],",
            "   cmd = 'touch result.txt result.log')",
            "",
            "genrule(name = 'bar',",
            "   srcs = ['input.csv'],",
            "   outs = [ 'graph.svg'],",
            "   cmd = 'touch graph.svg')");
    assertThat(stmts).hasSize(2);
  }

  @Test
  public void testParseBuildFileWithComments() throws Exception {
    StarlarkFile result =
        parseFile(
            "# Test BUILD file", //
            "# with multi-line comment",
            "",
            "genrule(name = 'foo',",
            "   srcs = ['input.csv'],",
            "   outs = [ 'result.txt',",
            "           'result.log'],",
            "   cmd = 'touch result.txt result.log')");
    assertThat(result.getStatements()).hasSize(1);
    assertThat(result.getComments()).hasSize(2);
  }

  @Test
  public void testParseBuildFileWithManyComments() throws Exception {
    StarlarkFile result =
        parseFile(
            "# 1", //
            "# 2",
            "",
            "# 4 ",
            "# 5",
            "#", // 6 - find empty comment for syntax highlighting
            "# 7 ",
            "# 8",
            "genrule(name = 'foo',",
            "   srcs = ['input.csv'],",
            "   # 11",
            "   outs = [ 'result.txt',",
            "           'result.log'], # 13",
            "   cmd = 'touch result.txt result.log')",
            "# 15");
    assertThat(result.getStatements()).hasSize(1); // Single genrule
    StringBuilder commentLines = new StringBuilder();
    for (Comment comment : result.getComments()) {
      // Comments start and end on the same line
      Location start = comment.getStartLocation();
      Location end = comment.getEndLocation();
      assertWithMessage("%s ends on %s", start.line(), end.line())
          .that(end.line())
          .isEqualTo(start.line());
      commentLines.append('(');
      commentLines.append(start.line());
      commentLines.append(',');
      commentLines.append(start.column());
      commentLines.append(") ");
    }
    assertWithMessage("Found: %s", commentLines)
        .that(result.getComments().size())
        .isEqualTo(10); // One per '#'
  }

  @Test
  public void testCommentNodes_doNotContainTerminatingNewline() throws Exception {
    String source =
        """
        # Ordinary comment 1
        a = 1  #: Trailing doc comment 2
        #: Doc comment 3
        #: Doc comment continued 4
        b = 2  # Ordinary trailing comment 5
        # Comment ending with EOF 6\
        """;
    StarlarkFile result = parseFile(source);
    assertThat(result.getComments()).hasSize(6);
    Comment lastComment = result.getComments().getLast();
    for (Comment comment : result.getComments()) {
      assertThat(comment.getText()).doesNotContain("\n");
      if (comment != lastComment) {
        assertThat(source.charAt(comment.getEndOffset())).isEqualTo('\n');
      }
    }
    assertThat(lastComment.getEndOffset()).isEqualTo(source.length());
  }

  @Test
  public void testComments_inMultilineExpressionWithSyntaxError() throws Exception {
    setFailFast(false);
    StarlarkFile result =
        parseFile(
            """
            x = (1 + *  # Ordinary comment
            # Ordinary comment
            2)

            y = (1 + *  #: Doc comment
            #: Doc comment
            2)
            """);
    assertThat(result.getComments()).hasSize(4);
    assertContainsError(":1:10: syntax error at '*': expected expression");
    assertContainsError(":5:10: syntax error at '*': expected expression");
  }

  @Test
  public void testDocComments_indentationDoesNotMatter() throws Exception {
    StarlarkFile result =
        parseFile(
            """
            #: Doc comment for a
                #: indent doesn't matter
            a = 1
            """);
    assertThat(result.getStatements()).hasSize(1);
    assertThat(result.getComments()).hasSize(2);
    assertThat(getDocComment(result.getStatements().get(0)))
        .isEqualTo("Doc comment for a\nindent doesn't matter");
  }

  @Test
  public void testDocComments_complexLhs() throws Exception {
    StarlarkFile result =
        parseFile(
            """
            #: Doc comment for a and b
            a, b = [1, 2]
            #: Doc comment for c and d
            c, \
            d = [3, 4]
            #: Doc comment for e and f
            (       #: ignored
                e,  #: ignored
                f,  #: ignored
            ) = [5, 6]
            """);
    assertThat(result.getStatements()).hasSize(3);
    assertThat(result.getComments()).hasSize(6);
    assertThat(getDocComment(result.getStatements().get(0))).isEqualTo("Doc comment for a and b");
    assertThat(getDocComment(result.getStatements().get(1))).isEqualTo("Doc comment for c and d");
    assertThat(getDocComment(result.getStatements().get(2))).isEqualTo("Doc comment for e and f");
  }

  @Test
  public void testDocComments_terminatedByBlankLineOrNonDocCommentLine() throws Exception {
    StarlarkFile result =
        parseFile(
            """
            #: Ignored - separated by newline from assignment statement

            a = 1
            #: Ignored - separated by non-doc comment line from assignment statement
            #
            b = 1
            #: Ignored

            #: Doc comment for c
            c = 2
            #: Ignored
            #
            #: Doc comment for d
            d = 2
            """);
    assertThat(result.getStatements()).hasSize(4);
    assertThat(result.getComments()).hasSize(8);
    assertThat(getDocComment(result.getStatements().get(0))).isNull();
    assertThat(getDocComment(result.getStatements().get(1))).isNull();
    assertThat(getDocComment(result.getStatements().get(2))).isEqualTo("Doc comment for c");
    assertThat(getDocComment(result.getStatements().get(3))).isEqualTo("Doc comment for d");
  }

  @Test
  public void testDocComments_trailing() throws Exception {
    StarlarkFile result =
        parseFile(
            """
            a = 1 #: Doc comment for a
                  #: Ignored; trailing comments are one-line only

            #: Ignored; trailing comments override leading comments
            b = 2 #: Doc comment for b
            """);
    assertThat(result.getStatements()).hasSize(2);
    assertThat(result.getComments()).hasSize(4);
    assertThat(getDocComment(result.getStatements().get(0))).isEqualTo("Doc comment for a");
    assertThat(getDocComment(result.getStatements().get(1))).isEqualTo("Doc comment for b");
  }

  @Test
  public void testDocComments_trailing_multistatementLine() throws Exception {
    StarlarkFile result =
        parseFile(
            """
            a = 1; b = 2; c = 3 #: Doc comment for c only

            d = 4; #: Ignored - no statement after the `;`
            """);
    assertThat(result.getStatements()).hasSize(4);
    assertThat(result.getComments()).hasSize(2);
    assertThat(getDocComment(result.getStatements().get(0))).isNull();
    assertThat(getDocComment(result.getStatements().get(1))).isNull();
    assertThat(getDocComment(result.getStatements().get(2))).isEqualTo("Doc comment for c only");
    assertThat(getDocComment(result.getStatements().get(3))).isNull();
  }

  @Test
  public void testDocComments_trailing_multilineStatement() throws Exception {
    StarlarkFile result =
        parseFile(
            """
            ( #: Ignored
                a, #: Ignored
                b  #: Ignored
            ) = foo(  #: Ignored
                #: Ignored
                x = 42  #: Ignored
            )[  #: Ignored
                1:2  #: Ignored
            ] #: Doc comment for a
            """);
    assertThat(result.getStatements()).hasSize(1);
    assertThat(result.getComments()).hasSize(9);
    assertThat(getDocComment(result.getStatements().get(0))).isEqualTo("Doc comment for a");
  }

  @Test
  public void testDocComments_leadingSpacesNormalized() throws Exception {
    StarlarkFile result =
        parseFile(
            """
            #:zero or
            #: one leading spaces
            #:  get stripped from each doc comment line
            a = 1
            """);
    assertThat(result.getStatements()).hasSize(1);
    assertThat(result.getComments()).hasSize(3);
    assertThat(getDocComment(result.getStatements().get(0)))
        .isEqualTo(
            "zero or\n" //
                + "one leading spaces\n" //
                + " get stripped from each doc comment line");
  }

  @Test
  public void testDocComments_inSuite() throws Exception {
    StarlarkFile result =
        parseFile(
            """
            def foo(): #: ignored
            #: Doc comment for a
                #: indent doesn't matter
                a = 1
                    #: Doc comment for b ignores indentation
                b = 2
                #: Applies to next statement
            #: indent doesn't matter
            x = 3
            """);
    assertThat(result.getStatements()).hasSize(2);
    assertThat(result.getComments()).hasSize(6);
    ImmutableList<Statement> body = ((DefStatement) result.getStatements().getFirst()).getBody();
    assertThat(body).hasSize(2);
    assertThat(getDocComment(body.get(0))).isEqualTo("Doc comment for a\nindent doesn't matter");
    assertThat(getDocComment(body.get(1))).isEqualTo("Doc comment for b ignores indentation");
    assertThat(getDocComment(result.getStatements().get(1)))
        .isEqualTo("Applies to next statement\nindent doesn't matter");
  }

  @Test
  public void testDocComments_allowedInNonAssignments() throws Exception {
    StarlarkFile result =
        parseFile(
            """
            #: ignored
            def foo( #: ignored
                #: ignored
                x, #: ignored
                #: ignored
                **kwags): #: ignored
                #: ignored
                return x #: ignored
                #: ignored
            #: ignored
            foo( #: ignored
                x = [ #: ignored
                    #: ignored
                    1, #: ignored
                    #: ignored
                    2, #: ignored
                    #: ignored
                ], #: ignored
                #: ignored
                y = { #: ignored
                    "z": 3, #: ignored
                    #: ignored
                }, #: ignored
                #: ignored
            ) #: ignored
            """);
    assertThat(result.getComments()).hasSize(25);
    assertThat(getDocComment(result.getStatements().get(0))).isNull();
    assertThat(getDocComment(result.getStatements().get(1))).isNull();
  }

  @Test
  public void testDocComments_notParsedInsideStrings() throws Exception {
    StarlarkFile result =
        parseFile(
            """
            "#: not parsed as doc comment"
            a = 1
            \"\"\"
            #: not parsed as doc comment
            \"\"\"
            b = 2
            '''
            #: not parsed as doc comment
            '''
            c = 3
            r'''
            #: not parsed as doc comment
            '''
            d = 4
            r\"\"\"
            #: not parsed as doc comment
            \"\"\"
            e = 5
            """);
    assertThat(result.getComments()).isEmpty();
    for (Statement stmt : result.getStatements()) {
      assertThat(getDocComment(stmt)).isNull();
    }
  }

  @Nullable
  private String getDocComment(Statement stmt) {
    if (stmt instanceof AssignmentStatement assign) {
      DocComments docComments = assign.getDocComments();
      if (docComments != null) {
        return docComments.getText();
      }
    }
    return null;
  }

  @Test
  public void testMissingComma() throws Exception {
    setFailFast(false);
    // Regression test.
    // Note: missing comma after name='foo'
    parseFile("genrule(name = 'foo'\n"
              + "      srcs = ['in'])");
    assertContainsError("syntax error at 'srcs'");
  }

  @Test
  public void testDoubleSemicolon() throws Exception {
    setFailFast(false);
    // Regression test.
    parseFile("x = 1; ; x = 2;");
    assertContainsError("syntax error at ';'");
  }

  @Test
  public void testDefSingleLine() throws Exception {
    List<Statement> statements = parseStatements("def foo(): x = 1; y = 2\n");
    DefStatement stmt = (DefStatement) statements.get(0);
    assertThat(stmt.getBody()).hasSize(2);
  }

  @Test
  public void testDef() throws Exception {
    DefStatement stmt =
        (DefStatement)
            parseStatement(
                "def f(a, *, b=1, *args, **kwargs):", //
                "  pass");

    assertThat(stmt.getParameters()).hasSize(5);

    assertThat(stmt.getParameters().get(0).getName()).isEqualTo("a");
    assertThat(stmt.getParameters().get(0)).isInstanceOf(Parameter.Mandatory.class);

    assertThat(stmt.getParameters().get(1).getName()).isNull();
    assertThat(stmt.getParameters().get(1)).isInstanceOf(Parameter.Star.class);

    assertThat(stmt.getParameters().get(2).getName()).isEqualTo("b");
    assertThat(stmt.getParameters().get(2)).isInstanceOf(Parameter.Optional.class);
    assertThat(stmt.getParameters().get(2).getDefaultValue().toString()).isEqualTo("1");

    assertThat(stmt.getParameters().get(3).getName()).isEqualTo("args");
    assertThat(stmt.getParameters().get(3)).isInstanceOf(Parameter.Star.class);

    assertThat(stmt.getParameters().get(4).getName()).isEqualTo("kwargs");
    assertThat(stmt.getParameters().get(4)).isInstanceOf(Parameter.StarStar.class);
  }

  @Test
  public void testDefAssignmentToStarArgs() throws Exception {
    setFailFast(false);
    parseStatement("def f(*args=1): pass");
    assertContainsError("syntax error at '=': expected ,");
  }

  @Test
  public void testDefAssignmentToBareStar() throws Exception {
    setFailFast(false);
    parseStatement("def f(* = 1): pass");
    assertContainsError("syntax error at '=': expected ,");
  }

  @Test
  public void testDefAssignmentToKwargs() throws Exception {
    setFailFast(false);
    parseStatement("def f(**kwargs=1): pass");
    assertContainsError("syntax error at '=': expected ,");
  }

  @Test
  public void testTypeExpression() throws Exception {
    setFileOptions(FileOptions.builder().allowTypeSyntax(true).build());
    // basic examples
    assertThat(parseTypeExpression("int")).isInstanceOf(Identifier.class);
    assertThat(parseTypeExpression("tuple[]")).isInstanceOf(TypeApplication.class);
    assertThat(parseTypeExpression("list[str]")).isInstanceOf(TypeApplication.class);
    assertThat(parseTypeExpression("dict[str, int]")).isInstanceOf(TypeApplication.class);
    // type expressions can use list literals
    assertThat(parseTypeExpression("Callable[[int, str], int]"))
        .isInstanceOf(TypeApplication.class);
    // type expressions can use dict literals
    assertThat(parseTypeExpression("TypedDict[{'a': int, 'b': bool}]"))
        .isInstanceOf(TypeApplication.class);
    // type expressions can use string literals
    assertThat(parseTypeExpression("Literal['abc']")).isInstanceOf(TypeApplication.class);
    // composition
    assertThat(parseTypeExpression("list[str, dict[str, bool]]"))
        .isInstanceOf(TypeApplication.class);
    // type unions
    assertThat(parseTypeExpression("str | int")).isInstanceOf(BinaryOperatorExpression.class);
    assertThat(parseTypeExpression("str | int | bool"))
        .isInstanceOf(BinaryOperatorExpression.class);
    // empty dict and list literals
    assertThat(parseTypeExpression("Callable[[], TypeDict[{}]]"))
        .isInstanceOf(TypeApplication.class);
    // trailing commas in dict and list arguments
    assertThat(parseTypeExpression("Callable[[int,],bool]")).isInstanceOf(TypeApplication.class);
    assertThat(parseTypeExpression("TypeDict[{'foo': int, }]")).isInstanceOf(TypeApplication.class);
  }

  @Test
  public void testIllegalTypeExpression_disallowed() throws Exception {
    setFileOptions(FileOptions.builder().allowTypeSyntax(true).build());
    setFailFast(false);
    parseStatement("def f(a : (lambda x: x)): pass");
    assertContainsError("syntax error at '(': expected a type");
  }

  @Test
  public void testIllegalTypeExpression_allowedWithFlag() throws Exception {
    setFileOptions(
        FileOptions.builder().allowTypeSyntax(true).allowArbitraryTypeExpressions(true).build());

    parseStatement("def f(a : (lambda x: x)): pass");
    assertThat(parseTypeExpression("lambda x: x")).isInstanceOf(LambdaExpression.class);

    // Annotations shouldn't consume adjacent params.
    Statement stmt = parseStatement("def f(p1 : x, p2): pass");
    assertThat(stmt.kind()).isEqualTo(Statement.Kind.DEF);
    assertThat(((DefStatement) stmt).getParameters().stream().map(p -> p.getName()))
        .containsExactly("p1", "p2")
        .inOrder();
  }

  @Test
  public void testDefWithTypeAnnotations() throws Exception {
    setFileOptions(FileOptions.builder().allowTypeSyntax(true).build());
    parseStatement("def f(a: int): pass");
    parseStatement("def f(a: tuple[]): pass");
    parseStatement("def f(a: list[str]): pass");
    parseStatement("def f(a: dict[str, int]): pass");

    // test with default values
    parseStatement("def f(a: int, *, b: bool = True, c): pass");

    // test args and kwargs
    parseStatement("def f(*args: list[int]): pass");
    parseStatement("def f(**kwargs: dict[str, Any]): pass");

    // Return type
    parseStatement("def f() -> int: pass");

    // Type parameters
    parseStatement("def f[T, U](x: dict[T, U]) -> dict[U, T]: pass");
  }

  @Test
  public void testDefBareStarCannotHaveTypeAnnotation() throws Exception {
    setFileOptions(FileOptions.builder().allowTypeSyntax(true).build());
    setFailFast(false);
    parseStatement("def f(a, *: int, b: bool): pass");
    assertContainsError("syntax error at ':': expected )");
  }

  // TODO(ilist): Python allows trailing commas in type arguments - we probably should too.
  @Test
  public void testTrailingCommaNotAllowedInTypeArgumentList() throws Exception {
    assertThat(parseTypeExpressionError("list[int,]"))
        .contains("syntax error at ']': expected a type argument");
  }

  @Test
  public void testDefWithDisallowedTypeAnnotations() throws Exception {
    setFileOptions(FileOptions.builder().allowTypeSyntax(false).build());
    setFailFast(false);
    parseStatement("def f(a: int): pass");
    assertContainsError("syntax error at ':': type annotations are disallowed.");
    events.clear();
    parseStatement("def f[T](): pass");
    assertContainsError("syntax error at '[': type annotations are disallowed.");
  }

  @Test
  public void testTypeApplicationsRequireConstructor() throws Exception {
    assertThat(parseTypeExpressionError("[int]")).contains("syntax error at '[': expected a type");
  }

  @Test
  public void testFunctionCallsNotAllowedInTypeExpressions() throws Exception {
    assertThat(parseTypeExpressionError("int[f(1)]")).contains("syntax error at '(': expected ,");
  }

  @Test
  public void testOnlyPipeOperatorsAllowedInTypeExpressions() throws Exception {
    ImmutableList<TokenKind> badOperators =
        ImmutableList.of(
            TokenKind.AMPERSAND,
            TokenKind.EQUALS,
            TokenKind.GREATER,
            TokenKind.LESS,
            TokenKind.MINUS,
            TokenKind.PLUS,
            TokenKind.SLASH,
            TokenKind.STAR);
    for (TokenKind op : badOperators) {
      assertThat(parseTypeExpressionError("int " + op + " str"))
          .contains("syntax error at '" + op + "'");
    }
  }

  @Test
  public void testTypeAliasStatement_basicFunctionality() throws Exception {
    setFileOptions(FileOptions.builder().allowTypeSyntax(true).build());
    TypeAliasStatement stmt = (TypeAliasStatement) parseStatement("type X = list");
    assertThat(stmt.getIdentifier().getName()).isEqualTo("X");
    assertThat(stmt.getParameters()).isEmpty();
    assertThat(stmt.getDefinition()).isInstanceOf(Identifier.class);
    assertThat(((Identifier) stmt.getDefinition()).getName()).isEqualTo("list");
  }

  @Test
  public void testTypeAliasStatement_typeParams() throws Exception {
    setFileOptions(FileOptions.builder().allowTypeSyntax(true).build());
    TypeAliasStatement stmt =
        (TypeAliasStatement) parseStatement("type my_nullable_dict[T, U] = dict[T, U] | None");
    assertThat(stmt.getIdentifier().getName()).isEqualTo("my_nullable_dict");
    assertThat(stmt.getParameters().stream().map(p -> p.getName()))
        .containsExactly("T", "U")
        .inOrder();
    assertThat(stmt.getDefinition()).isInstanceOf(BinaryOperatorExpression.class);
  }

  @Test
  public void testTypeAliasStatement_requiresTypeSyntax() throws Exception {
    setFileOptions(FileOptions.builder().allowTypeSyntax(false).build());
    setFailFast(false);
    parseStatement("type X = list");
    assertContainsError("syntax error at 'type': type annotations are disallowed");
  }

  @Test
  public void testTypeAliasStatement_requiresExactlyOneName() throws Exception {
    setFileOptions(FileOptions.builder().allowTypeSyntax(true).build());
    setFailFast(false);
    parseStatement("type X, Y = list, int");
    assertContainsError("syntax error at ',': expected =");
  }

  @Test
  public void testTypeAliasStatement_requiresDefinition() throws Exception {
    setFileOptions(FileOptions.builder().allowTypeSyntax(true).build());
    setFailFast(false);
    parseStatement("type X # = define_later");
    assertContainsError("syntax error at 'newline': expected =");
  }

  @Test
  public void testTypeAliasStatement_allowsParsingWithUnresolvableDefinition() throws Exception {
    setFileOptions(FileOptions.builder().allowTypeSyntax(true).build());
    parseStatement("type x = no_such_type");
  }

  @Test
  public void testTypeAliasStatement_disallowsIllegalDefinition() throws Exception {
    setFileOptions(FileOptions.builder().allowTypeSyntax(true).build());
    setFailFast(false);
    parseStatement("type X = lambda x: x");
    assertContainsError("syntax error at 'lambda': expected a type");
  }

  @Test
  public void testTypeAliasStatement_allowsIllegalDefinition_withFlag() throws Exception {
    setFileOptions(
        FileOptions.builder().allowTypeSyntax(true).allowArbitraryTypeExpressions(true).build());
    TypeAliasStatement stmt = (TypeAliasStatement) parseStatement("type X = lambda x: x");
    assertThat(stmt.getDefinition()).isInstanceOf(LambdaExpression.class);
  }

  @Test
  public void testTypeIsSoftKeyword() throws Exception {
    // Test that `type` may be used as an identifier in any context where identifiers are allowed.
    setFileOptions(FileOptions.builder().allowTypeSyntax(true).build());
    parseStatement("type = type.type(type)");
    parseStatement("type type = type");
  }

  private static enum TypeParamTestKind {
    DEF_STATEMENT,
    TYPE_ALIAS_STATEMENT
  }

  private Statement parseTypeParamTestCaseStatement(TypeParamTestKind testKind, String typeParams)
      throws Exception {
    return switch (testKind) {
      // Extra space in DEF_STATEMENT case to place typeParams at offset 6 in both cases
      case DEF_STATEMENT -> parseStatement(String.format("def  f%s(): pass", typeParams));
      case TYPE_ALIAS_STATEMENT -> parseStatement(String.format("type X%s = int", typeParams));
    };
  }

  @Test
  public void testTypeParams_mayBeUnused(@TestParameter TypeParamTestKind testKind)
      throws Exception {
    setFileOptions(FileOptions.builder().allowTypeSyntax(true).build());
    parseTypeParamTestCaseStatement(testKind, "[T, U]");
  }

  @Test
  public void testTypeParams_allowOnlyIdentifiers(@TestParameter TypeParamTestKind testKind)
      throws Exception {
    setFileOptions(FileOptions.builder().allowTypeSyntax(true).build());
    setFailFast(false);
    parseTypeParamTestCaseStatement(testKind, "[1]");
    assertContainsError("syntax error at '1': expected identifier");
    events.clear();
    parseTypeParamTestCaseStatement(testKind, "['two']");
    assertContainsError("syntax error at '\"two\"': expected identifier");
    events.clear();
    parseTypeParamTestCaseStatement(testKind, "[(THREE)]");
    assertContainsError("syntax error at '(': expected identifier");
  }

  @Test
  public void testTypeParams_disallowDuplicates(@TestParameter TypeParamTestKind testKind)
      throws Exception {
    setFileOptions(FileOptions.builder().allowTypeSyntax(true).build());
    setFailFast(false);
    parseTypeParamTestCaseStatement(testKind, "[T, U, T]");
    assertContainsError("1:14: syntax error at 'T': duplicate type parameter");
  }

  @Test
  public void testTypeParams_allowsTrailingCommas(@TestParameter TypeParamTestKind testKind)
      throws Exception {
    setFileOptions(FileOptions.builder().allowTypeSyntax(true).build());
    parseTypeParamTestCaseStatement(testKind, "[T, U,]");
  }

  @Test
  public void testTypeParams_cannotBeEmptyIfPresent(@TestParameter TypeParamTestKind testKind)
      throws Exception {
    setFileOptions(FileOptions.builder().allowTypeSyntax(true).build());
    setFailFast(false);
    parseTypeParamTestCaseStatement(testKind, "[]");
    assertContainsError("syntax error at ']': expected identifier");
    events.clear();
    parseTypeParamTestCaseStatement(testKind, "[,]");
    assertContainsError("syntax error at ',': expected identifier");
  }

  @Test
  public void testEllipsisNotAllowedInValueExpressions() throws Exception {
    setFileOptions(FileOptions.builder().allowTypeSyntax(true).build());
    assertThat(parseExpressionError("print(...)"))
        .contains("ellipsis ('...') is not allowed outside type expressions");
  }

  @Test
  public void testEllipsisAllowedInTypeExpressions() throws Exception {
    setFileOptions(
        FileOptions.builder().allowTypeSyntax(true).allowArbitraryTypeExpressions(true).build());
    parseStatement("x : Tuple[int, ...]");
  }

  @Test
  public void testCastExpression_basicFunctionality() throws Exception {
    setFileOptions(FileOptions.builder().allowTypeSyntax(true).build());
    CastExpression cast = (CastExpression) parseExpression("cast(list[int], foo())");
    assertThat(cast.getType()).isInstanceOf(TypeApplication.class);
    assertThat(cast.getValue()).isInstanceOf(CallExpression.class);
  }

  @Test
  public void testCastExpression_isExpression() throws Exception {
    setFileOptions(FileOptions.builder().allowTypeSyntax(true).build());
    parseStatement("cast(list, x) += cast(list[list], (cast(struct, y)).foo())[cast(int, z)]");
  }

  @Test
  public void testCast_isKeyword() throws Exception {
    setFileOptions(FileOptions.builder().allowTypeSyntax(true).build());
    setFailFast(false);
    assertThat(parseExpressionError("something.cast(list, x)"))
        .contains("syntax error at 'cast': expected identifier after dot");
    assertThat(parseExpressionError("(cast)(list, x)")).contains("expected (");
  }

  @Test
  public void testCastExpression_requiresTypeSyntax() throws Exception {
    // If type syntax is disabled, `cast` is treated as an ordinary identifier.
    setFileOptions(FileOptions.builder().allowTypeSyntax(false).build());
    assertThat(parseExpression("cast(list[str], foo())")).isInstanceOf(CallExpression.class);
  }

  @Test
  public void testCastExpression_goodSyntax() throws Exception {
    setFileOptions(FileOptions.builder().allowTypeSyntax(true).build());
    parseExpression(
        """
        cast(
            int,
            n
        )\
        """);
    parseExpression("cast(int, x,)");
  }

  @Test
  public void testCastExpression_badSyntax() throws Exception {
    setFileOptions(FileOptions.builder().allowTypeSyntax(true).build());
    setFailFast(false);
    assertThat(parseExpressionError("cast int, x")).contains("syntax error at 'int': expected (");
    assertThat(parseExpressionError("cast(int, x, y)")).contains("syntax error at 'y': expected )");
    assertThat(parseExpressionError("cast(int, x"))
        .contains("syntax error at 'newline': expected )");
    assertThat(parseExpressionError("cast(*args)"))
        .contains("syntax error at '*': expected a type");
    assertThat(parseExpressionError("cast(type=int, value=x"))
        .contains("syntax error at '=': expected ,");
  }

  @Test
  public void testIsInstanceExpression_basicFunctionality() throws Exception {
    setFileOptions(FileOptions.builder().allowTypeSyntax(true).build());
    IsInstanceExpression cast =
        (IsInstanceExpression) parseExpression("isinstance(foo(), list | tuple)");
    assertThat(cast.getType()).isInstanceOf(BinaryOperatorExpression.class);
    assertThat(cast.getValue()).isInstanceOf(CallExpression.class);
  }

  @Test
  public void testIsInstanceExpression_isExpression() throws Exception {
    setFileOptions(FileOptions.builder().allowTypeSyntax(true).build());
    parseStatement("if isinstance(isinstance(y, list), bool): isinstance(z, str)");
  }

  @Test
  public void testIsInstance_isKeyword() throws Exception {
    setFileOptions(FileOptions.builder().allowTypeSyntax(true).build());
    setFailFast(false);
    assertThat(parseExpressionError("something.isinstance(x, list)"))
        .contains("syntax error at 'isinstance': expected identifier after dot");
    assertThat(parseExpressionError("(isinstance)(x, list)")).contains("expected (");
  }

  @Test
  public void testIsInstanceExpression_requiresTypeSyntax() throws Exception {
    // If type syntax is disabled, `isinstance` is treated as an ordinary identifier.
    setFileOptions(FileOptions.builder().allowTypeSyntax(false).build());
    assertThat(parseExpression("isinstance(x, T)")).isInstanceOf(CallExpression.class);
  }

  @Test
  public void testIsInstanceExpression_goodSyntax() throws Exception {
    setFileOptions(FileOptions.builder().allowTypeSyntax(true).build());
    parseExpression(
        """
        isinstance(
            x,
            T | U[V]
        )\
        """);
    parseExpression("isinstance(x, tuple,)");
  }

  @Test
  public void testIsInstanceExpression_badSyntax() throws Exception {
    setFileOptions(FileOptions.builder().allowTypeSyntax(true).build());
    setFailFast(false);
    assertThat(parseExpressionError("isinstance x, int"))
        .contains("syntax error at 'x': expected (");
    assertThat(parseExpressionError("isinstance(x, y, int)"))
        .contains("syntax error at 'int': expected )");
    assertThat(parseExpressionError("isinstance(x, int"))
        .contains("syntax error at 'newline': expected )");
    assertThat(parseExpressionError("isinstance(*args)"))
        .contains("syntax error at '*': expected expression");
    assertThat(parseExpressionError("isinstance(value=x, type=int)"))
        .contains("syntax error at '=': expected ,");
  }

  @Test
  public void testLambda() throws Exception {
    parseExpression("lambda a, b=1, *args, **kwargs: a+b");
    parseExpression("lambda *, a, *b: 0");

    // lambda has lower predecence than binary or.
    assertThat(parseExpression("lambda: x or y").toString()).isEqualTo("lambda: (x or y)");

    // This is a well known parsing ambiguity in Python.
    // Python 2.7 accepts it but Python3 and Starlark reject it.
    parseExpressionError("[x for x in lambda: True, lambda: False if x()]");

    // ok in all dialects:
    parseExpression("[x for x in (lambda: True, lambda: False) if x()]");

    // An unparenthesized tuple is not allowed as the operand
    // of an 'if' clause in a comprehension, but a lambda is ok.
    assertThat(parseExpressionError("[a for b in c if 1, 2]"))
        .contains("expected ']', 'for' or 'if'");
    parseExpression("[a for b in c if lambda: d]");
    // But the body of the unparenthesized lambda may not be a conditional:
    parseExpression("[a for b in c if (lambda: d if e else f)]");
    assertThat(parseExpressionError("[a for b in c if lambda: d if e else f]"))
        .contains("expected ']', 'for' or 'if'");

    // A lambda is not allowed as the operand of a 'for' clause.
    assertThat(parseExpressionError("[a for b in lambda: c]")).contains("syntax error at 'lambda'");
  }

  @Test
  public void testForPass() throws Exception {
    List<Statement> statements = parseStatements("def foo():", "  pass\n");

    assertThat(statements).hasSize(1);
    DefStatement stmt = (DefStatement) statements.get(0);
    assertThat(stmt.getBody().get(0)).isInstanceOf(FlowStatement.class);
  }

  @Test
  public void testForLoopMultipleVariables() throws Exception {
    List<Statement> stmts1 = parseStatements("[ i for i, j, k in [(1, 2, 3)] ]\n");
    assertThat(stmts1).hasSize(1);

    List<Statement> stmts2 = parseStatements("[ i for i, j in [(1, 2, 3)] ]\n");
    assertThat(stmts2).hasSize(1);

    List<Statement> stmts3 = parseStatements("[ i for (i, j, k) in [(1, 2, 3)] ]\n");
    assertThat(stmts3).hasSize(1);
  }

  @Test
  public void testReturnNone() throws Exception {
    List<Statement> defNone = parseStatements("def foo():", "  return None\n");
    assertThat(defNone).hasSize(1);

    List<Statement> bodyNone = ((DefStatement) defNone.get(0)).getBody();
    assertThat(bodyNone).hasSize(1);

    ReturnStatement returnNone = (ReturnStatement) bodyNone.get(0);
    assertThat(((Identifier) returnNone.getResult()).getName()).isEqualTo("None");

    int i = 0;
    for (String end : new String[]{";", "\n"}) {
      List<Statement> defNoExpr = parseStatements("def bar" + i + "():", "  return" + end);
      i++;
      assertThat(defNoExpr).hasSize(1);

      List<Statement> bodyNoExpr = ((DefStatement) defNoExpr.get(0)).getBody();
      assertThat(bodyNoExpr).hasSize(1);

      ReturnStatement returnNoExpr = (ReturnStatement) bodyNoExpr.get(0);
      assertThat(returnNoExpr.getResult()).isNull();
    }
  }

  @Test
  public void testForLoopBadSyntax() throws Exception {
    setFailFast(false);
    parseFile("[1 for (a, b, c in var]\n");
    assertContainsError("syntax error");
  }

  @Test
  public void testForLoopBadSyntax2() throws Exception {
    setFailFast(false);
    parseFile("[1 for in var]\n");
    assertContainsError("syntax error");
  }

  @Test
  public void testFunCallBadSyntax() throws Exception {
    setFailFast(false);
    parseFile("f(1,\n");
    assertContainsError("syntax error");
  }

  @Test
  public void testFunCallBadSyntax2() throws Exception {
    setFailFast(false);
    parseFile("f(1, 5, ,)\n");
    assertContainsError("syntax error");
  }

  @Test
  public void testLoadNoSymbol() throws Exception {
    setFailFast(false);
    parseFile("load('//foo/bar:file.bzl')\n");
    assertContainsError("expected at least one symbol to load");
  }

  @Test
  public void testLoadOneSymbol() throws Exception {
    String text = "load('//foo/bar:file.bzl', 'fun_test')\n";
    List<Statement> statements = parseStatements(text);
    LoadStatement stmt = (LoadStatement) statements.get(0);
    assertThat(stmt.getImport().getValue()).isEqualTo("//foo/bar:file.bzl");
    assertThat(stmt.getBindings()).hasSize(1);
    Identifier sym = stmt.getBindings().get(0).getLocalName();
    assertThat(getText(text, sym)).isEqualTo("fun_test"); // apparent location within string literal
  }

  @Test
  public void testLoadOneSymbolWithTrailingComma() throws Exception {
    List<Statement> statements = parseStatements("load('//foo/bar:file.bzl', 'fun_test',)\n");
    LoadStatement stmt = (LoadStatement) statements.get(0);
    assertThat(stmt.getImport().getValue()).isEqualTo("//foo/bar:file.bzl");
    assertThat(stmt.getBindings()).hasSize(1);
  }

  @Test
  public void testLoadMultipleSymbols() throws Exception {
    List<Statement> statements = parseStatements("load(':file.bzl', 'foo', 'bar')\n");
    LoadStatement stmt = (LoadStatement) statements.get(0);
    assertThat(stmt.getImport().getValue()).isEqualTo(":file.bzl");
    assertThat(stmt.getBindings()).hasSize(2);
  }

  @Test
  public void testLoadLabelQuoteError() throws Exception {
    setFailFast(false);
    parseFile("load(non_quoted, 'a')\n");
    assertContainsError("syntax error");
  }

  @Test
  public void testLoadSymbolQuoteError() throws Exception {
    setFailFast(false);
    parseFile("load('label', non_quoted)\n");
    assertContainsError("syntax error");
  }

  @Test
  public void testLoadDisallowSameLine() throws Exception {
    setFailFast(false);
    parseFile("load('foo.bzl', 'foo') load('bar.bzl', 'bar')");
    assertContainsError("syntax error");
  }

  @Test
  public void testLoadNotAtTopLevel() throws Exception {
    // "This is not a parse error." --Magritte
    parseFile("if 1: load('', 'x')\n");
  }

  @Test
  public void testLoadModuleNotStringLiteral() throws Exception {
    setFailFast(false);
    parseFile("load(123, 'x')");
    assertContainsError("syntax error at '123': expected string literal");
  }

  @Test
  public void testLoadAlias() throws Exception {
    List<Statement> statements = parseStatements("load('//foo/bar:file.bzl', my_alias = 'lawl')\n");
    LoadStatement stmt = (LoadStatement) statements.get(0);
    ImmutableList<LoadStatement.Binding> actualSymbols = stmt.getBindings();

    assertThat(actualSymbols).hasSize(1);
    Identifier sym = actualSymbols.get(0).getLocalName();
    assertThat(sym.getName()).isEqualTo("my_alias");
    int startOffset = sym.getStartOffset();
    assertWithMessage("getStartOffset()").that(startOffset).isEqualTo(27);
    assertWithMessage("getEndOffset()").that(sym.getEndOffset()).isEqualTo(startOffset + 8);
  }

  @Test
  public void testLoadAliasMultiple() throws Exception {
    runLoadAliasTestForSymbols(
        "my_alias = 'lawl', 'lol', next_alias = 'rofl'", "my_alias", "lol", "next_alias");
  }

  private void runLoadAliasTestForSymbols(String loadSymbolString, String... expectedSymbols)
      throws SyntaxError.Exception {
    List<Statement> statements =
        parseStatements(String.format("load('//foo/bar:file.bzl', %s)\n", loadSymbolString));
    LoadStatement stmt = (LoadStatement) statements.get(0);

    List<String> actualSymbolNames = new ArrayList<>();
    for (LoadStatement.Binding binding : stmt.getBindings()) {
      actualSymbolNames.add(binding.getLocalName().getName());
    }
    assertThat(actualSymbolNames).containsExactly((Object[]) expectedSymbols);
  }

  @Test
  public void testLoadAliasSyntaxError() throws Exception {
    setFailFast(false);
    parseFile("load('//foo:bzl', test1 = )\n");
    assertContainsError("syntax error at ')': expected string");

    parseFile("load(':foo.bzl', test2 = 1)\n");
    assertContainsError("syntax error at '1': expected string");

    parseFile("load(':foo.bzl', test3 = old)\n");
    assertContainsError("syntax error at 'old': expected string");
  }

  @Test
  public void testLoadIsASmallStatement() throws Exception {
    // Regression test for b/148802200.
    parseFile("a=1; load('file', 'b'); c=3");
  }

  @Test
  public void testParseErrorNotComparison() throws Exception {
    setFailFast(false);
    parseFile("2 < not 3");
    assertContainsError("syntax error at 'not'");
  }

  @Test
  public void testNotWithArithmeticOperatorsBadSyntax() throws Exception {
    setFailFast(false);
    parseFile("0 + not 0");
    assertContainsError("syntax error at 'not'");
  }

  @Test
  public void testElseWithoutIf() throws Exception {
    setFailFast(false);
    parseFile(
        "def func(a):",
        // no if
        "  else: return a");
    assertContainsError("syntax error at 'else': expected expression");
  }

  @Test
  public void testForElse() throws Exception {
    setFailFast(false);
    parseFile(
        "def func(a):", //
        "  for i in range(a):",
        "    print(i)",
        "  else: return a");
    assertContainsError("syntax error at 'else': expected expression");
  }

  @Test
  public void testTryStatementInBuild() throws Exception {
    setFailFast(false);
    parseFile("try: pass");
    assertContainsError("'try' not supported, all exceptions are fatal");
  }

  @Test
  public void testClassDefinitionInBuild() throws Exception {
    setFailFast(false);
    parseFile("class test(object): pass");
    assertContainsError("keyword 'class' not supported");
  }

  @Test
  public void testClassDefinitionInStarlark() throws Exception {
    setFailFast(false);
    parseFile("class test(object): pass");
    assertContainsError("keyword 'class' not supported");
  }

  @Test
  public void testStringsAreDeduped() throws Exception {
    StarlarkFile file = parseFile("L1 = ['cat', 'dog', 'fish']", "L2 = ['dog', 'fish', 'cat']");
    Set<String> uniqueStringInstances = Sets.newIdentityHashSet();
    NodeVisitor collectAllStringsInStringLiteralsVisitor =
        new NodeVisitor() {
          @Override
          public void visit(StringLiteral stringLiteral) {
            uniqueStringInstances.add(stringLiteral.getValue());
          }
        };
    collectAllStringsInStringLiteralsVisitor.visit(file);
    assertThat(uniqueStringInstances).containsExactly("cat", "dog", "fish");
  }

  @Test
  public void testConditionalExpressions() throws Exception {
    assertThat(parseExpressionError("1 if 2"))
        .contains("missing else clause in conditional expression or semicolon before if");
  }

  @Test
  public void testPythonGeneratorsAreNotSupported() throws Exception {
    setFailFast(false);
    parseFile("y = (expr for vars in expr)");
    assertContainsError(
        "syntax error at 'for': Starlark does not support Python-style generator expressions");

    events.clear();
    parseFile("f(expr for vars in expr)");
    assertContainsError(
        "syntax error at 'for': Starlark does not support Python-style generator expressions");
  }

  @Test
  public void testParseFileStackOverflow() throws Exception {
    StarlarkFile file = StarlarkFile.parse(veryDeepExpression());
    SyntaxError ex = LexerTest.assertContainsError(file.errors(), "internal error: stack overflow");
    assertThat(ex.message()).contains("parseDictEntry"); // includes stack
    assertThat(ex.message()).contains("Please report the bug");
    assertThat(ex.message()).contains("include the text of foo.star"); // includes file name
  }

  @Test
  public void testParseExpressionStackOverflow() throws Exception {
    SyntaxError.Exception ex =
        assertThrows(SyntaxError.Exception.class, () -> Expression.parse(veryDeepExpression()));
    SyntaxError err = LexerTest.assertContainsError(ex.errors(), "internal error: stack overflow");
    assertThat(err.message()).contains("parseDictEntry"); // includes stack
    assertThat(err.message())
        .contains("while parsing Starlark expression <<{{{{"); // includes expression
    assertThat(err.message()).contains("Please report the bug");
  }

  private static ParserInput veryDeepExpression() {
    StringBuilder s = new StringBuilder();
    for (int i = 0; i < 5000; i++) {
      s.append("{");
    }
    return ParserInput.fromString(s.toString(), "foo.star");
  }
}
