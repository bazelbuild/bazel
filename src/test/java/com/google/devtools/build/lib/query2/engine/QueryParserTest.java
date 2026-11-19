// Copyright 2020 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.query2.engine;

import static com.google.common.truth.Truth.assertThat;
import static java.util.stream.Collectors.joining;
import static org.junit.Assert.assertThrows;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.Argument;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.ArgumentType;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryFunction;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryTaskFuture;
import com.google.devtools.build.lib.util.StringEncoding;
import java.util.List;
import java.util.stream.Stream;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests of parser and pretty-printer. */
@RunWith(JUnit4.class)
public final class QueryParserTest {
  private static class MockFunction implements QueryFunction {
    private final String name;
    private final int mandatoryArguments;
    private final List<ArgumentType> arguments;

    private MockFunction(String name, int mandatoryArguments, ArgumentType... arguments) {
      this.name = name;
      this.mandatoryArguments = mandatoryArguments;
      this.arguments = ImmutableList.copyOf(arguments);
    }

    @Override
    public String getName() {
      return name;
    }

    @Override
    public int getMandatoryArguments() {
      return mandatoryArguments;
    }

    @Override
    public List<ArgumentType> getArgumentTypes() {
      return arguments;
    }

    @Override
    public <T> QueryTaskFuture<Void> eval(
        QueryEnvironment<T> env,
        QueryExpressionContext<T> context,
        QueryExpression expression,
        List<Argument> args,
        Callback<T> callback) {
      throw new IllegalStateException();
    }
  }

  private static QueryEnvironment<?> mockEnvironment() {
    ImmutableList.Builder<QueryFunction> functions = ImmutableList.builder();
    functions.addAll(QueryEnvironment.DEFAULT_QUERY_FUNCTIONS);
    functions.add(new MockFunction("opt", 2,
        ArgumentType.WORD, ArgumentType.WORD, ArgumentType.WORD));

    QueryEnvironment<?> result = mock(QueryEnvironment.class);
    when(result.getFunctions()).thenReturn(functions.build());
    return result;
  }

  // Asserts that 'query' parses, and that when pretty-printed, yields 'query'.
  private static String checkPrettyPrint(String query) throws Exception {
    return checkPrettyPrint(query, query);
  }

  // Asserts that 'query' parses, and that when pretty-printed, yields
  // 'expectedPrettyPrintOutput' (both with and without placeholder parsing enabled).
  private static String checkPrettyPrint(String expectedPrettyPrintOutput, String query)
      throws Exception {
    QueryEnvironment<?> env = mockEnvironment();
    assertThat(QueryExpression.parse(query, env).toString()).isEqualTo(expectedPrettyPrintOutput);
    QueryAst astWithoutPlaceholders = QueryParser.parse(query, /* allowPlaceholders= */ false);
    QueryAst astWithPlaceholders = QueryParser.parse(query, /* allowPlaceholders= */ true);
    assertThat(astWithPlaceholders).isEqualTo(astWithoutPlaceholders);
    assertThat(QueryExpressionResolver.resolve(astWithPlaceholders, env).toString())
        .isEqualTo(expectedPrettyPrintOutput);
    return expectedPrettyPrintOutput;
  }

  public static void checkParseFails(String query, String expectedError) {
    QueryEnvironment<?> env = mockEnvironment();
    QuerySyntaxException e =
        assertThrows(QuerySyntaxException.class, () -> QueryExpression.parse(query, env));
    assertThat(e).hasMessageThat().isEqualTo(expectedError);
    QuerySyntaxException eWithPlaceholders =
        assertThrows(
            QuerySyntaxException.class,
            () ->
                QueryExpressionResolver.resolve(
                    QueryParser.parse(query, /* allowPlaceholders= */ true), env));
    assertThat(eWithPlaceholders).hasMessageThat().isEqualTo(expectedError);
  }

  @Test
  public void testOptionalArguments() throws Exception {
    checkPrettyPrint("opt('foo', 'bar')");
    checkPrettyPrint("opt('foo', 'bar', 'qux')");
    checkParseFails(
        "opt('foo', 'bar', 'qux', 'zyc')", "too many arguments to function 'opt' at ', zyc )'");
    checkParseFails("opt('foo')", "too few arguments to function 'opt' at ')'");
    checkParseFails("opt()", "too few arguments to function 'opt' at ')'");
  }

  @Test
  public void testUnknownFunction() throws Exception {
    String knownFunctions =
        Stream.concat(
                QueryEnvironment.DEFAULT_QUERY_FUNCTIONS.stream()
                    .map(f -> String.format("'%s'", f.getName())),
                Stream.of("'opt'"))
            .sorted()
            .collect(joining(", "));
    checkParseFails(
        "badfunc('foo', 'bar', 'qux', 'zyc')",
        String.format(
            "unknown function 'badfunc' at 'badfunc ( foo'; expected one of [%s]", knownFunctions));
  }

  @Test
  public void testTargetLiterals() throws Exception {
    checkPrettyPrint("x");
    checkPrettyPrint("//x");
    checkPrettyPrint("//x:y");
    checkPrettyPrint("x/...:all-targets");
    checkPrettyPrint("\"set\""); // reserved word
    checkPrettyPrint("\"\"");
  }

  @Test
  public void checkParseErrors() {
    checkParseFails("rdeps(", "premature end of input");
    checkParseFails("rdeps(,", "syntax error at ','");
    checkParseFails("rdeps(a", "premature end of input");
    checkParseFails("rdeps(a, ", "premature end of input");
    checkParseFails("rdeps(a, )", "syntax error at ')'");
    checkParseFails("rdeps(a, b", "premature end of input");
    checkParseFails("rdeps(a, b, ", "premature end of input");
    checkParseFails("rdeps(a, b, )", "syntax error at ')'");
    checkParseFails("rdeps(a, b, 3", "premature end of input");
    checkParseFails("rdeps(a, b, 3, ", "premature end of input");
    checkParseFails("rdeps(a, b, 3, 4)", "too many arguments to function 'rdeps' at ', 4 )'");
    checkParseFails("rdeps(a, b, c, d)", "expected an integer literal: 'c'");
    checkParseFails("set(", "premature end of input");
    checkParseFails("set(a", "premature end of input");
    checkParseFails("set(a b", "premature end of input");
    checkParseFails("set(a, ", "syntax error at ','");
    checkParseFails("set(a, b)", "syntax error at ', b )'");
  }

  @Test
  public void testBinaryOperators() throws Exception {
    checkParseFails("foo intersect", "premature end of input");

    checkPrettyPrint("(a - b)", "a - b");

    checkPrettyPrint("(a intersect b)", "a intersect b");
    checkPrettyPrint("(a intersect b intersect c)", "a intersect b intersect c");
    checkPrettyPrint("(a union b)", "a union b");
    checkPrettyPrint("(a union b union c)", "a union b union c");
    checkPrettyPrint("(a except b)", "a except b");
    checkPrettyPrint("(a except b except c)", "a except b except c");
    checkPrettyPrint("((a union b) except c)", "a union b except c");
    checkPrettyPrint("((a except b) union c)", "a except b union c");
  }

  @Test
  public void testOperators() throws Exception {
    checkPrettyPrint("some(x)");
    checkPrettyPrint("somepath(x, y)");
    checkPrettyPrint("allpaths(x, y)");
    checkPrettyPrint("deps(x)");
    checkPrettyPrint("deps(x, 1)");
    checkPrettyPrint("rdeps(x, y)");
    checkPrettyPrint("rdeps(x, y, 1)");
    checkPrettyPrint("kind('rule', x)", "kind(rule, x)");
    checkPrettyPrint("kind('source file', x)");
    checkPrettyPrint("kind('.*', x)");
    checkPrettyPrint("attr('linkshared', '1', x)", "attr(linkshared,1,x)");
    checkPrettyPrint("filter('jar$', x)", "filter(jar$, x)");
    checkPrettyPrint("let x = e1 in e2");
    checkPrettyPrint("labels('srcs', x)");
    checkPrettyPrint("tests(x)");
    checkPrettyPrint("executables(x)");
    checkPrettyPrint("set()");
    checkPrettyPrint("set(//a)");
    checkPrettyPrint("set(//a //b)");
  }

  @Test
  public void testMultipleOperatorParsing() throws Exception {
    checkPrettyPrint(checkPrettyPrint("kind('rule', x)", "kind(rule, x)"));
    checkPrettyPrint(checkPrettyPrint("attr('linkshared', '1', x)", "attr(linkshared,1,x)"));
    checkPrettyPrint(checkPrettyPrint("filter('jar$', x)", "filter(jar$, x)"));
  }

  @Test
  public void testMultipleBinaryOperatorParsing() throws Exception {
    checkPrettyPrint(checkPrettyPrint("((a union b) except c)", "a union b except c"));
    checkPrettyPrint(checkPrettyPrint("(a intersect b intersect c)", "a intersect b intersect c"));
    checkPrettyPrint(checkPrettyPrint("(a union b union c)", "a union b union c"));
    checkPrettyPrint(
        checkPrettyPrint(
            "((((a union b) intersect c) except d) intersect e)",
            "a union b intersect c except d intersect e"));
  }

  @Test
  public void testMultipleTargetLiteralParsing() throws Exception {
    checkPrettyPrint(checkPrettyPrint("//foo:.*@4", "\"//foo:.*@4\""));
    checkPrettyPrint(checkPrettyPrint("set(//foo)", "set(\"//foo\")"));
    checkPrettyPrint("\"set(//foo)\"");
    checkPrettyPrint("\"set('//foo')\"");
  }

  @Test
  public void testQuotedAndUnquotedMetacharacters() throws Exception {
    checkPrettyPrint("\"//foo:xx+xx\"");
    checkPrettyPrint(checkPrettyPrint("(//foo:xx + xx)", "//foo:xx+xx"));
    checkPrettyPrint("\"//foo:xx=xx\"");
    checkParseFails("//foo:xx=xx", "unexpected token '=' after query expression '//foo:xx'");
  }

  @Test
  public void testQuotedSpecialCharacters() throws Exception {
    checkPrettyPrint("\"foo[]^$asd.|asd?*+{})_asd()2\"", "'foo[]^$asd.|asd?*+{})_asd()2'");
    checkPrettyPrint("\"foo[]^$asd.|asd?*+{})_asd()2\"");
    checkPrettyPrint("\" #&()+,;<=>?[]{|}\"");
  }

  @Test
  public void testUnquotedSpecialCharacters() throws Exception {
    // All special characters in the Lexer#scanWord ./@_:~-*$
    checkPrettyPrint("a.b");
    checkPrettyPrint("a/b");
    checkPrettyPrint("a@b");
    checkPrettyPrint("a_b");
    checkPrettyPrint("a:b");
    checkPrettyPrint("a~b");
    checkPrettyPrint("a-b");
    checkPrettyPrint("a*b");
    checkPrettyPrint("a$b");
  }

  @Test
  public void testPreserveQuoting() throws Exception {
    checkPrettyPrint(checkPrettyPrint("\"a+b\""));
    // this should preserve quoting without being quoted in TargetLiteral#toString
    checkPrettyPrint(checkPrettyPrint("aaa", "\"aaa\""));
  }

  @Test
  public void testQuotedIllegalCharacters() throws Exception {
    checkParseFails("\"-x\"", "target literal must not begin with (-): -x");
    checkParseFails("\"*x\"", "target literal must not begin with (*): *x");
  }

  @Test
  public void testIllegalQuoting() throws Exception {
    checkParseFails("\"a", "unclosed quotation");
    checkParseFails("\'a", "unclosed quotation");
    checkParseFails("a\"a", "unclosed quotation");
    checkParseFails("a\'a", "unclosed quotation");
    checkParseFails("a\'\"a", "unclosed quotation");
    checkParseFails("a\"\'a", "unclosed quotation");
    checkParseFails("\'a\"\'a\'", "unclosed quotation");
    checkParseFails("\"a\'\"a\"", "unclosed quotation");
    checkParseFails(
        "\'\"a\" + \'a\'\'", "unexpected token 'a' after query expression ''\"a\" + ''");
    checkParseFails(
        "\"\'a\' + \"a\"\"", "unexpected token 'a' after query expression '\"'a' + \"'");
    checkParseFails(
        "\"set(\"//foo\" + \"bar\")\"",
        "unexpected token '//foo' after query expression '\"set(\"'");
    checkParseFails(
        "'set('//foo' + 'bar')'", "unexpected token '//foo' after query expression ''set(''");
  }

  @Test
  public void testUsingCorrectQuotingInTargetLiteralToString() throws Exception {
    // These tests all fall into the needsQuoting == true use case in TargetLiteral#toString
    checkPrettyPrint("'set(\"//foo\" + \"bar\")'");
    checkPrettyPrint("\"set('//foo' + 'bar')\"");
    checkPrettyPrint("\"a'a\"");
    checkPrettyPrint("\'a\"a\'");
  }

  @Test
  public void testUnicodeLabels() throws Exception {
    checkPrettyPrint(
        StringEncoding.unicodeToInternal("//:äöüÄÖÜß🌱"),
        StringEncoding.unicodeToInternal("'//:äöüÄÖÜß🌱'"));
    checkPrettyPrint(
        StringEncoding.unicodeToInternal("//:äöüÄÖÜß🌱"),
        StringEncoding.unicodeToInternal("//:äöüÄÖÜß🌱"));
  }

  @Test
  public void testBoundVariableInWordArgumentFails() throws Exception {
    checkParseFails(
        "let x = //foo in filter($x, //bar)",
        "variable '$x' cannot be used as an argument to function 'filter'; "
            + "variable references are only allowed in expression arguments");
    checkParseFails(
        "let x = //foo in kind($x, //bar)",
        "variable '$x' cannot be used as an argument to function 'kind'; "
            + "variable references are only allowed in expression arguments");
    checkParseFails(
        "let x = //foo in attr(tags, $x, //bar)",
        "variable '$x' cannot be used as an argument to function 'attr'; "
            + "variable references are only allowed in expression arguments");
    checkParseFails(
        "let x = //foo in labels($x, //bar)",
        "variable '$x' cannot be used as an argument to function 'labels'; "
            + "variable references are only allowed in expression arguments");
    checkParseFails(
        "let x = //foo in let y = //bar in filter($x, $y)",
        "variable '$x' cannot be used as an argument to function 'filter'; "
            + "variable references are only allowed in expression arguments");
    checkParseFails(
        "let x = //foo in (let x = //bar in //baz) union filter($x, //qux)",
        "variable '$x' cannot be used as an argument to function 'filter'; "
            + "variable references are only allowed in expression arguments");

    // Non-violating expressions:
    // Regex ending in $ parses without error.
    checkPrettyPrint("let x = //foo in filter('jar$', $x)", "let x = //foo in filter(jar$, $x)");
    // Unbound variable reference in word argument parses as literal word.
    checkPrettyPrint("filter('$unbound', //bar)", "filter($unbound, //bar)");
    // Out-of-scope variable reference in word argument parses as literal word.
    checkPrettyPrint(
        "(let x = //foo in //bar union filter('$x', //baz))",
        "(let x = //foo in //bar) union filter($x, //baz)");
    // Variable is not yet in scope in varExpr.
    checkPrettyPrint("let x = filter('$x', //bar) in //baz", "let x = filter($x, //bar) in //baz");
  }

  @Test
  public void testAstParsesUnresolvedFunctions() throws Exception {
    String query = "custom_func('*pattern', //foo + //bar, 3)";
    QueryAst ast = QueryParser.parse(query);
    assertThat(QueryParser.parse(query, /* allowPlaceholders= */ true)).isEqualTo(ast);
    assertThat(ast).isInstanceOf(QueryAst.FunctionCall.class);
    QueryAst.FunctionCall call = (QueryAst.FunctionCall) ast;
    assertThat(call.functionName()).isEqualTo("custom_func");
    assertThat(call.arguments()).hasSize(3);
    assertThat(call.arguments().get(0).expr()).isEqualTo(new QueryAst.Word("*pattern"));
    assertThat(call.arguments().get(1).expr())
        .isEqualTo(
            new QueryAst.BinaryOperator(
                Lexer.TokenKind.PLUS,
                ImmutableList.of(new QueryAst.Word("//foo"), new QueryAst.Word("//bar"))));
    assertThat(call.arguments().get(2).expr()).isEqualTo(new QueryAst.Word("3"));
  }

  @Test
  public void testAstStructureForLetSetAndParenthesized() throws Exception {
    String query = "let x = set(//a //b) in ($x intersect //c)";
    QueryAst ast = QueryParser.parse(query);
    assertThat(QueryParser.parse(query, /* allowPlaceholders= */ true)).isEqualTo(ast);
    assertThat(ast)
        .isEqualTo(
            new QueryAst.Let(
                "x",
                new QueryAst.Set(
                    ImmutableList.of(new QueryAst.Word("//a"), new QueryAst.Word("//b"))),
                new QueryAst.Parenthesized(
                    new QueryAst.BinaryOperator(
                        Lexer.TokenKind.INTERSECT,
                        ImmutableList.of(new QueryAst.Word("$x"), new QueryAst.Word("//c"))))));
  }

  @Test
  public void testFunctionAstRecordsExcludeErrorMessageFields() throws Exception {
    QueryAst.FunctionArgument arg1 =
        new QueryAst.FunctionArgument(new QueryAst.Word("//foo"), "( //foo )", "//foo )");
    QueryAst.FunctionArgument arg2 =
        new QueryAst.FunctionArgument(new QueryAst.Word("//foo"), ", //foo ,", "//foo , //bar");
    QueryAst.FunctionArgument differentArg =
        new QueryAst.FunctionArgument(new QueryAst.Word("//bar"), "( //foo )", "//foo )");

    assertThat(arg1).isEqualTo(arg2);
    assertThat(arg1.hashCode()).isEqualTo(arg2.hashCode());
    assertThat(arg1.toString()).isEqualTo("FunctionArgument[expr=Word[value=//foo]]");
    assertThat(arg2.toString()).isEqualTo("FunctionArgument[expr=Word[value=//foo]]");
    assertThat(arg1).isNotEqualTo(differentArg);

    QueryAst.FunctionCall call1 =
        new QueryAst.FunctionCall("deps", ImmutableList.of(arg1), "deps ( //foo", ")");
    QueryAst.FunctionCall call2 =
        new QueryAst.FunctionCall(
            "deps", ImmutableList.of(arg2), "other function context", ") union //bar");
    QueryAst.FunctionCall differentFunc =
        new QueryAst.FunctionCall("somepath", ImmutableList.of(arg1), "deps ( //foo", ")");
    QueryAst.FunctionCall differentCallArgs =
        new QueryAst.FunctionCall("deps", ImmutableList.of(differentArg), "deps ( //foo", ")");

    assertThat(call1).isEqualTo(call2);
    assertThat(call1.hashCode()).isEqualTo(call2.hashCode());
    assertThat(call1.toString())
        .isEqualTo(
            "FunctionCall[functionName=deps,"
                + " arguments=[FunctionArgument[expr=Word[value=//foo]]]]");
    assertThat(call2.toString()).isEqualTo(call1.toString());
    assertThat(call1).isNotEqualTo(differentFunc);
    assertThat(call1).isNotEqualTo(differentCallArgs);

    // Parsed FunctionCall nodes with different surrounding tokens remain equal.
    QueryAst.FunctionCall standalone = (QueryAst.FunctionCall) QueryParser.parse("deps(//foo)");
    QueryAst.FunctionCall inUnion =
        (QueryAst.FunctionCall)
            ((QueryAst.BinaryOperator) QueryParser.parse("deps(//foo) union //bar"))
                .operands()
                .get(0);
    assertThat(standalone.closeParenContextForErrorMessages())
        .isNotEqualTo(inUnion.closeParenContextForErrorMessages());
    assertThat(standalone).isEqualTo(inUnion);
    assertThat(standalone.hashCode()).isEqualTo(inUnion.hashCode());
    assertThat(standalone.toString()).isEqualTo(inUnion.toString());
  }

  @Test
  public void testResolverValidatesWordAndIntegerArguments() throws Exception {
    QueryAst resolvedWord = QueryParser.parse("filter('*_test', //foo)");
    assertThat(QueryParser.parse("filter('*_test', //foo)", /* allowPlaceholders= */ true))
        .isEqualTo(resolvedWord);
    assertThat(QueryExpressionResolver.resolve(resolvedWord, mockEnvironment()).toString())
        .isEqualTo("filter('*_test', //foo)");

    checkParseFails("deps('*_test')", "target literal must not begin with (*): *_test");
    checkParseFails("kind(('rule'), //foo)", "syntax error at '( rule )'");
    checkParseFails("deps(//foo, (1))", "syntax error at '( 1 )'");
  }

  @Test
  public void testAtTargetLiteralsUnchangedWhenPlaceholdersEnabled() throws Exception {
    checkPrettyPrint("@foo");
    checkPrettyPrint("@foo//:bar");
    checkPrettyPrint("@//pkg:target");
    checkPrettyPrint("@@foo");
    checkPrettyPrint(
        "\"@@rules_jvm_external++maven+maven//:bar\"", "@@rules_jvm_external++maven+maven//:bar");
    checkPrettyPrint("(@foo + @bar)", "@foo+@bar");
    checkPrettyPrint("set(@foo @bar)");
    checkPrettyPrint("\"@(foo)\"", "'@(foo)'");
    checkPrettyPrint("\"@(foo)\"");
    checkPrettyPrint("@");
  }

  @Test
  public void testPlaceholdersParsedWhenEnabled() throws Exception {
    assertThat(QueryParser.parse("@(foo)", /* allowPlaceholders= */ false))
        .isEqualTo(
            new QueryAst.FunctionCall(
                "@",
                ImmutableList.of(new QueryAst.FunctionArgument(new QueryAst.Word("foo"), "", "")),
                "",
                ""));

    String query =
        "let x = @(init) in rdeps(@(universe), $x) intersect (@(filter) + '@(literal)' + @repo"
            + " + @//a:b)";
    QueryAst ast = QueryParser.parse(query, /* allowPlaceholders= */ true);

    QueryAst.Let letAst = (QueryAst.Let) ast;
    assertThat(letAst.varName()).isEqualTo("x");
    assertThat(letAst.varExpr()).isEqualTo(new QueryAst.Placeholder("init", 8, 15));
    assertThat(query.substring(8, 15)).isEqualTo("@(init)");

    QueryAst.BinaryOperator intersect = (QueryAst.BinaryOperator) letAst.bodyExpr();
    assertThat(intersect.operator()).isEqualTo(Lexer.TokenKind.INTERSECT);
    assertThat(intersect.operands()).hasSize(2);

    QueryAst.FunctionCall rdepsCall = (QueryAst.FunctionCall) intersect.operands().get(0);
    assertThat(rdepsCall.functionName()).isEqualTo("rdeps");
    assertThat(rdepsCall.arguments()).hasSize(2);
    QueryAst.Placeholder universe = (QueryAst.Placeholder) rdepsCall.arguments().get(0).expr();
    assertThat(universe).isEqualTo(new QueryAst.Placeholder("universe", 25, 36));
    assertThat(query.substring(universe.startOffset(), universe.endOffset()))
        .isEqualTo("@(universe)");
    assertThat(rdepsCall.arguments().get(1).expr()).isEqualTo(new QueryAst.Word("$x"));

    assertThat(intersect.operands().get(1))
        .isEqualTo(
            new QueryAst.Parenthesized(
                new QueryAst.BinaryOperator(
                    Lexer.TokenKind.PLUS,
                    ImmutableList.of(
                        new QueryAst.Placeholder("filter", 53, 62),
                        new QueryAst.Word("@(literal)"),
                        new QueryAst.Word("@repo"),
                        new QueryAst.Word("@//a:b")))));
    assertThat(query.substring(53, 62)).isEqualTo("@(filter)");

    // Reserved words inside '@(...)' are valid identifier-shaped placeholder names.
    assertThat(QueryParser.parse("@(union) + @(in) + @(set)", /* allowPlaceholders= */ true))
        .isEqualTo(
            new QueryAst.BinaryOperator(
                Lexer.TokenKind.PLUS,
                ImmutableList.of(
                    new QueryAst.Placeholder("union", 0, 8),
                    new QueryAst.Placeholder("in", 11, 16),
                    new QueryAst.Placeholder("set", 19, 25))));
  }

  @Test
  public void testPlaceholdersRejectedInInvalidGrammarPositions() {
    QuerySyntaxException inSet =
        assertThrows(
            QuerySyntaxException.class,
            () -> QueryParser.parse("set(@(foo))", /* allowPlaceholders= */ true));
    assertThat(inSet).hasMessageThat().isEqualTo("syntax error at '@(foo) )'");

    QuerySyntaxException inLetVarName =
        assertThrows(
            QuerySyntaxException.class,
            () -> QueryParser.parse("let @(x) = //a in //b", /* allowPlaceholders= */ true));
    assertThat(inLetVarName).hasMessageThat().isEqualTo("syntax error at '@(x) = //a'");

    QuerySyntaxException asFunctionName =
        assertThrows(
            QuerySyntaxException.class,
            () -> QueryParser.parse("@(foo)(//a)", /* allowPlaceholders= */ true));
    assertThat(asFunctionName)
        .hasMessageThat()
        .isEqualTo("unexpected token '(' after query expression '@(foo)'");

    QuerySyntaxException trailingPlaceholder =
        assertThrows(
            QuerySyntaxException.class,
            () -> QueryParser.parse("//a @(b)", /* allowPlaceholders= */ true));
    assertThat(trailingPlaceholder)
        .hasMessageThat()
        .isEqualTo("unexpected token '@(b)' after query expression '//a'");

    QuerySyntaxException attachedAfterWord =
        assertThrows(
            QuerySyntaxException.class,
            () -> QueryParser.parse("a@(b)", /* allowPlaceholders= */ true));
    assertThat(attachedAfterWord)
        .hasMessageThat()
        .isEqualTo("unexpected token '@(b)' after query expression 'a'");

    QuerySyntaxException doubleAtBeforeParens =
        assertThrows(
            QuerySyntaxException.class,
            () -> QueryParser.parse("@@(b)", /* allowPlaceholders= */ true));
    assertThat(doubleAtBeforeParens)
        .hasMessageThat()
        .isEqualTo("unexpected token '@(b)' after query expression '@'");
  }

  @Test
  public void testResolverRejectsPlaceholders() throws Exception {
    QueryAst ast = QueryParser.parse("deps(@(targets))", /* allowPlaceholders= */ true);
    QuerySyntaxException e =
        assertThrows(
            QuerySyntaxException.class,
            () -> QueryExpressionResolver.resolve(ast, mockEnvironment()));
    assertThat(e).hasMessageThat().isEqualTo("placeholders are not supported: '@(targets)'");
  }

  @Test
  public void testIdentifierAndVariableReferenceHelpers() {
    assertThat(QueryAst.isValidIdentifier("foo")).isTrue();
    assertThat(QueryAst.isValidIdentifier("_foo_123")).isTrue();
    assertThat(QueryAst.isValidIdentifier("")).isFalse();
    assertThat(QueryAst.isValidIdentifier("1foo")).isFalse();
    assertThat(QueryAst.isValidIdentifier("foo-bar")).isFalse();

    QueryAst.Word varRef = new QueryAst.Word("$my_var");
    assertThat(varRef.isVariableReference()).isTrue();
    assertThat(varRef.variableName()).isEqualTo("my_var");

    assertThat(new QueryAst.Word("$").isVariableReference()).isFalse();
    assertThat(new QueryAst.Word("$1").isVariableReference()).isFalse();
    assertThat(new QueryAst.Word("jar$").isVariableReference()).isFalse();
    assertThat(new QueryAst.Word("//pkg:target").isVariableReference()).isFalse();
  }
}
