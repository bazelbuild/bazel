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
import com.google.devtools.build.lib.query2.engine.QueryParser.ParserOptions;
import com.google.devtools.build.lib.util.StringEncoding;
import java.util.List;
import java.util.stream.Stream;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests of parser and pretty-printer. */
@RunWith(JUnit4.class)
public final class QueryParserTest {
  private static final ParserOptions ALLOW_PLACEHOLDERS =
      ParserOptions.builder().setAllowPlaceholders(true).build();

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
    QueryAst astWithoutPlaceholders = QueryParser.parse(query, ParserOptions.DEFAULT);
    QueryAst astWithPlaceholders = QueryParser.parse(query, ALLOW_PLACEHOLDERS);
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
                QueryExpressionResolver.resolve(QueryParser.parse(query, ALLOW_PLACEHOLDERS), env));
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
    QueryAst ast = QueryParser.parse(query, ParserOptions.DEFAULT);
    assertThat(QueryParser.parse(query, ALLOW_PLACEHOLDERS)).isEqualTo(ast);
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
    QueryAst ast = QueryParser.parse(query, ParserOptions.DEFAULT);
    assertThat(QueryParser.parse(query, ALLOW_PLACEHOLDERS)).isEqualTo(ast);
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
    QueryAst.FunctionCall standalone =
        (QueryAst.FunctionCall) QueryParser.parse("deps(//foo)", ParserOptions.DEFAULT);
    QueryAst.FunctionCall inUnion =
        (QueryAst.FunctionCall)
            ((QueryAst.BinaryOperator)
                    QueryParser.parse("deps(//foo) union //bar", ParserOptions.DEFAULT))
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
    QueryAst resolvedWord = QueryParser.parse("filter('*_test', //foo)", ParserOptions.DEFAULT);
    assertThat(QueryParser.parse("filter('*_test', //foo)", ALLOW_PLACEHOLDERS))
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
    assertThat(QueryParser.parse("@(foo)", ParserOptions.DEFAULT))
        .isEqualTo(
            new QueryAst.FunctionCall(
                "@",
                ImmutableList.of(new QueryAst.FunctionArgument(new QueryAst.Word("foo"), "", "")),
                "",
                ""));

    String query =
        "let x = @(init) in rdeps(@(universe), $x) intersect (@(filter) + '@(literal)' + @repo"
            + " + @//a:b)";
    QueryAst ast = QueryParser.parse(query, ALLOW_PLACEHOLDERS);

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
    assertThat(QueryParser.parse("@(union) + @(in) + @(set)", ALLOW_PLACEHOLDERS))
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
            QuerySyntaxException.class, () -> QueryParser.parse("set(@(foo))", ALLOW_PLACEHOLDERS));
    assertThat(inSet).hasMessageThat().isEqualTo("syntax error at '@(foo) )'");

    QuerySyntaxException inLetVarName =
        assertThrows(
            QuerySyntaxException.class,
            () -> QueryParser.parse("let @(x) = //a in //b", ALLOW_PLACEHOLDERS));
    assertThat(inLetVarName).hasMessageThat().isEqualTo("syntax error at '@(x) = //a'");

    QuerySyntaxException asFunctionName =
        assertThrows(
            QuerySyntaxException.class, () -> QueryParser.parse("@(foo)(//a)", ALLOW_PLACEHOLDERS));
    assertThat(asFunctionName)
        .hasMessageThat()
        .isEqualTo("unexpected token '(' after query expression '@(foo)'");

    QuerySyntaxException trailingPlaceholder =
        assertThrows(
            QuerySyntaxException.class, () -> QueryParser.parse("//a @(b)", ALLOW_PLACEHOLDERS));
    assertThat(trailingPlaceholder)
        .hasMessageThat()
        .isEqualTo("unexpected token '@(b)' after query expression '//a'");

    QuerySyntaxException attachedAfterWord =
        assertThrows(
            QuerySyntaxException.class, () -> QueryParser.parse("a@(b)", ALLOW_PLACEHOLDERS));
    assertThat(attachedAfterWord)
        .hasMessageThat()
        .isEqualTo("unexpected token '@(b)' after query expression 'a'");

    QuerySyntaxException doubleAtBeforeParens =
        assertThrows(
            QuerySyntaxException.class, () -> QueryParser.parse("@@(b)", ALLOW_PLACEHOLDERS));
    assertThat(doubleAtBeforeParens)
        .hasMessageThat()
        .isEqualTo("unexpected token '@(b)' after query expression '@'");
  }

  @Test
  public void testResolverRejectsPlaceholders() throws Exception {
    QueryAst ast = QueryParser.parse("deps(@(targets))", ALLOW_PLACEHOLDERS);
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

  @Test
  public void testParserOptionsDefaultsAndValidation() {
    ParserOptions defaults = ParserOptions.builder().build();
    assertThat(defaults).isEqualTo(ParserOptions.DEFAULT);
    assertThat(defaults.allowPlaceholders()).isFalse();
    assertThat(defaults.maxQueryDepth()).isEqualTo(Integer.MAX_VALUE);

    ParserOptions custom =
        ParserOptions.builder().setAllowPlaceholders(true).setMaxQueryDepth(25).build();
    assertThat(custom.allowPlaceholders()).isTrue();
    assertThat(custom.maxQueryDepth()).isEqualTo(25);
    assertThat(custom.toBuilder().setMaxQueryDepth(10).build())
        .isEqualTo(new ParserOptions(/* allowPlaceholders= */ true, /* maxQueryDepth= */ 10));

    ParserOptions.Builder negativeDepthBuilder = ParserOptions.builder().setMaxQueryDepth(-1);
    assertThrows(IllegalArgumentException.class, negativeDepthBuilder::build);
    assertThrows(
        IllegalArgumentException.class,
        () -> new ParserOptions(/* allowPlaceholders= */ false, /* maxQueryDepth= */ -1));
    assertThrows(NullPointerException.class, () -> QueryParser.parse(null, ParserOptions.DEFAULT));
    assertThrows(NullPointerException.class, () -> QueryParser.parse("//a", null));
  }

  @Test
  public void testMaxQueryDepthZeroAllowsOnlyUnnestedLeaves() throws Exception {
    ParserOptions depthZero = ParserOptions.builder().setMaxQueryDepth(0).build();
    ParserOptions depthZeroWithPlaceholders =
        depthZero.toBuilder().setAllowPlaceholders(true).build();

    assertThat(QueryParser.parse("//a", depthZero)).isEqualTo(new QueryAst.Word("//a"));
    assertThat(QueryParser.parse("@(foo)", depthZeroWithPlaceholders))
        .isEqualTo(new QueryAst.Placeholder("foo", /* startOffset= */ 0, /* endOffset= */ 6));

    for (String nestedQuery :
        ImmutableList.of(
            "(//a)", "deps(//a)", "opt()", "set()", "set(//a)", "let x = //a in $x", "//a + //b")) {
      QuerySyntaxException e =
          assertThrows(
              QuerySyntaxException.class,
              () -> QueryParser.parse(nestedQuery, depthZeroWithPlaceholders));
      assertThat(e).hasMessageThat().isEqualTo("query exceeds maximum nesting depth of 0");
    }
  }

  @Test
  public void testMaxQueryDepthExactBoundariesAcrossConstructs() throws Exception {
    // Nested parentheses: depth 5.
    assertExactDepthBoundary("(".repeat(5) + "//a" + ")".repeat(5), /* exactDepth= */ 5);

    // Nested function calls: depth 5.
    assertExactDepthBoundary("deps(".repeat(5) + "//a" + ")".repeat(5), /* exactDepth= */ 5);

    // Multi-argument function where second argument determines depth: 1 (rdeps) + 4 (deps) = 5.
    assertExactDepthBoundary(
        "rdeps(//a, " + "deps(".repeat(4) + "//b" + ")".repeat(4) + ")", /* exactDepth= */ 5);

    // set(...) has depth 1; wrapped inside parens or a function call it reaches depth 2.
    assertExactDepthBoundary("set(//a //b)", /* exactDepth= */ 1);
    assertExactDepthBoundary("(set(//a))", /* exactDepth= */ 2);
    assertExactDepthBoundary("deps(set(//a))", /* exactDepth= */ 2);

    // Nested let bindings in bodyExpr and varExpr: depth 5.
    assertExactDepthBoundary("let x = //a in ".repeat(5) + "$x", /* exactDepth= */ 5);
    assertExactDepthBoundary(
        "let x = " + "(".repeat(4) + "//a" + ")".repeat(4) + " in $x", /* exactDepth= */ 5);

    // Alternating binary operators: each distinct operator group adds 1 to left-spine depth.
    assertExactDepthBoundary("//a + //b", /* exactDepth= */ 1);
    assertExactDepthBoundary("//a + //b - //c", /* exactDepth= */ 2);
    assertExactDepthBoundary("//a + //b0 - //c0 + //b1 - //c1 + //b2", /* exactDepth= */ 5);

    // Mixed deep subexpressions and binary operator alternations:
    // 1. Deep LHS wrapped by binary operators: deps(deps(//a)) has depth 2, + //b -> 3, - //c -> 4.
    assertExactDepthBoundary("deps(deps(//a)) + //b", /* exactDepth= */ 3);
    assertExactDepthBoundary("deps(deps(//a)) + //b - //c", /* exactDepth= */ 4);

    // 2. Deep RHS in first group followed by alternation: + group has depth 3, - wraps it -> 4.
    assertExactDepthBoundary("//a + deps(deps(//b)) - //c", /* exactDepth= */ 4);

    // 3. Deep RHS in second group (left-associative): (//a + //b) has depth 1, deps(deps(//c)) has
    // depth 2, so outer - group has depth max(1, 2) + 1 = 3.
    assertExactDepthBoundary("//a + //b - deps(deps(//c))", /* exactDepth= */ 3);

    // 4. Function argument starting with word or function call followed by alternating binary tail.
    assertExactDepthBoundary("deps(//a + //b - //c)", /* exactDepth= */ 3);
    assertExactDepthBoundary("deps(kind('rule', //a) + //b - //c)", /* exactDepth= */ 4);
  }

  @Test
  public void testMaxQueryDepthWideFlatExpressionsAllowed() throws Exception {
    ParserOptions depthOne = ParserOptions.builder().setMaxQueryDepth(1).build();
    ParserOptions depthTwo = ParserOptions.builder().setMaxQueryDepth(2).build();

    // 5,000 uniform '+' operators flatten into a single BinaryOperator node of depth 1.
    String uniformSum = "//a" + " + //b".repeat(5000);
    QueryAst sumAst = QueryParser.parse(uniformSum, depthOne);
    assertThat(((QueryAst.BinaryOperator) sumAst).operands()).hasSize(5001);

    // 5,000 words in set(...) have depth 1.
    String largeSet = "set(//a" + " //b".repeat(5000) + ")";
    QueryAst setAst = QueryParser.parse(largeSet, depthOne);
    assertThat(((QueryAst.Set) setAst).words()).hasSize(5001);

    // Sibling subexpressions do not accumulate depth across '+' or ','.
    String siblingParens = "(//a)" + " + (//b)".repeat(1000);
    assertThat(QueryParser.parse(siblingParens, depthTwo))
        .isInstanceOf(QueryAst.BinaryOperator.class);

    String siblingLetsInFunctionArgs = "rdeps(let x = //a in $x, let y = //b in $y)";
    assertThat(QueryParser.parse(siblingLetsInFunctionArgs, depthTwo))
        .isInstanceOf(QueryAst.FunctionCall.class);
  }

  @Test
  public void testMaxQueryDepthRejectsDeepInputsWithoutStackOverflow() throws Exception {
    ParserOptions maxDepth1000 = ParserOptions.builder().setMaxQueryDepth(1000).build();

    // Alternating binary operator chain with 10,000 alternations.
    StringBuilder alternatingOps = new StringBuilder("//a");
    for (int i = 0; i < 5000; i++) {
      alternatingOps.append(" + //b").append(i).append(" - //c").append(i);
    }
    String deepParens = "(".repeat(20_000) + "//a" + ")".repeat(20_000);
    String unclosedDeepParens = "(".repeat(20_000);
    String deepFunctionCalls = "deps(".repeat(20_000) + "//a" + ")".repeat(20_000);
    String unclosedDeepFunctionCalls = "deps(".repeat(20_000);
    String deepLetBodies = "let x = //a in ".repeat(20_000) + "$x";
    String unclosedDeepLetValues = "let x = ".repeat(20_000) + "//a";

    for (String deepQuery :
        ImmutableList.of(
            alternatingOps.toString(),
            deepParens,
            unclosedDeepParens,
            deepFunctionCalls,
            unclosedDeepFunctionCalls,
            deepLetBodies,
            unclosedDeepLetValues)) {
      QuerySyntaxException e =
          assertThrows(
              QuerySyntaxException.class, () -> QueryParser.parse(deepQuery, maxDepth1000));
      assertThat(e).hasMessageThat().isEqualTo("query exceeds maximum nesting depth of 1000");
    }

    // Verify an alternating binary operator chain right at depth 1000 parses and resolves cleanly.
    StringBuilder atLimitAlternating = new StringBuilder("//a");
    for (int i = 0; i < 500; i++) {
      atLimitAlternating.append(" + //b").append(i).append(" - //c").append(i);
    }
    QueryAst atLimitAst = QueryParser.parse(atLimitAlternating.toString(), maxDepth1000);
    QueryExpression resolved = QueryExpressionResolver.resolve(atLimitAst, mockEnvironment());
    assertThat(resolved.toString()).contains("//b499");
  }

  private static void assertExactDepthBoundary(String query, int exactDepth) throws Exception {
    ParserOptions atLimit = ParserOptions.builder().setMaxQueryDepth(exactDepth).build();
    assertThat(QueryParser.parse(query, atLimit))
        .isEqualTo(QueryParser.parse(query, ParserOptions.DEFAULT));

    ParserOptions oneBelowLimit = ParserOptions.builder().setMaxQueryDepth(exactDepth - 1).build();
    QuerySyntaxException e =
        assertThrows(QuerySyntaxException.class, () -> QueryParser.parse(query, oneBelowLimit));
    assertThat(e)
        .hasMessageThat()
        .isEqualTo("query exceeds maximum nesting depth of " + (exactDepth - 1));
  }
}
