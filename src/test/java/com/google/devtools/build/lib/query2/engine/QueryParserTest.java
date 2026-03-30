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
  // 'expectedPrettyPrintOutput'.
  private static String checkPrettyPrint(String expectedPrettyPrintOutput, String query)
      throws Exception {
    assertThat(QueryExpression.parse(query, mockEnvironment()).toString())
        .isEqualTo(expectedPrettyPrintOutput);
    return expectedPrettyPrintOutput;
  }

  public static void checkParseFails(String query, String expectedError) {
    QuerySyntaxException e =
        assertThrows(
            QuerySyntaxException.class, () -> QueryExpression.parse(query, mockEnvironment()));
    assertThat(e).hasMessageThat().isEqualTo(expectedError);
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
    checkParseFails("rdeps(a, b, 3, ", "too many arguments to function 'rdeps' at ','");
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
        "'set('//foo' + 'bar')'", "unexpected token '//foo' after query expression '\"set(\"'");
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
}
