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
import static org.junit.Assert.assertThrows;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for the query expression lexer. */
@RunWith(JUnit4.class)
public final class LexerTest {

  private String asString(Lexer.Token[] tokens) {
    StringBuilder buffer = new StringBuilder();
    for (Lexer.Token token : tokens) {
      if (buffer.length() > 0) {
        buffer.append(' ');
      }
      buffer.append(token);
    }
    return buffer.toString();
  }

  private Lexer.Token[] scan(String input) throws QuerySyntaxException {
    Lexer.Token[] tokens;
    try {
      tokens = Lexer.scan(input).toArray(new Lexer.Token[0]);
    } catch (QuerySyntaxException e) {
      QuerySyntaxException eWithPlaceholders =
          assertThrows(
              QuerySyntaxException.class, () -> Lexer.scan(input, /* allowPlaceholders= */ true));
      assertThat(eWithPlaceholders).hasMessageThat().isEqualTo(e.getMessage());
      throw e;
    }
    Lexer.Token[] tokensWithPlaceholders = scanWithPlaceholders(input);
    assertThat(tokensWithPlaceholders).hasLength(tokens.length);
    for (int i = 0; i < tokens.length; i++) {
      assertThat(tokensWithPlaceholders[i].kind).isEqualTo(tokens[i].kind);
      assertThat(tokensWithPlaceholders[i].word).isEqualTo(tokens[i].word);
      assertThat(tokensWithPlaceholders[i].startOffset).isEqualTo(tokens[i].startOffset);
      assertThat(tokensWithPlaceholders[i].endOffset).isEqualTo(tokens[i].endOffset);
    }
    return tokens;
  }

  private Lexer.Token[] scanWithPlaceholders(String input) throws QuerySyntaxException {
    return Lexer.scan(input, /* allowPlaceholders= */ true).toArray(new Lexer.Token[0]);
  }

  @Test
  public void testBasics() throws QuerySyntaxException {
    assertThat(asString(scan(""))).isEqualTo("EOF");
  }

  @Test
  public void testWordsAndKeywords() throws QuerySyntaxException {
    Lexer.Token[] tokens = scan("foo bar wiz intersect");
    assertThat(asString(tokens)).isEqualTo("foo bar wiz intersect EOF");
    assertThat(tokens[0].kind).isEqualTo(Lexer.TokenKind.WORD);
    assertThat(tokens[1].kind).isEqualTo(Lexer.TokenKind.WORD);
    assertThat(tokens[2].kind).isEqualTo(Lexer.TokenKind.WORD);
    assertThat(tokens[3].kind).isEqualTo(Lexer.TokenKind.INTERSECT);
    assertThat(tokens[4].kind).isEqualTo(Lexer.TokenKind.EOF);
  }

  @Test
  public void testPunctuationAndWordBoundaries() throws QuerySyntaxException {
    assertThat(asString(scan("foo(bar,wiz)deps=intersect")))
        .isEqualTo("foo ( bar , wiz ) deps = intersect EOF");
    assertThat(asString(scan("deps(//pkg:target)"))).isEqualTo("deps ( //pkg:target ) EOF");
  }

  @Test
  public void testWordsMayContainDashOrStarButNotStartWithThem() throws QuerySyntaxException {
    assertThat(asString(scan("* foo*"))).isEqualTo("* foo* EOF");
    assertThat(asString(scan("-foo foo-bar"))).isEqualTo("- foo foo-bar EOF");
  }

  @Test
  public void testDotDotDot() throws QuerySyntaxException {
    assertThat(asString(scan("..."))).isEqualTo("... EOF");
  }

  @Test
  public void testQuotation() throws QuerySyntaxException {
    Lexer.Token[] tokens = scan("foo bar 'foo bar'");
    assertThat(asString(tokens)).isEqualTo("foo bar foo bar EOF");
    assertThat(tokens[0].kind).isEqualTo(Lexer.TokenKind.WORD);
    assertThat(tokens[1].kind).isEqualTo(Lexer.TokenKind.WORD);
    assertThat(tokens[2].kind).isEqualTo(Lexer.TokenKind.WORD);
    assertThat(tokens[2].word).isEqualTo("foo bar");
  }

  @Test
  public void testQuotedWordsAreNotIdentifiers() throws QuerySyntaxException {
    Lexer.Token[] tokens = scan("set 'set' \"set\"");
    assertThat(asString(tokens)).isEqualTo("set set set EOF");
    assertThat(tokens[0].kind).isEqualTo(Lexer.TokenKind.SET);
    assertThat(tokens[1].kind).isEqualTo(Lexer.TokenKind.WORD);
    assertThat(tokens[2].kind).isEqualTo(Lexer.TokenKind.WORD);
  }

  @Test
  public void testUnterminatedQuotation() {
    QuerySyntaxException e = assertThrows(QuerySyntaxException.class, () -> scan("'foo"));
    assertThat(e).hasMessageThat().isEqualTo("unclosed quotation");
  }

  @Test
  public void testOperatorWithSpecialCharacters() throws QuerySyntaxException {
    Lexer.Token[] tokens = scan("set(//foo_bar:.*@4)");
    assertThat(asString(tokens)).isEqualTo("set ( //foo_bar:.*@4 ) EOF");
    assertThat(tokens[0].kind).isEqualTo(Lexer.TokenKind.SET);
    assertThat(tokens[1].kind).isEqualTo(Lexer.TokenKind.LPAREN);
    assertThat(tokens[2].kind).isEqualTo(Lexer.TokenKind.WORD);
    assertThat(tokens[3].kind).isEqualTo(Lexer.TokenKind.RPAREN);
  }

  @Test
  public void testOperatorWithQuotedExprWithSpecialCharacters() throws QuerySyntaxException {
    Lexer.Token[] tokens = scan("set(\"//foo_bar:.*@4\")");
    assertThat(asString(tokens)).isEqualTo("set ( //foo_bar:.*@4 ) EOF");
    assertThat(tokens[0].kind).isEqualTo(Lexer.TokenKind.SET);
    assertThat(tokens[1].kind).isEqualTo(Lexer.TokenKind.LPAREN);
    assertThat(tokens[2].kind).isEqualTo(Lexer.TokenKind.WORD);
    assertThat(tokens[3].kind).isEqualTo(Lexer.TokenKind.RPAREN);
  }

  @Test
  public void testOperatorWithQuotedExprWithMoreSpecialCharacters() throws QuerySyntaxException {
    Lexer.Token[] tokens = scan("set(\"//foo:foo=base/2~123[]+asd\")");
    assertThat(asString(tokens)).isEqualTo("set ( //foo:foo=base/2~123[]+asd ) EOF");
    assertThat(tokens[0].kind).isEqualTo(Lexer.TokenKind.SET);
    assertThat(tokens[1].kind).isEqualTo(Lexer.TokenKind.LPAREN);
    assertThat(tokens[2].kind).isEqualTo(Lexer.TokenKind.WORD);
    assertThat(tokens[3].kind).isEqualTo(Lexer.TokenKind.RPAREN);
  }

  @Test
  public void testOperatorWithUnquotedExprWithSpecialCharacters() throws QuerySyntaxException {
    Lexer.Token[] tokens = scan("set(//a:b=bar./@_:~-*$123[]+asd)");
    assertThat(asString(tokens)).isEqualTo("set ( //a:b = bar./@_:~-*$123[] + asd ) EOF");
    assertThat(tokens[0].kind).isEqualTo(Lexer.TokenKind.SET);
    assertThat(tokens[1].kind).isEqualTo(Lexer.TokenKind.LPAREN);
    assertThat(tokens[2].kind).isEqualTo(Lexer.TokenKind.WORD);
    assertThat(tokens[3].kind).isEqualTo(Lexer.TokenKind.EQUALS);
    assertThat(tokens[4].kind).isEqualTo(Lexer.TokenKind.WORD);
    assertThat(tokens[5].kind).isEqualTo(Lexer.TokenKind.PLUS);
    assertThat(tokens[6].kind).isEqualTo(Lexer.TokenKind.WORD);
    assertThat(tokens[7].kind).isEqualTo(Lexer.TokenKind.RPAREN);
  }

  @Test
  public void testUnquotedCanonicalLabels() throws QuerySyntaxException {
    Lexer.Token[] tokens =
        scan("somepath(@foo+@bar+//baz+@@foo +bar,  @@rules_jvm_external++maven+maven//:bar)");
    assertThat(asString(tokens))
        .isEqualTo(
            "somepath ( @foo + @bar + //baz + @@foo + bar , @@rules_jvm_external++maven+maven//:bar"
                + " ) EOF");
    assertThat(tokens[0].kind).isEqualTo(Lexer.TokenKind.WORD);
    assertThat(tokens[1].kind).isEqualTo(Lexer.TokenKind.LPAREN);
    assertThat(tokens[2].kind).isEqualTo(Lexer.TokenKind.WORD);
    assertThat(tokens[3].kind).isEqualTo(Lexer.TokenKind.PLUS);
    assertThat(tokens[4].kind).isEqualTo(Lexer.TokenKind.WORD);
    assertThat(tokens[5].kind).isEqualTo(Lexer.TokenKind.PLUS);
    assertThat(tokens[6].kind).isEqualTo(Lexer.TokenKind.WORD);
    assertThat(tokens[7].kind).isEqualTo(Lexer.TokenKind.PLUS);
    assertThat(tokens[8].kind).isEqualTo(Lexer.TokenKind.WORD);
    assertThat(tokens[9].kind).isEqualTo(Lexer.TokenKind.PLUS);
    assertThat(tokens[10].kind).isEqualTo(Lexer.TokenKind.WORD);
    assertThat(tokens[11].kind).isEqualTo(Lexer.TokenKind.COMMA);
    assertThat(tokens[12].kind).isEqualTo(Lexer.TokenKind.WORD);
    assertThat(tokens[13].kind).isEqualTo(Lexer.TokenKind.RPAREN);
  }

  @Test
  public void testPlaceholdersTokenizedWithOffsetsWhenEnabled() throws QuerySyntaxException {
    String input =
        "deps(@(foo))+@(_bar_2) + '@(foo)' + \"@(bar)\" + @foo + @foo//:bar + @//pkg:target"
            + " + //pkg:.*@4";
    Lexer.Token[] tokens = scanWithPlaceholders(input);
    assertThat(asString(tokens))
        .isEqualTo(
            "deps ( @(foo) ) + @(_bar_2) + @(foo) + @(bar) + @foo + @foo//:bar + @//pkg:target"
                + " + //pkg:.*@4 EOF");

    assertThat(tokens[2].kind).isEqualTo(Lexer.TokenKind.PLACEHOLDER);
    assertThat(tokens[2].word).isEqualTo("foo");
    assertThat(input.substring(tokens[2].startOffset, tokens[2].endOffset)).isEqualTo("@(foo)");

    assertThat(tokens[5].kind).isEqualTo(Lexer.TokenKind.PLACEHOLDER);
    assertThat(tokens[5].word).isEqualTo("_bar_2");
    assertThat(input.substring(tokens[5].startOffset, tokens[5].endOffset)).isEqualTo("@(_bar_2)");

    // Quoted @(foo) strings, unquoted @repo / @// labels, and internal @ characters remain WORDs.
    assertThat(tokens[7].kind).isEqualTo(Lexer.TokenKind.WORD);
    assertThat(tokens[7].word).isEqualTo("@(foo)");
    assertThat(tokens[9].kind).isEqualTo(Lexer.TokenKind.WORD);
    assertThat(tokens[9].word).isEqualTo("@(bar)");
    assertThat(tokens[11].kind).isEqualTo(Lexer.TokenKind.WORD);
    assertThat(tokens[11].word).isEqualTo("@foo");
    assertThat(tokens[13].kind).isEqualTo(Lexer.TokenKind.WORD);
    assertThat(tokens[13].word).isEqualTo("@foo//:bar");
    assertThat(tokens[15].kind).isEqualTo(Lexer.TokenKind.WORD);
    assertThat(tokens[15].word).isEqualTo("@//pkg:target");
    assertThat(tokens[17].kind).isEqualTo(Lexer.TokenKind.WORD);
    assertThat(tokens[17].word).isEqualTo("//pkg:.*@4");

    // "placeholder" itself is not a reserved keyword.
    Lexer.Token[] wordTokens = scan("placeholder");
    assertThat(wordTokens[0].kind).isEqualTo(Lexer.TokenKind.WORD);
  }

  @Test
  public void testPlaceholderOffsetsAfterUnicodeCharacters() throws QuerySyntaxException {
    String input = "'//:äöü🌱' + @(target_set)";
    Lexer.Token[] tokens = scanWithPlaceholders(input);
    assertThat(tokens[2].kind).isEqualTo(Lexer.TokenKind.PLACEHOLDER);
    assertThat(tokens[2].word).isEqualTo("target_set");
    assertThat(input.substring(tokens[2].startOffset, tokens[2].endOffset))
        .isEqualTo("@(target_set)");
  }

  @Test
  public void testInvalidPlaceholdersThrowWhenEnabled() {
    for (String badPlaceholder :
        new String[] {
          "@()", "@(123)", "@(foo-bar)", "@(foo:bar)", "@(//foo)", "@(@foo)", "@($foo)", "@( foo )"
        }) {
      QuerySyntaxException e =
          assertThrows(QuerySyntaxException.class, () -> scanWithPlaceholders(badPlaceholder));
      assertThat(e).hasMessageThat().isEqualTo("invalid placeholder '" + badPlaceholder + "'");
    }

    for (String unclosedPlaceholder : new String[] {"@(", "@(foo", "@(foo\n)", "@((foo))"}) {
      QuerySyntaxException e =
          assertThrows(QuerySyntaxException.class, () -> scanWithPlaceholders(unclosedPlaceholder));
      assertThat(e).hasMessageThat().startsWith("unclosed placeholder '@(");
    }
  }
}
