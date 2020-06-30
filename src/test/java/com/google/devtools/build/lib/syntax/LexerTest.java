// Copyright 2006 The Bazel Authors. All Rights Reserved.
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

import com.google.common.base.Joiner;
import java.util.ArrayList;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests of tokenization behavior of the {@link Lexer}.
 */
@RunWith(JUnit4.class)
public class LexerTest {

  // TODO(adonovan): make these these tests less unnecessarily stateful.

  private final List<SyntaxError> errors = new ArrayList<>();
  private String lastError;

  /**
   * Create a lexer which takes input from the specified string. Resets the
   * error handler beforehand.
   */
  private Lexer createLexer(String input) {
    ParserInput inputSource = ParserInput.fromString(input, "/some/path.txt");
    errors.clear();
    lastError = null;
    return new Lexer(inputSource, FileOptions.DEFAULT, errors);
  }

  private static class Token {
    TokenKind kind;
    int start;
    int end;
    Object value;

    @Override
    public String toString() {
      return kind == TokenKind.STRING
          ? "\"" + value + "\""
          : value == null ? kind.toString() : value.toString();
    }
  }

  private ArrayList<Token> allTokens(Lexer lexer) {
    ArrayList<Token> result = new ArrayList<>();
    do {
      lexer.nextToken();
      Token tok = new Token();
      tok.kind = lexer.kind;
      tok.start = lexer.start;
      tok.end = lexer.end;
      tok.value = lexer.value;
      result.add(tok);
    } while (lexer.kind != TokenKind.EOF);

    for (SyntaxError error : errors) {
      lastError = error.location().file() + ":" + error.location().line() + ": " + error.message();
    }

    return result;
  }

  private Token[] tokens(String input) {
    ArrayList<Token> result = allTokens(createLexer(input));
    return result.toArray(new Token[0]);
  }

  /**
   * Lexes the specified input string, and returns a string containing just the line numbers of each
   * token.
   */
  private String linenums(String input) {
    Lexer lexer = createLexer(input);
    StringBuilder buf = new StringBuilder();
    for (Token tok : allTokens(lexer)) {
      if (buf.length() > 0) {
        buf.append(' ');
      }
      int line = lexer.locs.getLocation(tok.start).line();
      buf.append(line);
    }
    return buf.toString();
  }

  /**
   * Returns a string containing the names of the tokens and their associated
   * values. (String-literals are printed without escaping.)
   */
  private static String values(Token[] tokens) {
    StringBuilder buffer = new StringBuilder();
    for (Token token : tokens) {
      if (buffer.length() > 0) {
        buffer.append(' ');
      }
      buffer.append(token.kind.name());
      if (token.value != null) {
        buffer.append('(').append(token.value).append(')');
      }
    }
    return buffer.toString();
  }

  /**
   * Returns a string containing just the names of the tokens.
   */
  private static String names(Token[] tokens) {
    StringBuilder buf = new StringBuilder();
    for (Token tok : tokens) {
      if (buf.length() > 0) {
        buf.append(' ');
      }
      buf.append(tok.kind.name());
    }
    return buf.toString();
  }

  /**
   * Returns a string containing just the half-open position intervals of each
   * token. e.g. "[3,4) [4,9)".
   */
  private static String positions(Token[] tokens) {
    StringBuilder buf = new StringBuilder();
    for (Token tok : tokens) {
      if (buf.length() > 0) {
        buf.append(' ');
      }
      buf.append('[').append(tok.start).append(',').append(tok.end).append(')');
    }
    return buf.toString();
  }

  @Test
  public void testBasics1() throws Exception {
    assertThat(names(tokens("wiz) "))).isEqualTo("IDENTIFIER RPAREN NEWLINE EOF");
    assertThat(names(tokens("wiz )"))).isEqualTo("IDENTIFIER RPAREN NEWLINE EOF");
    assertThat(names(tokens(" wiz)")))
        .isEqualTo("INDENT IDENTIFIER RPAREN NEWLINE OUTDENT NEWLINE EOF");
    assertThat(names(tokens(" wiz ) ")))
        .isEqualTo("INDENT IDENTIFIER RPAREN NEWLINE OUTDENT NEWLINE EOF");
    assertThat(names(tokens("wiz\t)"))).isEqualTo("IDENTIFIER RPAREN NEWLINE EOF");
  }

  @Test
  public void testBasics2() throws Exception {
    assertThat(names(tokens(")"))).isEqualTo("RPAREN NEWLINE EOF");
    assertThat(names(tokens(" )"))).isEqualTo("INDENT RPAREN NEWLINE OUTDENT NEWLINE EOF");
    assertThat(names(tokens(" ) "))).isEqualTo("INDENT RPAREN NEWLINE OUTDENT NEWLINE EOF");
    assertThat(names(tokens(") "))).isEqualTo("RPAREN NEWLINE EOF");
  }

  @Test
  public void testBasics3() throws Exception {
    assertThat(names(tokens("123#456\n789"))).isEqualTo("INT NEWLINE INT NEWLINE EOF");
    assertThat(names(tokens("123 #456\n789"))).isEqualTo("INT NEWLINE INT NEWLINE EOF");
    assertThat(names(tokens("123#456 \n789"))).isEqualTo("INT NEWLINE INT NEWLINE EOF");
    assertThat(names(tokens("123#456\n 789")))
        .isEqualTo("INT NEWLINE INDENT INT NEWLINE OUTDENT NEWLINE EOF");
    assertThat(names(tokens("123#456\n789 "))).isEqualTo("INT NEWLINE INT NEWLINE EOF");
  }

  @Test
  public void testBasics4() throws Exception {
    assertThat(names(tokens(""))).isEqualTo("NEWLINE EOF");
    assertThat(names(tokens("# foo"))).isEqualTo("NEWLINE EOF");
    assertThat(names(tokens("1 2 3 4"))).isEqualTo("INT INT INT INT NEWLINE EOF");
    assertThat(names(tokens("1.234"))).isEqualTo("INT DOT INT NEWLINE EOF");
    assertThat(names(tokens("foo(bar, wiz)")))
        .isEqualTo("IDENTIFIER LPAREN IDENTIFIER COMMA IDENTIFIER RPAREN " + "NEWLINE EOF");
  }

  @Test
  public void testNoWhiteSpaceBetweenTokens() throws Exception {
    assertThat(names(tokens("6or()"))).isEqualTo("INT OR LPAREN RPAREN NEWLINE EOF");
    assertThat(names(tokens("0in(''and[])")))
        .isEqualTo("INT IN LPAREN STRING AND LBRACKET RBRACKET RPAREN NEWLINE EOF");

    assertThat(values(tokens("0or()"))).isEqualTo("INT(0) IDENTIFIER(r) LPAREN RPAREN NEWLINE EOF");
    assertThat(lastError).isEqualTo("/some/path.txt:1: invalid base-8 integer constant: 0o");
  }

  @Test
  public void testNonAsciiIdentifiers() throws Exception {
    tokens("ümlaut");
    assertThat(lastError.toString()).contains("invalid character: 'ü'");
    tokens("umläut");
    assertThat(lastError.toString()).contains("invalid character: 'ä'");
  }

  @Test
  public void testCrLf() throws Exception {
    assertThat(names(tokens("\r\n\r\n"))).isEqualTo("NEWLINE EOF");
    assertThat(names(tokens("\r\n\r1\r\r\n"))).isEqualTo("INT NEWLINE EOF");
    assertThat(names(tokens("# foo\r\n# bar\r\n"))).isEqualTo("NEWLINE EOF");
  }

  @Test
  public void testIntegers() throws Exception {
    // Detection of MINUS immediately following integer constant proves we
    // don't consume too many chars.

    // decimal
    assertThat(values(tokens("12345-"))).isEqualTo("INT(12345) MINUS NEWLINE EOF");

    // octal
    assertThat(values(tokens("0o12345-"))).isEqualTo("INT(5349) MINUS NEWLINE EOF");
    assertThat(values(tokens("0O77"))).isEqualTo("INT(63) NEWLINE EOF");

    // octal (bad)
    assertThat(values(tokens("012349-"))).isEqualTo("INT(0) MINUS NEWLINE EOF");
    assertThat(lastError.toString())
        .isEqualTo("/some/path.txt:1: invalid base-8 integer constant: 012349");

    assertThat(values(tokens("0o"))).isEqualTo("INT(0) NEWLINE EOF");
    assertThat(lastError.toString())
        .isEqualTo("/some/path.txt:1: invalid base-8 integer constant: 0o");

    assertThat(values(tokens("012345"))).isEqualTo("INT(5349) NEWLINE EOF");
    assertThat(lastError.toString())
        .isEqualTo("/some/path.txt:1: invalid octal value `012345`, should be: `0o12345`");

    // hexadecimal (uppercase)
    assertThat(values(tokens("0X12345F-"))).isEqualTo("INT(1193055) MINUS NEWLINE EOF");

    // hexadecimal (lowercase)
    assertThat(values(tokens("0x12345f-"))).isEqualTo("INT(1193055) MINUS NEWLINE EOF");

    // hexadecimal (lowercase) [note: "g" cause termination of token]
    assertThat(values(tokens("0x12345g-"))).isEqualTo("INT(74565) IDENTIFIER(g) MINUS NEWLINE EOF");
  }

  @Test
  public void testIntegersAndDot() throws Exception {
    assertThat(values(tokens("1.2345"))).isEqualTo("INT(1) DOT INT(2345) NEWLINE EOF");

    assertThat(values(tokens("1.2.345"))).isEqualTo("INT(1) DOT INT(2) DOT INT(345) NEWLINE EOF");

    assertThat(values(tokens("1.0E10"))).isEqualTo("INT(1) DOT INT(0) NEWLINE EOF");
    assertThat(lastError.toString())
        .isEqualTo("/some/path.txt:1: invalid base-8 integer constant: 0E10");

    assertThat(values(tokens("1.03E-10"))).isEqualTo("INT(1) DOT INT(0) MINUS INT(10) NEWLINE EOF");
    assertThat(lastError.toString())
        .isEqualTo("/some/path.txt:1: invalid base-8 integer constant: 03E");

    assertThat(values(tokens(". 123"))).isEqualTo("DOT INT(123) NEWLINE EOF");
    assertThat(values(tokens(".123"))).isEqualTo("DOT INT(123) NEWLINE EOF");
    assertThat(values(tokens(".abc"))).isEqualTo("DOT IDENTIFIER(abc) NEWLINE EOF");

    assertThat(values(tokens("foo.123"))).isEqualTo("IDENTIFIER(foo) DOT INT(123) NEWLINE EOF");
    assertThat(values(tokens("foo.bcd")))
        .isEqualTo("IDENTIFIER(foo) DOT IDENTIFIER(bcd) NEWLINE EOF"); // 'b' are hex chars
    assertThat(values(tokens("foo.xyz")))
        .isEqualTo("IDENTIFIER(foo) DOT IDENTIFIER(xyz) NEWLINE EOF");
  }

  @Test
  public void testStringDelimiters() throws Exception {
    assertThat(values(tokens("\"foo\""))).isEqualTo("STRING(foo) NEWLINE EOF");
    assertThat(values(tokens("'foo'"))).isEqualTo("STRING(foo) NEWLINE EOF");
  }

  @Test
  public void testQuotesInStrings() throws Exception {
    assertThat(values(tokens("'foo\\'bar'"))).isEqualTo("STRING(foo'bar) NEWLINE EOF");
    assertThat(values(tokens("\"foo'bar\""))).isEqualTo("STRING(foo'bar) NEWLINE EOF");
    assertThat(values(tokens("'foo\"bar'"))).isEqualTo("STRING(foo\"bar) NEWLINE EOF");
    assertThat(values(tokens("\"foo\\\"bar\""))).isEqualTo("STRING(foo\"bar) NEWLINE EOF");
  }

  @Test
  public void testStringEscapes() throws Exception {
    assertThat(values(tokens("'a\\tb\\nc\\rd'")))
        .isEqualTo("STRING(a\tb\nc\rd) NEWLINE EOF"); // \t \r \n
    assertThat(values(tokens("'x\\hx'")))
        .isEqualTo("STRING(x\\hx) NEWLINE EOF"); // \h is unknown => "\h"
    assertThat(values(tokens("'\\$$'"))).isEqualTo("STRING(\\$$) NEWLINE EOF");
    assertThat(values(tokens("'a\\\nb'")))
        .isEqualTo("STRING(ab) NEWLINE EOF"); // escape end of line
    assertThat(values(tokens("\"ab\\ucd\""))).isEqualTo("STRING(abcd) NEWLINE EOF");
    assertThat(lastError).isEqualTo("/some/path.txt:1: invalid escape sequence: \\u");
  }

  @Test
  public void testEscapedCrlfInString() throws Exception {
    assertThat(values(tokens("'a\\\r\nb'"))).isEqualTo("STRING(ab) NEWLINE EOF");
    assertThat(values(tokens("\"a\\\r\nb\""))).isEqualTo("STRING(ab) NEWLINE EOF");
    assertThat(values(tokens("\"\"\"a\\\r\nb\"\"\""))).isEqualTo("STRING(ab) NEWLINE EOF");
    assertThat(values(tokens("'''a\\\r\nb'''"))).isEqualTo("STRING(ab) NEWLINE EOF");
    assertThat(values(tokens("r'a\\\r\nb'"))).isEqualTo("STRING(a\\\nb) NEWLINE EOF");
    assertThat(values(tokens("r\"a\\\r\nb\""))).isEqualTo("STRING(a\\\nb) NEWLINE EOF");
    assertThat(values(tokens("r\"a\\\r\n\\\nb\""))).isEqualTo("STRING(a\\\n\\\nb) NEWLINE EOF");
  }

  @Test
  public void testRawString() throws Exception {
    assertThat(values(tokens("r'abcd'"))).isEqualTo("STRING(abcd) NEWLINE EOF");
    assertThat(values(tokens("r\"abcd\""))).isEqualTo("STRING(abcd) NEWLINE EOF");
    assertThat(values(tokens("r'a\\tb\\nc\\rd'")))
        .isEqualTo("STRING(a\\tb\\nc\\rd) NEWLINE EOF"); // r'a\tb\nc\rd'
    assertThat(values(tokens("r\"a\\\"\""))).isEqualTo("STRING(a\\\") NEWLINE EOF"); // r"a\""
    assertThat(values(tokens("r'a\\\\b'"))).isEqualTo("STRING(a\\\\b) NEWLINE EOF"); // r'a\\b'
    assertThat(values(tokens("r'ab'r"))).isEqualTo("STRING(ab) IDENTIFIER(r) NEWLINE EOF");

    // Unterminated raw string
    values(tokens("r'\\'")); // r'\'
    assertThat(lastError.toString())
        .isEqualTo("/some/path.txt:1: unterminated string literal at eof");
  }

  @Test
  public void testTripleRawString() throws Exception {
    // r'''a\ncd'''
    assertThat(values(tokens("r'''ab\\ncd'''"))).isEqualTo("STRING(ab\\ncd) NEWLINE EOF");
    // r"""ab
    // cd"""
    assertThat(values(tokens("\"\"\"ab\ncd\"\"\""))).isEqualTo("STRING(ab\ncd) NEWLINE EOF");

    // Unterminated raw string
    values(tokens("r'''\\'''")); // r'''\'''
    assertThat(lastError.toString())
        .isEqualTo("/some/path.txt:1: unterminated string literal at eof");
  }

  @Test
  public void testOctalEscapes() throws Exception {
    // Regression test for a bug.
    assertThat(values(tokens("'\\0 \\1 \\11 \\77 \\111 \\1111 \\377'")))
        .isEqualTo("STRING(\0 \1 \t \u003f I I1 \u00ff) NEWLINE EOF");
    // Test boundaries (non-octal char, EOF).
    assertThat(values(tokens("'\\1b \\1'"))).isEqualTo("STRING(\1b \1) NEWLINE EOF");
  }

  @Test
  public void testOctalEscapeOutOfRange() throws Exception {
    assertThat(values(tokens("'\\777'"))).isEqualTo("STRING(\u00ff) NEWLINE EOF");
    assertThat(lastError.toString())
        .isEqualTo("/some/path.txt:1: octal escape sequence out of range (maximum is \\377)");
  }

  @Test
  public void testTripleQuotedStrings() throws Exception {
    assertThat(values(tokens("\"\"\"a\"b'c \n d\"\"e\"\"\"")))
        .isEqualTo("STRING(a\"b'c \n d\"\"e) NEWLINE EOF");
    assertThat(values(tokens("'''a\"b'c \n d\"\"e'''")))
        .isEqualTo("STRING(a\"b'c \n d\"\"e) NEWLINE EOF");
  }

  @Test
  public void testBadChar() throws Exception {
    assertThat(values(tokens("a$b"))).isEqualTo("IDENTIFIER(a) IDENTIFIER(b) NEWLINE EOF");
    assertThat(lastError.toString()).isEqualTo("/some/path.txt:1: invalid character: '$'");
  }

  @Test
  public void testIndentation() throws Exception {
    assertThat(values(tokens("1\n2\n3")))
        .isEqualTo("INT(1) NEWLINE INT(2) NEWLINE INT(3) NEWLINE EOF");
    assertThat(values(tokens("1\n  2\n  3\n4 ")))
        .isEqualTo(
            "INT(1) NEWLINE INDENT INT(2) NEWLINE INT(3) NEWLINE OUTDENT " + "INT(4) NEWLINE EOF");
    assertThat(values(tokens("1\n  2\n  3")))
        .isEqualTo("INT(1) NEWLINE INDENT INT(2) NEWLINE INT(3) NEWLINE OUTDENT " + "NEWLINE EOF");
    assertThat(values(tokens("1\n  2\n    3")))
        .isEqualTo(
            "INT(1) NEWLINE INDENT INT(2) NEWLINE INDENT INT(3) NEWLINE "
                + "OUTDENT OUTDENT NEWLINE EOF");
    assertThat(values(tokens("1\n  2\n    3\n  4\n5")))
        .isEqualTo(
            "INT(1) NEWLINE INDENT INT(2) NEWLINE INDENT INT(3) NEWLINE "
                + "OUTDENT INT(4) NEWLINE OUTDENT INT(5) NEWLINE EOF");

    assertThat(values(tokens("1\n  2\n    3\n   4\n5")))
        .isEqualTo(
            "INT(1) NEWLINE INDENT INT(2) NEWLINE INDENT INT(3) NEWLINE "
                + "OUTDENT INT(4) NEWLINE OUTDENT INT(5) NEWLINE EOF");
    assertThat(lastError.toString()).isEqualTo("/some/path.txt:4: indentation error");
  }

  @Test
  public void testIndentationWithTab() throws Exception {
    tokens("def x():\n\tpass");
    assertThat(lastError).contains("Tab characters are not allowed");
  }

  @Test
  public void testIndentationWithCrLf() throws Exception {
    assertThat(values(tokens("1\r\n  2\r\n")))
        .isEqualTo("INT(1) NEWLINE INDENT INT(2) NEWLINE OUTDENT NEWLINE EOF");
    assertThat(values(tokens("1\r\n  2\r\n\r\n")))
        .isEqualTo("INT(1) NEWLINE INDENT INT(2) NEWLINE OUTDENT NEWLINE EOF");
    assertThat(values(tokens("1\r\n  2\r\n    3\r\n  4\r\n5")))
        .isEqualTo(
            "INT(1) NEWLINE INDENT INT(2) NEWLINE INDENT INT(3) NEWLINE OUTDENT INT(4) "
                + "NEWLINE OUTDENT INT(5) NEWLINE EOF");
    assertThat(values(tokens("1\r\n  2\r\n\r\n  3\r\n4")))
        .isEqualTo(
            "INT(1) NEWLINE INDENT INT(2) NEWLINE INT(3) NEWLINE OUTDENT INT(4) NEWLINE EOF");
  }

  @Test
  public void testIndentationInsideParens() throws Exception {
    // Indentation is ignored inside parens:
    assertThat(values(tokens("1 (\n  2\n    3\n  4\n5")))
        .isEqualTo("INT(1) LPAREN INT(2) INT(3) INT(4) INT(5) NEWLINE EOF");
    assertThat(values(tokens("1 {\n  2\n    3\n  4\n5")))
        .isEqualTo("INT(1) LBRACE INT(2) INT(3) INT(4) INT(5) NEWLINE EOF");
    assertThat(values(tokens("1 [\n  2\n    3\n  4\n5")))
        .isEqualTo("INT(1) LBRACKET INT(2) INT(3) INT(4) INT(5) NEWLINE EOF");
    assertThat(values(tokens("1 [\n  2]\n    3\n    4\n5")))
        .isEqualTo(
            "INT(1) LBRACKET INT(2) RBRACKET NEWLINE INDENT INT(3) "
                + "NEWLINE INT(4) NEWLINE OUTDENT INT(5) NEWLINE EOF");
  }

  @Test
  public void testIndentationAtEOF() throws Exception {
    // Matching OUTDENTS are created at EOF:
    assertThat(values(tokens("\n  1"))).isEqualTo("INDENT INT(1) NEWLINE OUTDENT NEWLINE EOF");
  }

  @Test
  public void testIndentationOnFirstLine() throws Exception {
    assertThat(values(tokens("    1"))).isEqualTo("INDENT INT(1) NEWLINE OUTDENT NEWLINE EOF");
    assertThat(values(tokens("\n\n    1"))).isEqualTo("INDENT INT(1) NEWLINE OUTDENT NEWLINE EOF");
  }

  @Test
  public void testBlankLineIndentation() throws Exception {
    // Blank lines and comment lines should not generate any newlines indents
    // (but note that every input ends with NEWLINE EOF).
    assertThat(names(tokens("\n      #\n"))).isEqualTo("NEWLINE EOF");
    assertThat(names(tokens("      #"))).isEqualTo("NEWLINE EOF");
    assertThat(names(tokens("      #\n"))).isEqualTo("NEWLINE EOF");
    assertThat(names(tokens("      #comment\n"))).isEqualTo("NEWLINE EOF");
    assertThat(names(tokens("def f(x):\n" + "  # comment\n" + "\n" + "  \n" + "  return x\n")))
        .isEqualTo(
            "DEF IDENTIFIER LPAREN IDENTIFIER RPAREN COLON NEWLINE "
                + "INDENT RETURN IDENTIFIER NEWLINE "
                + "OUTDENT NEWLINE EOF");
  }

  @Test
  public void testBackslash() throws Exception {
    assertThat(names(tokens("a\\\nb"))).isEqualTo("IDENTIFIER IDENTIFIER NEWLINE EOF");
    assertThat(names(tokens("a\\\r\nb"))).isEqualTo("IDENTIFIER IDENTIFIER NEWLINE EOF");
    assertThat(names(tokens("a\\ b"))).isEqualTo("IDENTIFIER ILLEGAL IDENTIFIER NEWLINE EOF");
    assertThat(names(tokens("a(\\\n2)"))).isEqualTo("IDENTIFIER LPAREN INT RPAREN NEWLINE EOF");
  }

  @Test
  public void testTokenPositions() throws Exception {
    assertThat(positions(tokens("foo(bar, {1: 'quux'}, \"\"\"b\"\"\", r\"\")")))
        .isEqualTo(
            // foo (     bar   ,     {      1       :
            "[0,3) [3,4) [4,7) [7,8) [9,10) [10,11) [11,12)"
                //  'quux'  }       ,       """b""" ,       r""     )       NEWLINE EOF
                + " [13,19) [19,20) [20,21) [22,29) [29,30) [31,34) [34,35) [35,35) [35,35)");
  }

  @Test
  public void testLineNumbers() throws Exception {
    assertThat(linenums("foo = 1\nbar = 2\n\nwiz = 3")).isEqualTo("1 1 1 1 2 2 2 2 4 4 4 4 4");

    assertThat(values(tokens("foo = 1\nbar = 2\n\nwiz = $\nbar = 2")))
        .isEqualTo(
            "IDENTIFIER(foo) EQUALS INT(1) NEWLINE "
                + "IDENTIFIER(bar) EQUALS INT(2) NEWLINE "
                + "IDENTIFIER(wiz) EQUALS NEWLINE "
                + "IDENTIFIER(bar) EQUALS INT(2) NEWLINE EOF");
    assertThat(lastError.toString()).isEqualTo("/some/path.txt:4: invalid character: '$'");

    // '\\n' in string should not increment linenum:
    String s = "1\n'foo\\nbar'\3";
    assertThat(values(tokens(s))).isEqualTo("INT(1) NEWLINE STRING(foo\nbar) NEWLINE EOF");
    assertThat(linenums(s)).isEqualTo("1 1 2 2 2");
  }

  @Test
  public void testContainsErrors() throws Exception {
    Lexer lexerSuccess = createLexer("foo");
    allTokens(lexerSuccess); // ensure the file has been completely scanned
    assertThat(errors).isEmpty();

    Lexer lexerFail = createLexer("f$o");
    allTokens(lexerFail);
    assertThat(errors).isNotEmpty();

    String s = "'unterminated";
    lexerFail = createLexer(s);
    allTokens(lexerFail);
    assertThat(errors).isNotEmpty();
    assertThat(values(tokens(s))).isEqualTo("STRING(unterminated) NEWLINE EOF");
  }

  @Test
  public void testUnterminatedRawStringWithEscapingError() throws Exception {
    assertThat(names(tokens("r'\\"))).isEqualTo("STRING NEWLINE EOF");
    assertThat(lastError).isEqualTo("/some/path.txt:1: unterminated string literal at eof");
  }

  @Test
  public void testFirstCharIsTab() {
    assertThat(names(tokens("\t"))).isEqualTo("NEWLINE EOF");
    assertThat(lastError)
        .isEqualTo(
            "/some/path.txt:1: Tab characters are not allowed for indentation. Use spaces"
                + " instead.");
  }

  /**
   * Returns the first error whose string form contains the specified substring, or throws an
   * informative AssertionError if there is none.
   *
   * <p>Exposed for use by other frontend tests.
   */
  static SyntaxError assertContainsError(List<SyntaxError> errors, String substr) {
    for (SyntaxError error : errors) {
      if (error.toString().contains(substr)) {
        return error;
      }
    }
    if (errors.isEmpty()) {
      throw new AssertionError("no errors, want '" + substr + "'");
    } else {
      throw new AssertionError(
          "error '" + substr + "' not found, but got these:\n" + Joiner.on("\n").join(errors));
    }
  }
}
