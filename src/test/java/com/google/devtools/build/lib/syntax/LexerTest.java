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

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.vfs.PathFragment;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests of tokenization behavior of the {@link Lexer}.
 */
@RunWith(JUnit4.class)
public class LexerTest {
  private String lastError;
  private Location lastErrorLocation;

  /**
   * Create a lexer which takes input from the specified string. Resets the
   * error handler beforehand.
   */
  private Lexer createLexer(String input) {
    PathFragment somePath = PathFragment.create("/some/path.txt");
    ParserInputSource inputSource = ParserInputSource.create(input, somePath);
    Reporter reporter = new Reporter(new EventBus());
    reporter.addHandler(new EventHandler() {
      @Override
      public void handle(Event event) {
        if (EventKind.ERRORS.contains(event.getKind())) {
          lastErrorLocation = event.getLocation();
          lastError = lastErrorLocation.getPath() + ":"
              + event.getLocation().getStartLineAndColumn().getLine() + ": " + event.getMessage();
        }
      }
    });

    return new Lexer(inputSource, reporter);
  }

  public Token[] tokens(String input) {
    return createLexer(input).getTokens().toArray(new Token[0]);
  }

  /**
   * Lexes the specified input string, and returns a string containing just the
   * linenumbers of each token.
   */
  private String linenums(String input) {
    Lexer lexer = createLexer(input);
    StringBuilder buf = new StringBuilder();
    for (Token tok : lexer.getTokens()) {
      if (buf.length() > 0) {
        buf.append(' ');
      }
      int line =
        lexer.createLocation(tok.left, tok.left).getStartLineAndColumn().getLine();
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
      buf.append('[')
         .append(tok.left)
         .append(',')
         .append(tok.right)
         .append(')');
    }
    return buf.toString();
  }

  @Test
  public void testBasics1() throws Exception {
    assertEquals("IDENTIFIER RPAREN NEWLINE EOF", names(tokens("wiz) ")));
    assertEquals("IDENTIFIER RPAREN NEWLINE EOF", names(tokens("wiz )")));
    assertEquals("IDENTIFIER RPAREN NEWLINE EOF", names(tokens(" wiz)")));
    assertEquals("IDENTIFIER RPAREN NEWLINE EOF", names(tokens(" wiz ) ")));
    assertEquals("IDENTIFIER RPAREN NEWLINE EOF", names(tokens("wiz\t)")));
  }

  @Test
  public void testBasics2() throws Exception {
    assertEquals("RPAREN NEWLINE EOF", names(tokens(")")));
    assertEquals("RPAREN NEWLINE EOF", names(tokens(" )")));
    assertEquals("RPAREN NEWLINE EOF", names(tokens(" ) ")));
    assertEquals("RPAREN NEWLINE EOF", names(tokens(") ")));
  }

  @Test
  public void testBasics3() throws Exception {
    assertEquals("INT COMMENT NEWLINE INT NEWLINE EOF", names(tokens("123#456\n789")));
    assertEquals("INT COMMENT NEWLINE INT NEWLINE EOF", names(tokens("123 #456\n789")));
    assertEquals("INT COMMENT NEWLINE INT NEWLINE EOF", names(tokens("123#456 \n789")));
    assertEquals("INT COMMENT NEWLINE INDENT INT NEWLINE OUTDENT NEWLINE EOF",
                 names(tokens("123#456\n 789")));
    assertEquals("INT COMMENT NEWLINE INT NEWLINE EOF", names(tokens("123#456\n789 ")));
  }

  @Test
  public void testBasics4() throws Exception {
    assertEquals("NEWLINE EOF", names(tokens("")));
    assertEquals("COMMENT NEWLINE EOF", names(tokens("# foo")));
    assertEquals("INT INT INT INT NEWLINE EOF", names(tokens("1 2 3 4")));
    assertEquals("INT DOT INT NEWLINE EOF", names(tokens("1.234")));
    assertEquals("IDENTIFIER LPAREN IDENTIFIER COMMA IDENTIFIER RPAREN "
                 + "NEWLINE EOF", names(tokens("foo(bar, wiz)")));
  }

  @Test
  public void testCrLf() throws Exception {
    assertEquals("NEWLINE EOF", names(tokens("\r\n\r\n")));
    assertEquals("NEWLINE INT NEWLINE EOF", names(tokens("\r\n\r1\r\r\n")));
    assertEquals("COMMENT NEWLINE COMMENT NEWLINE EOF", names(tokens("# foo\r\n# bar\r\n")));
  }

  @Test
  public void testIntegers() throws Exception {
    // Detection of MINUS immediately following integer constant proves we
    // don't consume too many chars.

    // decimal
    assertEquals("INT(12345) MINUS NEWLINE EOF", values(tokens("12345-")));

    // octal
    assertEquals("INT(5349) MINUS NEWLINE EOF", values(tokens("012345-")));

    // octal (bad)
    assertEquals("INT(0) MINUS NEWLINE EOF", values(tokens("012349-")));
    assertEquals("/some/path.txt:1: invalid base-8 integer constant: 012349",
                 lastError.toString());

    // hexadecimal (uppercase)
    assertEquals("INT(1193055) MINUS NEWLINE EOF", values(tokens("0X12345F-")));

    // hexadecimal (lowercase)
    assertEquals("INT(1193055) MINUS NEWLINE EOF", values(tokens("0x12345f-")));

    // hexadecimal (lowercase) [note: "g" cause termination of token]
    assertEquals("INT(74565) IDENTIFIER(g) MINUS NEWLINE EOF",
                 values(tokens("0x12345g-")));
  }

  @Test
  public void testIntegersAndDot() throws Exception {
    assertEquals("INT(1) DOT INT(2345) NEWLINE EOF", values(tokens("1.2345")));

    assertEquals("INT(1) DOT INT(2) DOT INT(345) NEWLINE EOF",
                 values(tokens("1.2.345")));

    assertEquals("INT(1) DOT INT(0) NEWLINE EOF", values(tokens("1.23E10")));
    assertEquals("/some/path.txt:1: invalid base-10 integer constant: 23E10",
                 lastError.toString());

    assertEquals("INT(1) DOT INT(0) MINUS INT(10) NEWLINE EOF",
                 values(tokens("1.23E-10")));
    assertEquals("/some/path.txt:1: invalid base-10 integer constant: 23E",
                 lastError.toString());

    assertEquals("DOT INT(123) NEWLINE EOF", values(tokens(". 123")));
    assertEquals("DOT INT(123) NEWLINE EOF", values(tokens(".123")));
    assertEquals("DOT IDENTIFIER(abc) NEWLINE EOF", values(tokens(".abc")));

    assertEquals("IDENTIFIER(foo) DOT INT(123) NEWLINE EOF",
                 values(tokens("foo.123")));
    assertEquals("IDENTIFIER(foo) DOT IDENTIFIER(bcd) NEWLINE EOF",
                 values(tokens("foo.bcd"))); // 'b' are hex chars
    assertEquals("IDENTIFIER(foo) DOT IDENTIFIER(xyz) NEWLINE EOF",
                 values(tokens("foo.xyz")));
  }

  @Test
  public void testStringDelimiters() throws Exception {
    assertEquals("STRING(foo) NEWLINE EOF", values(tokens("\"foo\"")));
    assertEquals("STRING(foo) NEWLINE EOF", values(tokens("'foo'")));
  }

  @Test
  public void testQuotesInStrings() throws Exception {
    assertEquals("STRING(foo'bar) NEWLINE EOF", values(tokens("'foo\\'bar'")));
    assertEquals("STRING(foo'bar) NEWLINE EOF", values(tokens("\"foo'bar\"")));
    assertEquals("STRING(foo\"bar) NEWLINE EOF", values(tokens("'foo\"bar'")));
    assertEquals("STRING(foo\"bar) NEWLINE EOF",
                 values(tokens("\"foo\\\"bar\"")));
  }

  @Test
  public void testStringEscapes() throws Exception {
    assertEquals("STRING(a\tb\nc\rd) NEWLINE EOF",
                 values(tokens("'a\\tb\\nc\\rd'"))); // \t \r \n
    assertEquals("STRING(x\\hx) NEWLINE EOF",
                 values(tokens("'x\\hx'"))); // \h is unknown => "\h"
    assertEquals("STRING(\\$$) NEWLINE EOF", values(tokens("'\\$$'")));
    assertEquals("STRING(ab) NEWLINE EOF",
                 values(tokens("'a\\\nb'"))); // escape end of line
    assertEquals("STRING(abcd) NEWLINE EOF",
                 values(tokens("\"ab\\ucd\"")));
    assertEquals("/some/path.txt:1: escape sequence not implemented: \\u",
                 lastError.toString());
  }

  @Test
  public void testEscapedCrlfInString() throws Exception {
    assertEquals("STRING(ab) NEWLINE EOF",
                 values(tokens("'a\\\r\nb'")));
    assertEquals("STRING(ab) NEWLINE EOF",
                 values(tokens("\"a\\\r\nb\"")));
    assertEquals("STRING(ab) NEWLINE EOF",
                 values(tokens("\"\"\"a\\\r\nb\"\"\"")));
    assertEquals("STRING(ab) NEWLINE EOF",
                 values(tokens("'''a\\\r\nb'''")));
    assertEquals("STRING(a\\\nb) NEWLINE EOF",
                 values(tokens("r'a\\\r\nb'")));
    assertEquals("STRING(a\\\nb) NEWLINE EOF",
                 values(tokens("r\"a\\\r\nb\"")));
    assertEquals("STRING(a\\\n\\\nb) NEWLINE EOF",
                 values(tokens("r\"a\\\r\n\\\nb\"")));
  }

  @Test
  public void testRawString() throws Exception {
    assertEquals("STRING(abcd) NEWLINE EOF",
                 values(tokens("r'abcd'")));
    assertEquals("STRING(abcd) NEWLINE EOF",
                 values(tokens("r\"abcd\"")));
    assertEquals("STRING(a\\tb\\nc\\rd) NEWLINE EOF",
                 values(tokens("r'a\\tb\\nc\\rd'"))); // r'a\tb\nc\rd'
    assertEquals("STRING(a\\\") NEWLINE EOF",
                 values(tokens("r\"a\\\"\""))); // r"a\""
    assertEquals("STRING(a\\\\b) NEWLINE EOF",
                 values(tokens("r'a\\\\b'"))); // r'a\\b'
    assertEquals("STRING(ab) IDENTIFIER(r) NEWLINE EOF",
                 values(tokens("r'ab'r")));

    // Unterminated raw string
    values(tokens("r'\\'")); // r'\'
    assertEquals("/some/path.txt:1: unterminated string literal at eof",
                 lastError.toString());
  }

  @Test
  public void testTripleRawString() throws Exception {
    // r'''a\ncd'''
    assertEquals("STRING(ab\\ncd) NEWLINE EOF",
                 values(tokens("r'''ab\\ncd'''")));
    // r"""ab
    // cd"""
    assertEquals(
        "STRING(ab\ncd) NEWLINE EOF",
        values(tokens("\"\"\"ab\ncd\"\"\"")));

    // Unterminated raw string
    values(tokens("r'''\\'''")); // r'''\'''
    assertEquals("/some/path.txt:1: unterminated string literal at eof",
                 lastError.toString());
  }

  @Test
  public void testOctalEscapes() throws Exception {
    // Regression test for a bug.
    assertEquals("STRING(\0 \1 \t \u003f I I1 \u00ff \u00ff \u00fe) NEWLINE EOF",
                 values(tokens("'\\0 \\1 \\11 \\77 \\111 \\1111 \\377 \\777 \\776'")));
    // Test boundaries (non-octal char, EOF).
    assertEquals("STRING(\1b \1) NEWLINE EOF", values(tokens("'\\1b \\1'")));
  }

  @Test
  public void testTripleQuotedStrings() throws Exception {
    assertEquals("STRING(a\"b'c \n d\"\"e) NEWLINE EOF",
                 values(tokens("\"\"\"a\"b'c \n d\"\"e\"\"\"")));
    assertEquals("STRING(a\"b'c \n d\"\"e) NEWLINE EOF",
                 values(tokens("'''a\"b'c \n d\"\"e'''")));
  }

  @Test
  public void testBadChar() throws Exception {
    assertEquals("IDENTIFIER(a) IDENTIFIER(b) NEWLINE EOF",
                 values(tokens("a$b")));
    assertEquals("/some/path.txt:1: invalid character: '$'",
                 lastError.toString());
  }

  @Test
  public void testIndentation() throws Exception {
    assertEquals("INT(1) NEWLINE INT(2) NEWLINE INT(3) NEWLINE EOF",
                 values(tokens("1\n2\n3")));
    assertEquals("INT(1) NEWLINE INDENT INT(2) NEWLINE INT(3) NEWLINE OUTDENT "
                 + "INT(4) NEWLINE EOF", values(tokens("1\n  2\n  3\n4 ")));
    assertEquals("INT(1) NEWLINE INDENT INT(2) NEWLINE INT(3) NEWLINE OUTDENT "
                 + "NEWLINE EOF", values(tokens("1\n  2\n  3")));
    assertEquals("INT(1) NEWLINE INDENT INT(2) NEWLINE INDENT INT(3) NEWLINE "
                 + "OUTDENT OUTDENT NEWLINE EOF",
                 values(tokens("1\n  2\n    3")));
    assertEquals("INT(1) NEWLINE INDENT INT(2) NEWLINE INDENT INT(3) NEWLINE "
                 + "OUTDENT INT(4) NEWLINE OUTDENT INT(5) NEWLINE EOF",
                 values(tokens("1\n  2\n    3\n  4\n5")));

    assertEquals("INT(1) NEWLINE INDENT INT(2) NEWLINE INDENT INT(3) NEWLINE "
                 + "OUTDENT INT(4) NEWLINE OUTDENT INT(5) NEWLINE EOF",
                 values(tokens("1\n  2\n    3\n   4\n5")));
    assertEquals("/some/path.txt:4: indentation error", lastError.toString());
  }

  @Test
  public void testIndentationWithCrLf() throws Exception {
    assertEquals("INT(1) NEWLINE INDENT INT(2) NEWLINE OUTDENT NEWLINE EOF",
        values(tokens("1\r\n  2\r\n")));
    assertEquals("INT(1) NEWLINE INDENT INT(2) NEWLINE OUTDENT NEWLINE EOF",
        values(tokens("1\r\n  2\r\n\r\n")));
    assertEquals("INT(1) NEWLINE INDENT INT(2) NEWLINE INDENT INT(3) NEWLINE OUTDENT INT(4) "
        + "NEWLINE OUTDENT INT(5) NEWLINE EOF",
        values(tokens("1\r\n  2\r\n    3\r\n  4\r\n5")));
    assertEquals(
        "INT(1) NEWLINE INDENT INT(2) NEWLINE INT(3) NEWLINE OUTDENT INT(4) NEWLINE EOF",
        values(tokens("1\r\n  2\r\n\r\n  3\r\n4")));
  }

  @Test
  public void testIndentationInsideParens() throws Exception {
    // Indentation is ignored inside parens:
    assertEquals("INT(1) LPAREN INT(2) INT(3) INT(4) INT(5) NEWLINE EOF",
                 values(tokens("1 (\n  2\n    3\n  4\n5")));
    assertEquals("INT(1) LBRACE INT(2) INT(3) INT(4) INT(5) NEWLINE EOF",
                 values(tokens("1 {\n  2\n    3\n  4\n5")));
    assertEquals("INT(1) LBRACKET INT(2) INT(3) INT(4) INT(5) NEWLINE EOF",
                 values(tokens("1 [\n  2\n    3\n  4\n5")));
    assertEquals("INT(1) LBRACKET INT(2) RBRACKET NEWLINE INDENT INT(3) "
                 + "NEWLINE INT(4) NEWLINE OUTDENT INT(5) NEWLINE EOF",
                 values(tokens("1 [\n  2]\n    3\n    4\n5")));
  }

  @Test
  public void testIndentationAtEOF() throws Exception {
    // Matching OUTDENTS are created at EOF:
    assertEquals("INDENT INT(1) NEWLINE OUTDENT NEWLINE EOF",
                 values(tokens("\n  1")));
  }

  @Test
  public void testBlankLineIndentation() throws Exception {
    // Blank lines and comment lines should not generate any newlines indents
    // (but note that every input ends with NEWLINE EOF).
    assertEquals("COMMENT NEWLINE EOF", names(tokens("\n      #\n")));
    assertEquals("COMMENT NEWLINE EOF", names(tokens("      #")));
    assertEquals("COMMENT NEWLINE EOF", names(tokens("      #\n")));
    assertEquals("COMMENT NEWLINE EOF", names(tokens("      #comment\n")));
    assertEquals("DEF IDENTIFIER LPAREN IDENTIFIER RPAREN COLON NEWLINE "
                 + "COMMENT INDENT RETURN IDENTIFIER NEWLINE "
                 + "OUTDENT NEWLINE EOF",
                 names(tokens("def f(x):\n"
                              + "  # comment\n"
                              + "\n"
                              + "  \n"
                              + "  return x\n")));
  }

  @Test
  public void testMultipleCommentLines() throws Exception {
    assertEquals("COMMENT NEWLINE COMMENT COMMENT COMMENT "
                 + "DEF IDENTIFIER LPAREN IDENTIFIER RPAREN COLON NEWLINE "
                 + "INDENT RETURN IDENTIFIER NEWLINE OUTDENT NEWLINE EOF",
                 names(tokens("# Copyright\n"
                              + "#\n"
                              + "# A comment line\n"
                              + "# An adjoining line\n"
                              + "def f(x):\n"
                              + "  return x\n")));
  }

  @Test
  public void testBackslash() throws Exception {
    assertEquals("IDENTIFIER IDENTIFIER NEWLINE EOF",
                 names(tokens("a\\\nb")));
    assertEquals("IDENTIFIER IDENTIFIER NEWLINE EOF", names(tokens("a\\\r\nb")));
    assertEquals("IDENTIFIER ILLEGAL IDENTIFIER NEWLINE EOF",
                 names(tokens("a\\ b")));
    assertEquals("IDENTIFIER LPAREN INT RPAREN NEWLINE EOF",
                 names(tokens("a(\\\n2)")));
  }

  @Test
  public void testTokenPositions() throws Exception {
    //            foo   (     bar   ,     {      1       :
    assertEquals("[0,3) [3,4) [4,7) [7,8) [9,10) [10,11) [11,12)"
             //      'quux'  }       )       NEWLINE EOF
                 + " [13,19) [19,20) [20,21) [20,21) [21,21)",
                 positions(tokens("foo(bar, {1: 'quux'})")));
  }

  @Test
  public void testLineNumbers() throws Exception {
    assertEquals("1 1 1 1 2 2 2 2 4 4 4 4 4",
                 linenums("foo = 1\nbar = 2\n\nwiz = 3"));

    assertEquals("IDENTIFIER(foo) EQUALS INT(1) NEWLINE "
                 + "IDENTIFIER(bar) EQUALS INT(2) NEWLINE "
                 + "IDENTIFIER(wiz) EQUALS NEWLINE "
                 + "IDENTIFIER(bar) EQUALS INT(2) NEWLINE EOF",
                 values(tokens("foo = 1\nbar = 2\n\nwiz = $\nbar = 2")));
    assertEquals("/some/path.txt:4: invalid character: '$'",
                 lastError.toString());

    // '\\n' in string should not increment linenum:
    String s = "1\n'foo\\nbar'\3";
    assertEquals("INT(1) NEWLINE STRING(foo\nbar) NEWLINE EOF",
                 values(tokens(s)));
    assertEquals("1 1 2 2 2", linenums(s));
  }

  @Test
  public void testContainsErrors() throws Exception {
    Lexer lexerSuccess = createLexer("foo");
    assertFalse(lexerSuccess.containsErrors());

    Lexer lexerFail = createLexer("f$o");
    assertTrue(lexerFail.containsErrors());

    String s = "'unterminated";
    lexerFail = createLexer(s);
    assertTrue(lexerFail.containsErrors());
    assertEquals(0, lastErrorLocation.getStartOffset());
    assertEquals(s.length(), lastErrorLocation.getEndOffset());
    assertEquals("STRING(unterminated) NEWLINE EOF", values(tokens(s)));
  }

  @Test
  public void testUnterminatedRawStringWithEscapingError() throws Exception {
    assertEquals("STRING NEWLINE EOF", names(tokens("r'\\")));
    assertEquals("/some/path.txt:1: unterminated string literal at eof", lastError);
  }
}
