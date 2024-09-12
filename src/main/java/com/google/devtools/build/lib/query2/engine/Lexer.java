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
package com.google.devtools.build.lib.query2.engine;

import java.util.ArrayList;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * A tokenizer for the Blaze query language, revision 2.
 *
 * No string escapes are allowed ("\").  Given the domain, that's not currently
 * a problem.
 */
public final class Lexer {

  /**
   * Discriminator for different kinds of tokens.
   */
  public enum TokenKind {
    WORD("word"),
    EOF("EOF"),

    COMMA(","),
    EQUALS("="),
    LPAREN("("),
    MINUS("-"),
    PLUS("+"),
    RPAREN(")"),
    CARET("^"),

    __ALL_IDENTIFIERS_FOLLOW(""), // See below

    IN("in"),
    LET("let"),
    SET("set"),

    INTERSECT("intersect"),
    EXCEPT("except"),
    UNION("union");

    private final String prettyName;

    private TokenKind(String prettyName) {
      this.prettyName = prettyName;
    }

    public String getPrettyName() {
      return prettyName;
    }
  }

  public static final Set<TokenKind> BINARY_OPERATORS = EnumSet.of(
      TokenKind.INTERSECT,
      TokenKind.CARET,
      TokenKind.UNION,
      TokenKind.PLUS,
      TokenKind.EXCEPT,
      TokenKind.MINUS);

  private static final Map<String, TokenKind> keywordMap = new HashMap<>();
  static {
    for (TokenKind kind : EnumSet.allOf(TokenKind.class)) {
      if (kind.ordinal() > TokenKind.__ALL_IDENTIFIERS_FOLLOW.ordinal()) {
        keywordMap.put(kind.getPrettyName(), kind);
      }
    }
  }

  /**
   * Returns true iff 'word' is a reserved word of the language.
   */
  static boolean isReservedWord(String word) {
    return keywordMap.containsKey(word);
  }

  /**
   * Tokens returned by the Lexer.
   */
  static class Token {

    public final TokenKind kind;
    public final String word;

    Token(TokenKind kind) {
      this.kind = kind;
      this.word = null;
    }

    Token(String word) {
      this.kind = TokenKind.WORD;
      this.word = word;
    }

    @Override
    public String toString() {
      return kind == TokenKind.WORD ? word : kind.getPrettyName();
    }
  }

  /**
   * Entry point to the lexer. Returns the list of tokens for the specified input, or throws
   * QueryException.
   */
  static List<Token> scan(String input) throws QuerySyntaxException {
    Lexer lexer = new Lexer(input);
    lexer.tokenize();
    return lexer.tokens;
  }

  // Input buffer and position
  private String input;
  private int pos;

  private final List<Token> tokens = new ArrayList<>();

  private Lexer(String input) {
    this.input = input;
    this.pos = 0;
  }

  private void addToken(Token s) {
    tokens.add(s);
  }

  /**
   * Scans a quoted word delimited by 'quot'.
   *
   * <p>ON ENTRY: 'pos' is 1 + the index of the first delimiter ON EXIT: 'pos' is 1 + the index of
   * the last delimiter.
   *
   * @return the word token.
   */
  private Token quotedWord(char quot) throws QuerySyntaxException {
    int oldPos = pos - 1;
    while (pos < input.length()) {
      char c = input.charAt(pos++);
      switch (c) {
        case '\'', '"' -> {
          if (c == quot) {
            // close-quote, all done.
            return new Token(bufferSlice(oldPos + 1, pos - 1));
          }
        }
        default -> {}
      }
    }
    throw new QuerySyntaxException("unclosed quotation");
  }

  private TokenKind getTokenKindForWord(String word) {
    TokenKind kind = keywordMap.get(word);
    return kind == null ? TokenKind.WORD : kind;
  }

  private String scanWord(char firstChar) {
    int oldPos = pos - 1;
    boolean startsWithDoubleAt =
        firstChar == '@' && pos < input.length() && input.charAt(pos) == '@';
    while (pos < input.length()) {
      switch (input.charAt(pos)) {
        case 'a',
                'b',
                'c',
                'd',
                'e',
                'f',
                'g',
                'h',
                'i',
                'j',
                'k',
                'l',
                'm',
                'n',
                'o',
                'p',
                'q',
                'r',
                's',
                't',
                'u',
                'v',
                'w',
                'x',
                'y',
                'z',
                'A',
                'B',
                'C',
                'D',
                'E',
                'F',
                'G',
                'H',
                'I',
                'J',
                'K',
                'L',
                'M',
                'N',
                'O',
                'P',
                'Q',
                'R',
                'S',
                'T',
                'U',
                'V',
                'W',
                'X',
                'Y',
                'Z',
                '0',
                '1',
                '2',
                '3',
                '4',
                '5',
                '6',
                '7',
                '8',
                '9',
                '*',
                '/',
                '@',
                '.',
                '-',
                '_',
                ':',
                '$',
                '~',
                '[',
                ']' ->
            pos++;
        case '+' -> {
          if (startsWithDoubleAt) {
            // Allow unquoted canonical labels such as
            // @@rules_jvm_external++maven+maven//:bar, but still parse @foo+@bar as two separate
            // labels (here @foo refers to the @foo//:foo target).
            // If @@foo+bar is intended to mean @@foo + bar, it can be written as such with spaces.
            pos++;
          } else {
            return bufferSlice(oldPos, pos);
          }
        }
        default -> {
          return bufferSlice(oldPos, pos);
        }
      }
    }
    return bufferSlice(oldPos, pos);
  }

  /**
   * Scans a word or keyword.
   *
   * <p>ON ENTRY: 'pos' is 1 + the index of the first char in the word. ON EXIT: 'pos' is 1 + the
   * index of the last char in the word.
   *
   * @return the word or keyword token.
   */
  private Token wordOrKeyword(char firstChar) {
    String word = scanWord(firstChar);
    TokenKind kind = getTokenKindForWord(word);
    return kind == TokenKind.WORD ? new Token(word) : new Token(kind);
  }

  /** Performs tokenization of the character buffer of file contents provided to the constructor. */
  private void tokenize() throws QuerySyntaxException {
    while (pos < input.length()) {
      char c = input.charAt(pos);
      pos++;
      switch (c) {
        case '(' -> addToken(new Token(TokenKind.LPAREN));
        case ')' -> addToken(new Token(TokenKind.RPAREN));
        case ',' -> addToken(new Token(TokenKind.COMMA));
        case '+' -> addToken(new Token(TokenKind.PLUS));
        case '-' -> addToken(new Token(TokenKind.MINUS));
        case '=' -> addToken(new Token(TokenKind.EQUALS));
        case '^' -> addToken(new Token(TokenKind.CARET));
        case '\n', ' ', '\t', '\r' -> {
          /* ignore */
        }
        case '\'', '\"' -> addToken(quotedWord(c));
        default -> addToken(wordOrKeyword(c));
      }
    }

    addToken(new Token(TokenKind.EOF));

    this.input = null; // release buffer now that we have our tokens
  }

  private String bufferSlice(int start, int end) {
    return this.input.substring(start, end);
  }

}
