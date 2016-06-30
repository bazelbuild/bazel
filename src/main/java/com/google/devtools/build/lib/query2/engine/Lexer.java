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
 * Note, we can avoid a lot of quoting by noting that the characters [() ,] do
 * not appear in any label, filename, function name, or regular expression we care about.
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
   * Entry point to the lexer.  Returns the list of tokens for the specified
   * input, or throws QueryException.
   */
  public static List<Token> scan(char[] buffer) throws QueryException {
    Lexer lexer = new Lexer(buffer);
    lexer.tokenize();
    return lexer.tokens;
  }

  // Input buffer and position
  private char[] buffer;
  private int pos;

  private final List<Token> tokens = new ArrayList<>();

  private Lexer(char[] buffer) {
    this.buffer = buffer;
    this.pos = 0;
  }

  private void addToken(Token s) {
    tokens.add(s);
  }

  /**
   * Scans a quoted word delimited by 'quot'.
   *
   * ON ENTRY: 'pos' is 1 + the index of the first delimiter
   * ON EXIT: 'pos' is 1 + the index of the last delimiter.
   *
   * @return the word token.
   */
  private Token quotedWord(char quot) throws QueryException {
    int oldPos = pos - 1;
    while (pos < buffer.length) {
      char c = buffer[pos++];
      switch (c) {
        case '\'':
        case '"':
          if (c == quot) {
            // close-quote, all done.
            return new Token(bufferSlice(oldPos + 1, pos - 1));
          }
      }
    }
    throw new QueryException("unclosed quotation");
  }

  private TokenKind getTokenKindForWord(String word) {
    TokenKind kind = keywordMap.get(word);
    return kind == null ? TokenKind.WORD : kind;
  }

  // Unquoted words may contain [-*$], but not start with them.  For user convenience, unquoted
  // words must include UNIX filenames, labels and target label patterns, and simple regexps
  // (e.g. cc_.*). Keep consistent with TargetLiteral.toString()!
  private String scanWord() {
    int oldPos = pos - 1;
    while (pos < buffer.length) {
      switch (buffer[pos]) {
        case 'a': case 'b': case 'c': case 'd': case 'e': case 'f':
        case 'g': case 'h': case 'i': case 'j': case 'k': case 'l':
        case 'm': case 'n': case 'o': case 'p': case 'q': case 'r':
        case 's': case 't': case 'u': case 'v': case 'w': case 'x':
        case 'y': case 'z':
        case 'A': case 'B': case 'C': case 'D': case 'E': case 'F':
        case 'G': case 'H': case 'I': case 'J': case 'K': case 'L':
        case 'M': case 'N': case 'O': case 'P': case 'Q': case 'R':
        case 'S': case 'T': case 'U': case 'V': case 'W': case 'X':
        case 'Y': case 'Z':
        case '0': case '1': case '2': case '3': case '4': case '5':
        case '6': case '7': case '8': case '9':
        case '*': case '/': case '@': case '.': case '-': case '_':
        case ':': case '$':
          pos++;
          break;
       default:
          return bufferSlice(oldPos, pos);
      }
    }
    return bufferSlice(oldPos, pos);
  }

  /**
   * Scans a word or keyword.
   *
   * ON ENTRY: 'pos' is 1 + the index of the first char in the word.
   * ON EXIT: 'pos' is 1 + the index of the last char in the word.
   *
   * @return the word or keyword token.
   */
  private Token wordOrKeyword() {
    String word = scanWord();
    TokenKind kind = getTokenKindForWord(word);
    return kind == TokenKind.WORD ? new Token(word) : new Token(kind);
  }

  /**
   * Performs tokenization of the character buffer of file contents provided to
   * the constructor.
   */
  private void tokenize() throws QueryException {
    while (pos < buffer.length) {
      char c = buffer[pos];
      pos++;
      switch (c) {
      case '(': {
        addToken(new Token(TokenKind.LPAREN));
        break;
      }
      case ')': {
        addToken(new Token(TokenKind.RPAREN));
        break;
      }
      case ',': {
        addToken(new Token(TokenKind.COMMA));
        break;
      }
      case '+': {
        addToken(new Token(TokenKind.PLUS));
        break;
      }
      case '-': {
        addToken(new Token(TokenKind.MINUS));
        break;
      }
      case '=': {
        addToken(new Token(TokenKind.EQUALS));
        break;
      }
      case '^': {
        addToken(new Token(TokenKind.CARET));
        break;
      }
      case '\n':
      case ' ':
      case '\t':
      case '\r': {
        /* ignore */
        break;
      }
      case '\'':
      case '\"': {
        addToken(quotedWord(c));
        break;
      }
      default: {
        addToken(wordOrKeyword());
        break;
      } // default
      } // switch
    } // while

    addToken(new Token(TokenKind.EOF));

    this.buffer = null; // release buffer now that we have our tokens
  }

  private String bufferSlice(int start, int end) {
    return new String(this.buffer, start, end - start);
  }

}
