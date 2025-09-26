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

/** A TokenKind represents the kind of a lexical token. */
public enum TokenKind {
  AMPERSAND("&"),
  AMPERSAND_EQUALS("&="),
  AND("and"),
  AS("as"),
  ASSERT("assert"),
  BREAK("break"),
  CARET("^"),
  CARET_EQUALS("^="),
  CLASS("class"),
  COLON(":"),
  COMMA(","),
  CONTINUE("continue"),
  DEF("def"),
  DEL("del"),
  /**
   * A multiline block of Sphinx autodoc-style doc comments. Implicitly includes terminating
   * newline.
   */
  DOC_COMMENT_BLOCK("#:"),
  /**
   * Inline trailing doc comment which was preceded by non-whitespace tokens on the same line.
   * Doesn't include terminating newline.
   */
  DOC_COMMENT_TRAILING("trailing #: "),
  DOT("."),
  ELIF("elif"),
  ELSE("else"),
  EOF("EOF"),
  EQUALS("="),
  EQUALS_EQUALS("=="),
  EXCEPT("except"),
  FINALLY("finally"),
  FLOAT("float literal"),
  FOR("for"),
  FROM("from"),
  GLOBAL("global"),
  GREATER(">"),
  GREATER_EQUALS(">="),
  GREATER_GREATER(">>"),
  GREATER_GREATER_EQUALS(">>="),
  IDENTIFIER("identifier"),
  IF("if"),
  ILLEGAL("illegal character"),
  IMPORT("import"),
  IN("in"),
  INDENT("indent"),
  INT("integer literal"),
  IS("is"),
  LAMBDA("lambda"),
  LBRACE("{"),
  LBRACKET("["),
  LESS("<"),
  LESS_EQUALS("<="),
  LESS_LESS("<<"),
  LESS_LESS_EQUALS("<<="),
  LOAD("load"),
  LPAREN("("),
  MINUS("-"),
  MINUS_EQUALS("-="),
  NEWLINE("newline"),
  NONLOCAL("nonlocal"),
  NOT("not"),
  NOT_EQUALS("!="),
  NOT_IN("not in"),
  OR("or"),
  OUTDENT("outdent"),
  PASS("pass"),
  PERCENT("%"),
  PERCENT_EQUALS("%="),
  PIPE("|"),
  PIPE_EQUALS("|="),
  PLUS("+"),
  PLUS_EQUALS("+="),
  RAISE("raise"),
  RARROW("->"),
  RBRACE("}"),
  RBRACKET("]"),
  RETURN("return"),
  RPAREN(")"),
  SEMI(";"),
  SLASH("/"),
  SLASH_EQUALS("/="),
  SLASH_SLASH("//"),
  SLASH_SLASH_EQUALS("//="),
  STAR("*"),
  STAR_EQUALS("*="),
  STAR_STAR("**"),
  STRING("string literal"),
  TILDE("~"),
  TRY("try"),
  WHILE("while"),
  WITH("with"),
  YIELD("yield");

  private final String name;

  private TokenKind(String name) {
    this.name = name;
  }

  @Override
  public String toString() {
    return name;
  }
}
