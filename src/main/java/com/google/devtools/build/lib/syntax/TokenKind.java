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

package com.google.devtools.build.lib.syntax;

/**
 * A TokenKind is an enumeration of each different kind of lexical symbol.
 */
public enum TokenKind {

  AND("and"),
  AS("as"),
  ASSERT("assert"),
  BREAK("break"),
  CLASS("class"),
  COLON(":"),
  COMMA(","),
  COMMENT("comment"),
  CONTINUE("continue"),
  DEF("def"),
  DEL("del"),
  DOT("."),
  ELIF("elif"),
  ELSE("else"),
  EOF("EOF"),
  EQUALS("="),
  EQUALS_EQUALS("=="),
  EXCEPT("except"),
  FINALLY("finally"),
  FOR("for"),
  FROM("from"),
  GLOBAL("global"),
  GREATER(">"),
  GREATER_EQUALS(">="),
  IDENTIFIER("identifier"),
  IF("if"),
  ILLEGAL("illegal character"),
  IMPORT("import"),
  IN("in"),
  INDENT("indent"),
  INT("integer"),
  IS("is"),
  LAMBDA("lambda"),
  LBRACE("{"),
  LBRACKET("["),
  LESS("<"),
  LESS_EQUALS("<="),
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
  PLUS("+"),
  PLUS_EQUALS("+="),
  RAISE("raise"),
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
  STRING("string"),
  TRY("try"),
  WHILE("while"),
  WITH("with"),
  YIELD("yield");

  private final String prettyName;

  private TokenKind(String prettyName) {
    this.prettyName = prettyName;
  }

  /**
   * Returns the pretty name for this token, for use in error messages for the user.
   */
  public String getPrettyName() {
    return prettyName;
  }
}
