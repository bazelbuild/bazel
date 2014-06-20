// Copyright 2014 Google Inc. All rights reserved.
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
 * A Token represents an actual lexeme; that is, a lexical unit, its location in
 * the input text, its lexical kind (TokenKind) and any associated value.
 */
public class Token {

  public final TokenKind kind;
  public final int left;
  public final int right;
  public final Object value;

  public Token(TokenKind kind, int left, int right) {
    this(kind, left, right, null);
  }

  public Token(TokenKind kind, int left, int right, Object value) {
    this.kind = kind;
    this.left = left;
    this.right = right;
    this.value = value;
  }

  /**
   * Constructs an easy-to-read string representation of token, suitable for use
   * in user error messages.
   */
  @Override
  public String toString() {
    // TODO(bazel-team): do proper escaping of string literals
    return kind == TokenKind.STRING     ? ("\"" + value + "\"")
        : value == null                 ? kind.getPrettyName()
        : value.toString();
  }

}
