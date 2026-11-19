// Copyright 2026 The Bazel Authors. All rights reserved.
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

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.devtools.build.lib.query2.engine.Lexer.BINARY_OPERATORS;

import com.google.common.collect.ImmutableList;
import java.util.List;
import java.util.Objects;

/**
 * Unresolved abstract syntax tree for an expression in the Blaze query language, revision 2.
 *
 * <p>Unlike {@link QueryExpression}, a {@link QueryAst} represents the syntactic structure of a
 * query before function names and argument types are resolved against a set of {@code
 * QueryFunction} definitions.
 */
public sealed interface QueryAst
    permits QueryAst.Word,
        QueryAst.Placeholder,
        QueryAst.BinaryOperator,
        QueryAst.FunctionCall,
        QueryAst.Let,
        QueryAst.Set,
        QueryAst.Parenthesized {

  /**
   * Returns true if {@code name} is a valid C-style identifier ({@code [a-zA-Z_][a-zA-Z0-9_]*}), as
   * required for {@code let} variable names and {@code @(placeholder)} names.
   */
  static boolean isValidIdentifier(String name) {
    if (name.isEmpty()) {
      return false;
    }
    char first = name.charAt(0);
    if (!isAsciiAlpha(first) && first != '_') {
      return false;
    }
    for (int i = 1; i < name.length(); i++) {
      char c = name.charAt(i);
      if (!isAsciiAlpha(c) && !isAsciiDigit(c) && c != '_') {
        return false;
      }
    }
    return true;
  }

  private static boolean isAsciiAlpha(char c) {
    return ('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z');
  }

  private static boolean isAsciiDigit(char c) {
    return '0' <= c && c <= '9';
  }

  /** A single word token (a target pattern, variable reference, or word/integer function arg). */
  record Word(String value) implements QueryAst {
    public Word {
      checkNotNull(value);
    }

    /**
     * Returns true if this word has the syntax of a {@code let}-variable reference ({@code
     * $<ident>}).
     */
    public boolean isVariableReference() {
      return value.startsWith("$") && isValidIdentifier(value.substring(1));
    }

    /**
     * Returns the referenced variable name without the leading {@code $}.
     *
     * @throws IllegalArgumentException if {@link #isVariableReference()} is false
     */
    public String variableName() {
      checkArgument(isVariableReference(), "Not a variable reference: %s", value);
      return value.substring(1);
    }
  }

  /**
   * An unquoted {@code @(name)} parameter placeholder, recording its {@code [startOffset,
   * endOffset)} character span (including the surrounding {@code @(} and {@code )}) in the original
   * query string.
   */
  record Placeholder(String name, int startOffset, int endOffset) implements QueryAst {
    public Placeholder {
      checkArgument(isValidIdentifier(name), "Invalid placeholder name: %s", name);
      checkArgument(startOffset >= 0, "Invalid startOffset: %s", startOffset);
      checkArgument(
          endOffset > startOffset,
          "Expected endOffset (%s) > startOffset (%s)",
          endOffset,
          startOffset);
    }
  }

  /** A binary algebraic set operation ({@code intersect}, {@code union}, {@code except}, etc.). */
  record BinaryOperator(Lexer.TokenKind operator, ImmutableList<QueryAst> operands)
      implements QueryAst {
    public BinaryOperator {
      checkNotNull(operator);
      checkNotNull(operands);
      checkArgument(BINARY_OPERATORS.contains(operator), "Not a binary operator: %s", operator);
      checkArgument(operands.size() > 1, "Expected at least 2 operands, got %s", operands.size());
    }

    public BinaryOperator(Lexer.TokenKind operator, List<QueryAst> operands) {
      this(operator, ImmutableList.copyOf(operands));
    }
  }

  /**
   * An argument passed to a {@link FunctionCall}, paired with token context for diagnostics.
   *
   * <p>{@link #equals}, {@link #hashCode}, and {@link #toString} exclude the {@code
   * *ForErrorMessages} fields.
   */
  record FunctionArgument(
      QueryAst expr, String separatorContextForErrorMessages, String argContextForErrorMessages) {
    public FunctionArgument {
      checkNotNull(expr);
      checkNotNull(separatorContextForErrorMessages);
      checkNotNull(argContextForErrorMessages);
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      return o instanceof FunctionArgument that && expr.equals(that.expr);
    }

    @Override
    public int hashCode() {
      return expr.hashCode();
    }

    @Override
    public String toString() {
      return "FunctionArgument[expr=" + expr + "]";
    }
  }

  /**
   * An unresolved function call expression {@code name(arg1, arg2, ...)}.
   *
   * <p>{@link #equals}, {@link #hashCode}, and {@link #toString} exclude the {@code
   * *ForErrorMessages} fields.
   */
  record FunctionCall(
      String functionName,
      ImmutableList<FunctionArgument> arguments,
      String functionContextForErrorMessages,
      String closeParenContextForErrorMessages)
      implements QueryAst {
    public FunctionCall {
      checkNotNull(functionName);
      checkNotNull(arguments);
      checkNotNull(functionContextForErrorMessages);
      checkNotNull(closeParenContextForErrorMessages);
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      return o instanceof FunctionCall that
          && functionName.equals(that.functionName)
          && arguments.equals(that.arguments);
    }

    @Override
    public int hashCode() {
      return Objects.hash(functionName, arguments);
    }

    @Override
    public String toString() {
      return "FunctionCall[functionName=" + functionName + ", arguments=" + arguments + "]";
    }
  }

  /** A {@code let name = varExpr in bodyExpr} binding expression. */
  record Let(String varName, QueryAst varExpr, QueryAst bodyExpr) implements QueryAst {
    public Let {
      checkNotNull(varName);
      checkNotNull(varExpr);
      checkNotNull(bodyExpr);
    }
  }

  /** A {@code set(word1 word2 ...)} expression. */
  record Set(ImmutableList<Word> words) implements QueryAst {
    public Set {
      checkNotNull(words);
    }

    public Set(List<Word> words) {
      this(ImmutableList.copyOf(words));
    }
  }

  /** An explicitly parenthesized expression {@code (expr)}. */
  record Parenthesized(QueryAst inner) implements QueryAst {
    public Parenthesized {
      checkNotNull(inner);
    }
  }
}
