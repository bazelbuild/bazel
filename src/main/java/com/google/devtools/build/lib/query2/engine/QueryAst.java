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
        QueryAst.BinaryOperator,
        QueryAst.FunctionCall,
        QueryAst.Let,
        QueryAst.Set,
        QueryAst.Parenthesized {

  /** A single word token (a target pattern, variable reference, or word/integer function arg). */
  record Word(String value) implements QueryAst {
    public Word {
      checkNotNull(value);
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
