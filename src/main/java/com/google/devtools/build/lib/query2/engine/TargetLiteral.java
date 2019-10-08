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

import com.google.common.base.CharMatcher;
import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryTaskFuture;
import java.util.Collection;
import java.util.Set;

/**
 * A literal set of targets, using 'blaze build' syntax. Or, a reference to a variable name. (The
 * syntax of the string "pattern" determines which.)
 *
 * <p>TODO(bazel-team): Perhaps we should distinguish NAME from WORD in the parser, based on the
 * characters in it? Also, perhaps we should not allow NAMEs to be quoted like WORDs can be.
 *
 * <pre>expr ::= NAME | WORD</pre>
 */
public final class TargetLiteral extends QueryExpression {

  private final String pattern;

  public TargetLiteral(String pattern) {
    this.pattern = Preconditions.checkNotNull(pattern);
  }

  public String getPattern() {
    return pattern;
  }

  public boolean isVariableReference() {
    return LetExpression.isValidVarReference(pattern);
  }

  private <T> QueryTaskFuture<Void> evalVarReference(
      QueryEnvironment<T> env, QueryExpressionContext<T> context, Callback<T> callback) {
    String varName = LetExpression.getNameFromReference(pattern);
    Set<T> value = context.get(varName);
    if (value == null) {
      return env.immediateFailedFuture(
          new QueryException(this, "undefined variable '" + varName + "'"));
    }
    try {
      callback.process(value);
      return env.immediateSuccessfulFuture(null);
    } catch (QueryException e) {
      return env.immediateFailedFuture(e);
    } catch (InterruptedException e) {
      return env.immediateCancelledFuture();
    }
  }

  @Override
  public <T> QueryTaskFuture<Void> eval(
      QueryEnvironment<T> env, QueryExpressionContext<T> context, Callback<T> callback) {
    if (isVariableReference()) {
      return evalVarReference(env, context, callback);
    } else {
      return env.getTargetsMatchingPattern(this, pattern, callback);
    }
  }

  @Override
  public void collectTargetPatterns(Collection<String> literals) {
    if (!isVariableReference()) {
      literals.add(pattern);
    }
  }

  @Override
  public <T, C> T accept(QueryExpressionVisitor<T, C> visitor, C context) {
    return visitor.visit(this, context);
  }

  @Override
  public String toString() {
    // The character matching has to be in sync with LabelValidator#PUNCTUATION_REQUIRING_QUOTING
    // except for the special characters that Lexer#scanWord *\/@.-_:$~ consider to be a word.
    boolean needsQuoting =
        Lexer.isReservedWord(pattern)
            || pattern.isEmpty()
            || pattern.startsWith("-")
            || pattern.startsWith("*")
            || CharMatcher.anyOf(" \"#&'()+,;<=>?[]{|}").matchesAnyOf(pattern);

    if (!needsQuoting) {
      return pattern;
    }

    /**
     * If the word requires quoting, we want to quote the word such that the quoting character does
     * not lex the result differently if the result toString is fed back into the parser. For
     * example: If the following Java string that requires quoting is set("foo"), and we quote the
     * Java string with double quotes, "set("foo")" and feed it back into the lexer, the lexer will
     * parse the following word set(. So in this case we want to quote it with single quotes such
     * that it would look like 'set("foo")'. In the case that we find both single quote and double
     * quote, we would fail and use either of the two quotes.
     */
    char quote = pattern.contains("\"") ? '\'' : '"';
    return quote + pattern + quote;
  }
}
