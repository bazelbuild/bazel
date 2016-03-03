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

import com.google.devtools.build.lib.util.Preconditions;

import java.util.Collection;
import java.util.Set;

/**
 * A literal set of targets, using 'blaze build' syntax.  Or, a reference to a
 * variable name.  (The syntax of the string "pattern" determines which.)
 *
 * TODO(bazel-team): Perhaps we should distinguish NAME from WORD in the parser,
 * based on the characters in it?  Also, perhaps we should not allow NAMEs to
 * be quoted like WORDs can be.
 *
 * <pre>expr ::= NAME | WORD</pre>
 */
public final class TargetLiteral extends QueryExpression {

  private final String pattern;

  TargetLiteral(String pattern) {
    this.pattern = Preconditions.checkNotNull(pattern);
  }

  public String getPattern() {
    return pattern;
  }

  public boolean isVariableReference() {
    return LetExpression.isValidVarReference(pattern);
  }

  @Override
  public <T> void eval(QueryEnvironment<T> env, Callback<T> callback)
      throws QueryException, InterruptedException {
    if (isVariableReference()) {
      String varName = LetExpression.getNameFromReference(pattern);
      Set<T> value = env.getVariable(varName);
      if (value == null) {
        throw new QueryException(this, "undefined variable '" + varName + "'");
      }
      callback.process(value);
    } else {
      env.getTargetsMatchingPattern(this, pattern, callback);
    }
  }

  @Override
  public void collectTargetPatterns(Collection<String> literals) {
    if (!isVariableReference()) {
      literals.add(pattern);
    }
  }

  @Override
  public QueryExpression getMapped(QueryExpressionMapper mapper) {
    return mapper.map(this);
  }

  @Override
  public String toString() {
    // Keep predicate consistent with Lexer.scanWord!
    boolean needsQuoting = Lexer.isReservedWord(pattern)
                        || pattern.isEmpty()
                        || "$-*".indexOf(pattern.charAt(0)) != -1;
    return needsQuoting ? ("\"" + pattern + "\"") : pattern;
  }
}
