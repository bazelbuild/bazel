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
package com.google.devtools.build.lib.query2.engine;

import com.google.common.base.Joiner;

import java.util.Collection;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * A set(word, ..., word) expression, which computes the union of zero or more
 * target patterns separated by whitespace.  This is intended to support the
 * use-case in which a set of labels written to a file by a previous query
 * expression can be modified externally, then used as input to another query,
 * like so:
 *
 * <pre>
 * % blaze query 'somepath(foo, bar)' | grep ... | sed ... | awk ... >file
 * % blaze query "kind(qux_library, set($(<file)))"
 * </pre>
 *
 * <p>The grammar currently restricts the operands of set() to being zero or
 * more words (target patterns), with no intervening punctuation.  In principle
 * this could be extended to arbitrary expressions without grammatical
 * ambiguity, but this seems excessively general for now.
 *
 * <pre>expr ::= SET '(' WORD * ')'</pre>
 */
class SetExpression extends QueryExpression {

  private final List<TargetLiteral> words;

  SetExpression(List<TargetLiteral> words) {
    this.words = words;
  }

  @Override
  public <T> Set<T> eval(QueryEnvironment<T> env) throws QueryException {
    Set<T> result = new LinkedHashSet<>();
    for (TargetLiteral expr : words) {
      result.addAll(expr.eval(env));
    }
    return result;
  }

  @Override
  public void collectTargetPatterns(Collection<String> literals) {
    for (TargetLiteral expr : words) {
      expr.collectTargetPatterns(literals);
    }
  }

  @Override
  public String toString() {
    return "set(" + Joiner.on(' ').join(words) + ")";
  }
}
