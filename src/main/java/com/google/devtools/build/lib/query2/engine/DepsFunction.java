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

import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.Argument;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.ArgumentType;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryFunction;

import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * A "deps" query expression, which computes the dependencies of the argument. An optional
 * integer-literal second argument may be specified; its value bounds the search from the arguments.
 *
 * <pre>expr ::= DEPS '(' expr ')'</pre>
 * <pre>       | DEPS '(' expr ',' WORD ')'</pre>
 */
final class DepsFunction implements QueryFunction {
  DepsFunction() {
  }

  @Override
  public String getName() {
    return "deps";
  }

  @Override
  public int getMandatoryArguments() {
    return 1;  // last argument is optional
  }

  @Override
  public List<ArgumentType> getArgumentTypes() {
    return ImmutableList.of(ArgumentType.EXPRESSION, ArgumentType.INTEGER);
  }

  /**
   * Breadth-first search from the arguments.
   */
  @Override
  public <T> Set<T> eval(QueryEnvironment<T> env, QueryExpression expression, List<Argument> args)
      throws QueryException, InterruptedException {
    Set<T> argumentValue = args.get(0).getExpression().eval(env);
    int depthBound = args.size() > 1 ? args.get(1).getInteger() : Integer.MAX_VALUE;
    env.buildTransitiveClosure(expression, argumentValue, depthBound);

    Set<T> visited = new LinkedHashSet<>();
    Collection<T> current = argumentValue;

    // We need to iterate depthBound + 1 times.
    for (int i = 0; i <= depthBound; i++) {
      List<T> next = new ArrayList<>();
      // Filter already visited nodes: if we see a node in a later round, then we don't need to
      // visit it again, because the depth at which we see it at must be greater than or equal to
      // the last visit.
      next.addAll(env.getFwdDeps(Iterables.filter(current,
          Predicates.not(Predicates.in(visited)))));
      visited.addAll(current);
      if (next.isEmpty()) {
        // Exit when there are no more nodes to visit.
        break;
      }
      current = next;
    }

    return visited;
  }
}
