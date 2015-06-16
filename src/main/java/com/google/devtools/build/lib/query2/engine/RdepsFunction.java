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
 * An "rdeps" query expression, which computes the reverse dependencies of the argument within the
 * transitive closure of the universe. An optional integer-literal third argument may be
 * specified; its value bounds the search from the arguments.
 *
 * <pre>expr ::= RDEPS '(' expr ',' expr ')'</pre>
 * <pre>       | RDEPS '(' expr ',' expr ',' WORD ')'</pre>
 */
final class RdepsFunction implements QueryFunction {
  RdepsFunction() {
  }

  @Override
  public String getName() {
    return "rdeps";
  }

  @Override
  public int getMandatoryArguments() {
    return 2;  // last argument is optional
  }

  @Override
  public List<ArgumentType> getArgumentTypes() {
    return ImmutableList.of(
        ArgumentType.EXPRESSION, ArgumentType.EXPRESSION, ArgumentType.INTEGER);
  }

  /**
   * Compute the transitive closure of the universe, then breadth-first search from the argument
   * towards the universe while staying within the transitive closure.
   */
  @Override
  public <T> Set<T> eval(QueryEnvironment<T> env, QueryExpression expression, List<Argument> args)
      throws QueryException {
    Set<T> universeValue = args.get(0).getExpression().eval(env);
    Set<T> argumentValue = args.get(1).getExpression().eval(env);
    int depthBound = args.size() > 2 ? args.get(2).getInteger() : Integer.MAX_VALUE;

    env.buildTransitiveClosure(expression, universeValue, Integer.MAX_VALUE);

    Set<T> visited = new LinkedHashSet<>();
    Set<T> reachableFromUniverse = env.getTransitiveClosure(universeValue);
    Collection<T> current = argumentValue;

    // We need to iterate depthBound + 1 times.
    for (int i = 0; i <= depthBound; i++) {
      List<T> next = new ArrayList<>();
      // Restrict to nodes in our universe.
      Iterable<T> currentInUniverse = Iterables.filter(current,
          Predicates.in(reachableFromUniverse));
      // Filter already visited nodes: if we see a node in a later round, then we don't need to
      // visit it again, because the depth at which we see it at must be greater than or equal to
      // the last visit.
      next.addAll(env.getReverseDeps(Iterables.filter(currentInUniverse,
          Predicates.not(Predicates.in(visited)))));
      Iterables.addAll(visited, currentInUniverse);
      if (next.isEmpty()) {
        // Exit when there are no more nodes to visit.
        break;
      }
      current = next;
    }

    return visited;
  }
}
