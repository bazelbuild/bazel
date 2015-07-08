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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.Argument;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.ArgumentType;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryFunction;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Implementation of the <code>allpaths()</code> function.
 */
public class AllPathsFunction implements QueryFunction {
  AllPathsFunction() {
  }

  @Override
  public String getName() {
    return "allpaths";
  }

  @Override
  public int getMandatoryArguments() {
    return 2;
  }

  @Override
  public List<ArgumentType> getArgumentTypes() {
    return ImmutableList.of(ArgumentType.EXPRESSION, ArgumentType.EXPRESSION);
  }

  @Override
  public <T> Set<T> eval(QueryEnvironment<T> env, QueryExpression expression, List<Argument> args)
      throws QueryException, InterruptedException {
    QueryExpression from = args.get(0).getExpression();
    QueryExpression to = args.get(1).getExpression();

    Set<T> fromValue = from.eval(env);
    Set<T> toValue = to.eval(env);

    // Algorithm: compute "reachableFromX", the forward transitive closure of
    // the "from" set, then find the intersection of "reachableFromX" with the
    // reverse transitive closure of the "to" set.  The reverse transitive
    // closure and intersection operations are interleaved for efficiency.
    // "result" holds the intersection.

    env.buildTransitiveClosure(expression, fromValue, Integer.MAX_VALUE);

    Set<T> reachableFromX = env.getTransitiveClosure(fromValue);
    Set<T> result = intersection(reachableFromX, toValue);
    Collection<T> worklist = result;
    while (!worklist.isEmpty()) {
      Collection<T> reverseDeps = env.getReverseDeps(worklist);
      worklist = new ArrayList<>();
      for (T np : reverseDeps) {
        if (reachableFromX.contains(np)) {
          if (result.add(np)) {
            worklist.add(np);
          }
        }
      }
    }
    return result;
  }

  /**
   * Returns a (new, mutable, unordered) set containing the intersection of the
   * two specified sets.
   */
  private static <T> Set<T> intersection(Set<T> x, Set<T> y) {
    Set<T> result = new HashSet<>();
    if (x.size() > y.size()) {
      Sets.intersection(y, x).copyInto(result);
    } else {
      Sets.intersection(x, y).copyInto(result);
    }
    return result;
  }
}
