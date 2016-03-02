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

import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.Argument;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.ArgumentType;

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
public final class RdepsFunction extends AllRdepsFunction {
  public RdepsFunction() {}

  @Override
  public String getName() {
    return "rdeps";
  }

  @Override
  public int getMandatoryArguments() {
    return super.getMandatoryArguments() + 1;  // +1 for the universe.
  }

  @Override
  public List<ArgumentType> getArgumentTypes() {
    return ImmutableList.<ArgumentType>builder()
        .add(ArgumentType.EXPRESSION).addAll(super.getArgumentTypes()).build();
  }

  /**
   * Compute the transitive closure of the universe, then breadth-first search from the argument
   * towards the universe while staying within the transitive closure.
   */
  @Override
  public <T> void eval(QueryEnvironment<T> env, QueryExpression expression,
      List<Argument> args, Callback<T> callback)
      throws QueryException, InterruptedException {
    Set<T> universeValue = QueryUtil.evalAll(env, args.get(0).getExpression());
    env.buildTransitiveClosure(expression, universeValue, Integer.MAX_VALUE);

    Predicate<T> universe = Predicates.in(env.getTransitiveClosure(universeValue));
    eval(env, args.subList(1, args.size()), callback, universe);
  }
}
