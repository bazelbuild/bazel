// Copyright 2015 The Bazel Authors. All rights reserved.
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
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.Argument;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.ArgumentType;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryFunction;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryTaskFuture;
import java.util.ArrayList;
import java.util.List;
import java.util.OptionalInt;

/**
 * An "allrdeps" query expression, which computes the reverse dependencies of the argument within
 * the currently known universe. An optional integer-literal second argument may be specified; its
 * value bounds the search from the arguments.
 *
 * <pre>expr ::= ALLRDEPS '(' expr ')'</pre>
 * <pre>       | ALLRDEPS '(' expr ',' WORD ')'</pre>
 */
// Public because SkyQueryEnvironment needs to refer to it directly.
public class AllRdepsFunction implements QueryFunction {

  @Override
  public String getName() {
    return "allrdeps";
  }

  @Override
  public int getMandatoryArguments() {
    return 1; // last argument is optional
  }

  @Override
  public List<ArgumentType> getArgumentTypes() {
    return ImmutableList.of(ArgumentType.EXPRESSION, ArgumentType.INTEGER);
  }

  @Override
  public <T> QueryTaskFuture<Void> eval(
      QueryEnvironment<T> env,
      QueryExpressionContext<T> context,
      QueryExpression expression,
      List<Argument> args,
      Callback<T> callback) {
    OptionalInt maxDepth =
        args.size() == 1 ? OptionalInt.empty() : OptionalInt.of(args.get(1).getInteger());
    QueryExpression argumentExpression = args.get(0).getExpression();
    if (env instanceof StreamableQueryEnvironment<T> streamableEnv) {
      return maxDepth.isPresent()
          ? streamableEnv.getAllRdepsBoundedParallel(
              argumentExpression, maxDepth.getAsInt(), context, callback)
          : streamableEnv.getAllRdepsUnboundedParallel(argumentExpression, context, callback);
    } else {
      return eval(env, argumentExpression, Predicates.alwaysTrue(), context, callback, maxDepth);
    }
  }

  /** Common non-parallel implementation of depth-bounded allrdeps/deps. */
  static <T> QueryTaskFuture<Void> eval(
      QueryEnvironment<T> env,
      QueryExpression expression,
      Predicate<T> universe,
      QueryExpressionContext<T> context,
      Callback<T> callback,
      OptionalInt depth) {
    MinDepthUniquifier<T> minDepthUniquifier = env.createMinDepthUniquifier();
    return env.eval(
        expression,
        context,
        partialResult -> {
          Iterable<T> current = partialResult;
          int i = 0;
          while (QueryEnvironment.shouldVisit(depth, i++)) {
            List<T> next = new ArrayList<>();
            // Restrict to nodes satisfying the universe predicate.
            Iterable<T> currentInUniverse = Iterables.filter(current, universe);
            // Filter already visited nodes: if we see a node in a later round, then we don't
            // need to visit it again, because the depth at which we see it must be greater
            // than or equal to the last visit.
            Iterables.addAll(
                next,
                env.getReverseDeps(
                    minDepthUniquifier.uniqueAtDepthLessThanOrEqualTo(currentInUniverse, i),
                    context));
            callback.process(currentInUniverse);
            if (next.isEmpty()) {
              // Exit when there are no more nodes to visit.
              break;
            }
            current = next;
          }
        });
  }
}
