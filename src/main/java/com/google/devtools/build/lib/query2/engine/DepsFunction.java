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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.Argument;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.ArgumentType;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.CustomFunctionQueryEnvironment;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryFunction;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryTaskFuture;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.ThreadSafeMutableSet;
import java.util.List;

/**
 * A "deps" query expression, which computes the dependencies of the argument. An optional
 * integer-literal second argument may be specified; its value bounds the search from the arguments.
 *
 * <pre>expr ::= DEPS '(' expr ')'</pre>
 *
 * <pre>       | DEPS '(' expr ',' WORD ')'</pre>
 */
final class DepsFunction implements QueryFunction {
  DepsFunction() {}

  @Override
  public String getName() {
    return "deps";
  }

  @Override
  public int getMandatoryArguments() {
    return 1; // last argument is optional
  }

  @Override
  public List<ArgumentType> getArgumentTypes() {
    return ImmutableList.of(ArgumentType.EXPRESSION, ArgumentType.INTEGER);
  }

  /** Breadth-first search from the arguments. */
  @SuppressWarnings("unchecked")
  @Override
  public <T> QueryTaskFuture<Void> eval(
      final QueryEnvironment<T> env,
      QueryExpressionContext<T> context,
      final QueryExpression expression,
      List<Argument> args,
      final Callback<T> callback) {
    QueryExpression queryExpression = args.get(0).getExpression();
    final int depthBound = args.size() > 1 ? args.get(1).getInteger() : Integer.MAX_VALUE;
    if (env instanceof StreamableQueryEnvironment) {
      if (args.size() == 1) {
        return ((StreamableQueryEnvironment<T>) env)
            .getDepsUnboundedParallel(queryExpression, context, callback, expression);
      }
      return ((StreamableQueryEnvironment<T>) env)
          .getDepsBounded(queryExpression, context, callback, depthBound, expression);
    }

    if (env instanceof QueryEnvironment.CustomFunctionQueryEnvironment) {
      // Not all expressions generate a single future (e.g. SetExpression), as such, we should batch
      // them here before the heavy blocking work is done in the callback to deps.
      return ((QueryEnvironment.CustomFunctionQueryEnvironment) env)
          .eval(
              queryExpression,
              context,
              result ->
                  ((CustomFunctionQueryEnvironment<T>) env)
                      .deps(result, depthBound, expression, callback),
              /* batch= */ true);
    }

    final MinDepthUniquifier<T> minDepthUniquifier = env.createMinDepthUniquifier();
    return env.eval(
        queryExpression,
        context,
        partialResult -> {
          ThreadSafeMutableSet<T> current = env.createThreadSafeMutableSet();
          Iterables.addAll(current, partialResult);
          try (SilentCloseable closeable =
              Profiler.instance().profile("env.buildTransitiveClosure")) {
            env.buildTransitiveClosure(expression, current, depthBound);
          }

          // We need to iterate depthBound + 1 times.
          for (int i = 0; i <= depthBound; i++) {
            // Filter already visited nodes: if we see a node in a later round, then we don't need
            // to visit it again, because the depth at which we see it at must be greater than or
            // equal to the last visit.
            ImmutableList<T> toProcess =
                minDepthUniquifier.uniqueAtDepthLessThanOrEqualTo(current, i);
            callback.process(toProcess);
            current = env.createThreadSafeMutableSet();
            try (SilentCloseable closeable = Profiler.instance().profile("env.getFwdDeps")) {
              Iterables.addAll(current, env.getFwdDeps(toProcess, context));
            }
            if (current.isEmpty()) {
              // Exit when there are no more nodes to visit.
              break;
            }
          }
        });
  }
}
