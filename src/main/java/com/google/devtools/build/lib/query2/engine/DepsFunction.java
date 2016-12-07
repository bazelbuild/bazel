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
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.Argument;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.ArgumentType;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryFunction;
import com.google.devtools.build.lib.query2.engine.QueryUtil.ProcessorWithUniquifier;
import java.util.Collection;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ForkJoinPool;

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

  private static class ProcessorImpl<T> implements ProcessorWithUniquifier<T> {
    private final QueryExpression expression;
    private final int depthBound;
    private final QueryEnvironment<T> env;

    private ProcessorImpl(QueryExpression expression, int depthBound, QueryEnvironment<T> env) {
      this.expression = expression;
      this.depthBound = depthBound;
      this.env = env;
    }

    @Override
    public void process(Iterable<T> partialResult, Uniquifier<T> uniquifier, Callback<T> callback)
        throws QueryException, InterruptedException {
      Collection<T> current = Sets.newHashSet(partialResult);
      env.buildTransitiveClosure(expression, (Set<T>) current, depthBound);

      // We need to iterate depthBound + 1 times.
      for (int i = 0; i <= depthBound; i++) {
        // Filter already visited nodes: if we see a node in a later round, then we don't need to
        // visit it again, because the depth at which we see it at must be greater than or equal
        // to the last visit.
        ImmutableList<T> toProcess = uniquifier.unique(current);
        callback.process(toProcess);
        current = ImmutableList.copyOf(env.getFwdDeps(toProcess));
        if (current.isEmpty()) {
          // Exit when there are no more nodes to visit.
          break;
        }
      }
    }
  }

  private static <T> void doEval(
      QueryEnvironment<T> env,
      VariableContext<T> context,
      QueryExpression expression,
      List<Argument> args,
      Callback<T> callback) throws QueryException, InterruptedException {
    int depthBound = args.size() > 1 ? args.get(1).getInteger() : Integer.MAX_VALUE;
    env.eval(
        args.get(0).getExpression(),
        context,
        QueryUtil.compose(
            new ProcessorImpl<T>(expression, depthBound, env),
            env.createUniquifier(),
            callback));
  }

  /**
   * Breadth-first search from the arguments.
   */
  @Override
  public <T> void eval(
      QueryEnvironment<T> env,
      VariableContext<T> context,
      QueryExpression expression,
      List<Argument> args,
      Callback<T> callback) throws QueryException, InterruptedException {
    doEval(env, context, expression, args, callback);
  }

  @Override
  public <T> void parEval(
      QueryEnvironment<T> env,
      VariableContext<T> context,
      QueryExpression expression,
      List<Argument> args,
      ThreadSafeCallback<T> callback,
      ForkJoinPool forkJoinPool) throws QueryException, InterruptedException {
    doEval(env, context, expression, args, callback);
  }
}
