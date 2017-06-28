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

import com.google.common.base.Function;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.Argument;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.ArgumentType;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryFunction;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryTaskFuture;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.ThreadSafeMutableSet;
import java.util.List;
import java.util.Set;

/**
 * A visible(x, y) query expression, which computes the subset of nodes in y
 * that are visible from all nodes in x.
 *
 * <pre>expr ::= VISIBILE '(' expr ',' expr ')'</pre>
 *
 * <p>Example: return targets from the package //bar/baz that are visible to //foo.
 *
 * <pre>
 * visible(//foo, //bar/baz:*)
 * </pre>
 */
public class VisibleFunction implements QueryFunction {

  @Override
  public String getName() {
    return "visible";
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
  public <T> QueryTaskFuture<Void> eval(
      final QueryEnvironment<T> env,
      final VariableContext<T> context,
      QueryExpression expression,
      final List<Argument> args,
      final Callback<T> callback) {
    final QueryTaskFuture<ThreadSafeMutableSet<T>> toSetFuture =
        QueryUtil.evalAll(env, context, args.get(0).getExpression());
    Function<ThreadSafeMutableSet<T>, QueryTaskFuture<Void>> computeVisibleNodesAsyncFunction =
        new Function<ThreadSafeMutableSet<T>, QueryTaskFuture<Void>>() {
          @Override
          public QueryTaskFuture<Void> apply(final ThreadSafeMutableSet<T> toSet) {
            return env.eval(args.get(1).getExpression(), context, new Callback<T>() {
              @Override
              public void process(Iterable<T> partialResult)
                  throws QueryException, InterruptedException {
                for (T t : partialResult) {
                  if (visibleToAll(env, toSet, t)) {
                    callback.process(ImmutableList.of(t));
                  }
                }
              }
            });
          }
        };
    return env.transformAsync(toSetFuture, computeVisibleNodesAsyncFunction);
  }

  /** Returns true if {@code target} is visible to all targets in {@code toSet}. */
  private static <T> boolean visibleToAll(QueryEnvironment<T> env, Set<T> toSet, T target)
      throws QueryException, InterruptedException {
    for (T to : toSet) {
      if (!visible(env, to, target)) {
        return false;
      }
    }
    return true;
  }

  /** Returns true if the target {@code from} is visible to the target {@code to}. */
  public static <T> boolean visible(QueryEnvironment<T> env, T to, T from)
      throws QueryException, InterruptedException {
    Set<QueryVisibility<T>> visiblePackages = env.getAccessor().getVisibility(from);
    for (QueryVisibility<T> spec : visiblePackages) {
      if (spec.contains(to)) {
        return true;
      }
    }
    return false;
  }
}
