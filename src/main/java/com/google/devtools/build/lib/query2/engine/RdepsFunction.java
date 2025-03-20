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

import com.google.common.base.Function;
import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.Argument;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.ArgumentType;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.CustomFunctionQueryEnvironment;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryTaskFuture;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.ThreadSafeMutableSet;
import java.util.List;
import java.util.OptionalInt;

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

  @Override
  public <T> QueryTaskFuture<Void> eval(
      QueryEnvironment<T> env,
      QueryExpressionContext<T> context,
      QueryExpression expression,
      List<Argument> args,
      Callback<T> callback) {
    OptionalInt depth =
        args.size() == 2 ? OptionalInt.empty() : OptionalInt.of(args.get(2).getInteger());
    QueryExpression universeExpression = args.get(0).getExpression();
    QueryExpression argumentExpression = args.get(1).getExpression();
    if (env instanceof StreamableQueryEnvironment<T> streamableEnv) {
      return depth.isPresent()
          ? streamableEnv.getRdepsBoundedParallel(
              argumentExpression, depth.getAsInt(), universeExpression, context, callback)
          : streamableEnv.getRdepsUnboundedParallel(
              argumentExpression, universeExpression, context, callback);
    } else {
      return evalWithBoundedDepth(
          env, expression, context, argumentExpression, depth, universeExpression, callback);
    }
  }

  /**
   * Compute the transitive closure of the universe, then breadth-first search from the argument
   * towards the universe while staying within the transitive closure.
   */
  private static <T> QueryTaskFuture<Void> evalWithBoundedDepth(
      QueryEnvironment<T> env,
      QueryExpression rdepsFunctionExpressionForErrorMessages,
      QueryExpressionContext<T> context,
      QueryExpression argumentExpression,
      OptionalInt depth,
      QueryExpression universeExpression,
      Callback<T> callback) {
    QueryTaskFuture<ThreadSafeMutableSet<T>> universeValueFuture =
        QueryUtil.evalAll(env, context, universeExpression);

    if (env instanceof CustomFunctionQueryEnvironment) {
      QueryTaskFuture<ThreadSafeMutableSet<T>> fromValueFuture =
          QueryUtil.evalAll(env, context, argumentExpression);
      return env.whenAllSucceedCall(
          ImmutableList.of(fromValueFuture, universeValueFuture),
          () -> {
            ThreadSafeMutableSet<T> fromValue = fromValueFuture.getIfSuccessful();
            ThreadSafeMutableSet<T> universeValue = universeValueFuture.getIfSuccessful();
            ((CustomFunctionQueryEnvironment<T>) env)
                .rdeps(
                    fromValue,
                    universeValue,
                    depth,
                    rdepsFunctionExpressionForErrorMessages,
                    callback);
            return null;
          });
    }

    Function<ThreadSafeMutableSet<T>, QueryTaskFuture<Void>> evalInUniverseAsyncFunction =
        universeValue -> {
          Predicate<T> universe;
          try {
            env.buildTransitiveClosure(
                rdepsFunctionExpressionForErrorMessages, universeValue, OptionalInt.empty());
            universe = Predicates.in(env.getTransitiveClosure(universeValue, context));
          } catch (InterruptedException e) {
            return env.immediateCancelledFuture();
          } catch (QueryException e) {
            return env.immediateFailedFuture(e);
          }

          return AllRdepsFunction.eval(env, argumentExpression, universe, context, callback, depth);
        };
    return env.transformAsync(universeValueFuture, evalInUniverseAsyncFunction);
  }
}
