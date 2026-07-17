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
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.Argument;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.ArgumentType;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.EvaluateExpression;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryFunction;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryTaskFuture;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.ThreadSafeMutableSet;
import com.google.devtools.build.lib.server.FailureDetails.Query;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CancellationException;

/**
 * A some(x) filter expression, which returns certain number of arbitrary nodes in set x, or fails
 * if x is empty. An optional integer-literal second argument may be specified; it specifies number
 * of arbitrary nodes to be returned. If second argument is empty, return one node.
 *
 * <pre>expr ::= SOME '(' expr ')'</pre>
 *
 * <pre>       | SOME '(' expr ',' count ')'</pre>
 */
class SomeFunction implements QueryFunction {
  SomeFunction() {}

  @Override
  public String getName() {
    return "some";
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
      final QueryExpression expression,
      List<Argument> args,
      final Callback<T> callback) {

    // Add a second optional integer parameter indicating return size.
    final int resultMaxSize = args.size() > 1 ? args.get(1).getInteger() : 1;

    // Since the callback will be executed multiple times, so we need to avoid some target
    // being added multiple times in different callback execution. So we need to have a state
    // variable (`targetsSet` below) to track which ones are already added in order to avoid adding
    // duplicates.
    ThreadSafeMutableSet<T> targetsSet = env.createThreadSafeMutableSet();

    EvaluateExpression<T> evaluateExpression =
        env.createEvaluateExpression(args.get(0).getExpression(), context);
    QueryTaskFuture<Void> queryTaskFuture =
        evaluateExpression.eval(
            partialResult -> {
              if (Iterables.isEmpty(partialResult)) {
                return;
              }

              boolean shouldCancel = false;
              synchronized (targetsSet) {
                ArrayList<T> current = new ArrayList<>();
                for (T nextTarget : partialResult) {
                  if (targetsSet.size() >= resultMaxSize) {
                    break;
                  }
                  if (targetsSet.add(nextTarget)) {
                    current.add(nextTarget);
                  }
                }

                if (!current.isEmpty()) {
                  callback.process(current);
                }

                if (targetsSet.size() >= resultMaxSize) {
                  shouldCancel = true;
                }
              }

              if (shouldCancel) {
                var unused = evaluateExpression.gracefullyCancel();
              }
            });

    return env.whenSucceedsOrIsCancelledCall(
        queryTaskFuture,
        () -> {
          if (evaluateExpression.isUngracefullyCancelled()) {
            throw new CancellationException();
          }

          if (targetsSet.isEmpty()) {
            throw new QueryException(
                expression, "argument set is empty", Query.Code.ARGUMENTS_MISSING);
          }
          return null;
        });
  }
}
