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
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.Argument;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.ArgumentType;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryFunction;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryTaskCallable;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryTaskFuture;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * A some(x) filter expression, which returns an arbitrary node in set x, or
 * fails if x is empty.
 *
 * <pre>expr ::= SOME '(' expr ')'</pre>
 */
class SomeFunction implements QueryFunction {
  SomeFunction() {
  }

  @Override
  public String getName() {
    return "some";
  }

  @Override
  public int getMandatoryArguments() {
    return 1;
  }

  @Override
  public List<ArgumentType> getArgumentTypes() {
    return ImmutableList.of(ArgumentType.EXPRESSION);
  }

  @Override
  public <T> QueryTaskFuture<Void> eval(
      QueryEnvironment<T> env,
      QueryExpressionContext<T> context,
      final QueryExpression expression,
      List<Argument> args,
      final Callback<T> callback) {
    final AtomicBoolean someFound = new AtomicBoolean(false);
    QueryTaskFuture<Void> operandEvalFuture = env.eval(
        args.get(0).getExpression(),
        context,
        new Callback<T>() {
          @Override
          public void process(Iterable<T> partialResult)
              throws QueryException, InterruptedException {
            if (Iterables.isEmpty(partialResult) || !someFound.compareAndSet(false, true)) {
              return;
            }
            callback.process(ImmutableSet.of(partialResult.iterator().next()));
          }
        });
    return env.whenSucceedsCall(
        operandEvalFuture,
        new QueryTaskCallable<Void>() {
          @Override
          public Void call() throws QueryException {
            if (!someFound.get()) {
              throw new QueryException(expression, "argument set is empty");
            }
            return null;
          }
        });
  }
}
