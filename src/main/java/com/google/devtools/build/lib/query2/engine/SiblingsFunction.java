// Copyright 2017 The Bazel Authors. All rights reserved.
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
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryTaskFuture;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.TargetAccessor;
import java.util.List;
import java.util.Set;

/**
 * A "siblings" query expression, which computes all of the targets in all of the packages of all
 * the targets to which the argument evaluates.
 *
 * <pre>expr ::= SIBLINGS '(' expr ')'</pre>
 */
public class SiblingsFunction implements QueryFunction {
  @Override
  public String getName() {
    return "siblings";
  }

  @Override
  public int getMandatoryArguments() {
    return 1;
  }

  @Override
  public Iterable<ArgumentType> getArgumentTypes() {
    return ImmutableList.of(ArgumentType.EXPRESSION);
  }

  @Override
  public <T> QueryTaskFuture<Void> eval(
      QueryEnvironment<T> env,
      QueryExpressionContext<T> context,
      final QueryExpression expression,
      List<Argument> args,
      final Callback<T> callback) {
    final TargetAccessor<T> targetAccessor = env.getAccessor();
    Set<String> packageNames = Sets.newConcurrentHashSet();
    return env.eval(
        args.get(0).getExpression(),
        context,
        new Callback<T>() {
          @Override
          public void process(Iterable<T> partialResult)
              throws QueryException, InterruptedException {
            for (T target : partialResult) {
              if (packageNames.add(targetAccessor.getPackage(target))) {
                callback.process(env.getSiblingTargetsInPackage(target));
              }
            }
          }
        });
  }
}
