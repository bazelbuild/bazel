// Copyright 2018 The Bazel Authors. All rights reserved.
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
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.CustomFunctionQueryEnvironment;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryFunction;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryTaskFuture;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.ThreadSafeMutableSet;
import java.util.Collections;
import java.util.List;

/**
 * A "same_pkg_direct_rdeps" query expression, which computes all of the targets in the same package
 * of the given targets which directly depend on them.
 *
 * <pre>expr ::= SAME_PKG_DIRECT_RDEPS '(' expr ')'</pre>
 */
public class SamePkgDirectRdepsFunction implements QueryFunction {

  @Override
  public String getName() {
    return "same_pkg_direct_rdeps";
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
    if (env instanceof CustomFunctionQueryEnvironment) {
      return env.eval(
          args.get(0).getExpression(),
          context,
          partialResult -> {
            ((CustomFunctionQueryEnvironment<T>) env)
                .samePkgDirectRdeps(partialResult, expression, callback);
          });
    }
    Uniquifier<T> uniquifier = env.createUniquifier();
    return env.eval(
        args.get(0).getExpression(),
        context,
        partialResult -> {
          for (T target : partialResult) {
            ThreadSafeMutableSet<T> siblings = env.createThreadSafeMutableSet();
            siblings.addAll(env.getSiblingTargetsInPackage(target));
            env.buildTransitiveClosure(expression, siblings, /*maxDepth=*/ 1);
            Iterable<T> rdeps = env.getReverseDeps(Collections.singleton(target), context);
            callback.process(uniquifier.unique(Iterables.filter(rdeps, siblings::contains)));
          }
        });
  }
}
