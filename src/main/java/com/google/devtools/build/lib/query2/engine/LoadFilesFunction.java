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
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryTaskFuture;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.ThreadSafeMutableSet;
import java.util.List;

/**
 * A loadfiles(x) query expression, which computes the set of .bzl files
 * for each target in set x.  The result is unordered.  This
 * operator is typically used for determining what files or packages to check
 * out.
 *
 * <pre>expr ::= LOADFILES '(' expr ')'</pre>
 */
public class LoadFilesFunction implements QueryEnvironment.QueryFunction {
  LoadFilesFunction() {}

  @Override
  public String getName() {
    return "loadfiles";
  }

  @Override
  public <T> QueryTaskFuture<Void> eval(
      final QueryEnvironment<T> env,
      VariableContext<T> context,
      final QueryExpression expression,
      List<QueryEnvironment.Argument> args,
      final Callback<T> callback) {
    final Uniquifier<T> uniquifier = env.createUniquifier();
    return env.eval(
        args.get(0).getExpression(),
        context,
        new Callback<T>() {
          @Override
          public void process(Iterable<T> partialResult)
              throws QueryException, InterruptedException {
            ThreadSafeMutableSet<T> result = env.createThreadSafeMutableSet();
            Iterables.addAll(result, partialResult);
            callback.process(uniquifier.unique(
                env.getBuildFiles(
                    expression,
                    result,
                    /* BUILD */ false,
                    /* subinclude */ false,
                    /* load */ true)));
          }
        });
  }

  @Override
  public int getMandatoryArguments() {
    return 1;
  }

  @Override
  public List<QueryEnvironment.ArgumentType> getArgumentTypes() {
    return ImmutableList.of(QueryEnvironment.ArgumentType.EXPRESSION);
  }
}
