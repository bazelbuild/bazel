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
package com.google.devtools.build.lib.query2.cquery;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.query2.common.AbstractBlazeQueryEnvironment;
import com.google.devtools.build.lib.query2.engine.Callback;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.Argument;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.ArgumentType;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryFunction;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryTaskFuture;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.ThreadSafeMutableSet;
import com.google.devtools.build.lib.query2.engine.QueryExpression;
import com.google.devtools.build.lib.query2.engine.QueryExpressionContext;
import com.google.devtools.build.lib.query2.engine.QueryUtil;
import java.util.List;

/**
 * A "config" query expression for cquery. The first argument is the expression to be evaluated. The
 * second argument is either "host", "target", "null", or an arbitrary configuration's hash (the
 * same hash cquery annotates label outputs with) to specify which configuration the user is seeking
 * to query in. If some but not all results of expr can be found in the specified config, the subset
 * that can be is returned. If no results of expr can be found in the specified config, an error is
 * thrown.
 *
 * <pre> expr ::= CONFIG '(' expr ',' word ')'</pre>
 */
public final class ConfigFunction implements QueryFunction {

  public ConfigFunction() {
  }

  @Override
  public String getName() {
    return "config";
  }

  @Override
  public int getMandatoryArguments() {
    return 2;
  }

  @Override
  public List<ArgumentType> getArgumentTypes() {
    return ImmutableList.of(ArgumentType.EXPRESSION, ArgumentType.WORD);
  }

  /**
   * This function is only viable with ConfiguredTargetQueryEnvironment which extends {@link
   * AbstractBlazeQueryEnvironment <ConfiguredTarget>}
   */
  @Override
  @SuppressWarnings("unchecked")
  public <T> QueryTaskFuture<Void> eval(
      QueryEnvironment<T> env,
      QueryExpressionContext<T> context,
      QueryExpression expression,
      List<Argument> args,
      final Callback<T> callback) {

    Argument targetExpression = args.get(0);
    String configuration = args.get(1).toString();
    // Turn "'string'" to "string" (remove the surrounding apostrophes).
    configuration = configuration.substring(1, configuration.length() - 1);

    final QueryTaskFuture<ThreadSafeMutableSet<T>> targets =
        QueryUtil.evalAll(env, context, targetExpression.getExpression());

    return env.whenSucceedsCall(
        targets,
        ((ConfiguredTargetQueryEnvironment) env)
            .getConfiguredTargets(
                targetExpression.toString(),
                (ThreadSafeMutableSet<ConfiguredTarget>) targets.getIfSuccessful(),
                configuration,
                (Callback<ConfiguredTarget>) callback));
  }
}
