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

import com.google.devtools.build.lib.query2.engine.QueryEnvironment.Argument;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryFunction;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryTaskFuture;
import java.util.List;
import java.util.regex.Pattern;
import java.util.regex.PatternSyntaxException;

/** A function that carries out filtering on the action graph. */
public abstract class ActionFilterFunction implements QueryFunction {

  @Override
  public int getMandatoryArguments() {
    // The number of mandatory args was set to 1 to accommodate the --skyframe_state flag.
    // For all other purposes, the _actual_ number of mandatory args is 2.
    return 1;
  }

  @Override
  public <T> QueryTaskFuture<Void> eval(
      QueryEnvironment<T> env,
      QueryExpressionContext<T> context,
      QueryExpression expression,
      List<Argument> args,
      Callback<T> callback) {
    String filterPattern = args.get(0).getWord();
    try {
      Pattern.compile(filterPattern);
    } catch (PatternSyntaxException e) {
      return env.immediateFailedFuture(
          new QueryException(
              expression,
              String.format(
                  "Illegal '%s' pattern regexp '%s': %s",
                  getName(), filterPattern, e.getMessage())));
    }

    // The 2nd argument can only be empty in the case of --skyframe_state.
    if (args.size() != 2) {
      return env.immediateFailedFuture(
          new QueryException(
              expression,
              "aquery filter functions (inputs, outputs, mnemonics) must have exactly 2 arguments,"
                  + "except when --skyframe_state is used."));
    }

    // Do nothing, pass the expression along
    return env.eval(args.get(1).getExpression(), context, callback);
  }
}
