// Copyright 2024 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.query2;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.query2.engine.Callback;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.Argument;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.ArgumentType;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryFunction;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryTaskFuture;
import com.google.devtools.build.lib.query2.engine.QueryException;
import com.google.devtools.build.lib.query2.engine.QueryExpression;
import com.google.devtools.build.lib.query2.engine.QueryExpressionContext;
import com.google.devtools.build.lib.server.FailureDetails.Query;
import java.util.List;

public class ExtensionInputsFunction implements QueryFunction {

  @Override
  public String getName() {
    return "extension_inputs";
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
  @SuppressWarnings("unchecked")
  public <T> QueryTaskFuture<Void> eval(
      QueryEnvironment<T> env,
      QueryExpressionContext<T> context,
      QueryExpression expression,
      List<Argument> args,
      Callback<T> callback) {

    if (!(env instanceof SkyQueryEnvironment skyEnv)) {
      return env.immediateFailedFuture(
          new QueryException(
              expression,
              "extension_inputs() can only be used with SkyQueryEnvironment",
              Query.Code.EXTENSION_INPUTS_FUNCTION_REQUIRES_SKYQUERY));
    }

    return skyEnv.getExtensionInputs(
        args.get(0).getExpression(),
        (QueryExpressionContext<Target>) context,
        (Callback<Target>) callback);
  }
}
