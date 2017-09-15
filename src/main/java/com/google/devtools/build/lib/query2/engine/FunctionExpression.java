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

import static java.util.stream.Collectors.joining;

import com.google.common.base.Functions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.Argument;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.ArgumentType;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryFunction;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryTaskFuture;
import java.util.Collection;
import java.util.List;

/**
 * A query expression for user-defined query functions.
 */
public class FunctionExpression extends QueryExpression {
  QueryFunction function;
  List<Argument> args;

  public FunctionExpression(QueryFunction function, List<Argument> args) {
    this.function = function;
    this.args = ImmutableList.copyOf(args);
  }

  public QueryFunction getFunction() {
    return function;
  }

  public List<Argument> getArgs() {
    return args;
  }

  @Override
  public <T> QueryTaskFuture<Void> eval(
      QueryEnvironment<T> env, VariableContext<T> context, Callback<T> callback) {
    return function.eval(env, context, this, args, callback);
  }

  @Override
  public void collectTargetPatterns(Collection<String> literals) {
    for (Argument arg : args) {
      if (arg.getType() == ArgumentType.EXPRESSION) {
        arg.getExpression().collectTargetPatterns(literals);
      }
    }
  }

  @Override
  public <T> T accept(QueryExpressionVisitor<T> visitor) {
    return visitor.visit(this);
  }

  @Override
  public String toString() {
    return function.getName()
        + "("
        + args.stream().map(Functions.toStringFunction()).collect(joining(", "))
        + ")";
  }
}
