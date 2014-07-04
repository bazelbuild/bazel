// Copyright 2014 Google Inc. All rights reserved.
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
package com.google.devtools.build.lib.syntax;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.events.Location;

import java.util.List;
import java.util.Map;

/**
 * The actual function registered in the environment. This function is defined in the
 * parsed code using {@link FunctionDefStatement}.
 */
public class UserDefinedFunction extends AbstractFunction {

  // This Dummy type helps debugging.
  // TODO(bazel-team): Move NONE somewhere else
  /**
   * Dummy default return value.
   */
  public static final class SkylarkDefaultReturnValue {}
  public static final SkylarkDefaultReturnValue NONE = new SkylarkDefaultReturnValue();

  private final ImmutableList<Ident> listArgNames;
  private final ImmutableList<Statement> statements;
  private final Location location;

  protected UserDefinedFunction(Ident function, ImmutableList<Ident> listArgNames,
      ImmutableList<Statement> statements) {
    super(function.getName());
    this.location = function.getLocation();
    this.listArgNames = listArgNames;
    this.statements = statements;
  }

  public ImmutableList<Ident> getListArgNames() {
    return listArgNames;
  }

  ImmutableList<Statement> getStatements() {
    return statements;
  }

  Location getLocation() {
    return location;
  }

  @Override
  public Object call(List<Object> args, Map<String, Object> kwargs,
      FuncallExpression ast, Environment env)
      throws EvalException, InterruptedException {
    if (args.size() != listArgNames.size()) {
      throw new EvalException(ast.getLocation(), String.format(
          "Invalid number of arguments, got %s instead of %s", args.size(), listArgNames.size()));
    }
    // Creating an environment from this functions arguments and the global environment
    SkylarkEnvironment functionEnv = SkylarkEnvironment.createEnvironmentForFunctionCalling(
        (SkylarkEnvironment) env, this);
    int i = 0;
    // TODO(bazel-team): support kwargs

    // Registering the functions's arguments as variables in the local Environment
    for (Object arg : args) {
      functionEnv.update(listArgNames.get(i++).getName(), arg);
    }

    try {
      for (Statement stmt : statements) {
        stmt.exec(functionEnv);
      }
    } catch (ReturnStatement.ReturnException e) {
      return e.getValue();
    }
    return NONE;
  }
}
