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

import com.google.common.base.Function;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.events.Location;

/**
 * The actual function registered in the environment. This function is defined in the
 * parsed code using {@link FunctionDefStatement}.
 */
public class UserDefinedFunction extends MixedModeFunction {

  private final ImmutableList<Argument> args;
  private final ImmutableMap<String, Integer> argIndexes;
  private final ImmutableMap<String, Object> defaultValues;
  private final ImmutableList<Statement> statements;
  private final SkylarkEnvironment definitionEnv;

  private static ImmutableList<String> argumentToStringList(ImmutableList<Argument> args) {
    Function<Argument, String> function = new Function<Argument, String>() {
      @Override
      public String apply(Argument id) {
        return id.getArgName();
      }
    };
    return ImmutableList.copyOf(Lists.transform(args, function));
  }

  private static int mandatoryArgNum(ImmutableList<Argument> args) {
    int mandatoryArgNum = 0;
    for (Argument arg : args) {
      if (!arg.hasValue()) {
        mandatoryArgNum++;
      }
    }
    return mandatoryArgNum;
  }

  UserDefinedFunction(Ident function, ImmutableList<Argument> args,
      ImmutableMap<String, Object> defaultValues,
      ImmutableList<Statement> statements, SkylarkEnvironment definitionEnv) {
    super(function.getName(), argumentToStringList(args), mandatoryArgNum(args), false,
        function.getLocation());
    this.args = args;
    this.statements = statements;
    this.definitionEnv = definitionEnv;
    this.defaultValues = defaultValues;

    ImmutableMap.Builder<String, Integer> argIndexes = new ImmutableMap.Builder<> ();
    int i = 0;
    for (Argument arg : args) {
      if (!arg.isKwargs()) { // TODO(bazel-team): add varargs support?
        argIndexes.put(arg.getArgName(), i++);
      }
    }
    this.argIndexes = argIndexes.build();
  }

  public ImmutableList<Argument> getArgs() {
    return args;
  }

  public Integer getArgIndex(String s) {
    return argIndexes.get(s);
  }

  ImmutableMap<String, Object> getDefaultValues() {
    return defaultValues;
  }

  ImmutableList<Statement> getStatements() {
    return statements;
  }

  Location getLocation() {
    return location;
  }

  @Override
  public Object call(Object[] namedArguments, FuncallExpression ast, Environment env)
      throws EvalException, InterruptedException {
    SkylarkEnvironment functionEnv = SkylarkEnvironment.createEnvironmentForFunctionCalling(
        env, definitionEnv, this);

    // Registering the functions's arguments as variables in the local Environment
    int i = 0;
    for (Object arg : namedArguments) {
      functionEnv.update(args.get(i++).getArgName(), arg);
    }

    try {
      for (Statement stmt : statements) {
        stmt.exec(functionEnv);
      }
    } catch (ReturnStatement.ReturnException e) {
      return e.getValue();
    }
    return Environment.NONE;
  }
}
