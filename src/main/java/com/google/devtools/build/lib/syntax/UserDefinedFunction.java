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
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.events.Location;

import java.util.List;
import java.util.Map;

/**
 * The actual function registered in the environment. This function is defined in the
 * parsed code using {@link FunctionDefStatement}.
 */
public class UserDefinedFunction extends MixedModeFunction {

  private final ImmutableList<Ident> listArgNames;
  private final ImmutableList<Statement> statements;
  private final SkylarkEnvironment definitionEnv;

  private static ImmutableList<String> identToStringList(ImmutableList<Ident> listArgNames) {
    Function<Ident, String> function = new Function<Ident, String>() {
      @Override
      public String apply(Ident id) {
        return id.getName();
      }
    };
    return ImmutableList.copyOf(Lists.transform(listArgNames, function));
  }

  protected UserDefinedFunction(Ident function, ImmutableList<Ident> listArgNames,
      ImmutableList<Statement> statements, SkylarkEnvironment definitionEnv) {
    super(function.getName(), identToStringList(listArgNames), listArgNames.size(), false,
        function.getLocation());
    this.listArgNames = listArgNames;
    this.statements = statements;
    this.definitionEnv = definitionEnv;
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
  public Object call(Object[] namedArguments, List<Object> positionalArguments,
      Map<String, Object> keywordArguments, FuncallExpression ast, Environment env)
      throws EvalException, InterruptedException {
    // TODO(bazel-team): support default values
    SkylarkEnvironment functionEnv = SkylarkEnvironment.createEnvironmentForFunctionCalling(
        env, definitionEnv, this);

    // Registering the functions's arguments as variables in the local Environment
    int i = 0;
    for (Object arg : namedArguments) {
      functionEnv.update(listArgNames.get(i++).getName(), arg);
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
