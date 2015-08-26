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

/**
 * The actual function registered in the environment. This function is defined in the
 * parsed code using {@link FunctionDefStatement}.
 */
public class UserDefinedFunction extends BaseFunction {

  private final ImmutableList<Statement> statements;
  private final SkylarkEnvironment definitionEnv;

  protected UserDefinedFunction(Identifier function,
      FunctionSignature.WithValues<Object, SkylarkType> signature,
      ImmutableList<Statement> statements, SkylarkEnvironment definitionEnv) {
    super(function.getName(), signature, function.getLocation());

    this.statements = statements;
    this.definitionEnv = definitionEnv;
  }

  public FunctionSignature.WithValues<Object, SkylarkType> getFunctionSignature() {
    return signature;
  }

  ImmutableList<Statement> getStatements() {
    return statements;
  }

  Location getLocation() {
    return location;
  }

  @Override
  public Object call(Object[] arguments, FuncallExpression ast, Environment env)
      throws EvalException, InterruptedException {
    ImmutableList<String> names = signature.getSignature().getNames();

    // Registering the functions's arguments as variables in the local Environment
    int i = 0;
    for (String name : names) {
      env.update(name, arguments[i++]);
    }

    Statement lastStatement = null;
    try {
      for (Statement stmt : statements) {
        lastStatement = stmt;
        stmt.exec(env);
      }
    } catch (ReturnStatement.ReturnException e) {
      return e.getValue();
    } catch (EvalExceptionWithStackTrace ex) {
      // We need this block since the next "catch" must only catch EvalExceptions that don't have a
      // stack trace yet.
      throw ex;
    } catch (EvalException ex) {
      EvalExceptionWithStackTrace real =
          new EvalExceptionWithStackTrace(ex, lastStatement.getLocation());
      real.registerStatement(lastStatement);
      throw real;
    }
    return Environment.NONE;
  }

  /**
   * Creates a new environment for the execution of this function.
   */
  @Override
  protected Environment getOrCreateChildEnvironment(Environment parent) throws EvalException {
   return SkylarkEnvironment.createEnvironmentForFunctionCalling(
       parent, definitionEnv, this);
  }
}
