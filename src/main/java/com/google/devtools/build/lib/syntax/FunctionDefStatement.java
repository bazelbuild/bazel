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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.syntax.SkylarkType.SkylarkFunctionType;

import java.util.Collection;

/**
 * Syntax node for a function definition.
 */
public class FunctionDefStatement extends Statement {

  private final Ident ident;
  private final ImmutableList<Argument> args;
  private final ImmutableList<Statement> statements;

  public FunctionDefStatement(Ident ident, Collection<Argument> args,
      Collection<Statement> statements) {
    for (Argument arg : args) {
      Preconditions.checkArgument(arg.isNamed());
    }
    this.ident = ident;
    this.args = ImmutableList.copyOf(args);
    this.statements = ImmutableList.copyOf(statements);
  }

  @Override
  void exec(Environment env) throws EvalException, InterruptedException {
    ImmutableMap.Builder<String, Object> defaultValues = ImmutableMap.builder();
    for (Argument arg : args) {
      if (arg.hasValue()) {
        defaultValues.put(arg.getArgName(), arg.getValue().eval(env));
      }
    }
    env.update(ident.getName(), new UserDefinedFunction(
        ident, args, defaultValues.build(), statements, (SkylarkEnvironment) env));
  }

  @Override
  public String toString() {
    return "def " + ident + "(" + args + "):\n";
  }

  public Ident getIdent() {
    return ident;
  }

  public ImmutableList<Statement> getStatements() {
    return statements;
  }

  public ImmutableList<Argument> getArgs() {
    return args;
  }

  @Override
  public void accept(SyntaxTreeVisitor visitor) {
    visitor.visit(this);
  }

  @Override
  void validate(ValidationEnvironment env) throws EvalException {
    SkylarkFunctionType type = SkylarkFunctionType.of(ident.getName());
    ValidationEnvironment localEnv = new ValidationEnvironment(env, type);
    for (Argument i : args) {
      SkylarkType argType = SkylarkType.UNKNOWN;
      if (i.hasValue()) {
        argType = i.getValue().validate(env);
        if (argType.equals(SkylarkType.NONE)) {
          argType = SkylarkType.UNKNOWN;
        }
      }
      localEnv.update(i.getArgName(), argType, getLocation());
    }
    for (Statement stmts : statements) {
      stmts.validate(localEnv);
    }
    env.updateFunction(ident.getName(), type, getLocation());
    // Register a dummy return value with an incompatible type if there was no return statement.
    type.setReturnType(SkylarkType.NONE, getLocation());
  }
}
