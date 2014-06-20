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
import com.google.devtools.build.lib.syntax.UserDefinedFunction.SkylarkDefaultReturnValue;

import java.util.Collection;

/**
 * Syntax node for a function definition.
 */
public class FunctionDefStatement extends Statement {

  private final Ident ident;
  private final ImmutableList<Ident> arg;
  private final ImmutableList<Statement> statements;

  public FunctionDefStatement(Ident ident, Collection<Ident> arg,
      Collection<Statement> statements) {
    this.ident = ident;
    this.arg = ImmutableList.copyOf(arg);
    this.statements = ImmutableList.copyOf(statements);
  }

  @Override
  void exec(Environment env) throws EvalException, InterruptedException {
    env.update(ident.getName(), new UserDefinedFunction(ident, arg, statements));
  }

  @Override
  public String toString() {
    return "def " + ident + "(" + arg + "):\n";
  }

  public Ident getIdent() {
    return ident;
  }

  public ImmutableList<Statement> getStatements() {
    return statements;
  }

  public ImmutableList<Ident> getArg() {
    return arg;
  }

  @Override
  public void accept(SyntaxTreeVisitor visitor) {
    visitor.visit(this);
  }

  @Override
  void validate(ValidationEnvironment env) throws EvalException {
    env.setCurrentFunction(ident.getName());
    // TODO(bazel-team): set up local environment.
    for (Ident i : arg) {
      env.update(i.getName(), Object.class, getLocation());
    }
    for (Statement stmts : statements) {
      stmts.validate(env);
    }
    env.update(ident.getName(), UserDefinedFunction.class, getLocation());
    // Register a dummy return value with an incompatible type if there was no return statement.
    if (!env.hasVariable(ident.getName() + ".return")) {
      env.update(ident.getName() + ".return", SkylarkDefaultReturnValue.class, getLocation());
    }
  }
}
