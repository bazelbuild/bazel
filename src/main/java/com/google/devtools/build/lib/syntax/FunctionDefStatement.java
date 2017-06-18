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
package com.google.devtools.build.lib.syntax;

import com.google.common.collect.ImmutableList;
import java.util.ArrayList;
import java.util.List;

/**
 * Syntax node for a function definition.
 */
public final class FunctionDefStatement extends Statement {

  private final Identifier ident;
  private final FunctionSignature.WithValues<Expression, Expression> signature;
  private final ImmutableList<Statement> statements;
  private final ImmutableList<Parameter<Expression, Expression>> parameters;

  public FunctionDefStatement(Identifier ident,
      Iterable<Parameter<Expression, Expression>> parameters,
      FunctionSignature.WithValues<Expression, Expression> signature,
      Iterable<Statement> statements) {
    this.ident = ident;
    this.parameters = ImmutableList.copyOf(parameters);
    this.signature = signature;
    this.statements = ImmutableList.copyOf(statements);
  }

  @Override
  void doExec(Environment env) throws EvalException, InterruptedException {
    List<Expression> defaultExpressions = signature.getDefaultValues();
    ArrayList<Object> defaultValues = null;
    ArrayList<SkylarkType> types = null;

    if (defaultExpressions != null) {
      defaultValues = new ArrayList<>(defaultExpressions.size());
      for (Expression expr : defaultExpressions) {
        defaultValues.add(expr.eval(env));
      }
    }

    FunctionSignature sig = signature.getSignature();
    if (env.getSemantics().incompatibleDisallowKeywordOnlyArgs
        && sig.getShape().getMandatoryNamedOnly() > 0) {
      throw new EvalException(
          getLocation(),
          "Keyword-only argument is forbidden. You can temporarily disable this "
              + "error using the flag --incompatible_disallow_keyword_only_args=false");
    }

    env.update(
        ident.getName(),
        new UserDefinedFunction(
            ident,
            FunctionSignature.WithValues.<Object, SkylarkType>create(sig, defaultValues, types),
            statements,
            env.getGlobals()));
  }

  @Override
  public String toString() {
    return "def " + ident + "(" + signature + "):\n";
  }

  public Identifier getIdent() {
    return ident;
  }

  public ImmutableList<Statement> getStatements() {
    return statements;
  }

  public ImmutableList<Parameter<Expression, Expression>> getParameters() {
    return parameters;
  }

  public FunctionSignature.WithValues<Expression, Expression> getSignature() {
    return signature;
  }

  @Override
  public void accept(SyntaxTreeVisitor visitor) {
    visitor.visit(this);
  }

  @Override
  void validate(final ValidationEnvironment env) throws EvalException {
    ValidationEnvironment localEnv = new ValidationEnvironment(env);
    FunctionSignature sig = signature.getSignature();
    FunctionSignature.Shape shape = sig.getShape();
    ImmutableList<String> names = sig.getNames();
    List<Expression> defaultExpressions = signature.getDefaultValues();

    int positionals = shape.getPositionals();
    int mandatoryPositionals = shape.getMandatoryPositionals();
    int namedOnly = shape.getNamedOnly();
    int mandatoryNamedOnly = shape.getMandatoryNamedOnly();
    boolean starArg = shape.hasStarArg();
    boolean kwArg = shape.hasKwArg();
    int named = positionals + namedOnly;
    int args = named + (starArg ? 1 : 0) + (kwArg ? 1 : 0);
    int startOptionals = mandatoryPositionals;
    int endOptionals = named - mandatoryNamedOnly;

    int j = 0; // index for the defaultExpressions
    for (int i = 0; i < args; i++) {
      String name = names.get(i);
      if (startOptionals <= i && i < endOptionals) {
        defaultExpressions.get(j++).validate(env);
      }
      localEnv.declare(name, getLocation());
    }
    for (Statement stmts : statements) {
      stmts.validate(localEnv);
    }
  }
}
