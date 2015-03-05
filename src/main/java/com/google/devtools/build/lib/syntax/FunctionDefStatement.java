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
import com.google.devtools.build.lib.syntax.SkylarkType.SkylarkFunctionType;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;

/**
 * Syntax node for a function definition.
 */
public class FunctionDefStatement extends Statement {

  private final Ident ident;
  private final FunctionSignature.WithValues<Expression, Expression> args;
  private final ImmutableList<Statement> statements;

  public FunctionDefStatement(Ident ident,
      FunctionSignature.WithValues<Expression, Expression> args,
      Collection<Statement> statements) {

    // TODO(bazel-team): lift the following limitation from {@link MixedModeFunction}
    FunctionSignature.Shape shape = args.getSignature().getShape();
    Preconditions.checkArgument(!shape.hasKwArg() && !shape.hasStarArg()
        && shape.getNamedOnly() == 0, "no star, star-star or named-only parameters (for now)");

    this.ident = ident;
    this.args = args;
    this.statements = ImmutableList.copyOf(statements);
  }

  @Override
  void exec(Environment env) throws EvalException, InterruptedException {
    List<Expression> defaultExpressions = args.getDefaultValues();
    ArrayList<Object> defaultValues = null;
    ArrayList<SkylarkType> types = null;

    if (defaultExpressions != null) {
      defaultValues = new ArrayList<>(defaultExpressions.size());
      for (Expression expr : defaultExpressions) {
        defaultValues.add(expr.eval(env));
      }
    }
    env.update(ident.getName(), new UserDefinedFunction(
        ident, FunctionSignature.WithValues.<Object, SkylarkType>create(
            args.getSignature(), defaultValues, types),
        statements, (SkylarkEnvironment) env));
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

  public FunctionSignature.WithValues<Expression, Expression> getArgs() {
    return args;
  }

  @Override
  public void accept(SyntaxTreeVisitor visitor) {
    visitor.visit(this);
  }

  @Override
  void validate(final ValidationEnvironment env) throws EvalException {
    SkylarkFunctionType type = SkylarkFunctionType.of(ident.getName());
    ValidationEnvironment localEnv = new ValidationEnvironment(env, type);
    FunctionSignature sig = args.getSignature();
    FunctionSignature.Shape shape = sig.getShape();
    ImmutableList<String> names = sig.getNames();
    List<Expression> defaultExpressions = args.getDefaultValues();

    int positionals = shape.getPositionals();
    int mandatoryPositionals = shape.getMandatoryPositionals();
    int namedOnly = shape.getNamedOnly();
    int mandatoryNamedOnly = shape.getMandatoryNamedOnly();
    boolean starArg = shape.hasStarArg();
    boolean hasStar = starArg || (namedOnly > 0);
    boolean kwArg = shape.hasKwArg();
    int named = positionals + namedOnly;
    int args = named + (starArg ? 1 : 0) + (kwArg ? 1 : 0);
    int startOptionals = mandatoryPositionals;
    int endOptionals = named - mandatoryNamedOnly;
    int iStarArg = named;
    int iKwArg = args - 1;

    int j = 0; // index for the defaultExpressions
    for (int i = 0; i < args; i++) {
      String name = names.get(i);
      SkylarkType argType = SkylarkType.UNKNOWN;
      if (hasStar && i == iStarArg) {
        argType = SkylarkType.of(SkylarkList.class, Object.class);
      } else if (kwArg && i == iKwArg) {
        argType = SkylarkType.of(Map.class, Object.class);
      } else {
        if (startOptionals <= i && i < endOptionals) {
          argType = defaultExpressions.get(j++).validate(env);
          if (argType.equals(SkylarkType.NONE)) {
            argType = SkylarkType.UNKNOWN;
          }
        }
      }
      localEnv.update(name, argType, getLocation());
    }
    for (Statement stmts : statements) {
      stmts.validate(localEnv);
    }
    env.updateFunction(ident.getName(), type, getLocation());
    // Register a dummy return value with an incompatible type if there was no return statement.
    type.setReturnType(SkylarkType.NONE, getLocation());
  }
}
