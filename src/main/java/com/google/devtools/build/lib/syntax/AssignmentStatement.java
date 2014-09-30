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



/**
 * Syntax node for an assignment statement.
 */
public final class AssignmentStatement extends Statement {

  private final Expression lvalue;

  private final Expression expression;

  /**
   *  Constructs an assignment: "lvalue := value".
   */
  AssignmentStatement(Expression lvalue, Expression expression) {
    this.lvalue = lvalue;
    this.expression = expression;
  }

  /**
   *  Returns the LHS of the assignment.
   */
  public Expression getLValue() {
    return lvalue;
  }

  /**
   *  Returns the RHS of the assignment.
   */
  public Expression getExpression() {
    return expression;
  }

  @Override
  public String toString() {
    return lvalue + " = " + expression + '\n';
  }

  @Override
  void exec(Environment env) throws EvalException, InterruptedException {
    if (lvalue instanceof Ident) {
      Ident ident = (Ident) lvalue;
      Object result = expression.eval(env);
      if (result == null) {
        // This can happen during direct Java calls.
        throw new EvalException(getLocation(), "Assigning null variable.");
      }
      if (env.isSkylarkEnabled()) {
        // The variable may have been referenced successfully if a global variable
        // with the same name exists. In this case an Exception needs to be thrown.
        SkylarkEnvironment skylarkEnv = (SkylarkEnvironment) env;
        if (skylarkEnv.hasBeenReadGlobalVariable(ident.getName())) {
          throw new EvalException(getLocation(), "Variable '" + ident.getName()
              + "' is referenced before assignment."
              + "The variable is defined in the global scope.");
        }
        Class<?> variableType = skylarkEnv.getVariableType(ident.getName());
        Class<?> resultType = EvalUtils.getSkylarkType(result.getClass());
        if (variableType != null && !variableType.equals(resultType)) {
          throw new EvalException(getLocation(), String.format("Incompatible variable types, "
              + "trying to assign %s (type of %s) to variable %s which is already %s",
              EvalUtils.prettyPrintValue(result),
              EvalUtils.getDatatypeName(result),
              ident.getName(),
              EvalUtils.getDataTypeNameFromClass(variableType)));
        }
      }
      env.update(ident.getName(), result);
    } else {
      throw new EvalException(getLocation(), "'" + lvalue + "' is not a valid lvalue");
    }
  }

  @Override
  public void accept(SyntaxTreeVisitor visitor) {
    visitor.visit(this);
  }

  @Override
  void validate(ValidationEnvironment env) throws EvalException {
    // TODO(bazel-team): Implement other validations.
    if (lvalue instanceof Ident) {
      Ident ident = (Ident) lvalue;
      SkylarkType resultType = expression.validate(env);
      env.update(ident.getName(), resultType, getLocation());
    } else {
      if (lvalue instanceof FuncallExpression) {
        FuncallExpression func = (FuncallExpression) lvalue;
        SkylarkType resultType = func.getObject().validate(env);
        if (resultType.isStruct()) {
          throw new EvalException(getLocation(), "structs are immutable");
        }
      }
      throw new EvalException(getLocation(), "'" + lvalue + "' is not a valid lvalue");
    }
  }
}
