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

import com.google.devtools.build.lib.syntax.Environment.NoSuchVariableException;

import java.util.Map;


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
        // Skylark has readonly variables.
        if (skylarkEnv.isReadOnly(ident.getName())) {
          throw new EvalException(getLocation(),
              "Cannot assign readonly variable: '" + ident.getName() + "'");
        }
        Class<?> variableType = skylarkEnv.getVariableType(ident.getName());
        if (variableType != null && !variableType.equals(result.getClass())) {
          throw new EvalException(getLocation(), String.format("Incompatible variable types, "
              + "trying to assign %s (type of %s) to variable %s which is already %s",
              EvalUtils.prettyPrintValue(result),
              EvalUtils.getDatatypeName(result),
              ident.getName(),
              skylarkEnv.getVariableTypeName(ident.getName())));
        }
      }
      env.update(ident.getName(), result);
    } else if (isIndexFuncallExpression(lvalue)) {
      assignCollection(env);
    } else {
      throw new EvalException(getLocation(), "'" + lvalue + "' is not a valid lvalue");
    }
  }

  // TODO(bazel-team): This is not very safe. One can assign e.g. a List<String>  - created via
  // direct Java calls - to a variable, then add an Integer here (which is fine in Python but
  // Java doesn't allow it). We should make sure somehow to access only generics created with
  // java.lang.Object. Alternatively we can create a safe copy of the collection here but that
  // would be not very efficient.
  @SuppressWarnings("unchecked")
  private void assignCollection(Environment env) throws EvalException, InterruptedException {
    try {
      FuncallExpression func = (FuncallExpression) lvalue;
      Expression obj = func.getObject();
      if (obj instanceof Ident) {
        // This won't allow us to do things like '{1: 2}[1] = 3', but we don't want that anyway.
        Object key = getAssignArg(func, env);
        Object value = expression.eval(env);
        Object collection = env.lookup(((Ident) obj).getName());
        try {
          if (collection instanceof Map<?, ?>) {
            ((Map<Object, Object>) collection).put(key, value);
            return;
          }
        } catch (UnsupportedOperationException e) {
          // Both java.lang.collection.UnmodifiableList and Guava immutable collections
          // throw this exception, it should cover most cases. However it's hard to guarantee
          // that it works properly for every implementation.
          throw new EvalException(getLocation(), "collection is unmodifiable");
        }
      }
    } catch (NoSuchVariableException e) {
      throw new EvalException(getLocation(), e.getMessage());
    }
    throw new EvalException(getLocation(), "'" + lvalue + "' is not a valid lvalue");
  }

  /**
   * Checks if func has only one argument, evaluates and returns it.
   */
  private Object getAssignArg(FuncallExpression func, Environment env)
      throws EvalException, InterruptedException {
    if (func.getArguments().size() != 1) {
      throw new EvalException(getLocation(), "$index has to have exactly 1 argument");
    }
    return func.getArguments().get(0).getValue().eval(env);
  }

  /**
   * Checks if lvalue is an $index function.
   */
  private boolean isIndexFuncallExpression(Object lvalue) {
    if (lvalue instanceof FuncallExpression) {
      FuncallExpression expr = (FuncallExpression) lvalue;
      if (expr.getFunction() instanceof Ident) {
        return ((Ident) expr.getFunction()).getName().equals("$index");
      }
    }
    return false;
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
      Class<?> resultType = expression.validate(env);
      env.update(ident.getName(), resultType, getLocation());
    } else if (!isIndexFuncallExpression(lvalue)) {
      throw new EvalException(getLocation(), "'" + lvalue + "' is not a valid lvalue");
    }
  }
}
