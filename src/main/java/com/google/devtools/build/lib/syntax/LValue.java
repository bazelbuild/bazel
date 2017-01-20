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

import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.util.Preconditions;
import java.io.Serializable;
import java.util.Collection;

/**
 * Class representing an LValue.
 * It appears in assignment, for loop and comprehensions, e.g.
 *    lvalue = 2
 *    [for lvalue in exp]
 *    for lvalue in exp: pass
 * An LValue can be a simple variable or something more complex like a tuple.
 */
public class LValue implements Serializable {
  private final Expression expr;

  public LValue(Expression expr) {
    this.expr = expr;
  }

  public Expression getExpression() {
    return expr;
  }

  /**
   * Assign a value to an LValue and update the environment.
   */
  public void assign(Environment env, Location loc, Object result)
      throws EvalException, InterruptedException {
    assign(env, loc, expr, result);
  }

  private static void assign(Environment env, Location loc, Expression lvalue, Object result)
      throws EvalException, InterruptedException {
    if (lvalue instanceof Identifier) {
      assign(env, loc, (Identifier) lvalue, result);
      return;
    }

    if (lvalue instanceof ListLiteral) {
      ListLiteral variables = (ListLiteral) lvalue;
      Collection<?> rvalue = EvalUtils.toCollection(result, loc);
      int len = variables.getElements().size();
      if (len != rvalue.size()) {
        throw new EvalException(loc, String.format(
            "lvalue has length %d, but rvalue has has length %d", len, rvalue.size()));
      }
      int i = 0;
      for (Object o : rvalue) {
        assign(env, loc, variables.getElements().get(i), o);
        i++;
      }
      return;
    }

    // Support syntax for setting an element in an array, e.g. a[5] = 2
    // TODO: We currently do not allow slices (e.g. a[2:6] = [3]).
    if (lvalue instanceof IndexExpression) {
      IndexExpression expression = (IndexExpression) lvalue;
      Object key = expression.getKey().eval(env);
      Object evaluatedObject = expression.getObject().eval(env);
      assignItem(env, loc, evaluatedObject, key, result);
      return;
    }
    throw new EvalException(loc,
        "cannot assign to '" + lvalue + "'");
  }

  @SuppressWarnings("unchecked")
  private static void assignItem(
      Environment env, Location loc, Object o, Object key, Object value)
      throws EvalException, InterruptedException {
    if (o instanceof SkylarkDict) {
      SkylarkDict<Object, Object> dict = (SkylarkDict<Object, Object>) o;
      dict.put(key, value, loc, env);
    } else if (o instanceof SkylarkList) {
      SkylarkList<Object> list = (SkylarkList<Object>) o;
      list.set(key, value, loc, env);
    } else {
      throw new EvalException(
          loc,
          "can only assign an element in a dictionary or a list, not in a '"
              + EvalUtils.getDataTypeName(o)
              + "'");
    }
  }

  /**
   * Assign value to a single variable.
   */
  private static void assign(Environment env, Location loc, Identifier ident, Object result)
      throws EvalException, InterruptedException {
    Preconditions.checkNotNull(result, "trying to assign null to %s", ident);

    // The variable may have been referenced successfully if a global variable
    // with the same name exists. In this case an Exception needs to be thrown.
    if (env.isKnownGlobalVariable(ident.getName())) {
      throw new EvalException(
          loc,
          String.format(
              "Variable '%s' is referenced before assignment. "
                  + "The variable is defined in the global scope.",
              ident.getName()));
    }
    env.update(ident.getName(), result);
  }

  void validate(ValidationEnvironment env, Location loc) throws EvalException {
    validate(env, loc, expr);
  }

  private static void validate(ValidationEnvironment env, Location loc, Expression expr)
      throws EvalException {
    if (expr instanceof Identifier) {
      Identifier ident = (Identifier) expr;
      env.declare(ident.getName(), loc);
      return;
    }
    if (expr instanceof ListLiteral) {
      for (Expression e : ((ListLiteral) expr).getElements()) {
        validate(env, loc, e);
      }
      return;
    }
    if (expr instanceof IndexExpression) {
      expr.validate(env);
      return;
    }
    throw new EvalException(loc,
        "cannot assign to '" + expr + "'");
  }

  @Override
  public String toString() {
    return expr.toString();
  }

  /**
   * Checks that the size of a collection at runtime conforms to the amount of l-value expressions
   * we have to assign to.
   */
  public static void checkSize(int expected, Collection<?> collection, Location location)
      throws EvalException {
    int actual = collection.size();
    if (expected != actual) {
      throw new EvalException(
          location,
          String.format("lvalue has length %d, but rvalue has has length %d", expected, actual));
    }
  }
}
