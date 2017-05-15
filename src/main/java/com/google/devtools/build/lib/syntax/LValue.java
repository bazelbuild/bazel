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

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.util.Preconditions;
import java.util.Collection;

/**
 * Class representing an LValue.
 * It appears in assignment, for loop and comprehensions, e.g.
 *    lvalue = 2
 *    [for lvalue in exp]
 *    for lvalue in exp: pass
 * An LValue can be a simple variable or something more complex like a tuple.
 */
public class LValue extends ASTNode {
  private final Expression expr;

  public LValue(Expression expr) {
    this.expr = expr;
    setLocation(expr.getLocation());
  }

  public Expression getExpression() {
    return expr;
  }

  /**
   * Assign a value to an LValue and update the environment.
   */
  public void assign(Environment env, Location loc, Object result)
      throws EvalException, InterruptedException {
    doAssign(env, loc, expr, result);
  }

  /**
   * Evaluate a rhs using a lhs and the operator, then assign to the lhs.
   */
  public void assign(Environment env, Location loc, Expression rhs, Operator operator)
      throws EvalException, InterruptedException {
    if (expr instanceof Identifier) {
      Object result =
          BinaryOperatorExpression.evaluate(operator, expr.eval(env), rhs, env, loc, true);
      assign(env, loc, (Identifier) expr, result);
      return;
    }

    if (expr instanceof IndexExpression) {
      IndexExpression indexExpression = (IndexExpression) expr;
      // This object should be evaluated only once
      Object evaluatedLhsObject = indexExpression.getObject().eval(env);
      Object evaluatedLhs = indexExpression.eval(env, evaluatedLhsObject);
      Object key = indexExpression.getKey().eval(env);
      Object result =
          BinaryOperatorExpression.evaluate(operator, evaluatedLhs, rhs, env, loc, true);
      assignItem(env, loc, evaluatedLhsObject, key, result);
      return;
    }

    if (expr instanceof ListLiteral) {
      throw new EvalException(loc, "Cannot perform augment assignment on a list literal");
    }
  }

  /**
   *  Returns all names bound by this LValue.
   *
   *  Examples:
   *  <ul>
   *  <li><{@code x = ...} binds x.</li>
   *  <li><{@code x, [y,z] = ..} binds x, y, z.</li>
   *  <li><{@code x[5] = ..} does not bind any names.</li>
   *  </ul>
   */
  public ImmutableSet<String> boundNames() {
    ImmutableSet.Builder<String> result = ImmutableSet.builder();
    collectBoundNames(expr, result);
    return result.build();
  }

  private static void collectBoundNames(Expression lhs, ImmutableSet.Builder<String> result) {
    if (lhs instanceof Identifier) {
      result.add(((Identifier) lhs).getName());
      return;
    }
    if (lhs instanceof ListLiteral) {
      ListLiteral variables = (ListLiteral) lhs;
      for (Expression expression : variables.getElements()) {
        collectBoundNames(expression, result);
      }
    }
  }

  private static void doAssign(
      Environment env, Location loc, Expression lhs, Object result)
      throws EvalException, InterruptedException {
    if (lhs instanceof Identifier) {
      assign(env, loc, (Identifier) lhs, result);
      return;
    }

    if (lhs instanceof ListLiteral) {
      ListLiteral variables = (ListLiteral) lhs;
      Collection<?> rvalue = EvalUtils.toCollection(result, loc);
      int len = variables.getElements().size();
      if (len != rvalue.size()) {
        throw new EvalException(loc, String.format(
            "lvalue has length %d, but rvalue has has length %d", len, rvalue.size()));
      }
      int i = 0;
      for (Object o : rvalue) {
        doAssign(env, loc, variables.getElements().get(i), o);
        i++;
      }
      return;
    }

    // Support syntax for setting an element in an array, e.g. a[5] = 2
    // TODO: We currently do not allow slices (e.g. a[2:6] = [3]).
    if (lhs instanceof IndexExpression) {
      IndexExpression expression = (IndexExpression) lhs;
      Object key = expression.getKey().eval(env);
      Object evaluatedObject = expression.getObject().eval(env);
      assignItem(env, loc, evaluatedObject, key, result);
      return;
    }
    throw new EvalException(loc, "cannot assign to '" + lhs + "'");
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

  @Override
  public void accept(SyntaxTreeVisitor visitor) {
    visitor.visit(this);
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
}
