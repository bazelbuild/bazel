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
import com.google.devtools.build.lib.syntax.SkylarkList.MutableList;
import com.google.devtools.build.lib.util.Preconditions;
import java.io.IOException;
import java.util.Collection;

/**
 * A term that can appear on the left-hand side of an assignment statement, for loop, comprehension
 * clause, etc. E.g.,
 * <ul>
 *   <li>{@code lvalue = 2}
 *   <li>{@code [for lvalue in exp]}
 *   <li>{@code for lvalue in exp: pass}
 * </ul>
 *
 * <p>An {@code LValue}'s expression must have one of the following forms:
 * <ul>
 *   <li>(Variable assignment) an {@link Identifier};
 *   <li>(List or dictionary item assignment) an {@link IndexExpression}; or
 *   <li>(Sequence assignment) a non-empty {@link ListLiteral} (either list or tuple) of expressions
 *       that can themselves appear in an {@code LValue}.
 * </ul>
 * In particular and unlike Python, slice expressions, dot expressions, and starred expressions
 * cannot appear in {@code LValue}s.
 */
// TODO(bazel-team): Add support for assigning to slices (e.g. a[2:6] = [3]).
public final class LValue extends ASTNode {

  private final Expression expr;

  public LValue(Expression expr) {
    this.expr = expr;
    setLocation(expr.getLocation());
  }

  public Expression getExpression() {
    return expr;
  }

  /**
   * Updates the environment bindings, and possibly mutates objects, so as to assign the given value
   * to this {@code LValue}.
   */
  public void assign(Object value, Environment env, Location loc)
      throws EvalException, InterruptedException {
    assign(expr, value, env, loc);
  }

  /**
   * Updates the environment bindings, and possibly mutates objects, so as to assign the given
   * value to the given expression. The expression must be valid for an {@code LValue}.
   */
  private static void assign(Expression expr, Object value, Environment env, Location loc)
      throws EvalException, InterruptedException {
    if (expr instanceof Identifier) {
      assignIdentifier((Identifier) expr, value, env, loc);
    } else if (expr instanceof IndexExpression) {
      Object object = ((IndexExpression) expr).getObject().eval(env);
      Object key = ((IndexExpression) expr).getKey().eval(env);
      assignItem(object, key, value, env, loc);
    } else if (expr instanceof ListLiteral) {
      ListLiteral list = (ListLiteral) expr;
      assignList(list, value, env, loc);
    } else {
      // Not possible for validated ASTs.
      throw new EvalException(loc, "cannot assign to '" + expr + "'");
    }
  }

  /**
   * Binds a variable to the given value in the environment.
   *
   * @throws EvalException if we're currently in a function's scope, and the identifier has
   * previously resolved to a global variable in the same function
   */
  private static void assignIdentifier(
      Identifier ident, Object value, Environment env, Location loc)
      throws EvalException, InterruptedException {
    Preconditions.checkNotNull(value, "trying to assign null to %s", ident);

    if (env.isKnownGlobalVariable(ident.getName())) {
      throw new EvalException(
          loc,
          String.format(
              "Variable '%s' is referenced before assignment. "
                  + "The variable is defined in the global scope.",
              ident.getName()));
    }
    env.update(ident.getName(), value);
  }

  /**
   * Adds or changes an object-key-value relationship for a list or dict.
   *
   * <p>For a list, the key is an in-range index. For a dict, it is a hashable value.
   *
   * @throws EvalException if the object is not a list or dict
   */
  @SuppressWarnings("unchecked")
  private static void assignItem(
      Object object, Object key, Object value, Environment env, Location loc)
      throws EvalException, InterruptedException {
    if (object instanceof SkylarkDict) {
      SkylarkDict<Object, Object> dict = (SkylarkDict<Object, Object>) object;
      dict.put(key, value, loc, env);
    } else if (object instanceof MutableList) {
      MutableList<Object> list = (MutableList<Object>) object;
      int index = EvalUtils.getSequenceIndex(key, list.size(), loc);
      list.set(index, value, loc, env.mutability());
    } else {
      throw new EvalException(
          loc,
          "can only assign an element in a dictionary or a list, not in a '"
              + EvalUtils.getDataTypeName(object)
              + "'");
    }
  }

  /**
   * Recursively assigns an iterable value to a list literal.
   *
   * @throws EvalException if the list literal has length 0, or if the value is not an iterable of
   *     matching length
   */
  private static void assignList(ListLiteral list, Object value, Environment env, Location loc)
      throws EvalException, InterruptedException {
    Collection<?> collection = EvalUtils.toCollection(value, loc, env);
    int len = list.getElements().size();
    if (len == 0) {
      throw new EvalException(
          loc,
          "lists or tuples on the left-hand side of assignments must have at least one item");
    }
    if (len != collection.size()) {
      throw new EvalException(loc, String.format(
          "assignment length mismatch: left-hand side has length %d, but right-hand side evaluates "
              + "to value of length %d", len, collection.size()));
    }
    int i = 0;
    for (Object item : collection) {
      assign(list.getElements().get(i), item, env, loc);
      i++;
    }
  }

  /**
   * Evaluates an augmented assignment that mutates this {@code LValue} with the given right-hand
   * side's value.
   *
   * <p>The left-hand side expression is evaluated only once, even when it is an {@link
   * IndexExpression}. The left-hand side is evaluated before the right-hand side to match Python's
   * behavior (hence why the right-hand side is passed as an expression rather than as an evaluated
   * value).
   */
  public void assignAugmented(Operator operator, Expression rhs, Environment env, Location loc)
      throws EvalException, InterruptedException {
    if (expr instanceof Identifier) {
      Object result =
          BinaryOperatorExpression.evaluateAugmented(
              operator, expr.eval(env), rhs.eval(env), env, loc);
      assignIdentifier((Identifier) expr, result, env, loc);
    } else if (expr instanceof IndexExpression) {
      IndexExpression indexExpression = (IndexExpression) expr;
      // The object and key should be evaluated only once, so we don't use expr.eval().
      Object object = indexExpression.getObject().eval(env);
      Object key = indexExpression.getKey().eval(env);
      Object oldValue = IndexExpression.evaluate(object, key, env, loc);
      // Evaluate rhs after lhs.
      Object rhsValue = rhs.eval(env);
      Object result =
          BinaryOperatorExpression.evaluateAugmented(operator, oldValue, rhsValue, env, loc);
      assignItem(object, key, result, env, loc);
    } else if (expr instanceof ListLiteral) {
      throw new EvalException(loc, "cannot perform augmented assignment on a list literal");
    } else {
      // Not possible for validated ASTs.
      throw new EvalException(loc, "cannot perform augmented assignment on '" + expr + "'");
    }
  }

  /**
   * Returns all names bound by this LValue.
   *
   * <p>Examples:
   *
   * <ul>
   *   <li><{@code x = ...} binds x.
   *   <li><{@code x, [y,z] = ..} binds x, y, z.
   *   <li><{@code x[5] = ..} does not bind any names.
   * </ul>
   */
  public ImmutableSet<Identifier> boundIdentifiers() {
    ImmutableSet.Builder<Identifier> result = ImmutableSet.builder();
    collectBoundIdentifiers(expr, result);
    return result.build();
  }

  private static void collectBoundIdentifiers(
      Expression lhs, ImmutableSet.Builder<Identifier> result) {
    if (lhs instanceof Identifier) {
      result.add((Identifier) lhs);
      return;
    }
    if (lhs instanceof ListLiteral) {
      ListLiteral variables = (ListLiteral) lhs;
      for (Expression expression : variables.getElements()) {
        collectBoundIdentifiers(expression, result);
      }
    }
  }

  @Override
  public void accept(SyntaxTreeVisitor visitor) {
    visitor.visit(this);
  }

  @Override
  public void prettyPrint(Appendable buffer, int indentLevel) throws IOException {
    expr.prettyPrint(buffer);
  }
}
