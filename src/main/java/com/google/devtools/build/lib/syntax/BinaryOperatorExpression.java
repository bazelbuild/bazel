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

import com.google.common.base.Strings;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.syntax.Concatable.Concatter;
import com.google.devtools.build.lib.syntax.SkylarkList.MutableList;
import com.google.devtools.build.lib.syntax.SkylarkList.Tuple;
import java.io.IOException;
import java.util.IllegalFormatException;

/** Syntax node for a binary operator expression. */
public final class BinaryOperatorExpression extends Expression {

  private final Expression lhs;

  private final Expression rhs;

  private final Operator operator;

  public BinaryOperatorExpression(Operator operator, Expression lhs, Expression rhs) {
    this.lhs = lhs;
    this.rhs = rhs;
    this.operator = operator;
  }

  public Expression getLhs() {
    return lhs;
  }

  public Expression getRhs() {
    return rhs;
  }

  /**
   * Returns the operator kind for this binary operation.
   */
  public Operator getOperator() {
    return operator;
  }

  @Override
  public void prettyPrint(Appendable buffer) throws IOException {
    // TODO(bazel-team): Possibly omit parentheses when they are not needed according to operator
    // precedence rules. This requires passing down more contextual information.
    buffer.append('(');
    lhs.prettyPrint(buffer);
    buffer.append(' ');
    buffer.append(operator.toString());
    buffer.append(' ');
    rhs.prettyPrint(buffer);
    buffer.append(')');
  }

  @Override
  public String toString() {
    // This omits the parentheses for brevity, but is not correct in general due to operator
    // precedence rules.
    return lhs + " " + operator + " " + rhs;
  }

  /** Implements comparison operators. */
  private static int compare(Object lval, Object rval, Location location) throws EvalException {
    try {
      return EvalUtils.SKYLARK_COMPARATOR.compare(lval, rval);
    } catch (EvalUtils.ComparisonException e) {
      throw new EvalException(location, e);
    }
  }

  /** Implements the "in" operator. */
  private static boolean in(Object lval, Object rval, Environment env, Location location)
      throws EvalException {
    if (env.getSemantics().incompatibleDepsetIsNotIterable() && rval instanceof SkylarkNestedSet) {
      throw new EvalException(
          location,
          "argument of type '"
              + EvalUtils.getDataTypeName(rval)
              + "' is not iterable. "
              + "in operator only works on lists, tuples, dicts and strings. "
              + "Use --incompatible_depset_is_not_iterable=false to temporarily disable "
              + "this check.");
    } else if (rval instanceof SkylarkQueryable) {
      return ((SkylarkQueryable) rval).containsKey(lval, location);
    } else if (rval instanceof String) {
      if (lval instanceof String) {
        return ((String) rval).contains((String) lval);
      } else {
        throw new EvalException(
            location,
            "'in <string>' requires string as left operand, not '"
                + EvalUtils.getDataTypeName(lval)
                + "'");
      }
    } else {
      throw new EvalException(
          location,
          "argument of type '"
              + EvalUtils.getDataTypeName(rval)
              + "' is not iterable. "
              + "in operator only works on lists, tuples, dicts and strings.");
    }
  }

  /**
   * Evaluates a short-circuiting binary operator, i.e. boolean {@code and} or {@code or}.
   *
   * <p>In contrast to {@link #evaluate}, this method takes unevaluated expressions. The left-hand
   * side expression is evaluated exactly once, and the right-hand side expression is evaluated
   * either once or not at all.
   *
   * @throws IllegalArgumentException if {@code operator} is not {@link Operator#AND} or
   *     {@link Operator#OR}.
   */
  public static Object evaluateWithShortCircuiting(
      Operator operator,
      Expression lhs,
      Expression rhs,
      Environment env,
      Location loc)
      throws EvalException, InterruptedException {
    Object lval = lhs.eval(env);
    if (operator == Operator.AND) {
      return EvalUtils.toBoolean(lval) ? rhs.eval(env) : lval;
    } else if (operator == Operator.OR) {
      return EvalUtils.toBoolean(lval) ? lval : rhs.eval(env);
    } else {
      throw new IllegalArgumentException("Not a short-circuiting operator: " + operator);
    }
  }

  /**
   * Evaluates {@code lhs @ rhs}, where {@code @} is the operator, and returns the result.
   *
   * <p>This method does not implement any short-circuiting logic for boolean operations, as the
   * parameters are already evaluated.
   */
  public static Object evaluate(
      Operator operator,
      Object lhs,
      Object rhs,
      Environment env,
      Location loc)
      throws EvalException, InterruptedException {
    return evaluate(operator, lhs, rhs, env, loc, /*isAugmented=*/false);
  }

  private static Object evaluate(
      Operator operator,
      Object lhs,
      Object rhs,
      Environment env,
      Location location,
      boolean isAugmented)
      throws EvalException, InterruptedException {
    try {
      switch (operator) {
          // AND and OR are included for completeness, but should normally be handled using
          // evaluateWithShortCircuiting() instead of this method.

        case AND:
          return EvalUtils.toBoolean(lhs) ? rhs : lhs;

        case OR:
          return EvalUtils.toBoolean(lhs) ? lhs : rhs;

        case PLUS:
          return plus(lhs, rhs, env, location, isAugmented);

        case PIPE:
          return pipe(lhs, rhs, env, location);

        case MINUS:
          return minus(lhs, rhs, location);

        case MULT:
          return mult(lhs, rhs, env, location);

        case DIVIDE:
          if (env.getSemantics().incompatibleDisallowSlashOperator()) {
            throw new EvalException(
                location,
                "The `/` operator has been removed. Please use the `//` operator for integer "
                    + "division. You can temporarily enable the `/` operator by passing "
                    + "the flag --incompatible_disallow_slash_operator=false");
          }
          return divide(lhs, rhs, location);

        case FLOOR_DIVIDE:
          return divide(lhs, rhs, location);

        case PERCENT:
          return percent(lhs, rhs, location);

        case EQUALS_EQUALS:
          return lhs.equals(rhs);

        case NOT_EQUALS:
          return !lhs.equals(rhs);

        case LESS:
          return compare(lhs, rhs, location) < 0;

        case LESS_EQUALS:
          return compare(lhs, rhs, location) <= 0;

        case GREATER:
          return compare(lhs, rhs, location) > 0;

        case GREATER_EQUALS:
          return compare(lhs, rhs, location) >= 0;

        case IN:
          return in(lhs, rhs, env, location);

        case NOT_IN:
          return !in(lhs, rhs, env, location);

        default:
          throw new AssertionError("Unsupported binary operator: " + operator);
      } // endswitch
    } catch (ArithmeticException e) {
      throw new EvalException(location, e.getMessage());
    }
  }

  /**
   * Evaluates {@code lhs @= rhs} and returns the result, possibly mutating {@code lhs}.
   *
   * <p>Whether or not {@code lhs} is mutated depends on its type. If it is mutated, then it is also
   * the return value.
   */
  public static Object evaluateAugmented(
      Operator operator, Object lhs, Object rhs, Environment env, Location loc)
      throws EvalException, InterruptedException {
    return evaluate(operator, lhs, rhs, env, loc, /*isAugmented=*/ true);
  }

  @Override
  Object doEval(Environment env) throws EvalException, InterruptedException {
    if (operator == Operator.AND || operator == Operator.OR) {
      return evaluateWithShortCircuiting(operator, lhs, rhs, env, getLocation());
    } else {
      return evaluate(operator, lhs.eval(env), rhs.eval(env), env, getLocation());
    }
  }

  @Override
  public void accept(SyntaxTreeVisitor visitor) {
    visitor.visit(this);
  }

  @Override
  public Kind kind() {
    return Kind.BINARY_OPERATOR;
  }

  /** Implements Operator.PLUS. */
  private static Object plus(
      Object lval, Object rval, Environment env, Location location, boolean isAugmented)
      throws EvalException {
    // int + int
    if (lval instanceof Integer && rval instanceof Integer) {
      return Math.addExact((Integer) lval, (Integer) rval);
    }

    // string + string
    if (lval instanceof String && rval instanceof String) {
      return (String) lval + (String) rval;
    }

    if (lval instanceof SelectorValue || rval instanceof SelectorValue
        || lval instanceof SelectorList
        || rval instanceof SelectorList) {
      return SelectorList.concat(location, lval, rval);
    }

    if ((lval instanceof Tuple) && (rval instanceof Tuple)) {
      return Tuple.concat((Tuple<?>) lval, (Tuple<?>) rval);
    }

    if ((lval instanceof MutableList) && (rval instanceof MutableList)) {
      if (isAugmented) {
        @SuppressWarnings("unchecked")
        MutableList<Object> list = (MutableList) lval;
        list.addAll((MutableList<?>) rval, location, env.mutability());
        return list;
      } else {
        return MutableList.concat((MutableList<?>) lval, (MutableList<?>) rval, env.mutability());
      }
    }

    if (lval instanceof SkylarkDict && rval instanceof SkylarkDict) {
      if (env.getSemantics().incompatibleDisallowDictPlus()) {
        throw new EvalException(
            location,
            "The `+` operator for dicts is deprecated and no longer supported. Please use the "
                + "`update` method instead. You can temporarily enable the `+` operator by passing "
                + "the flag --incompatible_disallow_dict_plus=false");
      }
      return SkylarkDict.plus((SkylarkDict<?, ?>) lval, (SkylarkDict<?, ?>) rval, env);
    }

    if (lval instanceof Concatable && rval instanceof Concatable) {
      Concatable lobj = (Concatable) lval;
      Concatable robj = (Concatable) rval;
      Concatter concatter = lobj.getConcatter();
      if (concatter != null && concatter.equals(robj.getConcatter())) {
        return concatter.concat(lobj, robj, location);
      } else {
        throw typeException(lval, rval, Operator.PLUS, location);
      }
    }

    // TODO(bazel-team): Remove deprecated operator.
    if (lval instanceof SkylarkNestedSet) {
      if (env.getSemantics().incompatibleDepsetUnion()) {
        throw new EvalException(
            location,
            "`+` operator on a depset is forbidden. See "
                + "https://docs.bazel.build/versions/master/skylark/depsets.html for "
                + "recommendations. Use --incompatible_depset_union=false "
                + "to temporarily disable this check.");
      }
      return SkylarkNestedSet.of((SkylarkNestedSet) lval, rval, location);
    }
    throw typeException(lval, rval, Operator.PLUS, location);
  }

  /** Implements Operator.PIPE. */
  private static Object pipe(Object lval, Object rval, Environment env, Location location)
      throws EvalException {
    if (lval instanceof SkylarkNestedSet) {
      if (env.getSemantics().incompatibleDepsetUnion()) {
        throw new EvalException(
            location,
            "`|` operator on a depset is forbidden. See "
                + "https://docs.bazel.build/versions/master/skylark/depsets.html for "
                + "recommendations. Use --incompatible_depset_union=false "
                + "to temporarily disable this check.");
      }
      return SkylarkNestedSet.of((SkylarkNestedSet) lval, rval, location);
    }
    throw typeException(lval, rval, Operator.PIPE, location);
  }

  /** Implements Operator.MINUS. */
  private static Object minus(Object lval, Object rval, Location location) throws EvalException {
    if (lval instanceof Integer && rval instanceof Integer) {
      return Math.subtractExact((Integer) lval, (Integer) rval);
    }
    throw typeException(lval, rval, Operator.MINUS, location);
  }

  /** Implements Operator.MULT. */
  private static Object mult(Object lval, Object rval, Environment env, Location location)
      throws EvalException {
    Integer number = null;
    Object otherFactor = null;

    if (lval instanceof Integer) {
      number = (Integer) lval;
      otherFactor = rval;
    } else if (rval instanceof Integer) {
      number = (Integer) rval;
      otherFactor = lval;
    }

    if (number != null) {
      if (otherFactor instanceof Integer) {
        return Math.multiplyExact(number, (Integer) otherFactor);
      } else if (otherFactor instanceof String) {
        // Similar to Python, a factor < 1 leads to an empty string.
        return Strings.repeat((String) otherFactor, Math.max(0, number));
      } else if (otherFactor instanceof SkylarkList && !(otherFactor instanceof RangeList)) {
        // Similar to Python, a factor < 1 leads to an empty string.
        return ((SkylarkList<?>) otherFactor).repeat(number, env.mutability());
      }
    }
    throw typeException(lval, rval, Operator.MULT, location);
  }

  /** Implements Operator.DIVIDE. */
  private static Object divide(Object lval, Object rval, Location location) throws EvalException {
    // int / int
    if (lval instanceof Integer && rval instanceof Integer) {
      if (rval.equals(0)) {
        throw new EvalException(location, "integer division by zero");
      }
      // Integer division doesn't give the same result in Java and in Python 2 with
      // negative numbers.
      // Java:   -7/3 = -2
      // Python: -7/3 = -3
      // We want to follow Python semantics, so we use float division and round down.
      return (int) Math.floor(Double.valueOf((Integer) lval) / (Integer) rval);
    }
    throw typeException(lval, rval, Operator.DIVIDE, location);
  }

  /** Implements Operator.PERCENT. */
  private static Object percent(Object lval, Object rval, Location location)
      throws EvalException {
    // int % int
    if (lval instanceof Integer && rval instanceof Integer) {
      if (rval.equals(0)) {
        throw new EvalException(location, "integer modulo by zero");
      }
      // Python and Java implement division differently, wrt negative numbers.
      // In Python, sign of the result is the sign of the divisor.
      int div = (Integer) rval;
      int result = ((Integer) lval).intValue() % Math.abs(div);
      if (result > 0 && div < 0) {
        result += div; // make the result negative
      } else if (result < 0 && div > 0) {
        result += div; // make the result positive
      }
      return result;
    }

    // string % tuple, string % dict, string % anything-else
    if (lval instanceof String) {
      String pattern = (String) lval;
      try {
        if (rval instanceof Tuple) {
          return Printer.formatWithList(pattern, (Tuple) rval);
        }
        return Printer.format(pattern, rval);
      } catch (IllegalFormatException e) {
        throw new EvalException(location, e.getMessage());
      }
    }
    throw typeException(lval, rval, Operator.PERCENT, location);
  }

  /**
   * Throws an exception signifying incorrect types for the given operator.
   */
  private static EvalException typeException(
      Object lval, Object rval, Operator operator, Location location) {
    // NB: this message format is identical to that used by CPython 2.7.6 or 3.4.0,
    // though python raises a TypeError.
    // For more details, we'll hopefully have usable stack traces at some point.
    return new EvalException(
        location,
        String.format(
            "unsupported operand type(s) for %s: '%s' and '%s'",
            operator,
            EvalUtils.getDataTypeName(lval),
            EvalUtils.getDataTypeName(rval)));
  }
}
