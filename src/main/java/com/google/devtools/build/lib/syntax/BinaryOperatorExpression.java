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
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.syntax.Concatable.Concatter;
import com.google.devtools.build.lib.syntax.SkylarkList.MutableList;
import com.google.devtools.build.lib.syntax.SkylarkList.Tuple;
import java.util.Collections;
import java.util.IllegalFormatException;

/**
 * Syntax node for a binary operator expression.
 */
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
  public String toString() {
    return lhs + " " + operator + " " + rhs;
  }

  /** Implements comparison operators.  */
  private static int compare(Object lval, Object rval, Location location) throws EvalException {
    try {
      return EvalUtils.SKYLARK_COMPARATOR.compare(lval, rval);
    } catch (EvalUtils.ComparisonException e) {
      throw new EvalException(location, e);
    }
  }

  /** Implements the "in" operator. */
  private static boolean in(Object lval, Object rval, Location location, Environment env)
      throws EvalException {
    if (env.getSemantics().incompatibleDepsetIsNotIterable && rval instanceof SkylarkNestedSet) {
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

  /** Helper method. Reused from the LValue class. */
  public static Object evaluate(
      Operator operator,
      Object lval,
      Expression rhs,
      Environment env,
      Location location,
      boolean isAugmented)
      throws EvalException, InterruptedException {
    // Short-circuit operators
    if (operator == Operator.AND) {
      if (EvalUtils.toBoolean(lval)) {
        return rhs.eval(env);
      } else {
        return lval;
      }
    }

    if (operator == Operator.OR) {
      if (EvalUtils.toBoolean(lval)) {
        return lval;
      } else {
        return rhs.eval(env);
      }
    }

    Object rval = rhs.eval(env);

    switch (operator) {
      case PLUS:
        return plus(lval, rval, env, location, isAugmented);

      case PIPE:
        return pipe(lval, rval, location);

      case MINUS:
        return minus(lval, rval, location);

      case MULT:
        return mult(lval, rval, env, location);

      case DIVIDE:
      case FLOOR_DIVIDE:
        return divide(lval, rval, location);

      case PERCENT:
        return percent(lval, rval, location);

      case EQUALS_EQUALS:
        return lval.equals(rval);

      case NOT_EQUALS:
        return !lval.equals(rval);

      case LESS:
        return compare(lval, rval, location) < 0;

      case LESS_EQUALS:
        return compare(lval, rval, location) <= 0;

      case GREATER:
        return compare(lval, rval, location) > 0;

      case GREATER_EQUALS:
        return compare(lval, rval, location) >= 0;

      case IN:
        return in(lval, rval, location, env);

      case NOT_IN:
        return !in(lval, rval, location, env);

      default:
        throw new AssertionError("Unsupported binary operator: " + operator);
    } // endswitch
  }

  @Override
  Object doEval(Environment env) throws EvalException, InterruptedException {
    return evaluate(operator, lhs.eval(env), rhs, env, getLocation(), false);
  }

  @Override
  public void accept(SyntaxTreeVisitor visitor) {
    visitor.visit(this);
  }

  @Override
  void validate(ValidationEnvironment env) throws EvalException {
    lhs.validate(env);
    rhs.validate(env);
  }

  /** Implements Operator.PLUS. */
  private static Object plus(
      Object lval, Object rval, Environment env, Location location, boolean isAugmented)
      throws EvalException {
    // int + int
    if (lval instanceof Integer && rval instanceof Integer) {
      return ((Integer) lval).intValue() + ((Integer) rval).intValue();
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
      return Tuple.copyOf(Iterables.concat((Tuple) lval, (Tuple) rval));
    }

    if ((lval instanceof MutableList) && (rval instanceof MutableList)) {
      if (isAugmented && env.getSemantics().incompatibleListPlusEqualsInplace) {
        @SuppressWarnings("unchecked")
        MutableList<Object> list = (MutableList) lval;
        list.addAll((MutableList<?>) rval, location, env);
        return list;
      }
      return MutableList.concat((MutableList) lval, (MutableList) rval, env);
    }

    if (lval instanceof SkylarkDict && rval instanceof SkylarkDict) {
      if (env.getSemantics().incompatibleDisallowDictPlus) {
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

    // TODO(bazel-team): Remove this case. Union of sets should use '|' instead of '+'.
    if (lval instanceof SkylarkNestedSet) {
      return new SkylarkNestedSet((SkylarkNestedSet) lval, rval, location);
    }
    throw typeException(lval, rval, Operator.PLUS, location);
  }

  /** Implements Operator.PIPE. */
  private static Object pipe(Object lval, Object rval, Location location) throws EvalException {
    if (lval instanceof SkylarkNestedSet) {
      return new SkylarkNestedSet((SkylarkNestedSet) lval, rval, location);
    }
    throw typeException(lval, rval, Operator.PIPE, location);
  }

  /** Implements Operator.MINUS. */
  private static Object minus(Object lval, Object rval, Location location) throws EvalException {
    if (lval instanceof Integer && rval instanceof Integer) {
      return ((Integer) lval).intValue() - ((Integer) rval).intValue();
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
        return number.intValue() * ((Integer) otherFactor).intValue();
      } else if (otherFactor instanceof String) {
        // Similar to Python, a factor < 1 leads to an empty string.
        return Strings.repeat((String) otherFactor, Math.max(0, number.intValue()));
      } else if (otherFactor instanceof MutableList) {
        // Similar to Python, a factor < 1 leads to an empty string.
        return MutableList.duplicate(
            (MutableList<?>) otherFactor, Math.max(0, number.intValue()), env);
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
  private static Object percent(Object lval, Object rval, Location location) throws EvalException {
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
          return Printer.formatToString(pattern, (Tuple) rval);
        }
        return Printer.formatToString(pattern, Collections.singletonList(rval));
      } catch (IllegalFormatException e) {
        throw new EvalException(location, e.getMessage());
      }
    }
    throw typeException(lval, rval, Operator.PERCENT, location);
  }

  /**
   * Throws an exception signifying incorrect types for the given operator.
   */
  private static final EvalException typeException(
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
