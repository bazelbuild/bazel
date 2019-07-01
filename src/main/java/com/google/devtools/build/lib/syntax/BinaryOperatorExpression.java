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
import java.util.EnumSet;
import java.util.IllegalFormatException;

/** A BinaryExpression represents a binary operator expression 'x op y'. */
public final class BinaryOperatorExpression extends Expression {

  private final Expression x;
  private final TokenKind op; // one of 'operators'
  private final Expression y;

  /** operators is the set of valid binary operators. */
  public static final EnumSet<TokenKind> operators =
      EnumSet.of(
          TokenKind.AND,
          TokenKind.EQUALS_EQUALS,
          TokenKind.GREATER,
          TokenKind.GREATER_EQUALS,
          TokenKind.IN,
          TokenKind.LESS,
          TokenKind.LESS_EQUALS,
          TokenKind.MINUS,
          TokenKind.NOT_EQUALS,
          TokenKind.NOT_IN,
          TokenKind.OR,
          TokenKind.PERCENT,
          TokenKind.SLASH,
          TokenKind.SLASH_SLASH,
          TokenKind.PLUS,
          TokenKind.PIPE,
          TokenKind.STAR);

  public BinaryOperatorExpression(Expression x, TokenKind op, Expression y) {
    this.x = x;
    this.op = op;
    this.y = y;
  }

  /** getX returns the left operand. */
  public Expression getX() {
    return x;
  }

  /** getOperator returns the operator. */
  public TokenKind getOperator() {
    return op;
  }

  /** getY returns the right operand. */
  public Expression getY() {
    return y;
  }

  @Override
  public void prettyPrint(Appendable buffer) throws IOException {
    // TODO(bazel-team): retain parentheses in the syntax tree so we needn't
    // conservatively emit them here.
    buffer.append('(');
    x.prettyPrint(buffer);
    buffer.append(' ');
    buffer.append(op.toString());
    buffer.append(' ');
    y.prettyPrint(buffer);
    buffer.append(')');
  }

  @Override
  public String toString() {
    // This omits the parentheses for brevity, but is not correct in general due to operator
    // precedence rules.
    return x + " " + op + " " + y;
  }

  /** Implements comparison operators. */
  private static int compare(Object x, Object y, Location location) throws EvalException {
    try {
      return EvalUtils.SKYLARK_COMPARATOR.compare(x, y);
    } catch (EvalUtils.ComparisonException e) {
      throw new EvalException(location, e);
    }
  }

  /** Implements 'x in y'. */
  private static boolean in(Object x, Object y, Environment env, Location location)
      throws EvalException {
    if (env.getSemantics().incompatibleDepsetIsNotIterable() && y instanceof SkylarkNestedSet) {
      throw new EvalException(
          location,
          "argument of type '"
              + EvalUtils.getDataTypeName(y)
              + "' is not iterable. "
              + "in operator only works on lists, tuples, dicts and strings. "
              + "Use --incompatible_depset_is_not_iterable=false to temporarily disable "
              + "this check.");
    } else if (y instanceof SkylarkQueryable) {
      return ((SkylarkQueryable) y).containsKey(x, location, env.getStarlarkContext());
    } else if (y instanceof String) {
      if (x instanceof String) {
        return ((String) y).contains((String) x);
      } else {
        throw new EvalException(
            location,
            "'in <string>' requires string as left operand, not '"
                + EvalUtils.getDataTypeName(x)
                + "'");
      }
    } else {
      throw new EvalException(
          location,
          "argument of type '"
              + EvalUtils.getDataTypeName(y)
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
   * @throws IllegalArgumentException if {@code op} is not {@link Operator#AND} or {@link
   *     Operator#OR}.
   */
  public static Object evaluateWithShortCircuiting(
      TokenKind op, Expression x, Expression y, Environment env, Location loc)
      throws EvalException, InterruptedException {
    Object xval = x.eval(env);
    switch (op) {
      case AND:
        return EvalUtils.toBoolean(xval) ? y.eval(env) : xval;
      case OR:
        return EvalUtils.toBoolean(xval) ? xval : y.eval(env);
      default:
        throw new IllegalArgumentException("Not a short-circuiting operator: " + op);
    }
  }

  /**
   * Evaluates {@code x @ y}, where {@code @} is the operator, and returns the result.
   *
   * <p>This method does not implement any short-circuiting logic for boolean operations, as the
   * parameters are already evaluated.
   */
  public static Object evaluate(TokenKind op, Object x, Object y, Environment env, Location loc)
      throws EvalException, InterruptedException {
    return evaluate(op, x, y, env, loc, /*isAugmented=*/ false);
  }

  private static Object evaluate(
      TokenKind op, Object x, Object y, Environment env, Location location, boolean isAugmented)
      throws EvalException, InterruptedException {
    try {
      switch (op) {
          // AND and OR are included for completeness, but should normally be handled using
          // evaluateWithShortCircuiting() instead of this method.

        case AND:
          return EvalUtils.toBoolean(x) ? y : x;

        case OR:
          return EvalUtils.toBoolean(x) ? x : y;

        case PLUS:
          return plus(x, y, env, location, isAugmented);

        case PIPE:
          return pipe(x, y, env, location);

        case MINUS:
          return minus(x, y, location);

        case STAR:
          return mult(x, y, env, location);

        case SLASH:
          throw new EvalException(
              location,
              "The `/` operator is not allowed. Please use the `//` operator for integer "
                  + "division.");

        case SLASH_SLASH:
          return divide(x, y, location);

        case PERCENT:
          return percent(x, y, location);

        case EQUALS_EQUALS:
          return x.equals(y);

        case NOT_EQUALS:
          return !x.equals(y);

        case LESS:
          return compare(x, y, location) < 0;

        case LESS_EQUALS:
          return compare(x, y, location) <= 0;

        case GREATER:
          return compare(x, y, location) > 0;

        case GREATER_EQUALS:
          return compare(x, y, location) >= 0;

        case IN:
          return in(x, y, env, location);

        case NOT_IN:
          return !in(x, y, env, location);

        default:
          throw new AssertionError("Unsupported binary operator: " + op);
      } // endswitch
    } catch (ArithmeticException e) {
      throw new EvalException(location, e.getMessage());
    }
  }

  /**
   * Evaluates {@code x @= y} and returns the result, possibly mutating {@code x}.
   *
   * <p>Whether or not {@code x} is mutated depends on its type. If it is mutated, then it is also
   * the return value.
   */
  public static Object evaluateAugmented(
      TokenKind op, Object x, Object y, Environment env, Location loc)
      throws EvalException, InterruptedException {
    return evaluate(op, x, y, env, loc, /*isAugmented=*/ true);
  }

  @Override
  Object doEval(Environment env) throws EvalException, InterruptedException {
    if (op == TokenKind.AND || op == TokenKind.OR) {
      return evaluateWithShortCircuiting(op, x, y, env, getLocation());
    } else {
      return evaluate(op, x.eval(env), y.eval(env), env, getLocation());
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

  /** Implements 'x + y'. */
  private static Object plus(
      Object x, Object y, Environment env, Location location, boolean isAugmented)
      throws EvalException {
    // int + int
    if (x instanceof Integer && y instanceof Integer) {
      return Math.addExact((Integer) x, (Integer) y);
    }

    // string + string
    if (x instanceof String && y instanceof String) {
      return (String) x + (String) y;
    }

    if (x instanceof SelectorValue
        || y instanceof SelectorValue
        || x instanceof SelectorList
        || y instanceof SelectorList) {
      return SelectorList.concat(location, x, y);
    }

    if (x instanceof Tuple && y instanceof Tuple) {
      return Tuple.concat((Tuple<?>) x, (Tuple<?>) y);
    }

    if (x instanceof MutableList && y instanceof MutableList) {
      if (isAugmented) {
        @SuppressWarnings("unchecked")
        MutableList<Object> list = (MutableList) x;
        list.addAll((MutableList<?>) y, location, env.mutability());
        return list;
      } else {
        return MutableList.concat((MutableList<?>) x, (MutableList<?>) y, env.mutability());
      }
    }

    if (x instanceof SkylarkDict && y instanceof SkylarkDict) {
      if (env.getSemantics().incompatibleDisallowDictPlus()) {
        throw new EvalException(
            location,
            "The `+` operator for dicts is deprecated and no longer supported. Please use the "
                + "`update` method instead. You can temporarily enable the `+` operator by passing "
                + "the flag --incompatible_disallow_dict_plus=false");
      }
      return SkylarkDict.plus((SkylarkDict<?, ?>) x, (SkylarkDict<?, ?>) y, env);
    }

    if (x instanceof Concatable && y instanceof Concatable) {
      Concatable lobj = (Concatable) x;
      Concatable robj = (Concatable) y;
      Concatter concatter = lobj.getConcatter();
      if (concatter != null && concatter.equals(robj.getConcatter())) {
        return concatter.concat(lobj, robj, location);
      } else {
        throw typeException(x, y, TokenKind.PLUS, location);
      }
    }

    // TODO(bazel-team): Remove deprecated operator.
    if (x instanceof SkylarkNestedSet) {
      if (env.getSemantics().incompatibleDepsetUnion()) {
        throw new EvalException(
            location,
            "`+` operator on a depset is forbidden. See "
                + "https://docs.bazel.build/versions/master/skylark/depsets.html for "
                + "recommendations. Use --incompatible_depset_union=false "
                + "to temporarily disable this check.");
      }
      return SkylarkNestedSet.of((SkylarkNestedSet) x, y, location);
    }
    throw typeException(x, y, TokenKind.PLUS, location);
  }

  /** Implements 'x | y'. */
  private static Object pipe(Object x, Object y, Environment env, Location location)
      throws EvalException {
    if (x instanceof SkylarkNestedSet) {
      if (env.getSemantics().incompatibleDepsetUnion()) {
        throw new EvalException(
            location,
            "`|` operator on a depset is forbidden. See "
                + "https://docs.bazel.build/versions/master/skylark/depsets.html for "
                + "recommendations. Use --incompatible_depset_union=false "
                + "to temporarily disable this check.");
      }
      return SkylarkNestedSet.of((SkylarkNestedSet) x, y, location);
    }
    throw typeException(x, y, TokenKind.PIPE, location);
  }

  /** Implements 'x - y'. */
  private static Object minus(Object x, Object y, Location location) throws EvalException {
    if (x instanceof Integer && y instanceof Integer) {
      return Math.subtractExact((Integer) x, (Integer) y);
    }
    throw typeException(x, y, TokenKind.MINUS, location);
  }

  /** Implements 'x * y'. */
  private static Object mult(Object x, Object y, Environment env, Location location)
      throws EvalException {
    Integer number = null;
    Object otherFactor = null;

    if (x instanceof Integer) {
      number = (Integer) x;
      otherFactor = y;
    } else if (y instanceof Integer) {
      number = (Integer) y;
      otherFactor = x;
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
    throw typeException(x, y, TokenKind.STAR, location);
  }

  /** Implements 'x // y'. */
  private static Object divide(Object x, Object y, Location location) throws EvalException {
    // int / int
    if (x instanceof Integer && y instanceof Integer) {
      if (y.equals(0)) {
        throw new EvalException(location, "integer division by zero");
      }
      // Integer division doesn't give the same result in Java and in Python 2 with
      // negative numbers.
      // Java:   -7/3 = -2
      // Python: -7/3 = -3
      // We want to follow Python semantics, so we use float division and round down.
      return (int) Math.floor(Double.valueOf((Integer) x) / (Integer) y);
    }
    throw typeException(x, y, TokenKind.SLASH_SLASH, location);
  }

  /** Implements 'x % y'. */
  private static Object percent(Object x, Object y, Location location) throws EvalException {
    // int % int
    if (x instanceof Integer && y instanceof Integer) {
      if (y.equals(0)) {
        throw new EvalException(location, "integer modulo by zero");
      }
      // Python and Java implement division differently, wrt negative numbers.
      // In Python, sign of the result is the sign of the divisor.
      int div = (Integer) y;
      int result = ((Integer) x).intValue() % Math.abs(div);
      if (result > 0 && div < 0) {
        result += div; // make the result negative
      } else if (result < 0 && div > 0) {
        result += div; // make the result positive
      }
      return result;
    }

    // string % tuple, string % dict, string % anything-else
    if (x instanceof String) {
      String pattern = (String) x;
      try {
        if (y instanceof Tuple) {
          return Printer.formatWithList(pattern, (Tuple) y);
        }
        return Printer.format(pattern, y);
      } catch (IllegalFormatException e) {
        throw new EvalException(location, e.getMessage());
      }
    }
    throw typeException(x, y, TokenKind.PERCENT, location);
  }

  /** Throws an exception signifying incorrect types for the given operator. */
  private static EvalException typeException(Object x, Object y, TokenKind op, Location location) {
    // NB: this message format is identical to that used by CPython 2.7.6 or 3.4.0,
    // though python raises a TypeError.
    // For more details, we'll hopefully have usable stack traces at some point.
    return new EvalException(
        location,
        String.format(
            "unsupported operand type(s) for %s: '%s' and '%s'",
            op, EvalUtils.getDataTypeName(x), EvalUtils.getDataTypeName(y)));
  }
}
