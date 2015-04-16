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

import com.google.common.base.Strings;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.syntax.ClassObject.SkylarkClassObject;

import java.util.Collection;
import java.util.Collections;
import java.util.IllegalFormatException;
import java.util.List;
import java.util.Map;

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

  private int compare(Object lval, Object rval) throws EvalException {
    try {
      return EvalUtils.SKYLARK_COMPARATOR.compare(lval, rval);
    } catch (EvalUtils.ComparisonException e) {
      throw new EvalException(getLocation(), e);
    }
  }

  @Override
  Object eval(Environment env) throws EvalException, InterruptedException {
    Object lval = lhs.eval(env);

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
      case PLUS: {
        // int + int
        if (lval instanceof Integer && rval instanceof Integer) {
          return ((Integer) lval).intValue() + ((Integer) rval).intValue();
        }

        // string + string
        if (lval instanceof String && rval instanceof String) {
          return (String) lval + (String) rval;
        }

        // list + list, tuple + tuple (list + tuple, tuple + list => error)
        if (lval instanceof List<?> && rval instanceof List<?>) {
          List<?> llist = (List<?>) lval;
          List<?> rlist = (List<?>) rval;
          if (EvalUtils.isImmutable(llist) != EvalUtils.isImmutable(rlist)) {
            throw new EvalException(getLocation(), "can only concatenate "
                + EvalUtils.getDataTypeName(rlist) + " (not \""
                + EvalUtils.getDataTypeName(llist) + "\") to "
                + EvalUtils.getDataTypeName(rlist));
          }
          if (llist instanceof GlobList<?> || rlist instanceof GlobList<?>) {
            return GlobList.concat(llist, rlist);
          } else {
            List<Object> result = Lists.newArrayListWithCapacity(llist.size() + rlist.size());
            result.addAll(llist);
            result.addAll(rlist);
            return EvalUtils.makeSequence(result, EvalUtils.isImmutable(llist));
          }
        }

        if (lval instanceof SelectorValue || rval instanceof SelectorValue
            || lval instanceof SelectorList || rval instanceof SelectorList) {
          return SelectorList.concat(getLocation(), lval, rval);
        }

        if (lval instanceof SkylarkList && rval instanceof SkylarkList) {
          return SkylarkList.concat((SkylarkList) lval, (SkylarkList) rval, getLocation());
        }

        if (lval instanceof Map<?, ?> && rval instanceof Map<?, ?>) {
          Map<?, ?> ldict = (Map<?, ?>) lval;
          Map<?, ?> rdict = (Map<?, ?>) rval;
          Map<Object, Object> result = Maps.newHashMapWithExpectedSize(ldict.size() + rdict.size());
          result.putAll(ldict);
          result.putAll(rdict);
          return ImmutableMap.copyOf(result);
        }

        if (lval instanceof SkylarkClassObject && rval instanceof SkylarkClassObject) {
          return SkylarkClassObject.concat(
              (SkylarkClassObject) lval, (SkylarkClassObject) rval, getLocation());
        }

        if (lval instanceof SkylarkNestedSet) {
          return new SkylarkNestedSet((SkylarkNestedSet) lval, rval, getLocation());
        }
        break;
      }

      case MINUS: {
        if (lval instanceof Integer && rval instanceof Integer) {
          return ((Integer) lval).intValue() - ((Integer) rval).intValue();
        }
        break;
      }

      case MULT: {
        // int * int
        if (lval instanceof Integer && rval instanceof Integer) {
          return ((Integer) lval).intValue() * ((Integer) rval).intValue();
        }

        // string * int
        if (lval instanceof String && rval instanceof Integer) {
          return Strings.repeat((String) lval, ((Integer) rval).intValue());
        }

        // int * string
        if (lval instanceof Integer && rval instanceof String) {
          return Strings.repeat((String) rval, ((Integer) lval).intValue());
        }
        break;
      }

      case DIVIDE: {
        // int / int
        if (lval instanceof Integer && rval instanceof Integer) {
          if (rval.equals(0)) {
            throw new EvalException(getLocation(), "integer division by zero");
          }
          // Integer division doesn't give the same result in Java and in Python 2 with
          // negative numbers.
          // Java:   -7/3 = -2
          // Python: -7/3 = -3
          // We want to follow Python semantics, so we use float division and round down.
          return (int) Math.floor(new Double((Integer) lval) / (Integer) rval);
        }
      }

      case PERCENT: {
        // int % int
        if (lval instanceof Integer && rval instanceof Integer) {
          if (rval.equals(0)) {
            throw new EvalException(getLocation(), "integer modulo by zero");
          }
          // Python and Java implement division differently, wrt negative numbers.
          // In Python, sign of the result is the sign of the divisor.
          int div = (Integer) rval;
          int result = ((Integer) lval).intValue() % Math.abs(div);
          if (result > 0 && div < 0) {
            result += div;  // make the result negative
          } else if (result < 0 && div > 0) {
            result += div;  // make the result positive
          }
          return result;
        }

        // string % tuple, string % dict, string % anything-else
        if (lval instanceof String) {
          try {
            String pattern = (String) lval;
            if (rval instanceof List<?>) {
              List<?> rlist = (List<?>) rval;
              if (EvalUtils.isTuple(rlist)) {
                return EvalUtils.formatString(pattern, rlist);
              }
              /* string % list: fall thru */
            }
            if (rval instanceof SkylarkList) {
              SkylarkList rlist = (SkylarkList) rval;
              if (rlist.isTuple()) {
                return EvalUtils.formatString(pattern, rlist.toList());
              }
            }

            return EvalUtils.formatString(pattern,
                                          Collections.singletonList(rval));
          } catch (IllegalFormatException e) {
            throw new EvalException(getLocation(), e.getMessage());
          }
        }
        break;
      }

      case EQUALS_EQUALS: {
        return lval.equals(rval);
      }

      case NOT_EQUALS: {
        return !lval.equals(rval);
      }

      case LESS: {
        return compare(lval, rval) < 0;
      }

      case LESS_EQUALS: {
        return compare(lval, rval) <= 0;
      }

      case GREATER: {
        return compare(lval, rval) > 0;
      }

      case GREATER_EQUALS: {
        return compare(lval, rval) >= 0;
      }

      case IN: {
        if (rval instanceof SkylarkList) {
          for (Object obj : (SkylarkList) rval) {
            if (obj.equals(lval)) {
              return true;
            }
          }
          return false;
        } else if (rval instanceof Collection<?>) {
          return ((Collection<?>) rval).contains(lval);
        } else if (rval instanceof Map<?, ?>) {
          return ((Map<?, ?>) rval).containsKey(lval);
        } else if (rval instanceof SkylarkNestedSet) {
          return ((SkylarkNestedSet) rval).expandedSet().contains(lval);
        } else if (rval instanceof String) {
          if (lval instanceof String) {
            return ((String) rval).contains((String) lval);
          } else {
            throw new EvalException(getLocation(),
                "in operator only works on strings if the left operand is also a string");
          }
        } else {
          throw new EvalException(getLocation(),
              "in operator only works on lists, tuples, sets, dicts and strings");
        }
      }

      default: {
        throw new AssertionError("Unsupported binary operator: " + operator);
      }
    } // endswitch

    // NB: this message format is identical to that used by CPython 2.7.6 or 3.4.0,
    // though python raises a TypeError.
    // For more details, we'll hopefully have usable stack traces at some point.
    throw new EvalException(getLocation(),
        String.format("unsupported operand type(s) for %s: '%s' and '%s'",
            operator, EvalUtils.getDataTypeName(lval), EvalUtils.getDataTypeName(rval)));
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
}
