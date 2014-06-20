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

import com.google.common.collect.Lists;

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

  public BinaryOperatorExpression(Operator operator,
                                  Expression lhs,
                                  Expression rhs) {
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
    if (!(lval instanceof Comparable)) {
      throw new EvalException(getLocation(), lval + " is not comparable");
    }
    try {
      return ((Comparable) lval).compareTo(rval);
    } catch (ClassCastException e) {
      throw new EvalException(getLocation(), "Cannot compare " + EvalUtils.getDatatypeName(lval)
          + " with " + EvalUtils.getDatatypeName(rval));
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
                + EvalUtils.getDatatypeName(rlist) + " (not \""
                + EvalUtils.getDatatypeName(llist) + "\") to "
                + EvalUtils.getDatatypeName(rlist));
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
        break;
      }

      case MINUS: {
        if (lval instanceof Integer && rval instanceof Integer) {
          return ((Integer) lval).intValue() - ((Integer) rval).intValue();
        }
        break;
      }

      case PERCENT: {
        // int % int
        if (lval instanceof Integer && rval instanceof Integer) {
          return ((Integer) lval).intValue() % ((Integer) rval).intValue();
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
        if (rval instanceof Collection<?>) {
          return ((Collection<?>) rval).contains(lval);
        } else if (rval instanceof Map<?, ?>) {
          return ((Map<?, ?>) rval).containsKey(lval);
        } else if (rval instanceof String) {
          if (lval instanceof String) {
            return ((String) rval).contains((String) lval);
          } else {
            throw new EvalException(getLocation(),
                "in operator only works on strings if the left operand is also a string");
          }
        } else {
          throw new EvalException(getLocation(),
              "in operator only works on lists, tuples, dictionaries and strings");
        }
      }

      default: {
        throw new AssertionError("Unsupported binary operator: " + operator);
      }
    } // endswitch

    throw new EvalException(getLocation(),
        "unsupported operand types for '" + operator + "': '"
        + EvalUtils.getDatatypeName(lval) + "' and '"
        + EvalUtils.getDatatypeName(rval) + "'");
  }

  @Override
  public void accept(SyntaxTreeVisitor visitor) {
    visitor.visit(this);
  }

  @Override
  Class<?> validate(ValidationEnvironment env) throws EvalException {
    Class<?> ltype = lhs.validate(env);
    Class<?> rtype = rhs.validate(env);

    switch (operator) {
      case AND: {
        return Object.class;
      }

      case OR: {
        return Object.class;
      }

      case PLUS: {
        // int + int
        if (ltype.equals(Integer.class) && rtype.equals(Integer.class)) {
          return Integer.class;
        }

        // string + string
        if (ltype.equals(String.class) && rtype.equals(String.class)) {
          return String.class;
        }

        // list + list, tuple + tuple (list + tuple, tuple + list => error)
        if (List.class.isAssignableFrom(ltype) && List.class.isAssignableFrom(rtype)) {
          if (EvalUtils.isTuple(ltype) != EvalUtils.isTuple(rtype)) {
            throw new EvalException(getLocation(), "can only concatenate "
                + EvalUtils.getDataTypeNameFromClass(rtype) + " (not \""
                + EvalUtils.getDataTypeNameFromClass(ltype) + "\") to "
                + EvalUtils.getDataTypeNameFromClass(rtype));
          }
          return ltype;
        }
        break;
      }

      case MINUS: {
        if (ltype.equals(Integer.class) && rtype.equals(Integer.class)) {
          return Integer.class;
        }
        break;
      }

      case PERCENT: {
        // int % int
        if (ltype.equals(Integer.class) && rtype.equals(Integer.class)) {
          return Integer.class;
        }

        // string % tuple, string % dict, string % anything-else
        if (ltype.equals(String.class)) {
          return String.class;
        }
        break;
      }

      case EQUALS_EQUALS:
      case NOT_EQUALS:
      case LESS:
      case LESS_EQUALS:
      case GREATER:
      case GREATER_EQUALS: {
        if (!ltype.equals(Object.class) && !(Comparable.class.isAssignableFrom(ltype))) {
          throw new EvalException(getLocation(),
              EvalUtils.getDataTypeNameFromClass(ltype) + " is not comparable");
        }
        if (!ltype.equals(Object.class) && !rtype.equals(Object.class) && !ltype.equals(rtype)) {
          throw new EvalException(getLocation(),
              String.format("cannot compare a %s to a(n) %s",
                  EvalUtils.getDataTypeNameFromClass(ltype),
                  EvalUtils.getDataTypeNameFromClass(rtype)));
        }
        return Boolean.class;
      }

      case IN: {
        if (Collection.class.isAssignableFrom(rtype)
            || Map.class.isAssignableFrom(rtype)
            || String.class.isAssignableFrom(rtype)) {
          return Boolean.class;
        } else {
          if (!rtype.equals(Object.class)) {
            throw new EvalException(getLocation(), String.format(
                "operand 'in' only works on strings, dictionaries, lists or tuples, not on a(n) %s",
                EvalUtils.getDataTypeNameFromClass(rtype)));
          }
        }
      }
    } // endswitch

    if (!ltype.equals(Object.class) && !rtype.equals(Object.class)) {
      throw new EvalException(getLocation(),
          "unsupported operand types for '" + operator + "': '"
          + EvalUtils.getDataTypeNameFromClass(ltype) + "' and '"
          + EvalUtils.getDataTypeNameFromClass(rtype) + "'");
    }
    return Object.class;
  }
}
