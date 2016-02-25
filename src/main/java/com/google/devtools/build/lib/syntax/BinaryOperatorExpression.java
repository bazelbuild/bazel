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

import static com.google.devtools.build.lib.syntax.compiler.ByteCodeUtils.append;

import com.google.common.base.Strings;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.syntax.ClassObject.SkylarkClassObject;
import com.google.devtools.build.lib.syntax.SkylarkList.MutableList;
import com.google.devtools.build.lib.syntax.SkylarkList.Tuple;
import com.google.devtools.build.lib.syntax.compiler.ByteCodeMethodCalls;
import com.google.devtools.build.lib.syntax.compiler.ByteCodeUtils;
import com.google.devtools.build.lib.syntax.compiler.DebugInfo;
import com.google.devtools.build.lib.syntax.compiler.DebugInfo.AstAccessors;
import com.google.devtools.build.lib.syntax.compiler.Jump;
import com.google.devtools.build.lib.syntax.compiler.Jump.PrimitiveComparison;
import com.google.devtools.build.lib.syntax.compiler.LabelAdder;
import com.google.devtools.build.lib.syntax.compiler.VariableScope;

import net.bytebuddy.implementation.bytecode.ByteCodeAppender;
import net.bytebuddy.implementation.bytecode.Duplication;
import net.bytebuddy.implementation.bytecode.Removal;
import net.bytebuddy.implementation.bytecode.StackManipulation;

import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumSet;
import java.util.IllegalFormatException;
import java.util.List;

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

  /**
   * Implements comparison operators.
   *
   * <p>Publicly accessible for reflection and compiled Skylark code.
   */
  public static int compare(Object lval, Object rval, Location location) throws EvalException {
    try {
      return EvalUtils.SKYLARK_COMPARATOR.compare(lval, rval);
    } catch (EvalUtils.ComparisonException e) {
      throw new EvalException(location, e);
    }
  }

  /**
   * Implements the "in" operator.
   *
   * <p>Publicly accessible for reflection and compiled Skylark code.
   */
  public static boolean in(Object lval, Object rval, Location location) throws EvalException {
    if (rval instanceof SkylarkList) {
      for (Object obj : (SkylarkList) rval) {
        if (obj.equals(lval)) {
          return true;
        }
      }
      return false;
    } else if (rval instanceof SkylarkDict) {
      return ((SkylarkDict<?, ?>) rval).containsKey(lval);
    } else if (rval instanceof SkylarkNestedSet) {
      return ((SkylarkNestedSet) rval).expandedSet().contains(lval);
    } else if (rval instanceof String) {
      if (lval instanceof String) {
        return ((String) rval).contains((String) lval);
      } else {
        throw new EvalException(
            location, "in operator only works on strings if the left operand is also a string");
      }
    } else {
      throw new EvalException(
          location, "in operator only works on lists, tuples, sets, dicts and strings");
    }
  }

  @Override
  Object doEval(Environment env) throws EvalException, InterruptedException {
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
      case PLUS:
        return plus(lval, rval, env, getLocation());

      case PIPE:
        return pipe(lval, rval, getLocation());

      case MINUS:
        return minus(lval, rval, getLocation());

      case MULT:
        return mult(lval, rval, getLocation());

      case DIVIDE:
        return divide(lval, rval, getLocation());

      case PERCENT:
        return percent(lval, rval, getLocation());

      case EQUALS_EQUALS:
        return lval.equals(rval);

      case NOT_EQUALS:
        return !lval.equals(rval);

      case LESS:
        return compare(lval, rval, getLocation()) < 0;

      case LESS_EQUALS:
        return compare(lval, rval, getLocation()) <= 0;

      case GREATER:
        return compare(lval, rval, getLocation()) > 0;

      case GREATER_EQUALS:
        return compare(lval, rval, getLocation()) >= 0;

      case IN:
        return in(lval, rval, getLocation());

      case NOT_IN:
        return !in(lval, rval, getLocation());

      default:
        throw new AssertionError("Unsupported binary operator: " + operator);
    } // endswitch
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

  @Override
  ByteCodeAppender compile(VariableScope scope, DebugInfo debugInfo) throws EvalException {
    AstAccessors debugAccessors = debugInfo.add(this);
    List<ByteCodeAppender> code = new ArrayList<>();
    ByteCodeAppender leftCompiled = lhs.compile(scope, debugInfo);
    ByteCodeAppender rightCompiled = rhs.compile(scope, debugInfo);
    // generate byte code for short-circuiting operators
    if (EnumSet.of(Operator.AND, Operator.OR).contains(operator)) {
      LabelAdder after = new LabelAdder();
      code.add(leftCompiled);
      append(
          code,
          // duplicate the value, one to convert to boolean, one to leave on stack
          // assumes we don't compile Skylark values to long/double
          Duplication.SINGLE,
          EvalUtils.toBoolean,
          // short-circuit and jump behind second operand expression if first is false/true
          Jump.ifIntOperandToZero(
                  operator == Operator.AND
                      ? PrimitiveComparison.EQUAL
                      : PrimitiveComparison.NOT_EQUAL)
              .to(after),
          // remove the duplicated value from above, as only the rhs is still relevant
          Removal.SINGLE);
      code.add(rightCompiled);
      append(code, after);
    } else if (EnumSet.of(
            Operator.LESS, Operator.LESS_EQUALS, Operator.GREATER, Operator.GREATER_EQUALS)
        .contains(operator)) {
      compileComparison(debugAccessors, code, leftCompiled, rightCompiled);
    } else {
      code.add(leftCompiled);
      code.add(rightCompiled);
      switch (operator) {
        case PLUS:
          append(code, callImplementation(scope, debugAccessors, operator));
          break;
        case PIPE:
        case MINUS:
        case MULT:
        case DIVIDE:
        case PERCENT:
          append(code, callImplementation(debugAccessors, operator));
          break;
        case EQUALS_EQUALS:
          append(code, ByteCodeMethodCalls.BCObject.equals, ByteCodeMethodCalls.BCBoolean.valueOf);
          break;
        case NOT_EQUALS:
          append(
              code,
              ByteCodeMethodCalls.BCObject.equals,
              ByteCodeUtils.intLogicalNegation(),
              ByteCodeMethodCalls.BCBoolean.valueOf);
          break;
        case IN:
          append(
              code,
              callImplementation(debugAccessors, operator),
              ByteCodeMethodCalls.BCBoolean.valueOf);
          break;
        case NOT_IN:
          append(
              code,
              callImplementation(debugAccessors, Operator.IN),
              ByteCodeUtils.intLogicalNegation(),
              ByteCodeMethodCalls.BCBoolean.valueOf);
          break;
        default:
          throw new UnsupportedOperationException("Unsupported binary operator: " + operator);
      } // endswitch
    }
    return ByteCodeUtils.compoundAppender(code);
  }

  /**
   * Compile a comparison oer
   * @param debugAccessors
   * @param code
   * @param leftCompiled
   * @param rightCompiled
   * @throws Error
   */
  private void compileComparison(
      AstAccessors debugAccessors,
      List<ByteCodeAppender> code,
      ByteCodeAppender leftCompiled,
      ByteCodeAppender rightCompiled)
      throws Error {
    PrimitiveComparison byteCodeOperator = PrimitiveComparison.forOperator(operator);
    code.add(leftCompiled);
    code.add(rightCompiled);
    append(
        code,
        debugAccessors.loadLocation,
        ByteCodeUtils.invoke(
            BinaryOperatorExpression.class, "compare", Object.class, Object.class, Location.class),
        ByteCodeUtils.intToPrimitiveBoolean(byteCodeOperator),
        ByteCodeMethodCalls.BCBoolean.valueOf);
  }

  /**
   * Implements Operator.PLUS.
   *
   * <p>Publicly accessible for reflection and compiled Skylark code.
   */
  public static Object plus(Object lval, Object rval, Environment env, Location location)
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
      return MutableList.concat((MutableList) lval, (MutableList) rval, env);
    }

    if (lval instanceof SkylarkDict && rval instanceof SkylarkDict) {
      return SkylarkDict.plus((SkylarkDict<?, ?>) lval, (SkylarkDict<?, ?>) rval, env);
    }

    if (lval instanceof SkylarkClassObject && rval instanceof SkylarkClassObject) {
      return SkylarkClassObject.concat(
          (SkylarkClassObject) lval, (SkylarkClassObject) rval, location);
    }

    // TODO(bazel-team): Remove this case. Union of sets should use '|' instead of '+'.
    if (lval instanceof SkylarkNestedSet) {
      return new SkylarkNestedSet((SkylarkNestedSet) lval, rval, location);
    }
    throw typeException(lval, rval, Operator.PLUS, location);
  }

  /**
   * Implements Operator.PIPE.
   *
   * <p>Publicly accessible for reflection and compiled Skylark code.
   */
  public static Object pipe(Object lval, Object rval, Location location) throws EvalException {
    if (lval instanceof SkylarkNestedSet) {
      return new SkylarkNestedSet((SkylarkNestedSet) lval, rval, location);
    }
    throw typeException(lval, rval, Operator.PIPE, location);
  }

  /**
   * Implements Operator.MINUS.
   *
   * <p>Publicly accessible for reflection and compiled Skylark code.
   */
  public static Object minus(Object lval, Object rval, Location location) throws EvalException {
    if (lval instanceof Integer && rval instanceof Integer) {
      return ((Integer) lval).intValue() - ((Integer) rval).intValue();
    }
    throw typeException(lval, rval, Operator.MINUS, location);
  }

  /**
   * Implements Operator.MULT.
   *
   * <p>Publicly accessible for reflection and compiled Skylark code.
   */
  public static Object mult(Object lval, Object rval, Location location) throws EvalException {
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
    throw typeException(lval, rval, Operator.MULT, location);
  }

  /**
   * Implements Operator.DIVIDE.
   *
   * <p>Publicly accessible for reflection and compiled Skylark code.
   */
  public static Object divide(Object lval, Object rval, Location location) throws EvalException {
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
      return (int) Math.floor(new Double((Integer) lval) / (Integer) rval);
    }
    throw typeException(lval, rval, Operator.DIVIDE, location);
  }

  /**
   * Implements Operator.PERCENT.
   *
   * <p>Publicly accessible for reflection and compiled Skylark code.
   */
  public static Object percent(Object lval, Object rval, Location location) throws EvalException {
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
   * Returns a StackManipulation that calls the given operator's implementation method.
   *
   * <p> The method must be named exactly as the lower case name of the operator and in addition to
   * the operands require an Environment and Location.
   */
  private static StackManipulation callImplementation(
      VariableScope scope, AstAccessors debugAccessors, Operator operator) {
    Class<?>[] parameterTypes =
        new Class<?>[] {Object.class, Object.class, Environment.class, Location.class};
    return new StackManipulation.Compound(
        scope.loadEnvironment(),
        debugAccessors.loadLocation,
        ByteCodeUtils.invoke(
            BinaryOperatorExpression.class, operator.name().toLowerCase(), parameterTypes));
  }

  /**
   * Returns a StackManipulation that calls the given operator's implementation method.
   *
   * <p> The method must be named exactly as the lower case name of the operator and in addition to
   * the operands require a Location.
   */
  private static StackManipulation callImplementation(
      AstAccessors debugAccessors, Operator operator) {
    Class<?>[] parameterTypes = new Class<?>[] {Object.class, Object.class, Location.class};
    return new StackManipulation.Compound(
        debugAccessors.loadLocation,
        ByteCodeUtils.invoke(
            BinaryOperatorExpression.class, operator.name().toLowerCase(), parameterTypes));
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
