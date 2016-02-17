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

import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.syntax.compiler.ByteCodeUtils;
import com.google.devtools.build.lib.syntax.compiler.DebugInfo.AstAccessors;
import com.google.devtools.build.lib.syntax.compiler.Variable.InternalVariable;
import com.google.devtools.build.lib.syntax.compiler.VariableScope;
import com.google.devtools.build.lib.util.Preconditions;

import net.bytebuddy.implementation.bytecode.ByteCodeAppender;
import net.bytebuddy.implementation.bytecode.Removal;
import net.bytebuddy.implementation.bytecode.constant.IntegerConstant;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;

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
    // We currently do not allow slices (e.g. a[2:6] = [3]).
    if (lvalue instanceof FuncallExpression) {
      FuncallExpression func = (FuncallExpression) lvalue;
      List<Argument.Passed> args = func.getArguments();
      if (func.getFunction().getName().equals("$index")
          && func.getObject() instanceof Identifier
          && args.size() == 1) {
        Object key = args.get(0).getValue().eval(env);
        assignItem(env, loc, (Identifier) func.getObject(), key, result);
        return;
      }
    }

    throw new EvalException(loc,
        "can only assign to variables and tuples, not to '" + lvalue + "'");
  }

  // Since dict is still immutable, the expression 'a[x] = b' creates a new dictionary and
  // assigns it to 'a'.
  @SuppressWarnings("unchecked")
  private static void assignItem(
      Environment env, Location loc, Identifier ident, Object key, Object value)
      throws EvalException, InterruptedException {
    Object o = ident.eval(env);
    if (!(o instanceof SkylarkDict)) {
      throw new EvalException(
          loc,
          "can only assign an element in a dictionary, not in a '"
              + EvalUtils.getDataTypeName(o)
              + "'");
    }
    SkylarkDict<Object, Object> dict = (SkylarkDict<Object, Object>) o;
    dict.put(key, value, loc, env);
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
    if (expr instanceof FuncallExpression) {
      FuncallExpression func = (FuncallExpression) expr;
      if (func.getFunction().getName().equals("$index") && func.getObject() instanceof Identifier) {
        return;
      }
    }
    throw new EvalException(loc,
        "can only assign to variables and tuples, not to '" + expr + "'");
  }

  @Override
  public String toString() {
    return expr.toString();
  }

  /**
   * Compile an assignment within the given ASTNode to these l-values.
   *
   * <p>The value to possibly destructure and assign must already be on the stack.
   */
  public ByteCodeAppender compileAssignment(
      ASTNode node, AstAccessors debugAccessors, VariableScope scope) throws EvalException {
    List<ByteCodeAppender> code = new ArrayList<>();
    compileAssignment(node, debugAccessors, expr, scope, code);
    return ByteCodeUtils.compoundAppender(code);
  }

  /**
   * Called recursively to compile the tree of l-values we might have.
   */
  private static void compileAssignment(
      ASTNode node,
      AstAccessors debugAccessors,
      Expression leftValue,
      VariableScope scope,
      List<ByteCodeAppender> code)
      throws EvalException {
    if (leftValue instanceof Identifier) {
      code.add(compileAssignment(scope, (Identifier) leftValue));
    } else if (leftValue instanceof ListLiteral) {
      List<Expression> lValueExpressions = ((ListLiteral) leftValue).getElements();
      compileAssignment(node, debugAccessors, scope, lValueExpressions, code);
    } else {
      String message =
          String.format(
              "Can't assign to expression '%s', only to variables or nested tuples of variables",
              leftValue);
      throw new EvalExceptionWithStackTrace(new EvalException(node.getLocation(), message), node);
    }
  }

  /**
   * Assumes a collection of values on the top of the stack and assigns them to the l-value
   * expressions given.
   */
  private static void compileAssignment(
      ASTNode node,
      AstAccessors debugAccessors,
      VariableScope scope,
      List<Expression> lValueExpressions,
      List<ByteCodeAppender> code)
      throws EvalException {
    InternalVariable objects = scope.freshVariable(Collection.class);
    InternalVariable iterator = scope.freshVariable(Iterator.class);
    // convert the object on the stack into a collection and store it to a variable for loading
    // multiple times below below
    code.add(new ByteCodeAppender.Simple(debugAccessors.loadLocation, EvalUtils.toCollection));
    code.add(objects.store());
    append(
        code,
        // check that we got exactly the amount of objects in the collection that we need
        IntegerConstant.forValue(lValueExpressions.size()),
        objects.load(),
        debugAccessors.loadLocation, // TODO(bazel-team) load better location within tuple
        ByteCodeUtils.invoke(
            LValue.class, "checkSize", int.class, Collection.class, Location.class),
        // get an iterator to assign the objects
        objects.load(),
        ByteCodeUtils.invoke(Collection.class, "iterator"));
    code.add(iterator.store());
    // assign each object to the corresponding l-value
    for (Expression lValue : lValueExpressions) {
      code.add(
          new ByteCodeAppender.Simple(
              iterator.load(), ByteCodeUtils.invoke(Iterator.class, "next")));
      compileAssignment(node, debugAccessors, lValue, scope, code);
    }
  }

  /**
   * Compile assignment to a single identifier.
   */
  private static ByteCodeAppender compileAssignment(VariableScope scope, Identifier identifier) {
    // don't store to/create the _ "variable" the value is not needed, just remove it
    if (identifier.getName().equals("_")) {
      return new ByteCodeAppender.Simple(Removal.SINGLE);
    }
    return scope.getVariable(identifier).store();
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
