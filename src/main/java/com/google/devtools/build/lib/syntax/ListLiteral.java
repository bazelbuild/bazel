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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.syntax.SkylarkList.MutableList;
import com.google.devtools.build.lib.syntax.SkylarkList.Tuple;
import com.google.devtools.build.lib.syntax.compiler.ByteCodeMethodCalls;
import com.google.devtools.build.lib.syntax.compiler.ByteCodeUtils;
import com.google.devtools.build.lib.syntax.compiler.DebugInfo;
import com.google.devtools.build.lib.syntax.compiler.DebugInfo.AstAccessors;
import com.google.devtools.build.lib.syntax.compiler.NewObject;
import com.google.devtools.build.lib.syntax.compiler.VariableScope;
import com.google.devtools.build.lib.util.Preconditions;

import net.bytebuddy.implementation.bytecode.ByteCodeAppender;
import net.bytebuddy.implementation.bytecode.Duplication;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Syntax node for list and tuple literals.
 *
 * <p>(Note that during evaluation, both list and tuple values are represented by
 * java.util.List objects, the only difference between them being whether or not
 * they are mutable.)
 */
public final class ListLiteral extends Expression {

  /**
   * Types of the ListLiteral.
   */
  public static enum Kind {LIST, TUPLE}

  private final Kind kind;

  private final List<Expression> exprs;

  private ListLiteral(Kind kind, List<Expression> exprs) {
    this.kind = kind;
    this.exprs = exprs;
  }

  public static ListLiteral makeList(List<Expression> exprs) {
    return new ListLiteral(Kind.LIST, exprs);
  }

  public static ListLiteral makeTuple(List<Expression> exprs) {
    return new ListLiteral(Kind.TUPLE, exprs);
  }

  /** A new literal for an empty list, onto which a new location can be specified */
  public static ListLiteral emptyList() {
    return makeList(Collections.<Expression>emptyList());
  }

  /**
   * Returns the list of expressions for each element of the tuple.
   */
  public List<Expression> getElements() {
    return exprs;
  }

  /**
   * Returns true if this list is a tuple (a hash table, immutable list).
   */
  public boolean isTuple() {
    return kind == Kind.TUPLE;
  }

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder();
    Printer.printList(sb, exprs, isTuple(), '"', Printer.SUGGESTED_CRITICAL_LIST_ELEMENTS_COUNT,
        Printer.SUGGESTED_CRITICAL_LIST_ELEMENTS_STRING_LENGTH);
    return sb.toString();
  }

  @Override
  Object doEval(Environment env) throws EvalException, InterruptedException {
    List<Object> result = new ArrayList<>(exprs.size());
    for (Expression expr : exprs) {
      // Convert NPEs to EvalExceptions.
      if (expr == null) {
        throw new EvalException(getLocation(), "null expression in " + this);
      }
      result.add(expr.eval(env));
    }
    return isTuple() ? Tuple.copyOf(result) : new MutableList(result, env);
  }

  @Override
  public void accept(SyntaxTreeVisitor visitor) {
    visitor.visit(this);
  }

  @Override
  void validate(ValidationEnvironment env) throws EvalException {
    for (Expression expr : exprs) {
      expr.validate(env);
    }
  }

  @Override
  ByteCodeAppender compile(VariableScope scope, DebugInfo debugInfo) throws EvalException {
    AstAccessors debugAccessors = debugInfo.add(this);
    List<ByteCodeAppender> listConstruction = new ArrayList<>();
    if (isTuple()) {
      append(listConstruction, ByteCodeMethodCalls.BCImmutableList.builder);
    } else {
      append(
          listConstruction,
          // create a new MutableList object
          NewObject.fromConstructor(MutableList.class, Mutability.class)
              .arguments(
                  scope.loadEnvironment(), ByteCodeUtils.invoke(Environment.class, "mutability")));
    }

    for (Expression expression : exprs) {
      Preconditions.checkNotNull(
          expression, "List literal at %s contains null expression", getLocation());
      ByteCodeAppender compiledValue = expression.compile(scope, debugInfo);
      if (isTuple()) {
        listConstruction.add(compiledValue);
        append(
            listConstruction,
            // this re-adds the builder to the stack and we reuse it in the next iteration/after
            ByteCodeMethodCalls.BCImmutableList.Builder.add);
      } else {
        // duplicate the list reference on the stack for reuse in the next iteration/after
        append(listConstruction, Duplication.SINGLE);
        listConstruction.add(compiledValue);
        append(
            listConstruction,
            debugAccessors.loadLocation,
            scope.loadEnvironment(),
            ByteCodeUtils.cleanInvoke(
                MutableList.class, "add", Object.class, Location.class, Environment.class));
      }
    }
    if (isTuple()) {
      append(
          listConstruction,
          ByteCodeMethodCalls.BCImmutableList.Builder.build,
          ByteCodeUtils.invoke(Tuple.class, "create", ImmutableList.class));
    }
    return ByteCodeUtils.compoundAppender(listConstruction);
  }
}
