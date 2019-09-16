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

import com.google.devtools.build.lib.syntax.SkylarkList.MutableList;
import com.google.devtools.build.lib.syntax.SkylarkList.Tuple;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/** Syntax node for list and tuple expressions. */
public final class ListExpression extends Expression {

  // TODO(adonovan): split class into {List,Tuple}Expression, as a tuple may have no parens.
  private final boolean isTuple;
  private final List<Expression> elements;

  ListExpression(boolean isTuple, List<Expression> elements) {
    this.isTuple = isTuple;
    this.elements = elements;
  }

  public List<Expression> getElements() {
    return elements;
  }

  /** Reports whether this is a tuple expression. */
  public boolean isTuple() {
    return isTuple;
  }

  @Override
  public void prettyPrint(Appendable buffer) throws IOException {
    buffer.append(isTuple() ? '(' : '[');
    String sep = "";
    for (Expression e : elements) {
      buffer.append(sep);
      e.prettyPrint(buffer);
      sep = ", ";
    }
    if (isTuple() && elements.size() == 1) {
      buffer.append(',');
    }
    buffer.append(isTuple() ? ')' : ']');
  }

  @Override
  public String toString() {
    return Printer.printAbbreviatedList(
        elements,
        isTuple(),
        Printer.SUGGESTED_CRITICAL_LIST_ELEMENTS_COUNT,
        Printer.SUGGESTED_CRITICAL_LIST_ELEMENTS_STRING_LENGTH);
  }

  @Override
  Object doEval(Environment env) throws EvalException, InterruptedException {
    ArrayList<Object> result = new ArrayList<>(elements.size());
    for (Expression element : elements) {
      // Convert NPEs to EvalExceptions.
      if (element == null) {
        throw new EvalException(getLocation(), "null expression in " + this);
      }
      result.add(element.eval(env));
    }
    return isTuple() ? Tuple.copyOf(result) : MutableList.wrapUnsafe(env, result);
  }

  @Override
  public void accept(NodeVisitor visitor) {
    visitor.visit(this);
  }

  @Override
  public Expression.Kind kind() {
    return Expression.Kind.LIST_EXPR;
  }
}
