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

  public ListLiteral(Kind kind, List<Expression> exprs) {
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

  public Kind getKind() {
    return kind;
  }

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
  public void prettyPrint(Appendable buffer) throws IOException {
    buffer.append(isTuple() ? '(' : '[');
    String sep = "";
    for (Expression e : exprs) {
      buffer.append(sep);
      e.prettyPrint(buffer);
      sep = ", ";
    }
    if (isTuple() && exprs.size() == 1) {
      buffer.append(',');
    }
    buffer.append(isTuple() ? ')' : ']');
  }

  @Override
  public String toString() {
    return Printer.printAbbreviatedList(
        exprs,
        isTuple(),
        Printer.SUGGESTED_CRITICAL_LIST_ELEMENTS_COUNT,
        Printer.SUGGESTED_CRITICAL_LIST_ELEMENTS_STRING_LENGTH);
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
}
