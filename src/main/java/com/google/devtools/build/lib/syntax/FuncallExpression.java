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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import java.util.Collections;
import java.util.List;
import javax.annotation.Nullable;

/** Syntax node for a function call expression. */
// TODO(adonovan): rename CallExpression.
public final class FuncallExpression extends Expression {

  private final Expression function;
  private final ImmutableList<Argument> arguments;
  private final int numPositionalArgs;

  FuncallExpression(Expression function, ImmutableList<Argument> arguments) {
    this.function = Preconditions.checkNotNull(function);
    this.arguments = Preconditions.checkNotNull(arguments);
    this.numPositionalArgs = countPositionalArguments();
  }

  /** Returns the function that is called. */
  public Expression getFunction() {
    return this.function;
  }

  /**
   * Returns the number of positional arguments.
   */
  private int countPositionalArguments() {
    int num = 0;
    for (Argument arg : arguments) {
      if (arg instanceof Argument.Positional) {
        num++;
      }
    }
    return num;
  }

  /**
   * Returns an (immutable, ordered) list of function arguments. The first n are positional and the
   * remaining ones are keyword args, where n = getNumPositionalArguments().
   */
  public List<Argument> getArguments() {
    return Collections.unmodifiableList(arguments);
  }

  /**
   * Returns the number of arguments which are positional; the remainder are
   * keyword arguments.
   */
  public int getNumPositionalArguments() {
    return numPositionalArgs;
  }

  @Override
  public String toString() {
    StringBuilder buf = new StringBuilder();
    buf.append(function);
    buf.append('(');
    ListExpression.appendNodes(buf, arguments);
    buf.append(')');
    return buf.toString();
  }

  /**
   * Returns the value of the argument 'name' (or null if there is none). This function is used to
   * associate debugging information to rules created by skylark "macros".
   */
  // TODO(adonovan): move this into sole caller.
  @Nullable
  public String getNameArg() {
    for (Argument arg : arguments) {
      String name = arg.getName();
      if (name != null && name.equals("name")) {
        Expression expr = arg.getValue();
        return (expr instanceof StringLiteral) ? ((StringLiteral) expr).getValue() : null;
      }
    }
    return null;
  }

  @Override
  public void accept(NodeVisitor visitor) {
    visitor.visit(this);
  }

  @Override
  public Kind kind() {
    return Kind.FUNCALL;
  }
}
