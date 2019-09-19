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
import java.io.IOException;
import java.util.Collections;
import java.util.List;
import javax.annotation.Nullable;

/** Syntax node for a function call expression. */
// TODO(adonovan): rename CallExpression.
public final class FuncallExpression extends Expression {

  private final Expression function;
  private final ImmutableList<Argument.Passed> arguments;
  private final int numPositionalArgs;

  FuncallExpression(Expression function, ImmutableList<Argument.Passed> arguments) {
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
    for (Argument.Passed arg : arguments) {
      if (arg.isPositional()) {
        num++;
      }
    }
    return num;
  }

  /**
   * Returns an (immutable, ordered) list of function arguments. The first n are
   * positional and the remaining ones are keyword args, where n =
   * getNumPositionalArguments().
   */
  public List<Argument.Passed> getArguments() {
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
  public void prettyPrint(Appendable buffer) throws IOException {
    function.prettyPrint(buffer);
    buffer.append('(');
    String sep = "";
    for (Argument.Passed arg : arguments) {
      buffer.append(sep);
      arg.prettyPrint(buffer);
      sep = ", ";
    }
    buffer.append(')');
  }

  @Override
  public String toString() {
    Printer.LengthLimitedPrinter printer = new Printer.LengthLimitedPrinter();
    printer.append(function.toString());
    printer.printAbbreviatedList(arguments, "(", ", ", ")", null,
        Printer.SUGGESTED_CRITICAL_LIST_ELEMENTS_COUNT,
        Printer.SUGGESTED_CRITICAL_LIST_ELEMENTS_STRING_LENGTH);
    return printer.toString();
  }

  /**
   * Returns the value of the argument 'name' (or null if there is none). This function is used to
   * associate debugging information to rules created by skylark "macros".
   */
  // TODO(adonovan): move this into sole caller.
  @Nullable
  public String getNameArg() {
    for (Argument.Passed arg : arguments) {
      if (arg != null) {
        String name = arg.getName();
        if (name != null && name.equals("name")) {
          Expression expr = arg.getValue();
          return (expr instanceof StringLiteral) ? ((StringLiteral) expr).getValue() : null;
        }
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

  @Override
  protected boolean isNewScope() {
    return true;
  }
}
