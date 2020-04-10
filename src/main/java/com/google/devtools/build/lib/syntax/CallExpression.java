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

/** Syntax node for a function call expression. */
public final class CallExpression extends Expression {

  private final Expression function;
  private final Location lparenLocation;
  private final ImmutableList<Argument> arguments;
  private final int rparenOffset;

  private final int numPositionalArgs;

  CallExpression(
      FileLocations locs,
      Expression function,
      Location lparenLocation,
      ImmutableList<Argument> arguments,
      int rparenOffset) {
    super(locs);
    this.function = Preconditions.checkNotNull(function);
    this.lparenLocation = lparenLocation;
    this.arguments = arguments;
    this.rparenOffset = rparenOffset;

    int n = 0;
    for (Argument arg : arguments) {
      if (arg instanceof Argument.Positional) {
        n++;
      }
    }
    this.numPositionalArgs = n;
  }

  /** Returns the function that is called. */
  public Expression getFunction() {
    return this.function;
  }

  /** Returns the number of arguments of type {@code Argument.Positional}. */
  public int getNumPositionalArguments() {
    return numPositionalArgs;
  }

  /** Returns the function arguments. */
  public ImmutableList<Argument> getArguments() {
    return arguments;
  }

  @Override
  public int getStartOffset() {
    return function.getStartOffset();
  }

  @Override
  public int getEndOffset() {
    return rparenOffset + 1;
  }

  public Location getLparenLocation() {
    // Unlike all other getXXXLocation methods, this one returns a reference to
    // a previously materialized Location. getLparenLocation is unique among
    // locations because the tree-walking evaluator needs it frequently even
    // in the absence of errors. When we switch to a compiled representation
    // we can dispense with this optimization.
    return lparenLocation;
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

  @Override
  public void accept(NodeVisitor visitor) {
    visitor.visit(this);
  }

  @Override
  public Kind kind() {
    return Kind.CALL;
  }
}
