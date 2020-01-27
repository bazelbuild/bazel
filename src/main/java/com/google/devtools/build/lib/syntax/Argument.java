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
import javax.annotation.Nullable;

/**
 * Syntax node for an argument to a function.
 *
 * <p>Arguments may be of four forms, as in {@code f(expr, id=expr, *expr, **expr)}. These are
 * represented by the subclasses Positional, Keyword, Star, and StarStar.
 */
public abstract class Argument extends Node {

  private final Expression value;

  Argument(Expression value) {
    this.value = Preconditions.checkNotNull(value);
  }

  public final Expression getValue() {
    return value;
  }

  /** Return the name of this argument's parameter, or null if it is not a Keyword argument. */
  @Nullable
  public String getName() {
    return null;
  }

  /** Syntax node for a positional argument, {@code f(expr)}. */
  public static final class Positional extends Argument {
    Positional(Expression value) {
      super(value);
    }
  }

  /** Syntax node for a keyword argument, {@code f(id=expr)}. */
  public static final class Keyword extends Argument {

    // Unlike in Python, keyword arguments in Bazel BUILD files
    // are about 10x more numerous than positional arguments.

    final Identifier identifier;

    Keyword(Identifier identifier, Expression value) {
      super(value);
      this.identifier = identifier;
    }

    public Identifier getIdentifier() {
      return identifier;
    }

    @Override
    public String getName() {
      return identifier.getName();
    }
  }

  /** Syntax node for an argument of the form {@code f(*expr)}. */
  public static final class Star extends Argument {
    Star(Expression value) {
      super(value);
    }
  }

  /** Syntax node for an argument of the form {@code f(**expr)}. */
  public static final class StarStar extends Argument {
    StarStar(Expression value) {
      super(value);
    }
  }

  @Override
  public void accept(NodeVisitor visitor) {
    visitor.visit(this);
  }
}
