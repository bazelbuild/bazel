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

  protected final Expression value;

  Argument(FileLocations locs, Expression value) {
    super(locs);
    this.value = Preconditions.checkNotNull(value);
  }

  public final Expression getValue() {
    return value;
  }

  @Override
  public int getEndOffset() {
    return value.getEndOffset();
  }

  /** Return the name of this argument's parameter, or null if it is not a Keyword argument. */
  @Nullable
  public String getName() {
    return null;
  }

  /** Syntax node for a positional argument, {@code f(expr)}. */
  public static final class Positional extends Argument {
    Positional(FileLocations locs, Expression value) {
      super(locs, value);
    }

    @Override
    public int getStartOffset() {
      return value.getStartOffset();
    }
  }

  /** Syntax node for a keyword argument, {@code f(id=expr)}. */
  public static final class Keyword extends Argument {

    // Unlike in Python, keyword arguments in Bazel BUILD files
    // are about 10x more numerous than positional arguments.

    final Identifier id;

    Keyword(FileLocations locs, Identifier id, Expression value) {
      super(locs, value);
      this.id = id;
    }

    public Identifier getIdentifier() {
      return id;
    }

    @Override
    public String getName() {
      return id.getName();
    }

    @Override
    public int getStartOffset() {
      return id.getStartOffset();
    }
  }

  /** Syntax node for an argument of the form {@code f(*expr)}. */
  public static final class Star extends Argument {
    private final int starOffset;

    Star(FileLocations locs, int starOffset, Expression value) {
      super(locs, value);
      this.starOffset = starOffset;
    }

    @Override
    public int getStartOffset() {
      return starOffset;
    }
  }

  /** Syntax node for an argument of the form {@code f(**expr)}. */
  public static final class StarStar extends Argument {
    private final int starStarOffset;

    StarStar(FileLocations locs, int starStarOffset, Expression value) {
      super(locs, value);
      this.starStarOffset = starStarOffset;
    }

    @Override
    public int getStartOffset() {
      return starStarOffset;
    }
  }

  @Override
  public void accept(NodeVisitor visitor) {
    visitor.visit(this);
  }
}
