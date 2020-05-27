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

import javax.annotation.Nullable;

/**
 * Syntax node for a parameter in a function definition.
 *
 * <p>Parameters may be of four forms, as in {@code def f(a, b=c, *args, **kwargs)}. They are
 * represented by the subclasses Mandatory, Optional, Star, and StarStar.
 */
public abstract class Parameter extends Node {

  @Nullable private final Identifier id;

  private Parameter(FileLocations locs, @Nullable Identifier id) {
    super(locs);
    this.id = id;
  }

  @Nullable
  public String getName() {
    return id != null ? id.getName() : null;
  }

  @Nullable
  public Identifier getIdentifier() {
    return id;
  }

  @Nullable
  public Expression getDefaultValue() {
    return null;
  }

  /**
   * Syntax node for a mandatory parameter, {@code f(id)}. It may be positional or keyword-only
   * depending on its position.
   */
  public static final class Mandatory extends Parameter {
    Mandatory(FileLocations locs, Identifier id) {
      super(locs, id);
    }

    @Override
    public int getStartOffset() {
      return getIdentifier().getStartOffset();
    }

    @Override
    public int getEndOffset() {
      return getIdentifier().getEndOffset();
    }
  }

  /**
   * Syntax node for an optional parameter, {@code f(id=expr).}. It may be positional or
   * keyword-only depending on its position.
   */
  public static final class Optional extends Parameter {

    public final Expression defaultValue;

    Optional(FileLocations locs, Identifier id, @Nullable Expression defaultValue) {
      super(locs, id);
      this.defaultValue = defaultValue;
    }

    @Override
    @Nullable
    public Expression getDefaultValue() {
      return defaultValue;
    }

    @Override
    public int getStartOffset() {
      return getIdentifier().getStartOffset();
    }

    @Override
    public int getEndOffset() {
      return getDefaultValue().getEndOffset();
    }

    @Override
    public String toString() {
      return getName() + "=" + defaultValue;
    }
  }

  /** Syntax node for a star parameter, {@code f(*id)} or or {@code f(..., *, ...)}. */
  public static final class Star extends Parameter {
    private final int starOffset;

    Star(FileLocations locs, int starOffset, @Nullable Identifier id) {
      super(locs, id);
      this.starOffset = starOffset;
    }

    @Override
    public int getStartOffset() {
      return starOffset;
    }

    @Override
    public int getEndOffset() {
      return getIdentifier().getEndOffset();
    }
  }

  /** Syntax node for a parameter of the form {@code f(**id)}. */
  public static final class StarStar extends Parameter {
    private final int starStarOffset;

    StarStar(FileLocations locs, int starStarOffset, Identifier id) {
      super(locs, id);
      this.starStarOffset = starStarOffset;
    }

    @Override
    public int getStartOffset() {
      return starStarOffset;
    }

    @Override
    public int getEndOffset() {
      return getIdentifier().getEndOffset();
    }
  }

  @Override
  public void accept(NodeVisitor visitor) {
    visitor.visit(this);
  }
}
