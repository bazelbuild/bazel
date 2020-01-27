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
 *
 * <p>See FunctionSignature for how a valid list of Parameters is organized as a signature, e.g. def
 * foo(mandatory, optional = e1, *args, mandatorynamedonly, optionalnamedonly = e2, **kw): ...
 *
 * <p>V is the class of a defaultValue (Expression at compile-time, Object at runtime), T is the
 * class of a type (Expression at compile-time, SkylarkType at runtime).
 */
public abstract class Parameter extends Node {

  @Nullable private final Identifier identifier;

  private Parameter(@Nullable Identifier identifier) {
    this.identifier = identifier;
  }

  @Nullable
  public String getName() {
    return identifier != null ? identifier.getName() : null;
  }

  @Nullable
  public Identifier getIdentifier() {
    return identifier;
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
    Mandatory(Identifier identifier) {
      super(identifier);
    }
  }

  /**
   * Syntax node for an optional parameter, {@code f(id=expr).}. It may be positional or
   * keyword-only depending on its position.
   */
  public static final class Optional extends Parameter {

    public final Expression defaultValue;

    Optional(Identifier identifier, @Nullable Expression defaultValue) {
      super(identifier);
      this.defaultValue = defaultValue;
    }

    @Override
    @Nullable
    public Expression getDefaultValue() {
      return defaultValue;
    }

    @Override
    public String toString() {
      return getName() + "=" + defaultValue;
    }
  }

  /** Syntax node for a star parameter, {@code f(*identifier)} or or {@code f(..., *, ...)}. */
  public static final class Star extends Parameter {
    Star(@Nullable Identifier identifier) {
      super(identifier);
    }
  }

  /** Syntax node for a parameter of the form {@code f(**identifier)}. */
  public static final class StarStar extends Parameter {
    StarStar(Identifier identifier) {
      super(identifier);
    }
  }

  @Override
  public void accept(NodeVisitor visitor) {
    visitor.visit(this);
  }
}
