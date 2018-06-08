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

import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import java.io.IOException;
import javax.annotation.Nullable;

/**
 * Syntax node for a Parameter in a function (or lambda) definition; it's a subclass of Argument,
 * and contrasts with the class Argument.Passed of arguments in a function call.
 *
 * <p>There are four concrete subclasses of Parameter: Mandatory, Optional, Star, StarStar.
 *
 * <p>See FunctionSignature for how a valid list of Parameter's is organized as a signature, e.g.
 * def foo(mandatory, optional = e1, *args, mandatorynamedonly, optionalnamedonly = e2, **kw): ...
 *
 * <p>V is the class of a defaultValue (Expression at compile-time, Object at runtime),
 * T is the class of a type (Expression at compile-time, SkylarkType at runtime).
 */
public abstract class Parameter<V, T> extends Argument {

  @Nullable protected final Identifier identifier;
  @Nullable protected final T type;

  private Parameter(@Nullable Identifier identifier, @Nullable T type) {
    this.identifier = identifier;
    this.type = type;
  }

  public boolean isMandatory() {
    return false;
  }

  public boolean isOptional() {
    return false;
  }

  @Override
  public boolean isStar() {
    return false;
  }

  @Override
  public boolean isStarStar() {
    return false;
  }

  @Nullable
  public String getName() {
    return identifier != null ? identifier.getName() : null;
  }

  @Nullable
  public Identifier getIdentifier() {
    return identifier;
  }

  public boolean hasName() {
    return true;
  }

  @Nullable
  public T getType() {
    return type;
  }

  @Nullable
  public V getDefaultValue() {
    return null;
  }

  /** mandatory parameter (positional or key-only depending on position): Ident */
  @AutoCodec
  public static final class Mandatory<V, T> extends Parameter<V, T> {

    public Mandatory(Identifier identifier) {
      this(identifier, null);
    }

    @AutoCodec.Instantiator
    public Mandatory(Identifier identifier, @Nullable T type) {
      super(identifier, type);
    }

    @Override
    public boolean isMandatory() {
      return true;
    }

    @Override
    public void prettyPrint(Appendable buffer) throws IOException {
      buffer.append(getName());
    }
  }

  /** optional parameter (positional or key-only depending on position): Ident = Value */
  @AutoCodec
  public static final class Optional<V, T> extends Parameter<V, T> {

    public final V defaultValue;

    public Optional(Identifier identifier, @Nullable V defaultValue) {
      this(identifier, null, defaultValue);
    }

    @AutoCodec.Instantiator
    public Optional(Identifier identifier, @Nullable T type, @Nullable V defaultValue) {
      super(identifier, type);
      this.defaultValue = defaultValue;
    }

    @Override
    @Nullable
    public V getDefaultValue() {
      return defaultValue;
    }

    @Override
    public boolean isOptional() {
      return true;
    }

    @Override
    public void prettyPrint(Appendable buffer) throws IOException {
      buffer.append(getName());
      buffer.append('=');
      // This should only ever be used on a parameter representing static information, i.e. with V
      // and T instantiated as Expression.
      ((Expression) defaultValue).prettyPrint(buffer);
    }

    // Keep this as a separate method so that it can be used regardless of what V and T are
    // parameterized with.
    @Override
    public String toString() {
      return getName() + "=" + defaultValue;
    }
  }

  /** extra positionals parameter (star): *identifier */
  @AutoCodec
  public static final class Star<V, T> extends Parameter<V, T> {

    @AutoCodec.Instantiator
    public Star(@Nullable Identifier identifier, @Nullable T type) {
      super(identifier, type);
    }

    public Star(@Nullable Identifier identifier) {
      this(identifier, null);
    }

    @Override
    public boolean hasName() {
      return getName() != null;
    }

    @Override
    public boolean isStar() {
      return true;
    }

    @Override
    public void prettyPrint(Appendable buffer) throws IOException {
      buffer.append('*');
      if (getName() != null) {
        buffer.append(getName());
      }
    }
  }

  /** extra keywords parameter (star_star): **identifier */
  @AutoCodec
  public static final class StarStar<V, T> extends Parameter<V, T> {

    @AutoCodec.Instantiator
    public StarStar(Identifier identifier, @Nullable T type) {
      super(identifier, type);
    }

    public StarStar(Identifier identifier) {
      this(identifier, null);
    }

    @Override
    public boolean isStarStar() {
      return true;
    }

    @Override
    public void prettyPrint(Appendable buffer) throws IOException {
      buffer.append("**");
      buffer.append(getName());
    }
  }

  @Override
  @SuppressWarnings("unchecked")
  public void accept(SyntaxTreeVisitor visitor) {
    visitor.visit((Parameter<Expression, Expression>) this);
  }
}
