// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.util;

import java.util.Objects;
import java.util.function.Consumer;
import java.util.function.Function;

/**
 * A container wrapping a value of one of two types. An {@code Either<A, B>} instance either wraps
 * an instance of {@code A} or a instance of {@code B}.
 *
 * <p>Just as with {@link Pair}, this class is immutable, supports nullable values, and is
 * completely devoid of Bazel-business-logic-specific semantics. Avoid using it in public APIs.
 *
 * <p>This class is a a simple implementation of a general purpose "sum" type. In type theory, sum
 * types are the duals of product types -- the corresponding observation here is that {@link Either}
 * is the dual of {@link Pair}.
 */
public abstract class Either<A, B> {
  // Disallow subclasses outside of this file.
  private Either() {
  }

  /** Constructs a {@link Either} representing the left injection of {@code a}. */
  public static <A, B> Either<A, B> ofLeft(A a) {
    return new LeftEither<>(a);
  }

  /** Constructs a {@link Either} representing the right injection of {@code b}. */
  public static <A, B> Either<A, B> ofRight(B b) {
    return new RightEither<>(b);
  }

  /**
   * Consumes the value injected into this {@link Either}. A left injection is consumed using
   * {@code leftConsumer} and a right injection is consumed using {@code rightConsumer}.
   */
  public abstract void consume(Consumer<A> leftConsumer, Consumer<B> rightConsumer);

  /**
   * Maps the value injected into this {@link Either}. A left injection is mapped using
   * {@code leftFunction} and a right injection is mapped using {@code rightFunction}.
   */
  public abstract <C> C map(Function<A, C> leftFunction, Function<B, C> rightFunction);

  @Override
  public abstract int hashCode();

  @Override
  public abstract boolean equals(Object other);

  @Override
  public abstract String toString();

  private static class LeftEither<A, B> extends Either<A, B> {
    private final A a;

    private LeftEither(A a) {
      this.a = a;
    }

    @Override
    public void consume(Consumer<A> leftConsumer, Consumer<B> rightConsumer) {
      leftConsumer.accept(a);
    }

    @Override
    public <C> C map(Function<A, C> leftFunction, Function<B, C> rightFunction) {
      return leftFunction.apply(a);
    }

    @Override
    public int hashCode() {
      return Objects.hashCode(a);
    }

    @Override
    public boolean equals(Object other) {
      if (!(other instanceof LeftEither)) {
        return false;
      }
      return Objects.equals(this.a, ((LeftEither) other).a);
    }

    @Override
    public String toString() {
      return "left injection of " + a;
    }
  }

  private static class RightEither<A, B> extends Either<A, B> {
    private final B b;

    private RightEither(B b) {
      this.b = b;
    }

    @Override
    public void consume(Consumer<A> leftConsumer, Consumer<B> rightConsumer) {
      rightConsumer.accept(b);
    }

    @Override
    public <C> C map(Function<A, C> leftFunction, Function<B, C> rightFunction) {
      return rightFunction.apply(b);
    }

    @Override
    public int hashCode() {
      return Objects.hashCode(b);
    }

    @Override
    public boolean equals(Object other) {
      if (!(other instanceof RightEither)) {
        return false;
      }
      return Objects.equals(this.b, ((RightEither) other).b);
    }

    @Override
    public String toString() {
      return "right injection of " + b;
    }
  }
}
