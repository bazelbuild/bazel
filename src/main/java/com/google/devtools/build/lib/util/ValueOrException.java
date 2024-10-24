// Copyright 2024 The Bazel Authors. All rights reserved.
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

import static com.google.common.base.Preconditions.checkNotNull;

/**
 * Contains a value, or the exception encountered while obtaining that value.
 *
 * <p>This class is intended to be used when an operation computes multiple values, some of which
 * may have succeeded and others may have failed, and it is necessary to return all of these values
 * or failures without throwing at the first failure encountered.
 */
// TODO(b/331799946): try to consolidate Bazel's "value or exception" sum types into this one
public abstract sealed class ValueOrException<V, E extends Exception> {

  // Must be initialized through ofValue or ofException
  private ValueOrException() {}

  /** Constructs a ValueOrException holding a non-null value. */
  public static <V, E extends Exception> ValueOrException<V, E> ofValue(V value) {
    return new OfValue<>(checkNotNull(value));
  }

  /** Constructs a ValueOrException holding an exception. */
  public static <V, E extends Exception> ValueOrException<V, E> ofException(E exception) {
    return new OfException<>(checkNotNull(exception));
  }

  /** Returns true if the ValueOrException holds a value. */
  public abstract boolean isPresent();

  /**
   * Returns the value if the ValueOrException holds a value.
   *
   * @throws E if the ValueOrException holds an exception.
   */
  public abstract V get() throws E;

  /**
   * A variant of {@link #get} which throws an unchecked exception.
   *
   * @throws IllegalStateException if the ValueOrException holds an exception.
   */
  public V getUnchecked() {
    try {
      return get();
    } catch (Exception e) {
      throw new IllegalStateException(e);
    }
  }

  /**
   * Returns the exception if the ValueOrException holds an exception.
   *
   * @throws IllegalStateException if the ValueOrException does not hold an exception.
   */
  public abstract E getException();

  private static final class OfValue<V, E extends Exception> extends ValueOrException<V, E> {
    private final V value;

    OfValue(V value) {
      this.value = value;
    }

    @Override
    public boolean isPresent() {
      return true;
    }

    @Override
    public V get() {
      return value;
    }

    @Override
    public E getException() {
      throw new IllegalStateException(
          "ValueOrException.getException() cannot be called on a ValueOrException holding a value");
    }

    @Override
    public String toString() {
      return String.format("ValueOrException.OfValue[%s]", value);
    }

    /**
     * {@inheritDoc}
     *
     * <p>Value equality semantics; two {@code OfValue} objects are equal iff their contained values
     * are equal.
     */
    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (!(o instanceof OfValue<?, ?> rhs)) {
        return false;
      }
      return value.equals(rhs.value);
    }

    @Override
    public int hashCode() {
      return value.hashCode();
    }
  }

  private static final class OfException<V, E extends Exception> extends ValueOrException<V, E> {
    private final E exception;

    OfException(E exception) {
      this.exception = checkNotNull(exception);
    }

    @Override
    public boolean isPresent() {
      return false;
    }

    @Override
    public V get() throws E {
      throw exception;
    }

    @Override
    public E getException() {
      return exception;
    }

    @Override
    public String toString() {
      return String.format("ValueOrException.OfException[%s]", exception);
    }

    /**
     * {@inheritDoc}
     *
     * <p>Value equality semantics; two {@code OfException} objects are equal iff their contained
     * exception objects are equal.
     */
    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (!(o instanceof OfException<?, ?> rhs)) {
        return false;
      }
      return exception.equals(rhs.exception);
    }

    @Override
    public int hashCode() {
      return exception.hashCode();
    }
  }
}
