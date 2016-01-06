// Copyright 2015 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.util.Preconditions;

import java.io.Serializable;

/**
 * A Mutability is a resource associated with an {@link Environment} during an evaluation,
 * that gives those who possess it a revokable capability to mutate this Environment and
 * the objects created within that {@link Environment}. At the end of the evaluation,
 * the resource is irreversibly closed, at which point the capability is revoked,
 * and it is not possible to mutate either this {@link Environment} or these objects.
 *
 * <p>Evaluation in an {@link Environment} may thus mutate objects created in the same
 * {@link Environment}, but may not mutate {@link Freezable} objects (lists, sets, dicts)
 * created in a previous, concurrent or future {@link Environment}, and conversely,
 * the bindings and objects in this {@link Environment} will be protected from
 * mutation by evaluation in a different {@link Environment}.
 *
 * <p>Only a single thread may use the {@link Environment} and objects created within it while the
 * Mutability is still mutable as tested by {@link #isMutable}. Once the Mutability resource
 * is closed, the {@link Environment} and its objects are immutable and can be simultaneously used
 * by arbitrarily many threads.
 *
 * <p>The safe usage of a Mutability requires to always use try-with-resource style:
 * <code>try (Mutability mutability = Mutability.create(fmt, ...)) { ... }</code>
 * Thus, you can create a Mutability, build an {@link Environment}, mutate that {@link Environment}
 * and create mutable objects as you evaluate in that {@link Environment}, and finally return the
 * resulting {@link Environment}, at which point the resource is closed, and the {@link Environment}
 * and the objects it contains all become immutable.
 * (Unsafe usage is allowed only in test code that is not part of the Bazel jar.)
 */
// TODO(bazel-team): When we start using Java 8, this safe usage pattern can be enforced
// through the use of a higher-order function.
public final class Mutability implements AutoCloseable, Serializable {
  private boolean isMutable;
  private final String annotation; // For error reporting.

  /**
   * Creates a Mutability.
   * @param annotation an Object used for error reporting,
   * describing to the user the context in which this Mutability was active.
   */
  private Mutability(String annotation) {
    this.isMutable = true;
    this.annotation = Preconditions.checkNotNull(annotation);
  }

  /**
   * Creates a Mutability.
   * @param pattern is a {@link Printer#format} pattern used to lazily produce a string
   * for error reporting
   * @param arguments are the optional {@link Printer#format} arguments to produce that string
   */
  public static Mutability create(String pattern, Object... arguments) {
    // For efficiency, we could be lazy and use formattable instead of format,
    // but the result is going to be serialized, anyway.
    return new Mutability(Printer.format(pattern, arguments));
  }

  String getAnnotation() {
    return annotation;
  }

  @Override
  public String toString() {
    return String.format(isMutable ? "[%s]" : "(%s)", annotation);
  }

  boolean isMutable() {
    return isMutable;
  }

  /**
   * Freezes this Mutability, marking as immutable all {@link Freezable} objects that use it.
   */
  @Override
  public void close() {
    isMutable = false;
  }

  /**
   * Freezes this Mutability
   * @return it in fluent style.
   */
  public Mutability freeze() {
    close();
    return this;
  }

  /**
   * A MutabilityException will be thrown when the user attempts to mutate an object he shouldn't.
   */
  static class MutabilityException extends Exception {
    MutabilityException(String message) {
      super(message);
    }
  }

  /**
   * Each {@link Freezable} object possesses a revokable Mutability attribute telling whether
   * the object is still mutable. All {@link Freezable} objects created in the same
   * {@link Environment} will share the same Mutability, inherited from this {@link Environment}.
   * Only evaluation in the same {@link Environment} is allowed to mutate these objects,
   * and only until the Mutability is irreversibly revoked.
   */
  public interface Freezable {
    /**
     * Returns the {@link Mutability} capability associated with this Freezable object.
     */
    Mutability mutability();
  }

  /**
   * Checks that this Freezable object can be mutated from the given {@link Environment}.
   * @param object a Freezable object that we check is still mutable.
   * @param env the {@link Environment} attempting the mutation.
   * @throws MutabilityException when the object was frozen already, or is from another context.
   */
  public static void checkMutable(Freezable object, Environment env)
      throws MutabilityException {
    if (!object.mutability().isMutable()) {
      throw new MutabilityException("trying to mutate a frozen object");
    }
    // Consider an {@link Environment} e1, in which is created {@link UserDefinedFunction} f1,
    // that closes over some variable v1 bound to list l1. If somehow, via the magic of callbacks,
    // f1 or l1 is passed as argument to some function f2 evaluated in {@link environment} e2
    // while e1 is be mutable, e2, being a different {@link Environment}, should not be
    // allowed to mutate objects from e1. It's a bug, that shouldn't happen in our current code
    // base, so we throw an AssertionError. If in the future such situations are allowed to happen,
    // then we should throw a MutabilityException instead.
    if (!object.mutability().equals(env.mutability())) {
      throw new AssertionError("trying to mutate an object from a different context");
    }
  }

  public static final Mutability IMMUTABLE = create("IMMUTABLE").freeze();
}
