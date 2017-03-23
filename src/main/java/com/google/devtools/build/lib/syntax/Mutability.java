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

import com.google.common.base.Function;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.util.Preconditions;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;

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
 * <p>In addition, a Mutability tracks which of its objects are temporarily locked from mutation.
 * This is used to prevent modification of iterables during loops. A Freezable may be locked
 * multiple times (e.g., nested loops over the same iterable). Locking an object does not prohibit
 * mutating its deeply contained values, e.g. in the case of a list of lists.
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
  // For each locked Freezable, store all Locations where it is locked.
  // This field is set null once the Mutability is closed. This saves some
  // space, and avoids a concurrency bug from multiple Skylark modules
  // accessing the same Mutability at once.
  private IdentityHashMap<Freezable, List<Location>> lockedItems;
  private final String annotation; // For error reporting.

  /**
   * Creates a Mutability.
   * @param annotation an Object used for error reporting,
   * describing to the user the context in which this Mutability was active.
   */
  private Mutability(String annotation) {
    this.isMutable = true;
    // Seems unlikely that we'll often lock more than 10 things at once.
    this.lockedItems = new IdentityHashMap<>(10);
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
   * Return whether a {@link Freezable} belonging to this Mutability is currently locked.
   * Frozen objects are not considered locked, though they are of course immutable nonetheless.
   */
  public boolean isLocked(Freezable object) {
    if (!object.mutability().equals(this)) {
      throw new AssertionError("trying to check the lock of an object from a different context");
    }
    if (!isMutable) {
      return false;
    }
    return lockedItems.containsKey(object);
  }

  /**
   * For a locked {@link Freezable} that belongs to this Mutability, return a List of the
   * {@link Location}s corresponding to its current locks.
   */
  public List<Location> getLockLocations(Freezable object) {
    if (!isLocked(object)) {
      throw new AssertionError("trying to get lock locations for an object that is not locked");
    }
    return lockedItems.get(object);
  }

  /**
   * Add a lock on a {@link Freezable} belonging to this Mutability. The object cannot be
   * mutated until all locks on it are gone. For error reporting purposes each lock is
   * associated with its originating {@link Location}.
   */
  public void lock(Freezable object, Location loc) {
    if (!object.mutability().equals(this)) {
      throw new AssertionError("trying to lock an object from a different context");
    }
    if (!isMutable) {
      return;
    }
    List<Location> locList;
    if (!lockedItems.containsKey(object)) {
      locList = new ArrayList<>();
      lockedItems.put(object, locList);
    } else {
      locList = lockedItems.get(object);
    }
    locList.add(loc);
  }

  /**
   * Remove the lock for a given {@link Freezable} that is associated with the given
   * @{link Location}. It is an error if {@code object} does not belong to this mutability,
   * or has no lock corresponding to {@code loc}.
   */
  public void unlock(Freezable object, Location loc) {
    if (!object.mutability().equals(this)) {
      throw new AssertionError("trying to unlock an object from a different context");
    }
    if (!isMutable) {
      // It's okay if we somehow got frozen while there were still locked objects.
      return;
    }
    if (!lockedItems.containsKey(object)) {
      throw new AssertionError("trying to unlock an object that is not locked");
    }
    List<Location> locList = lockedItems.get(object);
    boolean changed = locList.remove(loc);
    if (!changed) {
      throw new AssertionError(Printer.format("trying to unlock an object for a location at which "
          + "it was not locked (%r)", loc));
    }
    if (locList.isEmpty()) {
      lockedItems.remove(object);
    }
  }

  /**
   * Freezes this Mutability, marking as immutable all {@link Freezable} objects that use it.
   */
  @Override
  public void close() {
    // No need to track per-Freezable info since everything is immutable now.
    lockedItems = null;
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
     * This should not change over the lifetime of the object.
     */
    Mutability mutability();
  }

  /**
   * Checks that this Freezable object can be mutated from the given {@link Environment}.
   * If the object is mutable, it must be from the environment.
   * @param object a Freezable object that we check is still mutable.
   * @param env the {@link Environment} attempting the mutation.
   * @throws MutabilityException when the object was frozen already, or is locked.
   */
  public static void checkMutable(Freezable object, Environment env)
      throws MutabilityException {
    if (!object.mutability().isMutable()) {
      // Throw MutabilityException, not AssertionError, even if the object was from
      // another context.
      throw new MutabilityException("trying to mutate a frozen object");
    }

    // Consider an {@link Environment} e1, in which is created {@link UserDefinedFunction} f1,
    // that closes over some variable v1 bound to list l1. If somehow, via the magic of callbacks,
    // f1 or l1 is passed as an argument to some function f2 evaluated in {@link Environment} e2
    // while e1 is still mutable, then e2, being a different {@link Environment}, should not be
    // allowed to mutate objects from e1. It's a bug, that shouldn't happen in our current code
    // base, so we throw an AssertionError. If in the future such situations are allowed to happen,
    // then we should throw a MutabilityException instead.
    if (!object.mutability().equals(env.mutability())) {
      throw new AssertionError("trying to mutate an object from a different context");
    }

    if (env.mutability().isLocked(object)) {
      Iterable<String> locs = Iterables.transform(env.mutability().getLockLocations(object),
          new Function<Location, String>() {
              @Override
              public String apply(Location loc) {
                return loc.print();
              }});
      throw new MutabilityException(
          "trying to mutate a locked object (is it currently being iterated over by a for loop "
          + "or comprehension?)\n"
          + "Object locked at the following location(s): "
          + String.join(", ", locs));
    }
  }

  public static final Mutability IMMUTABLE = create("IMMUTABLE").freeze();
}
