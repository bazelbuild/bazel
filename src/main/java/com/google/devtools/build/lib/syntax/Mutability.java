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

import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.util.Preconditions;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Formattable;
import java.util.IdentityHashMap;
import java.util.List;

/**
 * An object that manages the capability to mutate Skylark objects and their {@link Environment}s.
 * Collectively, the managed objects are called {@link Freezable}s.
 *
 * <p>Each {@code Environment}, and each of the mutable Skylark values (i.e., {@link
 * SkylarkMutable}s) that are created in that {@code Environment}, holds a pointer to the same
 * {@code Mutability} instance. Once the {@code Environment} is done evaluating, its {@code
 * Mutability} is irreversibly closed ("frozen"). At that point, it is no longer possible to change
 * either the bindings in that {@code Environment} or the state of its objects. This protects each
 * {@code Environment} from unintentional and unsafe modification.
 *
 * <p>{@code Mutability}s enforce isolation between {@code Environment}s; it is illegal for an
 * evaluation in one {@code Environment} to affect the bindings or values of another. In particular,
 * the {@code Environment} for any Skylark module is frozen before its symbols can be imported for
 * use by another module. Each individual {@code Environment}'s evaluation is single-threaded, so
 * this isolation also translates to thread safety. Any number of threads may simultaneously access
 * frozen data.
 *
 * <p>Although the mutability pointer of a {@code Freezable} contains some debugging information
 * about its context, this should not affect the {@code Freezable}'s semantics. From a behavioral
 * point of view, the only thing that matters is whether the {@code Mutability} is frozen, not what
 * particular {@code Mutability} object is pointed to.
 *
 * <p>A {@code Mutability} also tracks which {@code Freezable} objects in its {@code Environment}
 * are temporarily locked from mutation. This is used to prevent modification of iterables during
 * loops. A {@code Freezable} may be locked multiple times (e.g., nested loops over the same
 * iterable). Locking an object does not prohibit mutating its deeply contained values, such as in
 * the case of a list of lists.
 *
 * <p>We follow two disciplines to ensure safety. First, all mutation methods of a {@code Freezable}
 * must take in a {@code Mutability} as a parameter, and confirm that
 * <ol>
 *   <li>the {@code Freezable} is not yet frozen,
 *   <li>the given {@code Mutability} matches the one referred to by the {@code Freezable}, and
 *   <li>the {@code Freezable} is not locked.
 * </ol>
 * It is a high-level error ({@link MutabilityException}, which gets translated to {@link
 * EvalException}) to attempt to modify a frozen or locked value. But it is a low-level error
 * ({@link IllegalArgumentException}) to attempt to modify a value using the wrong {@link
 * Mutability} instance, since the user shouldn't be able to trigger this situation under normal
 * circumstances.
 *
 * <p>Second, {@code Mutability}s are created using the try-with-resource style:
 * <pre>{@code
 * try (Mutability mutability = Mutability.create(fmt, ...)) { ... }
 * }</pre>
 * The general pattern is to create a {@code Mutability}, build an {@code Environment}, mutate that
 * {@code Environment} and its objects, and possibly return the result from within the {@code try}
 * block, relying on the try-with-resource construct to ensure that everything gets frozen before
 * the result is used. The only code that should create a {@code Mutability} without using
 * try-with-resource is test code that is not part of the Bazel jar.
 *
 * We keep some (unchecked) invariants regarding where {@code Mutability} objects may appear in a
 * compound value.
 * <ol>
 *   <li>There is always at most one unfrozen {@code Mutability}, corresponding to the current
 *       {@code Environment}'s evaluation.
 *   <li>Whenever a new mutable Skylark value is created, its {@code Mutability} is either the
 *       current {@code Environment}'s {@code Mutability}, or else it is the special static
 *       instance, {@link #IMMUTABLE}, which represents that a value is at least shallowly
 *       immutable.
 * </ol>
 * It follows that an unfrozen value can never appear as the child of a frozen value unless the
 * frozen value's {@code Mutability} is {@code IMMUTABLE}. This can be used to prune traversals that
 * check whether a value is deeply immutable.
 */
// TODO(bazel-team): The safe try-with-resources usage pattern can be enforced through the use of a
// higher-order function.
public final class Mutability implements AutoCloseable, Serializable {

  /**
   * If true, mutation of any {@link Freezable} associated with this {@code Mutability} is
   * disallowed.
   */
  private boolean isFrozen;

  /**
   * For each locked {@link Freezable}, stores all {@link Location}s where it is locked.
   *
   * This field is set null once the {@code Mutability} is closed. This saves some space, and avoids
   * a concurrency bug from multiple Skylark modules accessing the same {@code Mutability} at once.
   */
  private IdentityHashMap<Freezable, List<Location>> lockedItems;

  /** For error reporting; a name for the context in which this {@code Mutability} is used. */
  private final Formattable annotation;

  private Mutability(Formattable annotation) {
    this.isFrozen = false;
    // Seems unlikely that we'll often lock more than 10 things at once.
    this.lockedItems = new IdentityHashMap<>(10);
    this.annotation = Preconditions.checkNotNull(annotation);
  }

  /**
   * Creates a Mutability.
   *
   * @param pattern is a {@link Printer#format} pattern used to lazily produce a string name
   *     for error reporting
   * @param arguments are the optional {@link Printer#format} arguments to produce that string
   */
  public static Mutability create(String pattern, Object... arguments) {
    return new Mutability(Printer.formattable(pattern, arguments));
  }

  public String getAnnotation() {
    return annotation.toString();
  }

  @Override
  public String toString() {
    return String.format(isFrozen ? "(%s)" : "[%s]", annotation);
  }

  public boolean isFrozen() {
    return isFrozen;
  }

  /**
   * Return whether a {@link Freezable} belonging to this {@code Mutability} is currently locked.
   * Frozen objects are not considered locked, though they are of course immutable nonetheless.
   *
   * @throws IllegalArgumentException if the {@code Freezable} does not belong to this {@code
   *     Mutability}
   */
  public boolean isLocked(Freezable object) {
    Preconditions.checkArgument(object.mutability().equals(this),
        "trying to check the lock of an object from a different context");
    if (isFrozen) {
      return false;
    }
    return lockedItems.containsKey(object);
  }

  /**
   * For a locked {@link Freezable} that belongs to this {@code Mutability}, return a List of the
   * {@link Location}s corresponding to its current locks.
   *
   * @throws IllegalArgumentException if the {@code Freezable} does not belong to this {@code
   *     Mutability}
   */
  public List<Location> getLockLocations(Freezable object) {
    Preconditions.checkArgument(isLocked(object),
        "trying to get lock locations for an object that is not locked");
    return lockedItems.get(object);
  }

  /**
   * Add a lock on a {@link Freezable} belonging to this {@code Mutability}. The object cannot be
   * mutated until all locks on it are gone. For error reporting purposes each lock is
   * associated with its originating {@link Location}.
   *
   * @throws IllegalArgumentException if the {@code Freezable} does not belong to this {@code
   *     Mutability}
   */
  public void lock(Freezable object, Location loc) {
    Preconditions.checkArgument(object.mutability().equals(this),
        "trying to lock an object from a different context");
    if (isFrozen) {
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
   * Remove the lock for a given {@link Freezable} that is associated with the given {@link
   * Location}.
   *
   * @throws IllegalArgumentException if the object does not belong to this {@code Mutability}, or
   *     if the object has no lock corresponding to {@code loc}
   */
  public void unlock(Freezable object, Location loc) {
    Preconditions.checkArgument(object.mutability().equals(this),
        "trying to unlock an object from a different context");
    if (isFrozen) {
      // It's okay if we somehow got frozen while there were still locked objects.
      return;
    }
    Preconditions.checkArgument(lockedItems.containsKey(object),
        "trying to unlock an object that is not locked");

    List<Location> locList = lockedItems.get(object);
    if (!locList.remove(loc)) {
      throw new IllegalArgumentException(
          Printer.format(
              "trying to unlock an object for a location at which it was not locked (%r)", loc));
    }
    if (locList.isEmpty()) {
      lockedItems.remove(object);
    }
  }

  /**
   * Freezes this {@code Mutability}, rendering all {@link Freezable} objects that refer to it
   * immutable.
   *
   * Note that freezing does not directly touch all the {@code Freezables}, so this operation is
   * constant-time.
   *
   * @return this object, in the fluent style
   */
  public Mutability freeze() {
    // No need to track per-Freezable info since everything is immutable now.
    lockedItems = null;
    isFrozen = true;
    return this;
  }

  @Override
  public void close() {
    freeze();
  }

  /** Indicates an illegal attempt to mutate a frozen or locked {@link Freezable}. */
  static class MutabilityException extends Exception {
    MutabilityException(String message) {
      super(message);
    }
  }

  /**
   * An object that refers to a {@link Mutability} to decide whether to allow mutation. All
   * {@link Freezable} Skylark objects created within a given {@link Environment} will share the
   * same {@code Mutability} as that {@code Environment}.
   */
  public interface Freezable {
    /**
     * Returns the {@link Mutability} associated with this {@code Freezable}. This should not change
     * over the lifetime of the object.
     */
    Mutability mutability();
  }

  /**
   * Checks that the given {@code Freezable} can be mutated using the given {@code Mutability}, and
   * throws an exception if it cannot.
   *
   * @throws MutabilityException if the object is either frozen or locked
   * @throws IllegalArgumentException if the given {@code Mutability} is not the same as the one
   *     the {@code Freezable} is associated with
   */
  public static void checkMutable(Freezable object, Mutability mutability)
      throws MutabilityException {
    if (object.mutability().isFrozen()) {
      // Throw MutabilityException, not IllegalArgumentException, even if the object was from
      // another context.
      throw new MutabilityException("trying to mutate a frozen object");
    }

    // Consider an {@link Environment} e1, in which is created {@link UserDefinedFunction} f1, that
    // closes over some variable v1 bound to list l1. If somehow, via the magic of callbacks, f1 or
    // l1 is passed as an argument to some function f2 evaluated in {@link Environment} e2 while e1
    // is still mutable, then e2, being a different {@link Environment}, should not be allowed to
    // mutate objects from e1. It's a bug, that shouldn't happen in our current code base, so we
    // throw an IllegalArgumentException. If in the future such situations are allowed to happen,
    // then we should throw a MutabilityException instead.
    if (!object.mutability().equals(mutability)) {
      throw new IllegalArgumentException("trying to mutate an object from a different context");
    }

    if (mutability.isLocked(object)) {
      Iterable<String> locs =
          Iterables.transform(mutability.getLockLocations(object), Location::print);
      throw new MutabilityException(
          "trying to mutate a locked object (is it currently being iterated over by a for loop "
          + "or comprehension?)\n"
          + "Object locked at the following location(s): "
          + String.join(", ", locs));
    }
  }

  /**
   * An instance indicating that a value is shallowly immutable. Its children may or may not be
   * mutable.
   *
   * <p>This instance is treated specially with regard to the {@code Mutability} invariant. Usually
   * an immutable value cannot directly or indirectly contain a mutable one. But an immutable value
   * with this {@code Mutability} may.
   *
   * <p>In practice, this instance is used as the {@code Mutability} for tuples. It may also be used
   * for certain lists and dictionaries that are immutable from creation -- though in general we
   * prefer to use tuples rather than always-frozen lists.
   */
  // TODO(bazel-team): We might be able to remove this instance, and instead have tuples and other
  // always-immutable things store the same Mutability as other values in that environment. Then we
  // can simplify the Mutability invariant, and implement deep-immutability checking in constant
  // time.
  //
  // This would also affect structs (SkylarkInfo). Maybe they would implement an interface similar
  // to SkylarkMutable, or the relevant methods could be worked into SkylarkValue.
  public static final Mutability IMMUTABLE = create("IMMUTABLE").freeze();
}
