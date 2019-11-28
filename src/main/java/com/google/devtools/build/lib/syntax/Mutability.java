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

import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.events.Location;
import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;

/**
 * An object that manages the capability to mutate Skylark objects and their {@link
 * StarlarkThread}s. Collectively, the managed objects are called {@link Freezable}s.
 *
 * <p>Each {@code StarlarkThread}, and each of the mutable Skylark values (i.e., {@link
 * StarlarkMutable}s) that are created in that {@code StarlarkThread}, holds a pointer to the same
 * {@code Mutability} instance. Once the {@code StarlarkThread} is done evaluating, its {@code
 * Mutability} is irreversibly closed ("frozen"). At that point, it is no longer possible to change
 * either the bindings in that {@code StarlarkThread} or the state of its objects. This protects
 * each {@code StarlarkThread} from unintentional and unsafe modification.
 *
 * <p>{@code Mutability}s enforce isolation between {@code StarlarkThread}s; it is illegal for an
 * evaluation in one {@code StarlarkThread} to affect the bindings or values of another. In
 * particular, the {@code StarlarkThread} for any Skylark module is frozen before its symbols can be
 * imported for use by another module. Each individual {@code StarlarkThread}'s evaluation is
 * single-threaded, so this isolation also translates to thread safety. Any number of threads may
 * simultaneously access frozen data. (The {@code Mutability} itself is also thread-safe if and only
 * if it is frozen.}
 *
 * <p>Although the mutability pointer of a {@code Freezable} contains some debugging information
 * about its context, this should not affect the {@code Freezable}'s semantics. From a behavioral
 * point of view, the only thing that matters is whether the {@code Mutability} is frozen, not what
 * particular {@code Mutability} object is pointed to.
 *
 * <p>A {@code Mutability} also tracks which {@code Freezable} objects in its {@code StarlarkThread}
 * are temporarily locked from mutation. This is used to prevent modification of iterables during
 * loops. A {@code Freezable} may be locked multiple times (e.g., nested loops over the same
 * iterable). Locking an object does not prohibit mutating its deeply contained values, such as in
 * the case of a list of lists.
 *
 * <p>We follow two disciplines to ensure safety. First, all mutation methods of a {@code Freezable}
 * must take in a {@code Mutability} as a parameter, and confirm that
 *
 * <ol>
 *   <li>the {@code Freezable} is not yet frozen,
 *   <li>the given {@code Mutability} matches the one referred to by the {@code Freezable}, and
 *   <li>the {@code Freezable} is not locked.
 * </ol>
 *
 * It is a high-level error ({@link MutabilityException}, which gets translated to {@link
 * EvalException}) to attempt to modify a frozen or locked value. But it is a low-level error
 * ({@link IllegalArgumentException}) to attempt to modify a value using the wrong {@link
 * Mutability} instance, since the user shouldn't be able to trigger this situation under normal
 * circumstances.
 *
 * <p>Second, {@code Mutability}s are created using the try-with-resource style:
 *
 * <pre>{@code
 * try (Mutability mutability = Mutability.create(name, ...)) { ... }
 * }</pre>
 *
 * The general pattern is to create a {@code Mutability}, build an {@code StarlarkThread}, mutate
 * that {@code StarlarkThread} and its objects, and possibly return the result from within the
 * {@code try} block, relying on the try-with-resource construct to ensure that everything gets
 * frozen before the result is used. The only code that should create a {@code Mutability} without
 * using try-with-resource is test code that is not part of the Bazel jar.
 *
 * <p>We keep some (unchecked) invariants regarding where {@code Mutability} objects may appear
 * within a compound value.
 *
 * <ol>
 *   <li>A compound value can never contain an unfrozen {@code Mutability} for any {@code
 *       StarlarkThread} except the one currently being evaluated.
 *   <li>If a value has the special {@link #IMMUTABLE} {@code Mutability}, all of its contents are
 *       themselves deeply immutable too (i.e. have frozen {@code Mutability}s).
 * </ol>
 *
 * It follows that, if these invariants hold, an unfrozen value cannot appear as the child of a
 * value whose {@code Mutability} is already frozen. This knowledge is used by {@link
 * StarlarkMutable#isImmutable} to prune traversals of a compound value.
 *
 * <p>There is a special API for freezing individual values rather than whole {@code
 * StarlarkThread}s. Because this API makes it easier to violate the above invariants, you should
 * avoid using it if at all possible; at the moment it is only used for serialization. Under this
 * API, you may call {@link Freezable#unsafeShallowFreeze} to reset a value's {@code Mutability}
 * pointer to be {@link #IMMUTABLE}. This operation has no effect on the {@code Mutability} itself.
 * It is up to the caller to preserve or restore the above invariants by ensuring that any deeply
 * contained values are also frozen. For safety and explicitness, this operation is disallowed
 * unless the {@code Mutability}'s {@link #allowsUnsafeShallowFreeze} method returns true.
 */
public final class Mutability implements AutoCloseable {

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

  // An optional list of values that are formatted with toString and joined with spaces to yield the
  // "annotation", an internal name describing the purpose of this Mutability.
  private final Object[] annotation;

  /** Controls access to {@link Freezable#unsafeShallowFreeze}. */
  private final boolean allowsUnsafeShallowFreeze;

  private Mutability(Object[] annotation, boolean allowsUnsafeShallowFreeze) {
    this.isFrozen = false;
    // Seems unlikely that we'll often lock more than 10 things at once.
    this.lockedItems = new IdentityHashMap<>(10);
    this.annotation = annotation;
    this.allowsUnsafeShallowFreeze = allowsUnsafeShallowFreeze;
  }

  /**
   * Creates a {@code Mutability}.
   *
   * @param annotation a list of objects whose toString representations are joined with spaces to
   *     yield the annotation, an internal name describing the purpose of this Mutability.
   */
  public static Mutability create(Object... annotation) {
    return new Mutability(annotation, /*allowsUnsafeShallowFreeze=*/ false);
  }

  /**
   * Creates a {@code Mutability} whose objects can be individually frozen; see docstrings for
   * {@link Mutability} and {@link Freezable#unsafeShallowFreeze}.
   */
  public static Mutability createAllowingShallowFreeze(Object... annotation) {
    return new Mutability(annotation, /*allowsUnsafeShallowFreeze=*/ true);
  }

  /** Returns the Mutability's "annotation", an internal name describing its purpose. */
  public String getAnnotation() {
    // The annotation string is computed when needed, typically never,
    // to avoid the performance penalty of materializing it eagerly.
    return Joiner.on(" ").join(annotation);
  }

  @Override
  public String toString() {
    return (isFrozen ? "(" : "[") + getAnnotation() + (isFrozen ? ")" : "]");
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
          Starlark.format(
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

  /**
   * Returns whether {@link Freezable}s having this {@code Mutability} allow the {@link
   * #unsafeShallowFreeze} operation.
   */
  public boolean allowsUnsafeShallowFreeze() {
    return allowsUnsafeShallowFreeze;
  }

  /** Indicates an illegal attempt to mutate a frozen or locked {@link Freezable}. */
  static class MutabilityException extends Exception {
    MutabilityException(String message) {
      super(message);
    }
  }

  /**
   * An object that refers to a {@link Mutability} to decide whether to allow mutation. All {@link
   * Freezable} Skylark objects created within a given {@link StarlarkThread} will share the same
   * {@code Mutability} as that {@code StarlarkThread}.
   */
  public interface Freezable {
    /**
     * Returns the {@link Mutability} associated with this {@code Freezable}. This should not change
     * over the lifetime of the object, except by calling {@link #shallowFreeze} if applicable.
     */
    Mutability mutability();

    /**
     * Freezes this object (and not its contents). Use with care.
     *
     * <p>This method is optional (i.e. may throw {@link NotImplementedException}).
     *
     * <p>If this object's {@link Mutability} is 1) not frozen, and 2) has {@link
     * #allowUnsafeShallowFreeze} return true, then the object's {@code Mutability} reference is
     * updated to point to {@link #IMMUTABLE}. Otherwise, this method throws {@link
     * IllegalArgumentException}.
     *
     * <p>It is up to the caller to ensure that any contents of this {@code Freezable} are also
     * frozen in order to preserve/restore the invariant that an immutable value cannot contain a
     * mutable one. Note that {@link StarlarkMutable#isImmutable} correctness and thread-safety are
     * not guaranteed otherwise.
     */
    default void unsafeShallowFreeze() {
      throw new UnsupportedOperationException();
    }

    /**
     * Throws {@link IllegalArgumentException} if the precondition for {@link #unsafeShallowFreeze}
     * is violated. To be used by implementors of {@link #unsafeShallowFreeze}.
     */
    static void checkUnsafeShallowFreezePrecondition(
        Freezable freezable) {
      Mutability mutability = freezable.mutability();
      if (mutability.isFrozen()) {
        // It's not safe to rewrite the Mutability pointer if this is already frozen, because we
        // could be accessed by multiple threads.
        throw new IllegalArgumentException(
            "cannot call unsafeShallowFreeze() on an object whose Mutability is already frozen");
      }
      if (!mutability.allowsUnsafeShallowFreeze()) {
        throw new IllegalArgumentException(
            "cannot call unsafeShallowFreeze() on a mutable object whose Mutability's "
                + "allowsUnsafeShallowFreeze() == false");
      }
    }
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

    // Consider an {@link StarlarkThread} e1, in which is created {@link StarlarkFunction} f1, that
    // closes over some variable v1 bound to list l1. If somehow, via the magic of callbacks, f1 or
    // l1 is passed as an argument to some function f2 evaluated in {@link StarlarkThread} e2 while
    // e1 is still mutable, then e2, being a different {@link StarlarkThread}, should not be allowed
    // to mutate objects from e1. It's a bug, that shouldn't happen in our current code base, so we
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
   * A {@code Mutability} indicating that a value is deeply immutable.
   *
   * <p>It is not associated with any particular {@link StarlarkThread}.
   */
  public static final Mutability IMMUTABLE = create("IMMUTABLE").freeze();
}
