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
import java.util.IdentityHashMap;

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
 * <p>When a Starlark program iterates over a mutable sequence value in a for-loop or comprehension,
 * the sequence value becomes temporarily immutable for the duration of the loop. Conceptually, the
 * value maintains a counter of active iterations, and the interpreter notifies the {@code
 * Freezable} value before and after the loop so that it can alter its counter by calling its {@code
 * updateIteratorCount} method. While the counter value is nonzero, the value should cause all
 * attempts to mutate it to fail. The temporary immutability applies only to the sequence itself,
 * not to its elements. Once a mutable sequence becomes frozen, there is no need to count active
 * iterators (and doing so would be racy as frozen objects may be published to other Starlark
 * threads). The default implementation of {@code updateIteratorCount} uses a set of counters in the
 * Mutability, but a Freezable object may define a more efficient intrusive counter implementation.
 *
 * <p>We follow two disciplines to ensure safety. First, all mutation methods of a Freezable value
 * must confirm that the value's Mutability is not yet frozen, nor is the value temporarily
 * immutable due to active iterators.
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

  // Maps each temporarily frozen Freezable value to the (positive) count of active iterators over
  // the value. This field is set to null when the Mutability becomes permanently frozen, at which
  // point there is no need to track iterators. This map does not contain Freezable values that
  // define their own implementation of updateIteratorCount.
  private IdentityHashMap<Freezable, Integer> iteratorCount =
      new IdentityHashMap<>(10); // 10 nested for-loops seems plenty

  // An optional list of values that are formatted with toString and joined with spaces to yield the
  // "annotation", an internal name describing the purpose of this Mutability.
  private final Object[] annotation;

  /** Controls access to {@link Freezable#unsafeShallowFreeze}. */
  private final boolean allowsUnsafeShallowFreeze;

  private Mutability(Object[] annotation, boolean allowsUnsafeShallowFreeze) {
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
    return (isFrozen() ? "(" : "[") + getAnnotation() + (isFrozen() ? ")" : "]");
  }

  public boolean isFrozen() {
    return this.iteratorCount == null;
  }

  // Defines the default behavior of mutable Freezable sequence values,
  // which become temporarily immutable while there are active iterators.
  private boolean updateIteratorCount(Freezable x, int delta) {
    if (isFrozen()) {
      return false;
    }
    int i = this.iteratorCount.getOrDefault(x, 0);
    if (delta > 0) {
      i++;
      this.iteratorCount.put(x, i);
    } else if (delta < 0) {
      i--;
      if (i == 0) {
        this.iteratorCount.remove(x);
      } else if (i > 0) {
        this.iteratorCount.put(x, i);
      } else {
        throw new IllegalStateException("zero value in this.iteratorCount");
      }
    }
    return i > 0;
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
    this.iteratorCount = null;
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
     * Registers a change to this Freezable's iterator count and reports whether it is temporarily
     * immutable.
     *
     * <p>If the value is permanently frozen ({@code mutability().isFrozen()), this function is a
     * no-op that returns false.
     *
     * <p>Otherwise, if delta is positive, this increments the count of active iterators over the
     * value, causing it to appear temporarily frozen (if it wasn't already). If delta is negative,
     * the counter is decremented, and if delta is zero the counter is unchanged. It is illegal to
     * decrement the counter if it was already zero. The return value is true if the count is
     * positive after the change, and false otherwise.
     *
     * <p>The default implementation stores the counter of iterators in a hash table in the
     * Mutability, but a subclass of Freezable may define a more efficient implementation such as an
     * integer field in the freezable value itself.
     *
     * <p>Call this function with a positive value when starting an iteration and with a negative
     * value when ending it.
     */
    default boolean updateIteratorCount(int delta) {
      return mutability().updateIteratorCount(this, delta);
    }

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
   * A {@code Mutability} indicating that a value is deeply immutable.
   *
   * <p>It is not associated with any particular {@link StarlarkThread}.
   */
  public static final Mutability IMMUTABLE = create("IMMUTABLE").freeze();
}
