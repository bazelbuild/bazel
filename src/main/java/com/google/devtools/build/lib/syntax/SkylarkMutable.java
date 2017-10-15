// Copyright 2016 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.Iterators;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.skylarkinterface.SkylarkValue;
import com.google.devtools.build.lib.syntax.Mutability.Freezable;
import com.google.devtools.build.lib.syntax.Mutability.MutabilityException;
import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.ListIterator;
import java.util.Map;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * Base class for data structures that are only mutable using a proper, unfrozen {@link Mutability}.
 */
public abstract class SkylarkMutable implements Freezable, SkylarkValue {

  /**
   * Checks whether this object is currently mutable in the given {@link Environment}, and throws
   * an exception if it is not.
   *
   * @deprecated prefer {@link #checkMutable(Location, Mutability)} instead
   */
  @Deprecated
  protected void checkMutable(Location loc, Environment env) throws EvalException {
    checkMutable(loc, env.mutability());
  }

  /**
   * Checks whether this object is currently mutable using the given {@link Mutability}, and throws
   * an exception if it is not.
   *
   * @throws EvalException if the object is not mutable. This may be because the object (i.e., its
   *     {@code Mutability} was frozen, or because it is temporarily locked from mutation (due to
   *     being iterated over by a loop), or because it is associated with a different {@code
   *     Mutability} than the one given.
   */
  protected void checkMutable(Location loc, Mutability mutability) throws EvalException {
    try {
      Mutability.checkMutable(this, mutability);
    } catch (MutabilityException ex) {
      throw new EvalException(loc, ex);
    }
  }

  @Override
  public boolean isImmutable() {
    return mutability().isFrozen();
  }

  /**
   * Add a new lock at {@code loc}. No effect if frozen.
   */
  public void lock(Location loc) {
    mutability().lock(this, loc);
  }

  /**
   * Remove the lock at {@code loc}; such a lock must already exist. No effect if frozen.
   */
  public void unlock(Location loc) {
    mutability().unlock(this, loc);
  }

  /**
   * Base class for a {@link SkylarkMutable} that implements a Java Collections Framework interface.
   * All of the interface's accessors should be supported, while its mutating methods must be
   * disallowed.
   *
   * <p>Think of this as similar to {@link Collections#unmodifiableList}, etc., except that it's an
   * extendable class rather than a method.
   *
   * <p>A subclass implements a specific data structure interface, say {@link List}, and refines the
   * return type of {@link #getContentsUnsafe} to be that interface. The subclass implements all of
   * the interface's accessors such that they defer to the result of {@code getContentsUnsafe}.
   * Accessors such as {@link Collection#iterator()} must return unmodifiable views. The subclass
   * implements final versions of all the interface's mutating methods such that they are marked
   * {@code @Deprecated} and throw {@link UnsupportedOperationException}.
   *
   * <p>A concrete subclass may provide alternative mutating methods that take in a {@link
   * Mutability} and validate that the mutation is allowed using {@link #checkMutable}. This
   * validation must occur <em>before</em> the mutation, not after, in order to ensure that a frozen
   * value cannot be mutated. (I.e., the fact that the check throws {@link EvalException} does not
   * excuse us from illegally mutating a frozen value, since {@code EvalException} is not a fatal
   * error.)
   *
   * <p>Subclasses need not overwrite the default methods added to some data structures in Java 8.
   * since these are defined in terms of the non-default methods.
   */
  abstract static class BaseMutableWrapper extends SkylarkMutable {

    /**
     * The underlying contents, to which read access is forwarded. This object must not be modified
     * without first calling {@link #checkMutable}.
     */
    protected abstract Object getContentsUnsafe();

    @Override
    public boolean equals(Object o) {
      return getContentsUnsafe().equals(o);
    }

    @Override
    public int hashCode() {
      return getContentsUnsafe().hashCode();
    }
  }

  /** Base class for a {@link SkylarkMutable} that is also a {@link Collection}. */
  abstract static class MutableCollection<E> extends BaseMutableWrapper implements Collection<E> {

    @Override
    protected abstract Collection<E> getContentsUnsafe();

    // Reading methods of Collection, in alphabetic order.

    @Override
    public boolean contains(@Nullable Object object) {
      return getContentsUnsafe().contains(object);
    }

    @Override
    public boolean containsAll(Collection<?> collection) {
      return getContentsUnsafe().containsAll(collection);
    }

    @Override
    public boolean isEmpty() {
      return getContentsUnsafe().isEmpty();
    }

    @Override
    public Iterator<E> iterator() {
      return Iterators.unmodifiableIterator(getContentsUnsafe().iterator());
    }

    @Override
    public int size() {
      return getContentsUnsafe().size();
    }

    // toArray() and toArray(T[]) return copies, so we don't need an unmodifiable view.
    @Override
    public Object[] toArray() {
      return getContentsUnsafe().toArray();
    }

    @Override
    public <T> T[] toArray(T[] other) {
      return getContentsUnsafe().toArray(other);
    }

    // (Disallowed) writing methods of Collection, in alphabetic order.

    @Deprecated
    @Override
    public final boolean add(E element) {
      throw new UnsupportedOperationException();
    }

    @Deprecated
    @Override
    public final boolean addAll(Collection<? extends E> collection) {
      throw new UnsupportedOperationException();
    }

    @Deprecated
    @Override
    public final void clear() {
      throw new UnsupportedOperationException();
    }

    @Deprecated
    @Override
    public final boolean remove(Object object) {
      throw new UnsupportedOperationException();
    }

    @Deprecated
    @Override
    public final boolean removeAll(Collection<?> collection) {
      throw new UnsupportedOperationException();
    }

    @Deprecated
    @Override
    public final boolean retainAll(Collection<?> collection) {
      throw new UnsupportedOperationException();
    }
  }

  /** Base class for a {@link SkylarkMutable} that is also a {@link List}. */
  abstract static class BaseMutableList<E> extends MutableCollection<E> implements List<E> {

    @Override
    protected abstract List<E> getContentsUnsafe();

    // Reading methods of List, in alphabetic order.

    @Override
    public E get(int i) {
      return getContentsUnsafe().get(i);
    }

    @Override
    public int indexOf(Object element) {
      return getContentsUnsafe().indexOf(element);
    }

    @Override
    public int lastIndexOf(Object element) {
      return getContentsUnsafe().lastIndexOf(element);
    }

    @Override
    public ListIterator<E> listIterator() {
      return Collections.unmodifiableList(getContentsUnsafe()).listIterator();
    }

    @Override
    public ListIterator<E> listIterator(int index) {
      return Collections.unmodifiableList(getContentsUnsafe()).listIterator(index);
    }

    @Override
    public List<E> subList(int fromIndex, int toIndex) {
      return Collections.unmodifiableList(getContentsUnsafe()).subList(fromIndex, toIndex);
    }

    // (Disallowed) writing methods of List, in alphabetic order.

    @Deprecated
    @Override
    public final void add(int index, E element) {
      throw new UnsupportedOperationException();
    }

    @Deprecated
    @Override
    public final boolean addAll(int index, Collection<? extends E> elements) {
      throw new UnsupportedOperationException();
    }

    @Deprecated
    @Override
    public final E remove(int index) {
      throw new UnsupportedOperationException();
    }

    @Deprecated
    @Override
    public final E set(int index, E element) {
      throw new UnsupportedOperationException();
    }
  }

  /** Base class for a {@link SkylarkMutable} that is also a {@link Map}. */
  abstract static class MutableMap<K, V> extends BaseMutableWrapper implements Map<K, V> {

    @Override
    protected abstract Map<K, V> getContentsUnsafe();

    // Reading methods of Map, in alphabetic order.

    @Override
    public boolean containsKey(Object key) {
      return getContentsUnsafe().containsKey(key);
    }

    @Override
    public boolean containsValue(Object value) {
      return getContentsUnsafe().containsValue(value);
    }

    @Override
    public Set<Map.Entry<K, V>> entrySet() {
      return Collections.unmodifiableMap(getContentsUnsafe()).entrySet();
    }

    @Override
    public V get(Object key) {
      return getContentsUnsafe().get(key);
    }

    @Override
    public boolean isEmpty() {
      return getContentsUnsafe().isEmpty();
    }

    @Override
    public Set<K> keySet() {
      return Collections.unmodifiableMap(getContentsUnsafe()).keySet();
    }

    @Override
    public int size() {
      return getContentsUnsafe().size();
    }

    @Override
    public Collection<V> values() {
      return Collections.unmodifiableMap(getContentsUnsafe()).values();
    }

    // (Disallowed) writing methods of Map, in alphabetic order.

    @Deprecated
    @Override
    public final void clear() {
      throw new UnsupportedOperationException();
    }

    @Deprecated
    @Override
    public final V put(K key, V value) {
      throw new UnsupportedOperationException();
    }

    @Deprecated
    @Override
    public final void putAll(Map<? extends K, ? extends V> map) {
      throw new UnsupportedOperationException();
    }

    @Deprecated
    @Override
    public final V remove(Object key) {
      throw new UnsupportedOperationException();
    }
  }
}
