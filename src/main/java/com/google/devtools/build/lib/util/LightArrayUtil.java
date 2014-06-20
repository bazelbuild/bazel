// Copyright 2014 Google Inc. All rights reserved.
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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.collect.Iterators;

import java.util.Iterator;

/**
 * This class creates a memory-efficient version of an {@link Iterable}, maintaining the order of
 * the passed-in iterable.
 *
 * @param <E> The element type for the Iterable
 */
public class LightArrayUtil<E> {

  private static final Object[] EMPTY = new Object[0];

  private final Class<E> clazz;

  public LightArrayUtil(Class<E> clazz) {
    Preconditions.checkArgument(!clazz.isArray(), "Iterables of arrays are not supported");
    this.clazz = clazz;
  }

  /**
   * Creates a representation of the Iterable in a memory-efficient way.
   */
  public Object create(Iterable<? extends E> iterable) {
    switch (Iterables.size(iterable)) {
      case 0:
        return EMPTY;
      case 1:
        return iterable.iterator().next();
      default:
        return Iterables.toArray(iterable, clazz);
    }
  }

  /**
   * Returns the object as an {@link Iterable}.
   *
   * <p>Note that the {@code iterable} should be created with {@link #create(Iterable)} method.</p>
   */
  public Iterable<E> iterable(Object iterable) {
    if (clazz.isInstance(iterable)) {
      return ImmutableList.of(asSingleton(iterable));
    }
    final E[] array = asArray(iterable);
    if (array.length == 0) {
      return ImmutableList.of();
    }
    return new Iterable<E>() {
      @Override
      public Iterator<E> iterator() {
        return Iterators.forArray(array);
      }
    };
  }

  /**
   * Returns true iff the objects inside {@code iterable} and {@code otherIterable} are equal and
   * are in the same order.
   *
   * </p>Note that the {@code iterable} Object should be created with {@link #create(Iterable)}
   * method.</p>
   */
  public boolean elementsEqual(Object iterable, Iterable<? extends E> otherIterable) {
    Iterator<? extends E> iterator = otherIterable.iterator();
    if (clazz.isInstance(iterable)) {
      return iterator.hasNext() && iterator.next().equals(iterable) && !iterator.hasNext();
    }
    for (E t : asArray(iterable)) {
      if (!(iterator.hasNext() && iterator.next().equals(t))) {
        return false;
      }
    }
    return !iterator.hasNext();
  }

  @SuppressWarnings("unchecked")
  private E[] asArray(Object iterable) {
    return (E[]) iterable;
  }

  @SuppressWarnings("unchecked")
  private E asSingleton(Object iterable) {
    return (E) iterable;
  }
}
