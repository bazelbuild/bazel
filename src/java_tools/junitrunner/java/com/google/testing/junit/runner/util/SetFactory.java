// Copyright 2016 The Bazel Authors. All Rights Reserved.
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

package com.google.testing.junit.runner.util;

import static java.util.Collections.emptySet;
import static java.util.Collections.unmodifiableSet;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * A {@link Factory} implementation used to implement {@link Set} bindings. This factory always
 * returns a new {@link Set} instance for each call to {@link #get} (as required by {@link Factory})
 * whose elements are populated by subsequent calls to their {@link Supplier#get} methods.
 */
public final class SetFactory<T> implements Factory<Set<T>> {
  /**
   * The maximum value for a signed 32-bit integer that is equal to a power of 2.
   */
  private static final int MAX_POWER_OF_TWO = 1 << (Integer.SIZE - 2);

  private static final Factory<Set<Object>> EMPTY_FACTORY =
      new Factory<Set<Object>>() {
        @Override
        public Set<Object> get() {
          return emptySet();
        }
      };

  @SuppressWarnings({"unchecked", "rawtypes"}) // safe covariant cast
  public static <T> Factory<Set<T>> empty() {
    return (Factory) EMPTY_FACTORY;
  }

  /**
   * Constructs a new {@link Builder} for a {@link SetFactory} with {@code individualSupplierSize}
   * individual {@code Supplier<T>} and {@code collectionSupplierSize} {@code
   * Supplier<Collection<T>>} instances.
   */
  public static <T> Builder<T> builder(int individualSupplierSize, int collectionSupplierSize) {
    return new Builder<T>(individualSupplierSize, collectionSupplierSize);
  }

  /**
   * A builder to accumulate {@code Supplier<T>} and {@code Supplier<Collection<T>>} instances.
   * These are only intended to be single-use and from within generated code. Do <em>NOT</em> add
   * providers after calling {@link #build()}.
   */
  public static final class Builder<T> {
    private final List<Supplier<T>> individualSuppliers;
    private final List<Supplier<Collection<T>>> collectionSuppliers;

    private Builder(int individualSupplierSize, int collectionSupplierSize) {
      individualSuppliers = presizedList(individualSupplierSize);
      collectionSuppliers = presizedList(collectionSupplierSize);
    }

    @SuppressWarnings("unchecked")
    public Builder<T> addSupplier(Supplier<? extends T> individualSupplier) {
      assert individualSupplier != null : "Codegen error? Null provider";
      // TODO(ronshapiro): Store a List<? extends Supplier<T>> and avoid the cast to Supplier<T>
      individualSuppliers.add((Supplier<T>) individualSupplier);
      return this;
    }

    @SuppressWarnings("unchecked")
    public Builder<T> addCollectionSupplier(
        Supplier<? extends Collection<? extends T>> collectionSupplier) {
      assert collectionSupplier != null : "Codegen error? Null provider";
      collectionSuppliers.add((Supplier<Collection<T>>) collectionSupplier);
      return this;
    }

    public SetFactory<T> build() {
      assert !hasDuplicates(individualSuppliers)
          : "Codegen error?  Duplicates in the provider list";
      assert !hasDuplicates(collectionSuppliers)
          : "Codegen error?  Duplicates in the provider list";

      return new SetFactory<T>(individualSuppliers, collectionSuppliers);
    }
  }

  private final List<Supplier<T>> individualSuppliers;
  private final List<Supplier<Collection<T>>> collectionSuppliers;

  private SetFactory(
      List<Supplier<T>> individualSuppliers, List<Supplier<Collection<T>>> collectionSuppliers) {
    this.individualSuppliers = individualSuppliers;
    this.collectionSuppliers = collectionSuppliers;
  }

  /**
   * Returns a {@link Set} whose iteration order is that of the elements given by each of the
   * providers, which are invoked in the order given at creation.
   *
   * @throws NullPointerException if any of the delegate {@link Set} instances or elements therein
   *     are {@code null}
   */
  @Override
  public Set<T> get() {
    int size = individualSuppliers.size();
    // Profiling revealed that this method was a CPU-consuming hotspot in some applications, so
    // these loops were changed to use c-style for.  Versus enhanced for-each loops, C-style for is
    // faster for ArrayLists, at least through Java 8.

    List<Collection<T>> providedCollections =
        new ArrayList<Collection<T>>(collectionSuppliers.size());
    for (int i = 0, c = collectionSuppliers.size(); i < c; i++) {
      Collection<T> providedCollection = collectionSuppliers.get(i).get();
      size += providedCollection.size();
      providedCollections.add(providedCollection);
    }

    Set<T> providedValues = newHashSetWithExpectedSize(size);
    for (int i = 0, c = individualSuppliers.size(); i < c; i++) {
      providedValues.add(checkNotNull(individualSuppliers.get(i).get()));
    }
    for (int i = 0, c = providedCollections.size(); i < c; i++) {
      for (T element : providedCollections.get(i)) {
        providedValues.add(checkNotNull(element));
      }
    }

    return unmodifiableSet(providedValues);
  }

  /**
   * Returns true if at least one pair of items in {@code list} are equals.
   */
  private static boolean hasDuplicates(List<?> list) {
    if (list.size() < 2) {
      return false;
    }
    Set<Object> asSet = new HashSet<Object>(list);
    return list.size() != asSet.size();
  }

  /**
   * Returns a new list that is pre-sized to {@code size}, or {@link Collections#emptyList()} if
   * empty. The list returned is never intended to grow beyond {@code size}, so adding to a list
   * when the size is 0 is an error.
   */
  private static <T> List<T> presizedList(int size) {
    if (size == 0) {
      return Collections.emptyList();
    }
    return new ArrayList<T>(size);
  }

  /**
   * Creates a {@link HashSet} instance, with a high enough "initial capcity" that it
   * <em>should</em> hold {@code expectedSize} elements without growth.
   */
  private static <T> HashSet<T> newHashSetWithExpectedSize(int expectedSize) {
    return new HashSet<T>(calculateInitialCapacity(expectedSize));
  }

  private static int calculateInitialCapacity(int expectedSize) {
    if (expectedSize < 3) {
      return expectedSize + 1;
    }
    if (expectedSize < MAX_POWER_OF_TWO) {
      // This is the calculation used in JDK8 to resize when a putAll
      // happens; it seems to be the most conservative calculation we
      // can make.  0.75 is the default load factor.
      return (int) (expectedSize / 0.75F + 1.0F);
    }
    return Integer.MAX_VALUE; // any large value
  }

  private static <T> T checkNotNull(T reference) {
    if (reference == null) {
      throw new NullPointerException();
    }
    return reference;
  }
}

