// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules.apple;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import javax.annotation.Nullable;

/**
 * Tests that a given comparator (or the implementation of {@link Comparable}) is correct. To use,
 * repeatedly call {@link #addEqualityGroup(Object...)} with sets of objects that should be equal.
 * The calls to {@link #addEqualityGroup(Object...)} must be made in sorted order. Then call {@link
 * #testCompare()} to test the comparison. For example:
 *
 * <pre>{@code
 * new ComparatorTester()
 *     .addEqualityGroup(1)
 *     .addEqualityGroup(2)
 *     .addEqualityGroup(3)
 *     .testCompare();
 * }</pre>
 */
public class ComparatorTester {
  @SuppressWarnings({"unchecked", "rawtypes"})
  private final @Nullable Comparator comparator;

  /** The items that we are checking, stored as a sorted set of equivalence classes. */
  private final List<List<Object>> equalityGroups;

  /**
   * Creates a new instance that tests the order of objects using the natural order (as defined by
   * {@link Comparable}).
   */
  public ComparatorTester() {
    this(null);
  }

  /**
   * Creates a new instance that tests the order of objects using the given comparator. Or, if the
   * comparator is {@code null}, the natural ordering (as defined by {@link Comparable})
   */
  public ComparatorTester(@Nullable Comparator<?> comparator) {
    this.equalityGroups = new ArrayList<>();
    this.comparator = comparator;
  }

  /**
   * Adds a set of objects to the test which should all compare as equal. All of the elements in
   * {@code objects} must be greater than any element of {@code objects} in a previous call to
   * {@link #addEqualityGroup(Object...)}.
   *
   * @return {@code this} (to allow chaining of calls)
   */
  public ComparatorTester addEqualityGroup(Object... objects) {
    Preconditions.checkNotNull(objects);
    Preconditions.checkArgument(objects.length > 0, "Array must not be empty");
    equalityGroups.add(ImmutableList.copyOf(objects));
    return this;
  }

  @SuppressWarnings({"unchecked", "rawtypes"})
  private int compare(Object a, Object b) {
    int compareValue;
    if (comparator == null) {
      compareValue = ((Comparable) a).compareTo(b);
    } else {
      compareValue = comparator.compare(a, b);
    }
    return compareValue;
  }

  public final void testCompare() {
    for (int referenceIndex = 0; referenceIndex < equalityGroups.size(); referenceIndex++) {
      for (Object reference : equalityGroups.get(referenceIndex)) {
        testNullCompare(reference);
        testClassCast(reference);
        for (int otherIndex = 0; otherIndex < equalityGroups.size(); otherIndex++) {
          for (Object other : equalityGroups.get(otherIndex)) {
            assertThat(compare(reference, other))
                .isEqualTo(Integer.compare(referenceIndex, otherIndex));
          }
        }
      }
    }
  }

  private void testNullCompare(Object obj) {
    // Comparator does not require any specific behavior for null.
    if (comparator == null) {
      assertThrows(
          "Expected NullPointerException in " + obj + ".compare(null)",
          NullPointerException.class,
          () -> compare(obj, null));
    }
  }

  @SuppressWarnings("unchecked")
  private void testClassCast(Object obj) {
    if (comparator == null) {
      assertThrows(
          "Expected ClassCastException in " + obj + ".compareTo(otherObject)",
          ClassCastException.class,
          () -> compare(obj, ICanNotBeCompared.INSTANCE));
    }
  }

  private static final class ICanNotBeCompared {
    static final ComparatorTester.ICanNotBeCompared INSTANCE = new ICanNotBeCompared();
  }
}
