// Copyright 2022 The Bazel Authors. All rights reserved.
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
//

package com.google.devtools.build.lib.util;

import com.google.common.base.Objects;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Sets;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * A set that either contains some elements or is the <i>complete</i> set (semantically contains
 * every possible value of the element type).
 */
public final class MaybeCompleteSet<T> {
  private static final MaybeCompleteSet<Object> COMPLETE = new MaybeCompleteSet<>(null);

  @Nullable private final ImmutableSet<T> internalSet;

  private MaybeCompleteSet(@Nullable ImmutableSet<T> nullableSet) {
    this.internalSet = nullableSet;
  }

  public boolean contains(T value) {
    return internalSet == null || internalSet.contains(value);
  }

  public boolean isComplete() {
    return internalSet == null;
  }

  public boolean isEmpty() {
    return internalSet != null && internalSet.isEmpty();
  }

  public ImmutableSet<T> getElementsIfNotComplete() {
    Preconditions.checkArgument(internalSet != null);
    return internalSet;
  }

  public static <T> MaybeCompleteSet<T> copyOf(Set<T> nonNullableSet) {
    return new MaybeCompleteSet<>(ImmutableSet.copyOf(nonNullableSet));
  }

  @SuppressWarnings("unchecked")
  public static <T> MaybeCompleteSet<T> completeSet() {
    return (MaybeCompleteSet<T>) COMPLETE;
  }

  public static <T> MaybeCompleteSet<T> unionElements(MaybeCompleteSet<T> set1, Set<T> set2) {
    if (set1.isComplete()) {
      return completeSet();
    }
    return copyOf(Sets.union(set1.getElementsIfNotComplete(), set2));
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (!(o instanceof MaybeCompleteSet)) {
      return false;
    }
    MaybeCompleteSet<?> that = (MaybeCompleteSet<?>) o;
    return Objects.equal(internalSet, that.internalSet);
  }

  @Override
  public int hashCode() {
    return Objects.hashCode("MaybeCompleteSet", internalSet);
  }
}
