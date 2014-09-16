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
package com.google.devtools.build.lib.collect.nestedset;

import com.google.common.collect.ImmutableList;

/**
 * Factory methods for creating {@link NestedSet}s of specific shapes. This allows the
 * implementation to be memory efficient (e.g. a specialized implementation for the case where
 * there are only direct elements, etc).
 *
 * <p>It's intended for each {@link Order} to have its own factory implementation. That way we can
 * be even more efficient since the {@link NestedSet}s instances don't need to store their
 * {@link Order}.
 */
interface NestedSetFactory {

  /** Create a NestedSet with just one direct element and not transitive elements. */
  <E> NestedSet<E> oneDirect(E element);

  /** Create a NestedSet with only direct elements. */
  <E> NestedSet<E> onlyDirects(Object[] directs);

  /** Create a NestedSet with only direct elements potentially sharing the ImmutableList. */
  <E> NestedSet<E> onlyDirects(ImmutableList<E> directs);

  /** Create a NestedSet with one direct element and one transitive {@code NestedSet}. */
  <E> NestedSet<E> oneDirectOneTransitive(E direct, NestedSet<E> transitive);

  /** Create a NestedSet with many direct elements and one transitive {@code NestedSet}. */
  <E> NestedSet<E> manyDirectsOneTransitive(Object[] direct, NestedSet<E> transitive);

  /** Create a NestedSet with no direct elements and one transitive {@code NestedSet.} */
  <E> NestedSet<E> onlyOneTransitive(NestedSet<E> transitive);

  /** Create a NestedSet with no direct elements and many transitive {@code NestedSet}s. */
  <E> NestedSet<E> onlyManyTransitives(NestedSet[] transitives);

  /** Create a NestedSet with one direct elements and many transitive {@code NestedSet}s. */
  <E> NestedSet<E> oneDirectManyTransitive(Object direct, NestedSet[] transitive);

  /** Create a NestedSet with many direct elements and many transitive {@code NestedSet}s. */
  <E> NestedSet<E> manyDirectManyTransitive(Object[] directs, NestedSet[] transitive);
}
