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

import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;

/**
 * A nested set expander that implements a variation of left-to-right preordering.
 *
 * <p>For example, for the nested set {A, C, {B, D}}, the iteration order is "A C B D"
 * (parent-first).
 *
 * <p>This type of set would typically be used for artifacts where elements of
 * nested sets go after the direct members of a set, for example when providing
 * a list of libraries to the C++ compiler.
 *
 * <p>The custom ordering has the property that elements of nested sets always come
 * before elements of descendant nested sets. Left-to-right order is preserved if
 * possible, both for items and for references to nested sets.
 *
 * <p>The left-to-right pre-order-like ordering is implemented by running a
 * right-to-left postorder traversal and then reversing the result.
 *
 * <p>The reason naive left-to left-to-right preordering is not used here is that
 * it does not handle diamond-like structures properly. For example, take the
 * following structure (nesting downwards):
 *
 * <pre>
 *    A
 *   / \
 *  B   C
 *   \ /
 *    D
 * </pre>
 *
 * <p>Naive preordering would produce "A B D C", which does not preserve the
 * "parent before child" property: C is a parent of D, so C should come before
 * D. Either "A B C D" or "A C B D" would be acceptable. This implementation
 * returns the first option of the two so that left-to-right order is preserved.
 *
 * <p>In case the nested sets form a tree, the ordering algorithm is equivalent to
 * standard left-to-right preorder.
 *
 * <p>Sometimes it may not be possible to preserve left-to-right order:
 *
 * <pre>
 *      A
 *    /   \
 *   B     C
 *  / \   / \
 *  \   E   /
 *   \     /
 *    \   /
 *      D
 * </pre>
 *
 * <p>The left branch (B) would indicate "D E" ordering and the right branch (C)
 * dictates "E D". In such cases ordering is decided by the rightmost branch
 * because of the list reversing behind the scenes, so the ordering in the final
 * enumeration will be "E D".
 */

final class LinkOrderExpander<E> implements NestedSetExpander<E> {
  @Override
  public void expandInto(NestedSet<E> nestedSet, Uniqueifier uniqueifier,
      ImmutableCollection.Builder<E> builder) {
    ImmutableList.Builder<E> result = ImmutableList.builder();
    internalEnumerate(nestedSet, uniqueifier, result);
    builder.addAll(result.build().reverse());
  }

  // We suppress unchecked warning so that we can access the internal raw structure of the
  // NestedSet.
  @SuppressWarnings("unchecked")
  private void internalEnumerate(NestedSet<E> set, Uniqueifier uniqueifier,
      ImmutableCollection.Builder<E> builder) {
    NestedSet[] transitiveSets = set.transitiveSets();
    for (int i = transitiveSets.length - 1; i >= 0; i--) {
      NestedSet<E> subset = transitiveSets[i];
      if (!subset.isEmpty() && uniqueifier.isUnique(subset)) {
        internalEnumerate(subset, uniqueifier, builder);
      }
    }

    Object[] directMembers = set.directMembers();
    for (int i = directMembers.length - 1; i >= 0; i--) {
      Object e = directMembers[i];
      if (uniqueifier.isUnique(e)) {
        builder.add((E) e);
      }
    }
  }
}
