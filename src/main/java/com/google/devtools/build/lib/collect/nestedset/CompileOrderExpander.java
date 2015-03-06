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

/**
 * A nested set expander that implements left-to-right postordering.
 *
 * <p>For example, for the nested set {B, D, {A, C}}, the iteration order is "A C B D"
 * (child-first).
 *
 * <p>This type of set would typically be used for artifacts where elements of nested sets go before
 * the direct members of a set, for example in the case of Javascript dependencies.
 */
final class CompileOrderExpander<E> implements NestedSetExpander<E> {

  // We suppress unchecked warning so that we can access the internal raw structure of the
  // NestedSet.
  @SuppressWarnings("unchecked")
  @Override
  public void expandInto(NestedSet<E> set, Uniqueifier uniqueifier,
      ImmutableCollection.Builder<E> builder) {
    for (NestedSet<E> subset : set.transitiveSets()) {
      if (!subset.isEmpty() && uniqueifier.isUnique(subset)) {
        expandInto(subset, uniqueifier, builder);
      }
    }

    // This switch is here to compress the memo used by the uniqueifier
    for (Object e : set.directMembers()) {
      if (uniqueifier.isUnique(e)) {
        builder.add((E) e);
      }
    }
  }
}
