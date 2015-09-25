// Copyright 2014 The Bazel Authors. All rights reserved.
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
 * A nested set expander that implements naive left-to-right preordering.
 *
 * <p>For example, for the nested set {B, D, {A, C}}, the iteration order is "B D A C".
 *
 * <p>This implementation is intended for backwards-compatible nested set replacements of code that
 * uses naive preordering.
 *
 * <p>The implementation is called naive because it does no special treatment of dependency graphs
 * that are not trees. For such graphs the property of parent-before-dependencies in the iteration
 * order will not be upheld. For example, the diamond-shape graph A->{B, C}, B->{D}, C->{D} will be
 * enumerated as "A B D C" rather than "A B C D" or "A C B D".
 *
 * <p>The difference from {@link LinkOrderNestedSet} is that this implementation gives priority to
 * left-to-right order over dependencies-after-parent ordering. Note that the latter is usually more
 * important, so please use {@link LinkOrderNestedSet} whenever possible.
 */
final class NaiveLinkOrderExpander<E> implements NestedSetExpander<E> {

  @SuppressWarnings("unchecked")
  @Override
  public void expandInto(NestedSet<E> set, Uniqueifier uniqueifier,
      ImmutableCollection.Builder<E> builder) {

    for (Object e : set.directMembers()) {
      if (uniqueifier.isUnique(e)) {
        builder.add((E) e);
      }
    }

    for (NestedSet<E> subset : set.transitiveSets()) {
      if (!subset.isEmpty() && uniqueifier.isUnique(subset)) {
        expandInto(subset, uniqueifier, builder);
      }
    }
  }
}
