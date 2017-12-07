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
package com.google.devtools.build.lib.collect.nestedset;

import static com.google.common.collect.ImmutableSet.toImmutableSet;

import com.google.common.collect.ImmutableSet;
import java.util.Arrays;
import java.util.Set;

/**
 * Class presenting the logical structure of a {@link NestedSet}.
 *
 * <p>The main use case for this class are situations were a larger number of related nested sets
 * needs to be serialized in an efficient way, as is the case when reporting artifacts in the build
 * event protocol.
 *
 * <p>Note that a {@link NestedSet} does not preserve all structure provided to the {@link
 * NestedSetBuilder}; in fact, it may decide to inline the contents of small nested sets as direct
 * members. This view class provides a view on the structure that is still present in a {@link
 * NestedSet}. As there is a fixed limit on the size a transitive member can have to still be
 * eligible for inlining, this is enough to allow an efficient deduplicated presentation.
 */
public class NestedSetView<E> {
  private final Object set;

  private NestedSetView(Object set) {
    this.set = set;
  }

  /** Construct a view of a given NestedSet. */
  public NestedSetView(NestedSet<E> set) {
    this(set.rawChildren());
  }

  /**
   * Return an object where the {@link equals()} method provides the correct notion of (intensional)
   * equality of the set viewed. Consumers of this method should not assume any properties of the
   * returned object apart from its {@link equals()} method.
   *
   * <p>The identifier is meant as an abstract, but memory efficient way of remembering nested sets
   * directly or indirectly seen. Storing the identifier of a nested-set view will not retain more
   * memory than storing the underlying nested set; in particular, it will not prevent the view
   * object from being garbage collected.
   *
   * <p>The equality of the view itself is the one inherited from Object, i.e., you can have many
   * views of the same set that are not equal as views.
   */
  public Object identifier() {
    return set;
  }

  /**
   * Return the set of transitive members.
   *
   * <p>This refers to the transitive members after any inlining that might have happened at
   * construction of the nested set.
   */
  public Set<NestedSetView<E>> transitives() {
    if (!(set instanceof Object[])) {
      return ImmutableSet.of();
    }
    return Arrays.stream((Object[]) set)
        .filter(c -> c instanceof Object[])
        .map(c -> new NestedSetView<E>(c))
        .collect(toImmutableSet());
  }

  /**
   * Return the set of direct members.
   *
   * <p>This refers to the direct members after any inlining that might have happened at
   * construction of the nested set.
   */
  @SuppressWarnings("unchecked")
  public Set<E> directs() {
    if (!(set instanceof Object[])) {
      return ImmutableSet.of((E) set);
    }
    return Arrays.stream((Object[]) set)
        .filter(c -> !(c instanceof Object[]))
        .map(c -> (E) c)
        .collect(toImmutableSet());
  }
}
