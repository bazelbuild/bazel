// Copyright 2026 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.actions;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.collect.Iterators;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import java.util.Iterator;
import java.util.List;

/**
 * Efficient representation of inputs to a {@link Spawn}, supporting the lazy union of multiple
 * nested sets and lists.
 *
 * <p>A common pattern for spawns is to take {@link Action#getInputs} and add a few additional files
 * such as parameter files. Creating a new {@link NestedSet} instance to include a few additional
 * files is suboptimal: it doesn't share a cached list representation with {@link Action#getInputs},
 * and it isn't expected to be fed into a larger {@link NestedSet}.
 *
 * <p>Calling {@link #flatten} calls {@link NestedSet#toList} on an <em>existing</em> {@link
 * NestedSet} (maximizing the likelihood of a cached representation), and lazily concatenates the
 * additional files using {@link Iterables#concat}.
 */
public final class SpawnInputs {

  private static final SpawnInputs EMPTY =
      new SpawnInputs(
          NestedSetBuilder.emptySet(Order.STABLE_ORDER),
          NestedSetBuilder.emptySet(Order.STABLE_ORDER),
          ImmutableList.of(),
          /* sizeOfRest= */ 0);

  /** Returns a canonical empty {@link SpawnInputs} instance. */
  public static SpawnInputs empty() {
    return EMPTY;
  }

  /** Creates {@link SpawnInputs} from a single {@link NestedSet}. */
  public static SpawnInputs of(NestedSet<? extends ActionInput> nestedSet) {
    return of(nestedSet, ImmutableList.of());
  }

  /** Creates {@link SpawnInputs} from a {@link NestedSet} plus some additional inputs. */
  public static SpawnInputs of(
      NestedSet<? extends ActionInput> nestedSet, List<? extends ActionInput> list) {
    return of(nestedSet, NestedSetBuilder.emptySet(Order.STABLE_ORDER), list);
  }

  /**
   * Creates {@link SpawnInputs} from two nested sets plus some additional inputs.
   *
   * <p>This is useful for input-discovering actions to union mandatory inputs and discovered
   * inputs.
   */
  public static SpawnInputs of(
      NestedSet<? extends ActionInput> nestedSet1,
      NestedSet<? extends ActionInput> nestedSet2,
      List<? extends ActionInput> list) {
    return new SpawnInputs(nestedSet1, nestedSet2, list, list.size());
  }

  private final NestedSet<? extends ActionInput> subset1;
  private final NestedSet<? extends ActionInput> subset2;
  private final Iterable<? extends ActionInput> rest;
  private final int sizeOfRest;

  private SpawnInputs(
      NestedSet<? extends ActionInput> subset1,
      NestedSet<? extends ActionInput> subset2,
      Iterable<? extends ActionInput> rest,
      int sizeOfRest) {
    this.subset1 = subset1;
    this.subset2 = subset2;
    this.rest = rest;
    this.sizeOfRest = sizeOfRest;
  }

  /**
   * Returns a single {@link NestedSet} containing all of the spawn's inputs.
   *
   * <p>Prefer {@link #flatten} for iterating over the inputs, as it is more likely to reuse cached
   * list representations. This method should only be used by callers that need to decompose the
   * {@link NestedSet} structurally.
   */
  public NestedSet<ActionInput> asNestedSet() {
    return NestedSetBuilder.<ActionInput>fromNestedSet(subset1)
        .addTransitive(subset2)
        .addAll(rest)
        .build();
  }

  /**
   * Flattens contained nested sets and returns a {@link FlattenedInputs} suitable for iteration.
   *
   * <p>Note that the returned {@link FlattenedInputs} may contain duplicate inputs if the
   * underlying sets or lists contain duplicates.
   */
  public FlattenedInputs flatten() {
    return new FlattenedInputs(subset1.toList(), subset2.toList(), rest, sizeOfRest);
  }

  /**
   * Returns a {@link SpawnInputs} with all the inputs in this instance plus the given additional
   * inputs.
   *
   * <p>Does not mutate this instance.
   */
  public SpawnInputs plus(List<ActionInput> additional) {
    if (additional.isEmpty()) {
      return this;
    }
    return new SpawnInputs(
        subset1, subset2, Iterables.concat(rest, additional), sizeOfRest + additional.size());
  }

  /**
   * Flattened representation of {@link SpawnInputs} that is ready for iteration.
   *
   * <p>This iterable may contain duplicate inputs.
   */
  public static final class FlattenedInputs implements Iterable<ActionInput> {
    private final List<? extends ActionInput> list1;
    private final List<? extends ActionInput> list2;
    private final Iterable<? extends ActionInput> rest;
    private final int sizeOfRest;

    private FlattenedInputs(
        List<? extends ActionInput> list1,
        List<? extends ActionInput> list2,
        Iterable<? extends ActionInput> rest,
        int sizeOfRest) {
      this.list1 = list1;
      this.list2 = list2;
      this.rest = rest;
      this.sizeOfRest = sizeOfRest;
    }

    /** Returns the number of inputs. */
    public int size() {
      return list1.size() + list2.size() + sizeOfRest;
    }

    /** Returns true if there are no inputs. */
    public boolean isEmpty() {
      return list1.isEmpty() && list2.isEmpty() && sizeOfRest == 0;
    }

    @Override
    public Iterator<ActionInput> iterator() {
      return Iterators.concat(list1.iterator(), list2.iterator(), rest.iterator());
    }
  }
}
