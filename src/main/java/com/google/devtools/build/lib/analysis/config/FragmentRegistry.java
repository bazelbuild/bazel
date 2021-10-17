// Copyright 2021 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis.config;

import com.google.common.collect.ImmutableSortedSet;
import com.google.common.collect.Sets;
import java.util.List;

/** A registry of all {@link Fragment} and {@link FragmentOptions} classes registered at startup. */
public final class FragmentRegistry {

  /**
   * Creates a {@code FragmentRegistry}.
   *
   * <p>Order of elements in the given lists does not matter - the resulting registry will contain
   * deterministically ordered sets.
   *
   * @param allFragments all registered fragment classes, including {@code universalFragments}
   * @param universalFragments fragment classes that should be available to all rules even when not
   *     explicitly required
   * @param additionalOptions any additional options classes not accounted for by a {@link
   *     RequiresOptions} annotation on a {@link Fragment} class in {@code allFragments}
   */
  public static FragmentRegistry create(
      List<Class<? extends Fragment>> allFragments,
      List<Class<? extends Fragment>> universalFragments,
      List<Class<? extends FragmentOptions>> additionalOptions) {
    FragmentClassSet allFragmentsSet = FragmentClassSet.of(allFragments);
    FragmentClassSet universalFragmentsSet = FragmentClassSet.of(universalFragments);
    if (!allFragmentsSet.containsAll(universalFragmentsSet)) {
      throw new IllegalArgumentException(
          "Missing universally required fragments: "
              + Sets.difference(universalFragmentsSet, allFragmentsSet));
    }

    ImmutableSortedSet.Builder<Class<? extends FragmentOptions>> optionsClasses =
        ImmutableSortedSet.orderedBy(BuildOptions.LEXICAL_FRAGMENT_OPTIONS_COMPARATOR);
    for (Class<? extends Fragment> fragment : allFragmentsSet) {
      optionsClasses.addAll(Fragment.requiredOptions(fragment));
    }
    optionsClasses.addAll(additionalOptions);

    return new FragmentRegistry(allFragmentsSet, universalFragmentsSet, optionsClasses.build());
  }

  private final FragmentClassSet allFragments;
  private final FragmentClassSet universalFragments;
  private final ImmutableSortedSet<Class<? extends FragmentOptions>> optionsClasses;

  private FragmentRegistry(
      FragmentClassSet allFragments,
      FragmentClassSet universalFragments,
      ImmutableSortedSet<Class<? extends FragmentOptions>> optionsClasses) {
    this.allFragments = allFragments;
    this.universalFragments = universalFragments;
    this.optionsClasses = optionsClasses;
  }

  /** Returns the set of all registered configuration fragments. */
  public FragmentClassSet getAllFragments() {
    return allFragments;
  }

  /**
   * Returns a subset of {@link #getAllFragments} that should be available to all rules even when
   * not explicitly required.
   */
  public FragmentClassSet getUniversalFragments() {
    return universalFragments;
  }

  /**
   * Returns the set of all registered {@link FragmentOptions} classes.
   *
   * <p>Includes at least all options classes {@linkplain RequiresOptions required} by fragments in
   * {@link #getAllFragments}.
   */
  public ImmutableSortedSet<Class<? extends FragmentOptions>> getOptionsClasses() {
    return optionsClasses;
  }
}
