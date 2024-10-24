// Copyright 2018 The Bazel Authors. All rights reserved.
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

import static com.google.common.base.Predicates.not;

import com.google.common.collect.Collections2;
import com.google.common.collect.ImmutableSortedSet;
import com.google.common.collect.Interner;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.SerializationConstant;
import com.google.devtools.build.lib.util.ClassName;
import java.util.AbstractSet;
import java.util.Collection;
import java.util.Comparator;
import java.util.Iterator;
import javax.annotation.concurrent.Immutable;

/**
 * A wrapper class for an {@code ImmutableSortedSet<Class<? extends Fragment>>}. Interning these
 * objects allows us to do cheap reference equality checks when these sets are in frequently used
 * keys.
 */
@Immutable
public final class FragmentClassSet extends AbstractSet<Class<? extends Fragment>> {

  /**
   * Sorts fragments by class name. This produces a stable order which, e.g., facilitates consistent
   * output from buildMnemonic.
   */
  @SerializationConstant
  public static final Comparator<Class<? extends Fragment>> LEXICAL_FRAGMENT_SORTER =
      Comparator.comparing(Class::getName);

  private static final Interner<FragmentClassSet> interner = BlazeInterners.newWeakInterner();

  public static FragmentClassSet of(Collection<Class<? extends Fragment>> fragments) {
    ImmutableSortedSet<Class<? extends Fragment>> sortedFragments =
        ImmutableSortedSet.copyOf(LEXICAL_FRAGMENT_SORTER, fragments);
    return interner.intern(new FragmentClassSet(sortedFragments, sortedFragments.hashCode()));
  }

  private final ImmutableSortedSet<Class<? extends Fragment>> fragments;
  private final int hashCode;

  private FragmentClassSet(ImmutableSortedSet<Class<? extends Fragment>> fragments, int hashCode) {
    this.fragments = fragments;
    this.hashCode = hashCode;
  }

  @Override
  public int size() {
    return fragments.size();
  }

  @Override
  public boolean contains(Object o) {
    return fragments.contains(o);
  }

  /** Returns a set of fragment classes identical to this one but without the given fragment. */
  public FragmentClassSet trim(Class<? extends Fragment> fragment) {
    if (!contains(fragment)) {
      return this;
    }
    return of(Sets.filter(fragments, not(fragment::equals)));
  }

  @Override
  public Iterator<Class<? extends Fragment>> iterator() {
    return fragments.iterator();
  }

  @Override
  public boolean equals(Object other) {
    if (this == other) {
      return true;
    }
    if (!(other instanceof FragmentClassSet that)) {
      return false;
    }
    return hashCode == that.hashCode && fragments.equals(that.fragments);
  }

  @Override
  public int hashCode() {
    return hashCode;
  }

  @Override
  public String toString() {
    return Collections2.transform(fragments, ClassName::getSimpleNameWithOuter).toString();
  }
}
