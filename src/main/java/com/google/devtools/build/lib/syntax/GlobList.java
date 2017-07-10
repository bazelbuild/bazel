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

package com.google.devtools.build.lib.syntax;

import com.google.common.base.Joiner;
import com.google.common.base.Splitter;
import com.google.common.collect.ForwardingList;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableList.Builder;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkPrinter;
import com.google.devtools.build.lib.skylarkinterface.SkylarkValue;
import com.google.devtools.build.lib.util.Preconditions;
import java.util.ArrayList;
import java.util.List;

/**
 * Glob matches and information about glob patterns, which are useful to
 * ide_build_info. Its implementation of the List interface is as an immutable
 * list of the matching files. Glob criteria can be retrieved through
 * {@link #getCriteria}.
 *
 * @param <E> the element this List contains (generally either String or Label)
 */
@SkylarkModule(
    name = "glob list",
    doc = "",
    documented = false)
public final class GlobList<E> extends ForwardingList<E> implements SkylarkValue {

  /** Include/exclude criteria. */
  private final ImmutableList<GlobCriteria> criteria;

  /** Matching files (usually either String or Label). */
  private final ImmutableList<E> matches;

  /**
   * Constructs a list with {@code glob()} call results.
   *
   * @param includes the patterns that the glob includes
   * @param excludes the patterns that the glob excludes
   * @param matches the filenames that matched the includes/excludes criteria
   */
  public static <T> GlobList<T> captureResults(List<String> includes,
      List<String> excludes, List<T> matches) {
    GlobCriteria criteria = GlobCriteria.fromGlobCall(
        ImmutableList.copyOf(includes), ImmutableList.copyOf(excludes));
    return new GlobList<>(ImmutableList.of(criteria), matches);
  }

  /**
   * Parses a GlobInfo from its {@link #toExpression} representation.
   */
  public static GlobList<String> parse(String text) {
    List<GlobCriteria> criteria = new ArrayList<>();
    Iterable<String> globs = Splitter.on(" + ").split(text);
    for (String glob : globs) {
      criteria.add(GlobCriteria.parse(glob));
    }
    return new GlobList<>(criteria, ImmutableList.<String>of());
  }

  /**
   * Concatenates two lists into a new GlobList. If either of the lists is a
   * GlobList, its GlobCriteria are preserved. Otherwise a simple GlobCriteria
   * is created to represent the fixed list.
   */
  public static <T> GlobList<T> concat(
      List<? extends T> list1, List<? extends T> list2) {
    // we add the list to both includes and matches, preserving order
    Builder<GlobCriteria> criteriaBuilder = ImmutableList.<GlobCriteria>builder();
    if (list1 instanceof GlobList<?>) {
      criteriaBuilder.addAll(((GlobList<?>) list1).criteria);
    } else {
      criteriaBuilder.add(GlobCriteria.fromList(list1));
    }
    if (list2 instanceof GlobList<?>) {
      criteriaBuilder.addAll(((GlobList<?>) list2).criteria);
    } else {
      criteriaBuilder.add(GlobCriteria.fromList(list2));
    }
    List<T> matches = ImmutableList.copyOf(Iterables.concat(list1, list2));
    return new GlobList<>(criteriaBuilder.build(), matches);
  }

  /**
   * Constructs a list with given criteria and matches.
   */
  public GlobList(List<GlobCriteria> criteria, List<E> matches) {
    Preconditions.checkNotNull(criteria);
    Preconditions.checkNotNull(matches);
    this.criteria = ImmutableList.copyOf(criteria);
    this.matches = ImmutableList.copyOf(matches);
  }

  /**
   * Returns the criteria used to create this list, from which the
   * includes/excludes can be retrieved.
   */
  public ImmutableList<GlobCriteria> getCriteria() {
    return criteria;
  }

  /**
   * Returns a String that represents this glob list as a BUILD expression.
   */
  public String toExpression() {
    return Joiner.on(" + ").join(criteria);
  }

  @Override
  protected ImmutableList<E> delegate() {
    return matches;
  }

  @Override
  public boolean isImmutable() {
    return false;
  }

  @Override
  public void repr(SkylarkPrinter printer) {
    printer.printList(this, false);
  }
}
