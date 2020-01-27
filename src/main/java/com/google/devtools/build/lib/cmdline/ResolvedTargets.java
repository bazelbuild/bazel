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
package com.google.devtools.build.lib.cmdline;

import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Sets;
import java.util.Collection;
import java.util.LinkedHashSet;
import java.util.Set;
import javax.annotation.concurrent.Immutable;

/**
 * Contains the result of the target pattern evaluation. This is a specialized container class for
 * the result of target pattern resolution. There is no restriction on the element type, but it will
 * usually be {@code Target} or {@code Label}.
 */
@Immutable
public final class ResolvedTargets<T> {
  private static final ResolvedTargets<?> FAILED_RESULT =
      new ResolvedTargets<>(ImmutableSet.of(), ImmutableSet.of(), true);

  private static final ResolvedTargets<?> EMPTY_RESULT =
      new ResolvedTargets<>(ImmutableSet.of(), ImmutableSet.of(), false);

  @SuppressWarnings("unchecked")
  public static <T> ResolvedTargets<T> failed() {
    return (ResolvedTargets<T>) FAILED_RESULT;
  }

  @SuppressWarnings("unchecked")
  public static <T> ResolvedTargets<T> empty() {
    return (ResolvedTargets<T>) EMPTY_RESULT;
  }

  public static <T> ResolvedTargets<T> of(T target) {
    return new ResolvedTargets<>(ImmutableSet.<T>of(target), false);
  }

  private final boolean hasError;
  private final ImmutableSet<T> targets;
  private final ImmutableSet<T> filteredTargets;

  public ResolvedTargets(Set<T> targets, Set<T> filteredTargets, boolean hasError) {
    this.targets = ImmutableSet.copyOf(targets);
    this.filteredTargets = ImmutableSet.copyOf(filteredTargets);
    this.hasError = hasError;
  }

  public ResolvedTargets(Set<T> targets, boolean hasError) {
    this.targets = ImmutableSet.copyOf(targets);
    this.filteredTargets = ImmutableSet.of();
    this.hasError = hasError;
  }

  @Override
  public String toString() {
    return "ResolvedTargets(" + targets + ", filtered=" + filteredTargets
        + ", hasError=" + hasError + ")";
  }

  public boolean hasError() {
    return hasError;
  }

  public ImmutableSet<T> getTargets() {
    return targets;
  }

  public ImmutableSet<T> getFilteredTargets() {
    return filteredTargets;
  }

  /**
   * Returns a builder using concurrent sets, as long as you don't call filter.
   */
  public static <T> ResolvedTargets.Builder<T> concurrentBuilder() {
    return new ResolvedTargets.Builder<>(
        Sets.<T>newConcurrentHashSet(),
        Sets.<T>newConcurrentHashSet());
  }

  public static <T> ResolvedTargets.Builder<T> builder() {
    return new ResolvedTargets.Builder<>();
  }

  public static final class Builder<T> {
    private Set<T> targets;
    private Set<T> filteredTargets;
    private volatile boolean hasError = false;

    private Builder() {
      this(new LinkedHashSet<>(), new LinkedHashSet<>());
    }

    private Builder(Set<T> targets, Set<T> filteredTargets) {
      this.targets = targets;
      this.filteredTargets = filteredTargets;
    }

    public ResolvedTargets<T> build() {
      return new ResolvedTargets<>(targets, filteredTargets, hasError);
    }

    public Builder<T> merge(ResolvedTargets<T> other) {
      removeAll(other.filteredTargets);
      addAll(other.targets);
      if (other.hasError) {
        hasError = true;
      }
      return this;
    }

    public Builder<T> add(T target) {
      targets.add(target);
      filteredTargets.remove(target);
      return this;
    }

    public Builder<T> addAll(Collection<T> targets) {
      this.targets.addAll(targets);
      this.filteredTargets.removeAll(targets);
      return this;
    }

    public void remove(T target) {
      targets.remove(target);
      filteredTargets.add(target);
    }

    public Builder<T> removeAll(Collection<T> targets) {
      this.filteredTargets.addAll(targets);
      this.targets.removeAll(targets);
      return this;
    }

    public Builder<T> filter(Predicate<T> predicate) {
      Set<T> oldTargets = targets;
      targets = Sets.newLinkedHashSet();
      for (T target : oldTargets) {
        if (predicate.apply(target)) {
          add(target);
        } else {
          remove(target);
        }
      }
      return this;
    }

    public Builder<T> setError() {
      this.hasError = true;
      return this;
    }

    public Builder<T> mergeError(boolean hasError) {
      this.hasError |= hasError;
      return this;
    }

    public boolean isEmpty() {
      return targets.isEmpty();
    }
  }
}
