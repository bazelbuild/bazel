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
package com.google.devtools.build.lib.skyframe;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.util.StringCanonicalizer;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import java.util.Collection;
import java.util.Objects;
import java.util.Set;

import javax.annotation.Nullable;

/**
 * A <i>transitive</i> target reference that, when built in skyframe, loads the entire transitive
 * closure of a target. Retains the first error message found during the transitive traversal,
 * and a set of names of providers if the target is a {@link Rule}.
 */
@Immutable
@ThreadSafe
public class TransitiveTraversalValue implements SkyValue {

  @Nullable private final ImmutableSet<String> providers;
  @Nullable private final String firstErrorMessage;

  private TransitiveTraversalValue(
      @Nullable Iterable<String> providers, @Nullable String firstErrorMessage) {
    this.providers = (providers == null) ? null : canonicalSet(providers);
    this.firstErrorMessage =
        (firstErrorMessage == null) ? null : StringCanonicalizer.intern(firstErrorMessage);
  }

  public static TransitiveTraversalValue unsuccessfulTransitiveTraversal(String firstErrorMessage) {
    return new TransitiveTraversalValue(null, Preconditions.checkNotNull(firstErrorMessage));
  }

  public static TransitiveTraversalValue forTarget(
      Target target, @Nullable String firstErrorMessage) {
    if (target instanceof Rule) {
      Rule rule = (Rule) target;
      return new TransitiveTraversalValue(
          toStringSet(rule.getRuleClassObject().getAdvertisedProviders()), firstErrorMessage);
    }
    return new TransitiveTraversalValue(ImmutableList.<String>of(), firstErrorMessage);
  }

  public static TransitiveTraversalValue withProviders(
      Collection<String> providers, @Nullable String firstErrorMessage) {
    return new TransitiveTraversalValue(ImmutableSet.copyOf(providers), firstErrorMessage);
  }

  private static ImmutableSet<String> canonicalSet(Iterable<String> strIterable) {
    ImmutableSet.Builder<String> builder = new ImmutableSet.Builder<>();
    for (String str : strIterable) {
      builder.add(StringCanonicalizer.intern(str));
    }
    return builder.build();
  }

  private static ImmutableSet<String> toStringSet(Iterable<Class<?>> providers) {
    ImmutableSet.Builder<String> pBuilder = new ImmutableSet.Builder<>();
    if (providers != null) {
      for (Class<?> clazz : providers) {
        pBuilder.add(StringCanonicalizer.intern(clazz.getName()));
      }
    }
    return pBuilder.build();
  }

  /**
   * Returns the set of provider names from the target, if the target is a {@link Rule}. If there
   * were errors loading the target, returns {@code null}.
   */
  @Nullable
  public Set<String> getProviders() {
    return providers;
  }

  /**
   * Returns the first error message, if any, from loading the target and its transitive
   * dependencies.
   */
  @Nullable
  public String getFirstErrorMessage() {
    return firstErrorMessage;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (!(o instanceof TransitiveTraversalValue)) {
      return false;
    }
    TransitiveTraversalValue that = (TransitiveTraversalValue) o;
    return Objects.equals(this.firstErrorMessage, that.firstErrorMessage)
        && Objects.equals(this.providers, that.providers);
  }

  @Override
  public int hashCode() {
    return 31 * Objects.hashCode(firstErrorMessage) + Objects.hashCode(providers);
  }

  @ThreadSafe
  public static SkyKey key(Label label) {
    return SkyKey.create(SkyFunctions.TRANSITIVE_TRAVERSAL, label);
  }
}
