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
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.util.StringCanonicalizer;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

import javax.annotation.Nullable;

/**
 * A <i>transitive</i> target reference that, when built in skyframe, loads the entire transitive
 * closure of a target. Retains the first error message found during the transitive traversal, and a
 * set of names of providers if the target is a {@link Rule}.
 *
 * <p>Interns values for error-free traversal nodes that correspond to built-in rules.
 */
@Immutable
@ThreadSafe
public class TransitiveTraversalValue implements SkyValue {
  private static final TransitiveTraversalValue EMPTY_VALUE =
      new TransitiveTraversalValue(false, ImmutableSet.<String>of(), null);
  // A quick-lookup cache that allows us to get the value for a given RuleClass, assuming no error
  // messages for the target. Only stores built-in RuleClass objects to avoid memory bloat.
  private static final ConcurrentMap<RuleClass, TransitiveTraversalValue> VALUES_BY_RULE_CLASS =
      new ConcurrentHashMap<>();
  /**
   * A strong interner of TransitiveTargetValue objects. Because we only wish to intern values for
   * built-in rules, we need an interner with an additional method to return the canonical
   * representative if it is present without interning our sample. This is only mutated in {@link
   * #forTarget}, and read in {@link #forTarget} and {@link #create}.
   */
  private static final InternerWithPresenceCheck<TransitiveTraversalValue> VALUE_INTERNER =
      new InternerWithPresenceCheck<>();

  static {
    VALUE_INTERNER.intern(EMPTY_VALUE);
  }

  private final boolean canHaveAnyProvider;
  private final ImmutableSet<String> providers;
  @Nullable private final String firstErrorMessage;

  private TransitiveTraversalValue(
      boolean canHaveAnyProvider,
      ImmutableSet<String> providers,
      @Nullable String firstErrorMessage) {
    this.canHaveAnyProvider = canHaveAnyProvider;
    this.providers = Preconditions.checkNotNull(providers);
    this.firstErrorMessage =
        (firstErrorMessage == null) ? null : StringCanonicalizer.intern(firstErrorMessage);
  }

  static TransitiveTraversalValue unsuccessfulTransitiveTraversal(String firstErrorMessage) {
    return new TransitiveTraversalValue(
        false, ImmutableSet.<String>of(), Preconditions.checkNotNull(firstErrorMessage));
  }

  static TransitiveTraversalValue forTarget(Target target, @Nullable String firstErrorMessage) {
    if (target instanceof Rule) {
      Rule rule = (Rule) target;
      RuleClass ruleClass = rule.getRuleClassObject();
      if (firstErrorMessage == null && !ruleClass.isSkylark()) {
        TransitiveTraversalValue value = VALUES_BY_RULE_CLASS.get(ruleClass);
        if (value != null) {
          return value;
        }
        ImmutableSet<String> providers = canonicalSet(toList(ruleClass.getAdvertisedProviders()));
        value = new TransitiveTraversalValue(ruleClass.canHaveAnyProvider(), providers, null);
        // May already be there from another RuleClass or a concurrent put.
        value = VALUE_INTERNER.intern(value);
        // May already be there from a concurrent put.
        VALUES_BY_RULE_CLASS.putIfAbsent(ruleClass, value);
        return value;
      } else {
        // If this is a Skylark rule, we may still get a cache hit from another RuleClass with the
        // same providers.
        return TransitiveTraversalValue.create(
            ruleClass.canHaveAnyProvider(),
            toList(rule.getRuleClassObject().getAdvertisedProviders()),
            firstErrorMessage);
      }
    }
    if (firstErrorMessage == null) {
      return EMPTY_VALUE;
    } else {
      return new TransitiveTraversalValue(false, ImmutableSet.<String>of(), firstErrorMessage);
    }
  }

  public static TransitiveTraversalValue create(
      boolean canHaveAnyProvider,
      Collection<String> providers,
      @Nullable String firstErrorMessage) {
    TransitiveTraversalValue value =
        new TransitiveTraversalValue(
            canHaveAnyProvider, canonicalSet(providers), firstErrorMessage);
    if (firstErrorMessage == null) {
      TransitiveTraversalValue oldValue = VALUE_INTERNER.getCanonical(value);
      return oldValue == null ? value : oldValue;
    }
    return value;
  }

  private static ImmutableSet<String> canonicalSet(Iterable<String> strIterable) {
    ImmutableSet.Builder<String> builder = new ImmutableSet.Builder<>();
    for (String str : strIterable) {
      builder.add(StringCanonicalizer.intern(str));
    }
    return builder.build();
  }

  private static List<String> toList(Collection<Class<?>> providers) {
    if (providers == null) {
      return ImmutableList.of();
    }
    List<String> strings = new ArrayList<>(providers.size());
    for (Class<?> clazz : providers) {
      strings.add(clazz.getName());
    }
    return strings;
  }

  /**
   * Returns if the associated target can have any provider. True for "alias" rules.
   */
  public boolean canHaveAnyProvider() {
    return canHaveAnyProvider;
  }

  /**
   * Returns the set of provider names from the target, if the target is a {@link Rule}. Otherwise
   * returns the empty set.
   */
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
    return this.canHaveAnyProvider == that.canHaveAnyProvider
        && Objects.equals(this.firstErrorMessage, that.firstErrorMessage)
        && this.providers.equals(that.providers);
  }

  @Override
  public int hashCode() {
    return Objects.hash(firstErrorMessage, providers, canHaveAnyProvider);
  }

  @ThreadSafe
  public static SkyKey key(Label label) {
    Preconditions.checkArgument(!label.getPackageIdentifier().getRepository().isDefault());
    return SkyKey.create(SkyFunctions.TRANSITIVE_TRAVERSAL, label);
  }
}
