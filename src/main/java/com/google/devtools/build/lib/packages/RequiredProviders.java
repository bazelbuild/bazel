// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.packages;

import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.util.Preconditions;

/**
 * Represents a constraint on a set of providers required by a dependency (of a rule
 * or an aspect).
 *
 * Currently we support three kinds of constraints:
 * <ul>
 *   <li>accept any dependency.</li>
 *   <li>accept no dependency (used for aspects-on-aspects to indicate
 *   that an aspect never wants to see any other aspect applied to a target.</li>
 *   <li>accept a dependency that provides all providers from one of several sets of providers.
 *       It just so happens that in all current usages these sets are either all
 *       native providers or all Skylark providers, so this is the only use case this
 *       class currently supports.
 *   </li>
 * </ul>
 */
@Immutable
public final class RequiredProviders {
  /** A constraint: either ANY, NONE, or RESTRICTED */
  private final Constraint constraint;
  /**
   * Sets of native providers.
   * If non-empty, {@link #constraint} is {@link Constraint#RESTRICTED}
   */
  private final ImmutableList<ImmutableSet<Class<?>>> nativeProviders;
  /**
   * Sets of native providers.
   * If non-empty, {@link #constraint} is {@link Constraint#RESTRICTED}
   */
  private final ImmutableList<ImmutableSet<SkylarkProviderIdentifier>> skylarkProviders;

  /**
   * Represents one of the constraints as desctibed in {@link RequiredProviders}
   */
  private enum Constraint {
    /** Accept any dependency */
    ANY {
      @Override
      public boolean satisfies(AdvertisedProviderSet advertisedProviderSet,
          RequiredProviders requiredProviders) {
        return true;
      }

      @Override
      public boolean satisfies(Predicate<Class<?>> hasNativeProvider,
          Predicate<SkylarkProviderIdentifier> hasSkylarkProvider,
          RequiredProviders requiredProviders) {
        return true;
      }
    },
    /** Accept no dependency */
    NONE {
      @Override
      public boolean satisfies(AdvertisedProviderSet advertisedProviderSet,
          RequiredProviders requiredProviders) {
        return false;
      }

      @Override
      public boolean satisfies(Predicate<Class<?>> hasNativeProvider,
          Predicate<SkylarkProviderIdentifier> hasSkylarkProvider,
          RequiredProviders requiredProviders) {
        return false;
      }
    },
    /** Accept a dependency that has all providers from one of the sets. */
    RESTRICTED {
      @Override
      public boolean satisfies(Predicate<Class<?>> hasNativeProvider,
          Predicate<SkylarkProviderIdentifier> hasSkylarkProvider,
          RequiredProviders requiredProviders) {
        for (ImmutableSet<Class<?>> nativeProviderSet : requiredProviders.nativeProviders) {
          if (Iterables.all(nativeProviderSet, hasNativeProvider)) {
            return true;
          }
        }

        for (ImmutableSet<SkylarkProviderIdentifier> skylarkProviderSet
            : requiredProviders.skylarkProviders) {
          if (Iterables.all(skylarkProviderSet, hasSkylarkProvider)) {
            return true;
          }
        }
        return false;
      }
    };

    /** Checks if {@code advertisedProviderSet} satisfies these {@code RequiredProviders} */
    public boolean satisfies(final AdvertisedProviderSet advertisedProviderSet,
        RequiredProviders requiredProviders) {
      if (advertisedProviderSet.canHaveAnyProvider()) {
        return true;
      }
      return satisfies(
          new Predicate<Class<?>>() {
            @Override
            public boolean apply(Class<?> aClass) {
              return advertisedProviderSet.getNativeProviders().contains(aClass);
            }
          },
          new Predicate<SkylarkProviderIdentifier>() {
            @Override
            public boolean apply(SkylarkProviderIdentifier skylarkProviderIdentifier) {
              if (!skylarkProviderIdentifier.isLegacy()) {
                return false;
              }
              return advertisedProviderSet.getSkylarkProviders()
                  .contains(skylarkProviderIdentifier.getLegacyId());
            }
          },
          requiredProviders
      );
    }

    /**
     * Checks if a set of providers encoded by predicates {@code hasNativeProviders}
     * and {@code hasSkylarkProvider} satisfies these {@code RequiredProviders}
     */
    abstract boolean satisfies(Predicate<Class<?>> hasNativeProvider,
        Predicate<SkylarkProviderIdentifier> hasSkylarkProvider,
        RequiredProviders requiredProviders);
  }

  /** Checks if {@code advertisedProviderSet} satisfies this {@code RequiredProviders} instance. */
  public boolean isSatisfiedBy(AdvertisedProviderSet advertisedProviderSet) {
    return constraint.satisfies(advertisedProviderSet, this);
  }

  /**
   * Checks if a set of providers encoded by predicates {@code hasNativeProviders}
   * and {@code hasSkylarkProvider} satisfies this {@code RequiredProviders} instance.
   */
  public boolean isSatisfiedBy(
      Predicate<Class<?>> hasNativeProvider,
      Predicate<SkylarkProviderIdentifier> hasSkylarkProvider) {
    return constraint.satisfies(hasNativeProvider, hasSkylarkProvider, this);
  }


  private RequiredProviders(
      Constraint constraint,
      ImmutableList<ImmutableSet<Class<?>>> nativeProviders,
      ImmutableList<ImmutableSet<SkylarkProviderIdentifier>> skylarkProviders) {
    this.constraint = constraint;

    Preconditions.checkState(constraint.equals(Constraint.RESTRICTED)
        || (nativeProviders.isEmpty() && skylarkProviders.isEmpty())
    );

    this.nativeProviders = nativeProviders;
    this.skylarkProviders = skylarkProviders;
  }

  /**
   * A builder for {@link RequiredProviders} that accepts any dependency
   * unless restriction provider sets are added.
   */
  public static Builder acceptAnyBuilder() {
    return new Builder(false);
  }

  /**
   * A builder for {@link RequiredProviders} that accepts no dependency
   * unless restriction provider sets are added.
   */
  public static Builder acceptNoneBuilder() {
    return new Builder(true);
  }

  /** A builder for {@link RequiredProviders} */
  public static class Builder {
    private final ImmutableList.Builder<ImmutableSet<Class<?>>> nativeProviders;
    private final ImmutableList.Builder<ImmutableSet<SkylarkProviderIdentifier>> skylarkProviders;
    private Constraint constraint;

    private Builder(boolean acceptNone) {
      constraint = acceptNone ? Constraint.NONE : Constraint.ANY;
      nativeProviders = ImmutableList.builder();
      skylarkProviders = ImmutableList.builder();
    }

    /**
     * Add an alternative set of Skylark providers.
     *
     * If all of these providers are present in the dependency, the dependency satisfies
     * {@link RequiredProviders}.
     */
    public Builder addSkylarkSet(ImmutableSet<SkylarkProviderIdentifier> skylarkProviderSet) {
      constraint = Constraint.RESTRICTED;
      Preconditions.checkState(!skylarkProviderSet.isEmpty());
      this.skylarkProviders.add(skylarkProviderSet);
      return this;
    }

    /**
     * Add an alternative set of native providers.
     *
     * If all of these providers are present in the dependency, the dependency satisfies
     * {@link RequiredProviders}.
     */
    public Builder addNativeSet(ImmutableSet<Class<?>> nativeProviderSet) {
      constraint = Constraint.RESTRICTED;
      Preconditions.checkState(!nativeProviderSet.isEmpty());
      this.nativeProviders.add(nativeProviderSet);
      return this;
    }

    public RequiredProviders build() {
      return new RequiredProviders(constraint, nativeProviders.build(), skylarkProviders.build());
    }
  }
}
