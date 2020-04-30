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

import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.VisibleForSerialization;
import java.util.Objects;
import java.util.function.Function;
import java.util.function.Predicate;
import javax.annotation.Nullable;

/**
 * Represents a constraint on a set of providers required by a dependency (of a rule or an aspect).
 *
 * <p>Currently we support three kinds of constraints:
 *
 * <ul>
 *   <li>accept any dependency.
 *   <li>accept no dependency (used for aspects-on-aspects to indicate that an aspect never wants to
 *       see any other aspect applied to a target.
 *   <li>accept a dependency that provides all providers from one of several sets of providers. It
 *       just so happens that in all current usages these sets are either all native providers or
 *       all Starlark providers, so this is the only use case this class currently supports.
 * </ul>
 */
@Immutable
@AutoCodec
public final class RequiredProviders {
  /** A constraint: either ANY, NONE, or RESTRICTED */
  private final Constraint constraint;
  /**
   * Sets of native providers. If non-empty, {@link #constraint} is {@link Constraint#RESTRICTED}
   */
  private final ImmutableList<ImmutableSet<Class<? extends TransitiveInfoProvider>>>
      nativeProviders;
  /**
   * Sets of native providers. If non-empty, {@link #constraint} is {@link Constraint#RESTRICTED}
   */
  private final ImmutableList<ImmutableSet<StarlarkProviderIdentifier>> skylarkProviders;

  public String getDescription() {
    return constraint.getDescription(this);
  }

  @Override
  public String toString() {
    return getDescription();
  }

  /** Represents one of the constraints as desctibed in {@link RequiredProviders} */
  @VisibleForSerialization
  enum Constraint {
    /** Accept any dependency */
    ANY {
      @Override
      public boolean satisfies(
          AdvertisedProviderSet advertisedProviderSet,
          RequiredProviders requiredProviders,
          Builder missing) {
        return true;
      }

      @Override
      public boolean satisfies(
          Predicate<Class<? extends TransitiveInfoProvider>> hasNativeProvider,
          Predicate<StarlarkProviderIdentifier> hasSkylarkProvider,
          RequiredProviders requiredProviders,
          Builder missingProviders) {
        return true;
      }

      @Override
      Builder copyAsBuilder(RequiredProviders providers) {
        return acceptAnyBuilder();
      }

      @Override
      public String getDescription(RequiredProviders providers) {
        return "no providers required";
      }
    },
    /** Accept no dependency */
    NONE {
      @Override
      public boolean satisfies(
          AdvertisedProviderSet advertisedProviderSet,
          RequiredProviders requiredProviders,
          Builder missing) {
        return false;
      }

      @Override
      public boolean satisfies(
          Predicate<Class<? extends TransitiveInfoProvider>> hasNativeProvider,
          Predicate<StarlarkProviderIdentifier> hasSkylarkProvider,
          RequiredProviders requiredProviders,
          Builder missingProviders) {
        return false;
      }

      @Override
      Builder copyAsBuilder(RequiredProviders providers) {
        return acceptNoneBuilder();
      }

      @Override
      public String getDescription(RequiredProviders providers) {
        return "no providers accepted";
      }
    },

    /** Accept a dependency that has all providers from one of the sets. */
    RESTRICTED {
      @Override
      public boolean satisfies(
          final AdvertisedProviderSet advertisedProviderSet,
          RequiredProviders requiredProviders,
          Builder missing) {
        if (advertisedProviderSet.canHaveAnyProvider()) {
          return true;
        }
        return satisfies(
            advertisedProviderSet.getNativeProviders()::contains,
            advertisedProviderSet.getSkylarkProviders()::contains,
            requiredProviders,
            missing);
      }

      @Override
      public boolean satisfies(
          Predicate<Class<? extends TransitiveInfoProvider>> hasNativeProvider,
          Predicate<StarlarkProviderIdentifier> hasSkylarkProvider,
          RequiredProviders requiredProviders,
          Builder missingProviders) {
        for (ImmutableSet<Class<? extends TransitiveInfoProvider>> nativeProviderSet :
            requiredProviders.nativeProviders) {
          if (nativeProviderSet.stream().allMatch(hasNativeProvider)) {
            return true;
          }

          // Collect missing providers
          if (missingProviders != null) {
            missingProviders.addNativeSet(
                nativeProviderSet
                    .stream()
                    .filter(hasNativeProvider.negate())
                    .collect(ImmutableSet.toImmutableSet()));
          }
        }

        for (ImmutableSet<StarlarkProviderIdentifier> skylarkProviderSet :
            requiredProviders.skylarkProviders) {
          if (skylarkProviderSet.stream().allMatch(hasSkylarkProvider)) {
            return true;
          }
          // Collect missing providers
          if (missingProviders != null) {
            missingProviders.addSkylarkSet(
                skylarkProviderSet
                    .stream()
                    .filter(hasSkylarkProvider.negate())
                    .collect(ImmutableSet.toImmutableSet()));
          }
        }
        return false;
      }

      @Override
      Builder copyAsBuilder(RequiredProviders providers) {
        Builder result = acceptAnyBuilder();
        for (ImmutableSet<Class<? extends TransitiveInfoProvider>> nativeProviderSet :
            providers.nativeProviders) {
          result.addNativeSet(nativeProviderSet);
        }
        for (ImmutableSet<StarlarkProviderIdentifier> skylarkProviderSet :
            providers.skylarkProviders) {
          result.addSkylarkSet(skylarkProviderSet);
        }
        return result;
      }

      @Override
      public String getDescription(RequiredProviders providers) {
        StringBuilder result = new StringBuilder();
        describe(result, providers.nativeProviders, Class::getSimpleName);
        describe(result, providers.skylarkProviders, id -> "'" + id.toString() + "'");
        return result.toString();
      }
    };

    /** Checks if {@code advertisedProviderSet} satisfies these {@code RequiredProviders} */
    public abstract boolean satisfies(
        AdvertisedProviderSet advertisedProviderSet,
        RequiredProviders requiredProviders,
        Builder missing);

    /**
     * Checks if a set of providers encoded by predicates {@code hasNativeProviders} and {@code
     * hasSkylarkProvider} satisfies these {@code RequiredProviders}
     */
    abstract boolean satisfies(
        Predicate<Class<? extends TransitiveInfoProvider>> hasNativeProvider,
        Predicate<StarlarkProviderIdentifier> hasSkylarkProvider,
        RequiredProviders requiredProviders,
        @Nullable Builder missingProviders);

    abstract Builder copyAsBuilder(RequiredProviders providers);

    /** Returns a string describing the providers that can be presented to the user. */
    abstract String getDescription(RequiredProviders providers);
  }

  /** Checks if {@code advertisedProviderSet} satisfies this {@code RequiredProviders} instance. */
  public boolean isSatisfiedBy(AdvertisedProviderSet advertisedProviderSet) {
    return constraint.satisfies(advertisedProviderSet, this, null);
  }

  /**
   * Checks if a set of providers encoded by predicates {@code hasNativeProviders} and {@code
   * hasSkylarkProvider} satisfies this {@code RequiredProviders} instance.
   */
  public boolean isSatisfiedBy(
      Predicate<Class<? extends TransitiveInfoProvider>> hasNativeProvider,
      Predicate<StarlarkProviderIdentifier> hasSkylarkProvider) {
    return constraint.satisfies(hasNativeProvider, hasSkylarkProvider, this, null);
  }

  /**
   * Returns providers that are missing. If none are missing, returns {@code RequiredProviders} that
   * accept anything.
   */
  public RequiredProviders getMissing(
      Predicate<Class<? extends TransitiveInfoProvider>> hasNativeProvider,
      Predicate<StarlarkProviderIdentifier> hasSkylarkProvider) {
    Builder builder = acceptAnyBuilder();
    if (constraint.satisfies(hasNativeProvider, hasSkylarkProvider, this, builder)) {
      // Ignore all collected missing providers.
      return acceptAnyBuilder().build();
    }
    return builder.build();
  }

  /**
   * Returns providers that are missing. If none are missing, returns {@code RequiredProviders} that
   * accept anything.
   */
  public RequiredProviders getMissing(AdvertisedProviderSet set) {
    Builder builder = acceptAnyBuilder();
    if (constraint.satisfies(set, this, builder)) {
      // Ignore all collected missing providers.
      return acceptAnyBuilder().build();
    }
    return builder.build();
  }

  /** Returns true if this {@code RequiredProviders} instance accept any set of providers. */
  public boolean acceptsAny() {
    return constraint.equals(Constraint.ANY);
  }

  @VisibleForSerialization
  RequiredProviders(
      Constraint constraint,
      ImmutableList<ImmutableSet<Class<? extends TransitiveInfoProvider>>> nativeProviders,
      ImmutableList<ImmutableSet<StarlarkProviderIdentifier>> skylarkProviders) {
    this.constraint = constraint;

    Preconditions.checkState(constraint.equals(Constraint.RESTRICTED)
        || (nativeProviders.isEmpty() && skylarkProviders.isEmpty())
    );

    this.nativeProviders = nativeProviders;
    this.skylarkProviders = skylarkProviders;
  }

  /** Helper method to describe lists of sets of things. */
  private static <T> void describe(
      StringBuilder result,
      ImmutableList<ImmutableSet<T>> listOfSets,
      Function<T, String> describeOne) {
    Joiner joiner = Joiner.on(", ");
    for (ImmutableSet<T> ids : listOfSets) {
      if (result.length() > 0) {
        result.append(" or ");
      }
      result.append((ids.size() > 1) ? "[" : "");
      joiner.appendTo(result, ids.stream().map(describeOne).iterator());
      result.append((ids.size() > 1) ? "]" : "");
    }
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }
    RequiredProviders that = (RequiredProviders) o;
    return constraint == that.constraint
        && Objects.equals(nativeProviders, that.nativeProviders)
        && Objects.equals(skylarkProviders, that.skylarkProviders);
  }

  @Override
  public int hashCode() {
    return Objects.hash(constraint, nativeProviders, skylarkProviders);
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

  /** Returns a Builder initialized to the same value as this {@code RequiredProvider} */
  public Builder copyAsBuilder() {
    return constraint.copyAsBuilder(this);
  }

  /** A builder for {@link RequiredProviders} */
  public static class Builder {
    private final ImmutableList.Builder<ImmutableSet<Class<? extends TransitiveInfoProvider>>>
        nativeProviders;
    private final ImmutableList.Builder<ImmutableSet<StarlarkProviderIdentifier>> skylarkProviders;
    private Constraint constraint;

    private Builder(boolean acceptNone) {
      constraint = acceptNone ? Constraint.NONE : Constraint.ANY;
      nativeProviders = ImmutableList.builder();
      skylarkProviders = ImmutableList.builder();
    }

    /**
     * Add an alternative set of Starlark providers.
     *
     * <p>If all of these providers are present in the dependency, the dependency satisfies {@link
     * RequiredProviders}.
     */
    public Builder addSkylarkSet(ImmutableSet<StarlarkProviderIdentifier> skylarkProviderSet) {
      constraint = Constraint.RESTRICTED;
      Preconditions.checkState(!skylarkProviderSet.isEmpty());
      this.skylarkProviders.add(skylarkProviderSet);
      return this;
    }

    /**
     * Add an alternative set of native providers.
     *
     * <p>If all of these providers are present in the dependency, the dependency satisfies {@link
     * RequiredProviders}.
     */
    public Builder addNativeSet(
        ImmutableSet<Class<? extends TransitiveInfoProvider>> nativeProviderSet) {
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
