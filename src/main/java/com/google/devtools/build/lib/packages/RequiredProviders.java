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
 *       just so happens that in all current usages these sets are either all builtin providers or
 *       all Starlark providers, so this is the only use case this class currently supports.
 * </ul>
 */
@Immutable
@AutoCodec
public final class RequiredProviders {
  /** A constraint: either ANY, NONE, or RESTRICTED */
  private final Constraint constraint;
  /**
   * Sets of builtin providers. If non-empty, {@link #constraint} is {@link Constraint#RESTRICTED}
   */
  private final ImmutableList<ImmutableSet<Class<? extends TransitiveInfoProvider>>>
      builtinProviders;
  /**
   * Sets of builtin providers. If non-empty, {@link #constraint} is {@link Constraint#RESTRICTED}
   */
  private final ImmutableList<ImmutableSet<StarlarkProviderIdentifier>> starlarkProviders;

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
          Predicate<Class<? extends TransitiveInfoProvider>> hasBuiltinProvider,
          Predicate<StarlarkProviderIdentifier> hasStarlarkProvider,
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
          Predicate<Class<? extends TransitiveInfoProvider>> hasBuiltinProvider,
          Predicate<StarlarkProviderIdentifier> hasStarlarkProvider,
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
            advertisedProviderSet.getBuiltinProviders()::contains,
            advertisedProviderSet.getStarlarkProviders()::contains,
            requiredProviders,
            missing);
      }

      @Override
      boolean satisfies(
          Predicate<Class<? extends TransitiveInfoProvider>> hasBuiltinProvider,
          Predicate<StarlarkProviderIdentifier> hasStarlarkProvider,
          RequiredProviders requiredProviders,
          Builder missingProviders) {
        for (ImmutableSet<Class<? extends TransitiveInfoProvider>> builtinProviderSet :
            requiredProviders.builtinProviders) {
          if (builtinProviderSet.stream().allMatch(hasBuiltinProvider)) {
            return true;
          }

          // Collect missing providers
          if (missingProviders != null) {
            missingProviders.addBuiltinSet(
                builtinProviderSet.stream()
                    .filter(hasBuiltinProvider.negate())
                    .collect(ImmutableSet.toImmutableSet()));
          }
        }

        for (ImmutableSet<StarlarkProviderIdentifier> starlarkProviderSet :
            requiredProviders.starlarkProviders) {
          if (starlarkProviderSet.stream().allMatch(hasStarlarkProvider)) {
            return true;
          }
          // Collect missing providers
          if (missingProviders != null) {
            missingProviders.addStarlarkSet(
                starlarkProviderSet.stream()
                    .filter(hasStarlarkProvider.negate())
                    .collect(ImmutableSet.toImmutableSet()));
          }
        }
        return false;
      }

      @Override
      Builder copyAsBuilder(RequiredProviders providers) {
        Builder result = acceptAnyBuilder();
        for (ImmutableSet<Class<? extends TransitiveInfoProvider>> builtinProviderSet :
            providers.builtinProviders) {
          result.addBuiltinSet(builtinProviderSet);
        }
        for (ImmutableSet<StarlarkProviderIdentifier> starlarkProviderSet :
            providers.starlarkProviders) {
          result.addStarlarkSet(starlarkProviderSet);
        }
        return result;
      }

      @Override
      public String getDescription(RequiredProviders providers) {
        StringBuilder result = new StringBuilder();
        describe(result, providers.builtinProviders, Class::getSimpleName);
        describe(result, providers.starlarkProviders, id -> "'" + id.toString() + "'");
        return result.toString();
      }
    };

    /** Checks if {@code advertisedProviderSet} satisfies these {@code RequiredProviders} */
    protected abstract boolean satisfies(
        AdvertisedProviderSet advertisedProviderSet,
        RequiredProviders requiredProviders,
        Builder missing);

    /**
     * Checks if a set of providers encoded by predicates {@code hasBuiltinProvider} and {@code
     * hasStarlarkProvider} satisfies these {@code RequiredProviders}
     */
    abstract boolean satisfies(
        Predicate<Class<? extends TransitiveInfoProvider>> hasBuiltinProvider,
        Predicate<StarlarkProviderIdentifier> hasStarlarkProvider,
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
   * Checks if a set of providers encoded by predicates {@code hasBuiltinProvider} and {@code
   * hasStarlarkProvider} satisfies this {@code RequiredProviders} instance.
   */
  public boolean isSatisfiedBy(
      Predicate<Class<? extends TransitiveInfoProvider>> hasBuiltinProvider,
      Predicate<StarlarkProviderIdentifier> hasStarlarkProvider) {
    return constraint.satisfies(hasBuiltinProvider, hasStarlarkProvider, this, null);
  }

  /**
   * Returns providers that are missing. If none are missing, returns {@code RequiredProviders} that
   * accept anything.
   */
  public RequiredProviders getMissing(
      Predicate<Class<? extends TransitiveInfoProvider>> hasBuiltinProvider,
      Predicate<StarlarkProviderIdentifier> hasStarlarkProvider) {
    Builder builder = acceptAnyBuilder();
    if (constraint.satisfies(hasBuiltinProvider, hasStarlarkProvider, this, builder)) {
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
      ImmutableList<ImmutableSet<Class<? extends TransitiveInfoProvider>>> builtinProviders,
      ImmutableList<ImmutableSet<StarlarkProviderIdentifier>> starlarkProviders) {
    this.constraint = constraint;

    Preconditions.checkState(
        constraint.equals(Constraint.RESTRICTED)
            || (builtinProviders.isEmpty() && starlarkProviders.isEmpty()));

    this.builtinProviders = builtinProviders;
    this.starlarkProviders = starlarkProviders;
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
        && Objects.equals(builtinProviders, that.builtinProviders)
        && Objects.equals(starlarkProviders, that.starlarkProviders);
  }

  @Override
  public int hashCode() {
    return Objects.hash(constraint, builtinProviders, starlarkProviders);
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
        builtinProviders;
    private final ImmutableList.Builder<ImmutableSet<StarlarkProviderIdentifier>> starlarkProviders;
    private Constraint constraint;

    private Builder(boolean acceptNone) {
      constraint = acceptNone ? Constraint.NONE : Constraint.ANY;
      builtinProviders = ImmutableList.builder();
      starlarkProviders = ImmutableList.builder();
    }

    /**
     * Add an alternative set of Starlark providers.
     *
     * <p>If all of these providers are present in the dependency, the dependency satisfies {@link
     * RequiredProviders}.
     */
    public Builder addStarlarkSet(ImmutableSet<StarlarkProviderIdentifier> starlarkProviderSet) {
      constraint = Constraint.RESTRICTED;
      Preconditions.checkState(!starlarkProviderSet.isEmpty());
      this.starlarkProviders.add(starlarkProviderSet);
      return this;
    }

    /**
     * Add an alternative set of builtin providers.
     *
     * <p>If all of these providers are present in the dependency, the dependency satisfies {@link
     * RequiredProviders}.
     */
    public Builder addBuiltinSet(
        ImmutableSet<Class<? extends TransitiveInfoProvider>> builtinProviderSet) {
      constraint = Constraint.RESTRICTED;
      Preconditions.checkState(!builtinProviderSet.isEmpty());
      this.builtinProviders.add(builtinProviderSet);
      return this;
    }

    public RequiredProviders build() {
      return new RequiredProviders(constraint, builtinProviders.build(), starlarkProviders.build());
    }
  }
}
