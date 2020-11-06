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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import java.util.ArrayList;
import java.util.Objects;

/**
 * Captures the set of providers rules and aspects can advertise. It is either of:
 *
 * <ul>
 *   <li>a set of builtin and Starlark providers
 *   <li>"can have any provider" set that alias rules have.
 * </ul>
 *
 * <p>Built-in providers should in theory only contain subclasses of {@link
 * com.google.devtools.build.lib.analysis.TransitiveInfoProvider}, but our current dependency
 * structure does not allow a reference to that class here.
 */
@Immutable
public final class AdvertisedProviderSet {
  private final boolean canHaveAnyProvider;
  private final ImmutableSet<Class<?>> builtinProviders;
  private final ImmutableSet<StarlarkProviderIdentifier> starlarkProviders;

  private AdvertisedProviderSet(
      boolean canHaveAnyProvider,
      ImmutableSet<Class<?>> builtinProviders,
      ImmutableSet<StarlarkProviderIdentifier> starlarkProviders) {
    this.canHaveAnyProvider = canHaveAnyProvider;
    this.builtinProviders = builtinProviders;
    this.starlarkProviders = starlarkProviders;
  }

  public static final AdvertisedProviderSet ANY =
      new AdvertisedProviderSet(
          true, ImmutableSet.<Class<?>>of(), ImmutableSet.<StarlarkProviderIdentifier>of());
  public static final AdvertisedProviderSet EMPTY =
      new AdvertisedProviderSet(
          false, ImmutableSet.<Class<?>>of(), ImmutableSet.<StarlarkProviderIdentifier>of());

  public static AdvertisedProviderSet create(
      ImmutableSet<Class<?>> builtinProviders,
      ImmutableSet<StarlarkProviderIdentifier> starlarkProviders) {
    if (builtinProviders.isEmpty() && starlarkProviders.isEmpty()) {
      return EMPTY;
    }
    return new AdvertisedProviderSet(false, builtinProviders, starlarkProviders);
  }

  @Override
  public int hashCode() {
    return Objects.hash(canHaveAnyProvider, builtinProviders, starlarkProviders);
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }

    if (!(obj instanceof AdvertisedProviderSet)) {
      return false;
    }

    AdvertisedProviderSet that = (AdvertisedProviderSet) obj;
    return Objects.equals(this.canHaveAnyProvider, that.canHaveAnyProvider)
        && Objects.equals(this.builtinProviders, that.builtinProviders)
        && Objects.equals(this.starlarkProviders, that.starlarkProviders);
  }

  @Override
  public String toString() {
    if (canHaveAnyProvider()) {
      return "Any Provider";
    }
    return String.format(
        "allowed built-in providers=%s, allowed Starlark providers=%s",
        getBuiltinProviders(), getStarlarkProviders());
  }

  /** Checks whether the rule can have any provider.
   *
   *  Used for alias rules.
   */
  public boolean canHaveAnyProvider() {
    return canHaveAnyProvider;
  }

  /** Get all advertised built-in providers. */
  public ImmutableSet<Class<?>> getBuiltinProviders() {
    return builtinProviders;
  }

  /** Get all advertised Starlark providers. */
  public ImmutableSet<StarlarkProviderIdentifier> getStarlarkProviders() {
    return starlarkProviders;
  }

  public static Builder builder() {
    return new Builder();
  }

  /**
   * Returns {@code true} if this provider set can have any provider, or if it advertises the
   * specific built-in provider requested.
   */
  public boolean advertises(Class<?> builtinProviderClass) {
    if (canHaveAnyProvider()) {
      return true;
    }
    return builtinProviders.contains(builtinProviderClass);
  }

  /**
   * Returns {@code true} if this provider set can have any provider, or if it advertises the
   * specific Starlark provider requested.
   */
  public boolean advertises(StarlarkProviderIdentifier starlarkProvider) {
    if (canHaveAnyProvider()) {
      return true;
    }
    return starlarkProviders.contains(starlarkProvider);
  }

  /** Builder for {@link AdvertisedProviderSet} */
  public static class Builder {
    private boolean canHaveAnyProvider;
    private final ArrayList<Class<?>> builtinProviders;
    private final ArrayList<StarlarkProviderIdentifier> starlarkProviders;

    private Builder() {
      builtinProviders = new ArrayList<>();
      starlarkProviders = new ArrayList<>();
    }

    /**
     * Advertise all providers inherited from a parent rule.
     */
    public Builder addParent(AdvertisedProviderSet parentSet) {
      Preconditions.checkState(!canHaveAnyProvider, "Alias rules inherit from no other rules");
      Preconditions.checkState(!parentSet.canHaveAnyProvider(),
          "Cannot inherit from alias rules");
      builtinProviders.addAll(parentSet.getBuiltinProviders());
      starlarkProviders.addAll(parentSet.getStarlarkProviders());
      return this;
    }

    public Builder addBuiltin(Class<?> builtinProvider) {
      this.builtinProviders.add(builtinProvider);
      return this;
    }

    public void canHaveAnyProvider() {
      Preconditions.checkState(builtinProviders.isEmpty() && starlarkProviders.isEmpty());
      this.canHaveAnyProvider = true;
    }

    public AdvertisedProviderSet build() {
      if (canHaveAnyProvider) {
        Preconditions.checkState(builtinProviders.isEmpty() && starlarkProviders.isEmpty());
        return ANY;
      }
      return AdvertisedProviderSet.create(
          ImmutableSet.copyOf(builtinProviders), ImmutableSet.copyOf(starlarkProviders));
    }

    public Builder addStarlark(String providerName) {
      starlarkProviders.add(StarlarkProviderIdentifier.forLegacy(providerName));
      return this;
    }

    public Builder addStarlark(StarlarkProviderIdentifier id) {
      starlarkProviders.add(id);
      return this;
    }

    public Builder addStarlark(Provider.Key id) {
      starlarkProviders.add(StarlarkProviderIdentifier.forKey(id));
      return this;
    }
  }
}
