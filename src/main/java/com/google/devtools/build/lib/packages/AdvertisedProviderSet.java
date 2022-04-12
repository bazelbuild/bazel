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
import com.google.devtools.build.lib.util.Fingerprint;
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
      new AdvertisedProviderSet(true, ImmutableSet.of(), ImmutableSet.of());
  public static final AdvertisedProviderSet EMPTY =
      new AdvertisedProviderSet(false, ImmutableSet.of(), ImmutableSet.of());

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
    return this.canHaveAnyProvider == that.canHaveAnyProvider
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
        builtinProviders, starlarkProviders);
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

  /**
   * Adds the fingerprints of this {@link AdvertisedProviderSet} into {@code fp}.
   *
   * <p>Fingerprints of {@link AdvertisedProviderSet} must have the following properties:
   *
   * <ul>
   *   <li>If {@code aps1.equals(aps2)} then {@code aps1} and {@code aps2} have the same
   *       fingerprint.
   *   <li>If {@code !aps1.equals(aps2)} then {@code aps1} and {@code aps2} don't have the same
   *       fingerprint (except for unintentional digest collisions).
   * </ul>
   *
   * <p>In other words, this method is a proxy for {@link #equals}. These properties *do not* need
   * to be maintained across Blaze versions (e.g. there's no need to worry about historical
   * serialized fingerprints).
   */
  public void fingerprint(Fingerprint fp) {
    fp.addBoolean(canHaveAnyProvider);
    // #builtinProviders and #starlarkProviders are ordered according to the calls to the builder
    // methods, and that order is assumed to be deterministic.
    builtinProviders.forEach(clazz -> fp.addString(clazz.getCanonicalName()));
    starlarkProviders.forEach(starlarkProvider -> starlarkProvider.fingerprint(fp));
  }

  public static Builder builder() {
    return new Builder();
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
  }
}
