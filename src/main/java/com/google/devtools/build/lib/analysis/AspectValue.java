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

package com.google.devtools.build.lib.analysis;

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.devtools.build.lib.actions.BasicActionLookupValue;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.packages.Aspect;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.skyframe.AspectKeyCreator.AspectKey;
import javax.annotation.Nullable;
import net.starlark.java.syntax.Location;

/** An aspect in the context of the Skyframe graph. */
public class AspectValue extends BasicActionLookupValue
    implements ConfiguredAspect, RuleConfiguredObjectValue {

  public static AspectValue create(
      AspectKey key,
      Aspect aspect,
      Location location,
      ConfiguredAspect configuredAspect,
      @Nullable NestedSet<Package> transitivePackages) {
    return transitivePackages == null
        ? new AspectValue(key, aspect, location, configuredAspect)
        : new AspectValueWithTransitivePackages(
            key, aspect, location, configuredAspect, transitivePackages);
  }

  // These variables are only non-final because they may be clear()ed to save memory. They are null
  // only after they are cleared except for transitivePackagesForPackageRootResolution.
  @Nullable private Aspect aspect;
  @Nullable private Location location;
  @Nullable private TransitiveInfoProviderMap providers;
  // Normally the key used to evaluate this value in AspectFunction#compute. But in the case of a
  // top-level StarlarkAspectKey, the AspectValue will be this value but the key will be the
  // associated aspect key from StarlarkAspectKey#toAspectkey.
  @Nullable private AspectKey key;

  private AspectValue(
      AspectKey key, Aspect aspect, Location location, ConfiguredAspect configuredAspect) {
    super(configuredAspect.getActions());
    this.key = checkNotNull(key);
    this.aspect = checkNotNull(aspect);
    this.location = checkNotNull(location);
    this.providers = configuredAspect.getProviders();
  }

  public final Location getLocation() {
    return checkNotNull(location);
  }

  public final AspectKey getKey() {
    return checkNotNull(key);
  }

  public final Aspect getAspect() {
    return checkNotNull(aspect);
  }

  @Override
  public TransitiveInfoProviderMap getProviders() {
    return checkNotNull(providers);
  }

  @Override
  public void clear(boolean clearEverything) {
    if (clearEverything) {
      aspect = null;
      location = null;
      providers = null;
      key = null;
    }
  }

  @Nullable
  @Override
  public NestedSet<Package> getTransitivePackages() {
    return null;
  }

  @Override
  public final ProviderCollection getConfiguredObject() {
    return this;
  }

  @Override
  public final String toString() {
    return getStringHelper()
        .add("key", key)
        .add("location", location)
        .add("aspect", aspect)
        .add("providers", providers)
        .toString();
  }

  private static final class AspectValueWithTransitivePackages extends AspectValue {
    @Nullable private transient NestedSet<Package> transitivePackages; // Null after clear().

    private AspectValueWithTransitivePackages(
        AspectKey key,
        Aspect aspect,
        Location location,
        ConfiguredAspect configuredAspect,
        NestedSet<Package> transitivePackages) {
      super(key, aspect, location, configuredAspect);
      this.transitivePackages = checkNotNull(transitivePackages);
    }

    @Override
    public NestedSet<Package> getTransitivePackages() {
      return transitivePackages;
    }

    @Override
    public void clear(boolean clearEverything) {
      super.clear(clearEverything);
      transitivePackages = null;
    }
  }
}
