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

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.actions.BasicActionLookupValue;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.packages.Aspect;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.skyframe.AspectValueKey.AspectKey;
import com.google.devtools.build.lib.syntax.Location;
import javax.annotation.Nullable;

/** An aspect in the context of the Skyframe graph. */
public final class AspectValue extends BasicActionLookupValue implements ConfiguredObjectValue {
  // These variables are only non-final because they may be clear()ed to save memory. They are null
  // only after they are cleared except for transitivePackagesForPackageRootResolution.
  @Nullable private Aspect aspect;
  @Nullable private Location location;
  // Normally the key used to evaluate this value in AspectFunction#compute. But in the case of a
  // top-level SkylarkAspectKey, the AspectValue will be this value but the key will be the
  // associated aspect key from SkylarkAspectKey#toAspectkey.
  @Nullable private AspectKey key;
  @Nullable private ConfiguredAspect configuredAspect;
  // May be null either after clearing or because transitive packages are not tracked.
  @Nullable private transient NestedSet<Package> transitivePackagesForPackageRootResolution;

  public AspectValue(
      AspectKey key,
      Aspect aspect,
      Location location,
      ConfiguredAspect configuredAspect,
      NestedSet<Package> transitivePackagesForPackageRootResolution) {
    super(configuredAspect.getActions());
    this.key = key;
    this.aspect = Preconditions.checkNotNull(aspect, location);
    this.location = Preconditions.checkNotNull(location, aspect);
    this.configuredAspect = Preconditions.checkNotNull(configuredAspect, location);
    this.transitivePackagesForPackageRootResolution = transitivePackagesForPackageRootResolution;
  }

  public ConfiguredAspect getConfiguredAspect() {
    return Preconditions.checkNotNull(configuredAspect);
  }

  public Location getLocation() {
    return Preconditions.checkNotNull(location);
  }

  public AspectKey getKey() {
    return Preconditions.checkNotNull(key);
  }

  public Aspect getAspect() {
    return Preconditions.checkNotNull(aspect);
  }

  @Override
  public void clear(boolean clearEverything) {
    Preconditions.checkNotNull(aspect, this);
    Preconditions.checkNotNull(location, this);
    Preconditions.checkNotNull(key, this);
    Preconditions.checkNotNull(configuredAspect, this);
    if (clearEverything) {
      aspect = null;
      location = null;
      key = null;
      configuredAspect = null;
    }
    transitivePackagesForPackageRootResolution = null;
  }

  @Override
  public NestedSet<Package> getTransitivePackagesForPackageRootResolution() {
    return Preconditions.checkNotNull(transitivePackagesForPackageRootResolution);
  }

  @Override
  public String toString() {
    return getStringHelper()
        .add("key", key)
        .add("location", location)
        .add("aspect", aspect)
        .add("configuredAspect", configuredAspect)
        .toString();
  }

  @Override
  public ProviderCollection getConfiguredObject() {
    return getConfiguredAspect();
  }
}
