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

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.actions.BasicActionLookupValue;
import com.google.devtools.build.lib.analysis.ConfiguredAspect;
import com.google.devtools.build.lib.analysis.ProviderCollection;
import com.google.devtools.build.lib.cmdline.Label;
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
  @Nullable private Label label;
  @Nullable private Aspect aspect;
  @Nullable private Location location;
  @Nullable private AspectKey key;
  @Nullable private ConfiguredAspect configuredAspect;
  // May be null either after clearing or because transitive packages are not tracked.
  @Nullable private transient NestedSet<Package> transitivePackagesForPackageRootResolution;

  public AspectValue(
      AspectKey key,
      Aspect aspect,
      Label label,
      Location location,
      ConfiguredAspect configuredAspect,
      NestedSet<Package> transitivePackagesForPackageRootResolution) {
    super(configuredAspect.getActions());
    this.label = Preconditions.checkNotNull(label, actions);
    this.aspect = Preconditions.checkNotNull(aspect, label);
    this.location = Preconditions.checkNotNull(location, label);
    this.key = Preconditions.checkNotNull(key, label);
    this.configuredAspect = Preconditions.checkNotNull(configuredAspect, label);
    this.transitivePackagesForPackageRootResolution = transitivePackagesForPackageRootResolution;
  }

  public ConfiguredAspect getConfiguredAspect() {
    return Preconditions.checkNotNull(configuredAspect);
  }

  public Label getLabel() {
    return Preconditions.checkNotNull(label);
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
    Preconditions.checkNotNull(label, this);
    Preconditions.checkNotNull(aspect, this);
    Preconditions.checkNotNull(location, this);
    Preconditions.checkNotNull(key, this);
    Preconditions.checkNotNull(configuredAspect, this);
    if (clearEverything) {
      label = null;
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
        .add("label", label)
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
