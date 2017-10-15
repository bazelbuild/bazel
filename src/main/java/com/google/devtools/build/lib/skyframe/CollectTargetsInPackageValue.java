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
package com.google.devtools.build.lib.skyframe;

import com.google.auto.value.AutoValue;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.pkgcache.FilteringPolicy;
import com.google.devtools.build.skyframe.LegacySkyKey;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.io.Serializable;

/** Singleton result of {@link CollectTargetsInPackageFunction}. */
public class CollectTargetsInPackageValue implements SkyValue {
  public static final CollectTargetsInPackageValue INSTANCE = new CollectTargetsInPackageValue();

  private CollectTargetsInPackageValue() {}

  /**
   * Creates a key for evaluation of {@link CollectTargetsInPackageFunction}. See that class's
   * comment for what callers should have done beforehand.
   */
  public static SkyKey key(PackageIdentifier packageId, FilteringPolicy filteringPolicy) {
    return LegacySkyKey.create(
        SkyFunctions.COLLECT_TARGETS_IN_PACKAGE,
        CollectTargetsInPackageKey.create(packageId, filteringPolicy));
  }

  /** {@link SkyKey} argument. */
  @AutoValue
  public abstract static class CollectTargetsInPackageKey implements Serializable {
    public static CollectTargetsInPackageKey create(
        PackageIdentifier packageId, FilteringPolicy filteringPolicy) {
      return new AutoValue_CollectTargetsInPackageValue_CollectTargetsInPackageKey(
          packageId, filteringPolicy);
    }

    public abstract PackageIdentifier getPackageId();

    public abstract FilteringPolicy getFilteringPolicy();
  }
}
