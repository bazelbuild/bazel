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
import com.google.common.collect.Interner;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.pkgcache.FilteringPolicy;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

/** Singleton result of {@link CollectTargetsInPackageFunction}. */
public class CollectTargetsInPackageValue implements SkyValue {
  @AutoCodec
  public static final CollectTargetsInPackageValue INSTANCE = new CollectTargetsInPackageValue();

  private CollectTargetsInPackageValue() {}

  /**
   * Creates a key for evaluation of {@link CollectTargetsInPackageFunction}. See that class's
   * comment for what callers should have done beforehand.
   */
  public static CollectTargetsInPackageKey key(
      PackageIdentifier packageId, FilteringPolicy filteringPolicy) {
    return CollectTargetsInPackageKey.create(packageId, filteringPolicy);
  }

  /** {@link SkyKey} argument. */
  @AutoValue
  @AutoCodec
  public abstract static class CollectTargetsInPackageKey implements SkyKey {
    private static final Interner<CollectTargetsInPackageKey> interner =
        BlazeInterners.newWeakInterner();

    @AutoCodec.VisibleForSerialization
    @AutoCodec.Instantiator
    public static CollectTargetsInPackageKey create(
        PackageIdentifier packageId, FilteringPolicy filteringPolicy) {
      return interner.intern(
          new AutoValue_CollectTargetsInPackageValue_CollectTargetsInPackageKey(
              packageId, filteringPolicy));
    }

    public abstract PackageIdentifier getPackageId();

    public abstract FilteringPolicy getFilteringPolicy();

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.COLLECT_TARGETS_IN_PACKAGE;
    }
  }
}
