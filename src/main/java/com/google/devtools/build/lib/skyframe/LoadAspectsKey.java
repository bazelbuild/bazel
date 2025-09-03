// Copyright 2025 The Bazel Authors. All rights reserved.
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

import static com.google.common.base.Preconditions.checkArgument;

import com.google.common.base.MoreObjects;
import com.google.common.base.Objects;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.packages.AspectClass;
import com.google.devtools.build.lib.skyframe.serialization.VisibleForSerialization;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyKey.SkyKeyInterner;
import javax.annotation.Nullable;

/** {@link SkyKey} for building top-level aspects details. */
@AutoCodec
public final class LoadAspectsKey implements SkyKey {
  private static final SkyKeyInterner<LoadAspectsKey> interner = SkyKey.newInterner();

  private final ImmutableList<AspectClass> topLevelAspectsClasses;
  private final ImmutableMap<String, String> topLevelAspectsParameters;
  private final int hashCode;

  public static LoadAspectsKey create(
      ImmutableList<AspectClass> topLevelAspectsClasses,
      ImmutableMap<String, String> topLevelAspectsParameters) {
    return interner.intern(
        new LoadAspectsKey(
            topLevelAspectsClasses,
            topLevelAspectsParameters,
            Objects.hashCode(topLevelAspectsClasses, topLevelAspectsParameters)));
  }

  @VisibleForSerialization
  @AutoCodec.Interner
  static LoadAspectsKey intern(LoadAspectsKey key) {
    return interner.intern(key);
  }

  private LoadAspectsKey(
      ImmutableList<AspectClass> topLevelAspectsClasses,
      @Nullable ImmutableMap<String, String> topLevelAspectsParameters,
      int hashCode) {
    checkArgument(!topLevelAspectsClasses.isEmpty(), "No aspects");
    this.topLevelAspectsClasses = topLevelAspectsClasses;
    this.topLevelAspectsParameters = topLevelAspectsParameters;
    this.hashCode = hashCode;
  }

  @Override
  public SkyFunctionName functionName() {
    return SkyFunctions.LOAD_ASPECTS;
  }

  ImmutableList<AspectClass> getTopLevelAspectsClasses() {
    return topLevelAspectsClasses;
  }

  ImmutableMap<String, String> getTopLevelAspectsParameters() {
    return topLevelAspectsParameters;
  }

  @Override
  public int hashCode() {
    return hashCode;
  }

  @Override
  public boolean equals(Object o) {
    if (o == this) {
      return true;
    }
    if (!(o instanceof LoadAspectsKey that)) {
      return false;
    }
    return hashCode == that.hashCode
        && topLevelAspectsClasses.equals(that.topLevelAspectsClasses)
        && topLevelAspectsParameters.equals(that.topLevelAspectsParameters);
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("topLevelAspectsClasses", topLevelAspectsClasses)
        .add("topLevelAspectsParameters", topLevelAspectsParameters)
        .toString();
  }

  @Override
  public SkyKeyInterner<LoadAspectsKey> getSkyKeyInterner() {
    return interner;
  }
}
