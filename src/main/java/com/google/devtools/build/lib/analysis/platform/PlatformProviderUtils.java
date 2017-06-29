// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis.platform;

import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.analysis.SkylarkProviderCollection;
import com.google.devtools.build.lib.util.Preconditions;

/** Utility methods to help locate platform-related providers. */
public class PlatformProviderUtils {

  /** Retrieves and casts the {@link PlatformInfo} provider from the given target. */
  public static PlatformInfo platform(SkylarkProviderCollection target) {
    Object provider = target.get(PlatformInfo.SKYLARK_IDENTIFIER);
    if (provider == null) {
      return null;
    }
    Preconditions.checkState(provider instanceof PlatformInfo);
    return (PlatformInfo) provider;
  }

  /** Retrieves and casts {@link PlatformInfo} providers from the given targets. */
  public static Iterable<PlatformInfo> platforms(
      Iterable<? extends SkylarkProviderCollection> targets) {
    return Iterables.transform(targets, PlatformProviderUtils::platform);
  }

  /** Retrieves and casts the {@link ConstraintSettingInfo} provider from the given target. */
  public static ConstraintSettingInfo constraintSetting(SkylarkProviderCollection target) {
    Object provider = target.get(ConstraintSettingInfo.SKYLARK_IDENTIFIER);
    if (provider == null) {
      return null;
    }
    Preconditions.checkState(provider instanceof ConstraintSettingInfo);
    return (ConstraintSettingInfo) provider;
  }

  /** Retrieves and casts {@link ConstraintSettingInfo} providers from the given targets. */
  public static Iterable<ConstraintSettingInfo> constraintSettings(
      Iterable<? extends SkylarkProviderCollection> targets) {
    return Iterables.transform(targets, PlatformProviderUtils::constraintSetting);
  }

  /** Retrieves and casts the {@link ConstraintValueInfo} provider from the given target. */
  public static ConstraintValueInfo constraintValue(SkylarkProviderCollection target) {
    Object provider = target.get(ConstraintValueInfo.SKYLARK_IDENTIFIER);
    if (provider == null) {
      return null;
    }
    Preconditions.checkState(provider instanceof ConstraintValueInfo);
    return (ConstraintValueInfo) provider;
  }

  /** Retrieves and casts {@link ConstraintValueInfo} providers from the given targets. */
  public static Iterable<ConstraintValueInfo> constraintValues(
      Iterable<? extends SkylarkProviderCollection> targets) {
    return Iterables.transform(targets, PlatformProviderUtils::constraintValue);
  }
}
