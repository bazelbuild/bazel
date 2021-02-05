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

import static com.google.common.base.Predicates.notNull;
import static com.google.common.collect.ImmutableList.toImmutableList;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.ProviderCollection;
import java.util.List;
import javax.annotation.Nullable;

/** Utility methods to help locate platform-related providers. */
public class PlatformProviderUtils {

  /** Retrieves and casts the {@link PlatformInfo} provider from the given target. */
  @Nullable
  public static PlatformInfo platform(@Nullable ProviderCollection target) {
    if (target == null) {
      return null;
    }
    return target.get(PlatformInfo.PROVIDER);
  }

  /** Retrieves and casts {@link PlatformInfo} providers from the given targets. */
  public static ImmutableList<PlatformInfo> platforms(List<? extends ProviderCollection> targets) {
    return targets.stream()
        .map(PlatformProviderUtils::platform)
        .filter(notNull())
        .collect(toImmutableList());
  }

  /** Retrieves and casts the {@link ConstraintSettingInfo} provider from the given target. */
  @Nullable
  public static ConstraintSettingInfo constraintSetting(@Nullable ProviderCollection target) {
    if (target == null) {
      return null;
    }
    return target.get(ConstraintSettingInfo.PROVIDER);
  }

  /** Retrieves and casts {@link ConstraintSettingInfo} providers from the given targets. */
  public static ImmutableList<ConstraintSettingInfo> constraintSettings(
      List<? extends ProviderCollection> targets) {
    return targets.stream()
        .map(PlatformProviderUtils::constraintSetting)
        .filter(notNull())
        .collect(toImmutableList());
  }

  /** Retrieves and casts the {@link ConstraintValueInfo} provider from the given target. */
  @Nullable
  public static ConstraintValueInfo constraintValue(@Nullable ProviderCollection target) {
    if (target == null) {
      return null;
    }
    return target.get(ConstraintValueInfo.PROVIDER);
  }

  /** Returns if a target provides {@link ConstraintValueInfo}. * */
  public static boolean hasConstraintValue(ProviderCollection target) {
    return target.get(ConstraintValueInfo.PROVIDER) != null;
  }

  /** Retrieves and casts {@link ConstraintValueInfo} providers from the given targets. */
  public static ImmutableList<ConstraintValueInfo> constraintValues(
      List<? extends ProviderCollection> targets) {
    return targets.stream()
        .map(PlatformProviderUtils::constraintValue)
        .filter(notNull())
        .collect(toImmutableList());
  }

  /**
   * Retrieves and casts the {@link DeclaredToolchainInfo} from {@link
   * com.google.devtools.build.lib.rules.platform.Toolchain} rule.
   *
   * <p>Returns null if the rule isn't a toolchain.
   */
  @Nullable
  public static DeclaredToolchainInfo declaredToolchainInfo(@Nullable ProviderCollection target) {
    if (target == null) {
      return null;
    }
    return target.getProvider(DeclaredToolchainInfo.class);
  }

  /** Retrieves and casts the {@link ToolchainInfo} provider from the given target. */
  @Nullable
  public static ToolchainInfo toolchain(@Nullable ProviderCollection target) {
    if (target == null) {
      return null;
    }
    return target.get(ToolchainInfo.PROVIDER);
  }

  /** Retrieves and casts the {@link ToolchainTypeInfo} provider from the given target. */
  public static ToolchainTypeInfo toolchainType(ProviderCollection target) {
    return target.get(ToolchainTypeInfo.PROVIDER);
  }
}
