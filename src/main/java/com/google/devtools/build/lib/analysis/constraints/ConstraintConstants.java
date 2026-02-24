// Copyright 2020 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis.constraints;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.platform.ConstraintSettingInfo;
import com.google.devtools.build.lib.analysis.platform.ConstraintValueInfo;
import com.google.devtools.build.lib.analysis.platform.PlatformInfo;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.util.OS;
import java.util.Map;

/** Constants needed for use of the constraints system. */
public final class ConstraintConstants {

  public static final String ENVIRONMENT_RULE = "environment";

  private static final ConstraintSettingInfo OS_CONSTRAINT_SETTING =
      ConstraintSettingInfo.create(
          Label.parseCanonicalUnchecked("@platforms//os:os"));

  public static final ConstraintSettingInfo CPU_CONSTRAINT_SETTING =
      ConstraintSettingInfo.create(
          Label.parseCanonicalUnchecked("@platforms//cpu:cpu"));

  // Standard mapping between OS and the corresponding platform constraints.
  private static final ImmutableMap<ConstraintValueInfo, OS> CONSTRAINT_VALUE_TO_OS =
      ImmutableMap.of(
          ConstraintValueInfo.create(
              OS_CONSTRAINT_SETTING,
              Label.parseCanonicalUnchecked("@platforms//os:linux")),
          OS.LINUX,
          ConstraintValueInfo.create(
              OS_CONSTRAINT_SETTING,
              Label.parseCanonicalUnchecked("@platforms//os:osx")),
          OS.DARWIN,
          ConstraintValueInfo.create(
              OS_CONSTRAINT_SETTING,
              Label.parseCanonicalUnchecked("@platforms//os:macos")),
          OS.DARWIN,
          ConstraintValueInfo.create(
              OS_CONSTRAINT_SETTING,
              Label.parseCanonicalUnchecked("@platforms//os:windows")),
          OS.WINDOWS,
          ConstraintValueInfo.create(
              OS_CONSTRAINT_SETTING,
              Label.parseCanonicalUnchecked("@platforms//os:freebsd")),
          OS.FREEBSD,
          ConstraintValueInfo.create(
              OS_CONSTRAINT_SETTING,
              Label.parseCanonicalUnchecked("@platforms//os:openbsd")),
          OS.OPENBSD,
          ConstraintValueInfo.create(
              OS_CONSTRAINT_SETTING,
              Label.parseCanonicalUnchecked("@platforms//os:none")),
          OS.UNKNOWN);

  // Only used for testing, so we accept the ambiguity of multiple constraints representing the same
  // OS.
  @VisibleForTesting
  public static final ImmutableMap<OS, ConstraintValueInfo> OS_TO_DEFAULT_CONSTRAINT_VALUE =
      CONSTRAINT_VALUE_TO_OS.entrySet().stream()
          .collect(
              ImmutableMap.toImmutableMap(Map.Entry::getValue, Map.Entry::getKey, (a, b) -> a));

  /**
   * Returns the OS corresponding to the given platform's constraint collection based on the
   * contained platform constraint, falling back to the host platform if none is found.
   */
  public static OS getOsFromConstraintsOrHost(PlatformInfo platformInfo) {
    var osConstraintValue = platformInfo.constraints().get(OS_CONSTRAINT_SETTING);
    if (osConstraintValue == null) {
      // The platform doesn't specify any OS constraint, which makes it difficult to say how the
      // parts of Bazel that are OS-specific should behave. Purely for backwards compatibility and
      // to avoid unexpected breakages, we fall back to the host OS in this case.
      return OS.getCurrent();
    }
    // If the constraint value isn't known to Bazel, it is certainly distinct from all the values
    // Bazel specifically cares about (e.g. for Windows- or macOS-specific behavior). This is best
    // modeled by returning UNKNOWN, which is distinct from all the specific OS values in the enum.
    return CONSTRAINT_VALUE_TO_OS.getOrDefault(osConstraintValue, OS.UNKNOWN);
  }

  // No-op constructor to keep this from being instantiated.
  private ConstraintConstants() {}
}
