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

import com.google.common.collect.ImmutableBiMap;
import com.google.devtools.build.lib.analysis.platform.ConstraintCollection;
import com.google.devtools.build.lib.analysis.platform.ConstraintSettingInfo;
import com.google.devtools.build.lib.analysis.platform.ConstraintValueInfo;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.util.OS;

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
  public static final ImmutableBiMap<OS, ConstraintValueInfo> OS_TO_CONSTRAINTS =
      ImmutableBiMap.of(
          OS.LINUX,
          ConstraintValueInfo.create(
              OS_CONSTRAINT_SETTING,
              Label.parseCanonicalUnchecked("@platforms//os:linux")),
          OS.DARWIN,
          ConstraintValueInfo.create(
              OS_CONSTRAINT_SETTING,
              Label.parseCanonicalUnchecked("@platforms//os:osx")),
          OS.WINDOWS,
          ConstraintValueInfo.create(
              OS_CONSTRAINT_SETTING,
              Label.parseCanonicalUnchecked("@platforms//os:windows")),
          OS.FREEBSD,
          ConstraintValueInfo.create(
              OS_CONSTRAINT_SETTING,
              Label.parseCanonicalUnchecked("@platforms//os:freebsd")),
          OS.OPENBSD,
          ConstraintValueInfo.create(
              OS_CONSTRAINT_SETTING,
              Label.parseCanonicalUnchecked("@platforms//os:openbsd")),
          OS.UNKNOWN,
          ConstraintValueInfo.create(
              OS_CONSTRAINT_SETTING,
              Label.parseCanonicalUnchecked("@platforms//os:none")));

  /**
   * Returns the OS corresponding to the given constraint collection based on the contained platform
   * constraint.
   */
  public static OS getOsFromConstraints(ConstraintCollection constraintCollection) {
    if (!constraintCollection.has(OS_CONSTRAINT_SETTING)) {
      return OS.getCurrent();
    }
    return OS_TO_CONSTRAINTS
        .inverse()
        .getOrDefault(constraintCollection.get(OS_CONSTRAINT_SETTING), OS.getCurrent());
  }

  // No-op constructor to keep this from being instantiated.
  private ConstraintConstants() {}
}
