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

import com.google.common.collect.ImmutableMap;
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

  // Standard mapping between OS and the corresponding platform constraints.
  public static final ImmutableMap<OS, ConstraintValueInfo> OS_TO_CONSTRAINTS =
      ImmutableMap.<OS, ConstraintValueInfo>builder()
          .put(
              OS.DARWIN,
              ConstraintValueInfo.create(
                  OS_CONSTRAINT_SETTING,
                  Label.parseCanonicalUnchecked("@platforms//os:osx")))
          .put(
              OS.WINDOWS,
              ConstraintValueInfo.create(
                  OS_CONSTRAINT_SETTING,
                  Label.parseCanonicalUnchecked("@platforms//os:windows")))
          .put(
              OS.FREEBSD,
              ConstraintValueInfo.create(
                  OS_CONSTRAINT_SETTING,
                  Label.parseCanonicalUnchecked("@platforms//os:freebsd")))
          .put(
              OS.OPENBSD,
              ConstraintValueInfo.create(
                  OS_CONSTRAINT_SETTING,
                  Label.parseCanonicalUnchecked("@platforms//os:openbsd")))
          .put(
              OS.UNKNOWN,
              ConstraintValueInfo.create(
                  OS_CONSTRAINT_SETTING,
                  Label.parseCanonicalUnchecked("@platforms//os:none")))
          .buildOrThrow();

  // No-op constructor to keep this from being instantiated.
  private ConstraintConstants() {}
}
