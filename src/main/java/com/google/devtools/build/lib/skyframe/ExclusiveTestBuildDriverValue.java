// Copyright 2022 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.skyframe.SkyValue;

/** A subclass of BuildDriverValue, meant to represent an exclusive test. */
public class ExclusiveTestBuildDriverValue extends BuildDriverValue {

  private final ConfiguredTarget exclusiveTestConfiguredTarget;

  ExclusiveTestBuildDriverValue(
      SkyValue wrappedSkyValue, ConfiguredTarget exclusiveTestConfiguredTarget) {
    // If an exclusive test was run at all, it wasn't skipped.
    super(wrappedSkyValue, /*skipped=*/ false);
    this.exclusiveTestConfiguredTarget = exclusiveTestConfiguredTarget;
  }

  public ConfiguredTarget getExclusiveTestConfiguredTarget() {
    return exclusiveTestConfiguredTarget;
  }
}
