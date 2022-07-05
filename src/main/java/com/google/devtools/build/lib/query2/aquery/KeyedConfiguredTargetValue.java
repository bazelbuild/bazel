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
package com.google.devtools.build.lib.query2.aquery;

import com.google.auto.value.AutoValue;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.ConfiguredTargetValue;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;

/**
 * An object wrapping a ConfiguredTargetValue and its ConfiguredTargetKey.
 *
 * <p>WARNING: Strictly only for aquery. Do not use this elsewhere.
 *
 * <p>In an actual build, it's too expensive to establish this link, but we need it for correctness
 * in aquery.
 */
@AutoValue
public abstract class KeyedConfiguredTargetValue {
  public abstract ConfiguredTargetValue getConfiguredTargetValue();

  public abstract ConfiguredTargetKey getConfiguredTargetKey();

  public ConfiguredTarget getConfiguredTarget() {
    return getConfiguredTargetValue().getConfiguredTarget();
  }

  public static KeyedConfiguredTargetValue create(
      ConfiguredTargetValue configuredTargetValue, ConfiguredTargetKey configuredTargetKey) {
    return new AutoValue_KeyedConfiguredTargetValue(configuredTargetValue, configuredTargetKey);
  }
}
