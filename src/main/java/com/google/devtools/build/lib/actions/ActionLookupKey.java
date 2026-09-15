// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.actions;

import com.google.devtools.build.lib.skyframe.config.BuildConfigurationKey;
import com.google.devtools.build.skyframe.CPUHeavySkyKey;
import com.google.devtools.build.skyframe.SkyKey;
import javax.annotation.Nullable;

/**
 * {@link SkyKey} for an "analysis object": either an {@link ActionLookupValue} or a {@link
 * com.google.devtools.build.lib.analysis.ConfiguredTargetValue}.
 *
 * <p>Whether a configured target creates actions cannot be inferred from its {@link
 * com.google.devtools.build.lib.cmdline.Label} without performing analysis, so this class is used
 * for both types. Only {@link ActionLookupValue} nodes are accessed during the execution phase.
 *
 * <p>All subclasses of {@link ActionLookupValue} "own" artifacts with {@link ArtifactOwner}s that
 * are subclasses of {@link ActionLookupKey}. This allows callers to easily find the value key,
 * while remaining agnostic to what action lookup values actually exist.
 */
public interface ActionLookupKey extends ArtifactOwner, CPUHeavySkyKey {
  /**
   * Returns the {@link BuildConfigurationKey} for the configuration associated with this key, or
   * {@code null} if this key has no associated configuration.
   */
  @Nullable
  BuildConfigurationKey getConfigurationKey();

  /**
   * Returns {@code true} if the actions <em>may</em> own shareable actions, as determined by {@link
   * ActionLookupData#valueIsShareable}.
   *
   * <p>Returns {@code false} for some non-standard keys such as the build info key and coverage
   * report key.
   *
   * <p>A return of {@code true} still requires checking {@link ActionLookupData#valueIsShareable}
   * to determine whether the individual action can be shared - notably, for a test target,
   * compilation actions are shareable, but test actions are not.
   */
  default boolean mayOwnShareableActions() {
    return getLabel() != null;
  }
}
