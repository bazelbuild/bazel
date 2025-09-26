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

package com.google.devtools.build.lib.analysis;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.collect.ImmutableSet;
import javax.annotation.Nullable;

/**
 * Provides the transitive visibility groups that a target belongs to. If a target belongs to a
 * transitive visibility group, it may only be depended on by other targets that also belong to the
 * same group.
 */
public class TransitiveVisibilityProvider implements TransitiveInfoProvider {
  @Nullable private final ImmutableSet<PackageSpecificationProvider> transitiveVisibility;

  /**
   * Creates a new {@link TransitiveVisibilityProvider} from a set of transitive visibility labels.
   */
  public TransitiveVisibilityProvider(
      ImmutableSet<PackageSpecificationProvider> transitiveVisibility) {
    // We should only try to create a provider if there is a non-empty transitive visibility.
    checkNotNull(transitiveVisibility);
    checkArgument(!transitiveVisibility.isEmpty());

    this.transitiveVisibility = transitiveVisibility;
  }

  /** Returns the set of transitive visibility groups for the target. */
  @Nullable
  ImmutableSet<PackageSpecificationProvider> getTransitiveVisibility() {
    return transitiveVisibility;
  }
}
