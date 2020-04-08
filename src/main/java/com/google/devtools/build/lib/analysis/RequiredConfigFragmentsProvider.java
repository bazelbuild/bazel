// Copyright 2019 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import java.util.List;

/**
 * Provides a user-friendly list of the
 * {@link com.google.devtools.build.lib.analysis.config.BuildConfiguration.Fragment}s and
 * {@link com.google.devtools.build.lib.analysis.config.FragmentOptions} required by this target
 * and its transitive dependencies.
 *
 * <p>See {@link ConfiguredTargetFactory#getRequiredConfigFragments) for details.
 */
@Immutable
public class RequiredConfigFragmentsProvider implements TransitiveInfoProvider {
  private final ImmutableSet<String> requiredConfigFragments;

  public RequiredConfigFragmentsProvider(ImmutableSet<String> requiredConfigFragments) {
    this.requiredConfigFragments = requiredConfigFragments;
  }

  public ImmutableSet<String> getRequiredConfigFragments() {
    return requiredConfigFragments;
  }

  /** Merges the values of multiple {@link RequiredConfigFragmentsProvider}s. */
  public static RequiredConfigFragmentsProvider merge(
      List<RequiredConfigFragmentsProvider> providers) {
    checkArgument(!providers.isEmpty());
    if (providers.size() == 1) {
      return providers.get(0);
    }
    ImmutableSet.Builder<String> merged = ImmutableSet.builder();
    for (RequiredConfigFragmentsProvider provider : providers) {
      merged.addAll(provider.getRequiredConfigFragments());
    }
    return new RequiredConfigFragmentsProvider(merged.build());
  }
}
