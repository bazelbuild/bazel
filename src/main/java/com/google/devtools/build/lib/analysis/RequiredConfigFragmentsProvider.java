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

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;

/**
 * Provides a user-friendly list of the {@link BuildConfiguration.Fragment}s and
 * {@link FragmentOptions} required by this target and its transitive dependencies.
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
}
