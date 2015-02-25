// Copyright 2014 Google Inc. All rights reserved.
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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;

/**
 * A {@link TransitiveInfoProvider} that has baseline coverage artifacts.
 */
@Immutable
public final class BaselineCoverageArtifactsProvider implements TransitiveInfoProvider {
  private final ImmutableList<Artifact> baselineCoverageArtifacts;

  public BaselineCoverageArtifactsProvider(ImmutableList<Artifact> baselineCoverageArtifacts) {
    this.baselineCoverageArtifacts = baselineCoverageArtifacts;
  }

  /**
   * Returns a set of baseline coverage artifacts for a given set of configured targets.
   *
   * <p>These artifacts represent "empty" code coverage data for non-test libraries and binaries and
   * used to establish correct baseline when calculating code coverage ratios since they would cover
   * completely non-tested code as well.
   */
  public ImmutableList<Artifact> getBaselineCoverageArtifacts() {
    return baselineCoverageArtifacts;
  }
}
