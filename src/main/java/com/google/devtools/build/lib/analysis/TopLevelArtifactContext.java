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

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;

import java.util.Objects;
import java.util.Set;

/**
 * Contains options which control the set of artifacts to build for top-level targets.
 */
@Immutable
public final class TopLevelArtifactContext {

  public static final TopLevelArtifactContext DEFAULT = new TopLevelArtifactContext(
      /*buildDefaultArtifacts=*/true, /*runTestsExclusively=*/false,
      /*outputGroups=*/ImmutableSet.<String>of());

  private final boolean buildDefaultArtifacts;
  private final boolean runTestsExclusively;
  private final ImmutableSet<String> outputGroups;

  public TopLevelArtifactContext(boolean filesToRun,
      boolean runTestsExclusively, ImmutableSet<String> outputGroups) {
    this.buildDefaultArtifacts = filesToRun;
    this.runTestsExclusively = runTestsExclusively;
    this.outputGroups = outputGroups;
  }

  /** Returns the value of the (undocumented) --build_default_artifacts flag. */
  public boolean buildDefaultArtifacts() {
    return buildDefaultArtifacts;
  }

  /** Whether to run tests in exclusive mode. */
  public boolean runTestsExclusively() {
    return runTestsExclusively;
  }

  /** Returns the value of the --output_groups flag. */
  public Set<String> outputGroups() {
    return outputGroups;
  }

  // TopLevelArtifactContexts are stored in maps in BuildView,
  // so equals() and hashCode() need to work.
  @Override
  public boolean equals(Object other) {
    if (other instanceof TopLevelArtifactContext) {
      TopLevelArtifactContext otherContext = (TopLevelArtifactContext) other;
      return buildDefaultArtifacts == otherContext.buildDefaultArtifacts
          && runTestsExclusively == otherContext.runTestsExclusively
          && outputGroups.equals(otherContext.outputGroups);
    } else {
      return false;
    }
  }

  @Override
  public int hashCode() {
    return Objects.hash(buildDefaultArtifacts, runTestsExclusively, outputGroups);
  }
}
