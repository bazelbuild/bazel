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

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;

import java.util.Map;

/**
 * {@code ConfiguredTarget}s implementing this interface can provide artifacts that <b>can</b> be
 * built when the target is mentioned on the command line (as opposed to being always built, like
 * {@link com.google.devtools.build.lib.analysis.FileProvider})
 *
 * <p>The artifacts are grouped into "output groups". Which output groups are built is controlled
 * by the {@code --output_groups} undocumented command line option, which in turn is added to the
 * command line at the discretion of the build command being run.
 */
@Immutable
public final class TopLevelArtifactProvider implements TransitiveInfoProvider {
  private final ImmutableMap<String, NestedSet<Artifact>> outputGroups;

  private TopLevelArtifactProvider(ImmutableMap<String, NestedSet<Artifact>> outputGroups) {
    this.outputGroups = outputGroups;
  }

  /**
   * Create a {@link com.google.devtools.build.lib.analysis.TopLevelArtifactProvider} with a single
   * output group.
   *
   * <p>The artifacts in the second argument will be built if the {@code --output_groups=KEY}
   * argument is present on the command line.
   */
  public static TopLevelArtifactProvider of(String key, NestedSet<Artifact> artifacts) {
    return new TopLevelArtifactProvider(ImmutableMap.of(key, artifacts));
  }

  /**
   * Create a {@link com.google.devtools.build.lib.analysis.TopLevelArtifactProvider} with multiple
   * output groups.
   *
   * <p>Each output group will be built if a corresponding {@code --output_groups=KEY} argument is
   * present on the command line.
   */
  public static TopLevelArtifactProvider of(Map<String, NestedSet<Artifact>> groups) {
    return new TopLevelArtifactProvider(ImmutableMap.copyOf(groups));
  }

  /** Return the artifacts in a particular output group. */
  public NestedSet<Artifact> getOutputGroup(String outputGroupName) {
    return outputGroups.get(outputGroupName);
  }
}
