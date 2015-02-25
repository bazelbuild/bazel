// Copyright 2014 Google Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.analysis;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;

/**
 * Artifacts that should be built when a target is mentioned in the command line, but are neither in
 * the {@code filesToBuild} nor in the runfiles.
 *
 * <p>
 * Link actions, may not run a link for their transitive dependencies, so it does not force the
 * source files in the transitive closure to be built by default. However, users expect builds to
 * fail when there is an error in a dependent library, so we use this mechanism to force their
 * compilation.
 */
@Immutable
public final class AlwaysBuiltArtifactsProvider implements TransitiveInfoProvider {

  private final NestedSet<Artifact> artifactsToAlwaysBuild;

  public AlwaysBuiltArtifactsProvider(NestedSet<Artifact> artifactsToAlwaysBuild) {
    this.artifactsToAlwaysBuild = artifactsToAlwaysBuild;
  }

  /**
   * Returns the collection of artifacts to be built.
   */
  public NestedSet<Artifact> getArtifactsToAlwaysBuild() {
    return artifactsToAlwaysBuild;
  }
}
