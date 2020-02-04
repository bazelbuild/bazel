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

package com.google.devtools.build.lib.bazel.rules.ninja.actions;

import com.google.common.collect.ImmutableSortedMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.vfs.PathFragment;
import javax.annotation.Nullable;

/**
 * Provider to pass the information between ninja_graph ({@link NinjaGraphRule}) and ninja_build
 * targets.
 */
@Immutable
public final class NinjaGraphProvider implements TransitiveInfoProvider {
  private final PathFragment outputRoot;
  private final PathFragment workingDirectory;
  private final ImmutableSortedMap<PathFragment, NestedSet<Artifact>> phonyTargets;

  /**
   * Constructor
   *
   * @param outputRoot name of output directory for Ninja actions under execroot
   * @param workingDirectory relative path under execroot, the root for interpreting all paths
   * @param phonyTargets mapping between phony target names and their input files
   */
  NinjaGraphProvider(
      PathFragment outputRoot,
      PathFragment workingDirectory,
      ImmutableSortedMap<PathFragment, NestedSet<Artifact>> phonyTargets) {
    this.outputRoot = outputRoot;
    this.workingDirectory = workingDirectory;
    this.phonyTargets = phonyTargets;
  }

  public PathFragment getOutputRoot() {
    return outputRoot;
  }

  public PathFragment getWorkingDirectory() {
    return workingDirectory;
  }

  @Nullable
  public NestedSet<Artifact> getPhonyArtifacts(PathFragment fragment) {
    return phonyTargets.get(fragment);
  }
}
