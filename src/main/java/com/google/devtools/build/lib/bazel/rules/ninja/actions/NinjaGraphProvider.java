// Copyright 2020 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedMap;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.bazel.rules.ninja.parser.NinjaTarget;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.vfs.PathFragment;

/**
 * Provider for passing information between {@link NinjaGraphRule} and {@link NinjaBuildRule}.
 * Represents all regular and phony {@link NinjaTarget}s from the Ninja graph.
 */
@Immutable
public final class NinjaGraphProvider implements TransitiveInfoProvider {
  private final PathFragment outputRoot;
  private final PathFragment workingDirectory;
  private final ImmutableSortedMap<PathFragment, NinjaTarget> targetsMap;
  private final ImmutableSortedMap<PathFragment, PhonyTarget> phonyTargetsMap;
  private final ImmutableSet<PathFragment> outputRootSymlinks;
  private final ImmutableSet<PathFragment> outputRootInputsSymlinks;

  public NinjaGraphProvider(
      PathFragment outputRoot,
      PathFragment workingDirectory,
      ImmutableSortedMap<PathFragment, NinjaTarget> targetsMap,
      ImmutableSortedMap<PathFragment, PhonyTarget> phonyTargetsMap,
      ImmutableSet<PathFragment> outputRootSymlinks,
      ImmutableSet<PathFragment> outputRootInputsSymlinks) {

    this.outputRoot = outputRoot;
    this.workingDirectory = workingDirectory;
    this.targetsMap = targetsMap;
    this.phonyTargetsMap = phonyTargetsMap;
    this.outputRootSymlinks = outputRootSymlinks;
    this.outputRootInputsSymlinks = outputRootInputsSymlinks;
  }

  public PathFragment getOutputRoot() {
    return outputRoot;
  }

  public PathFragment getWorkingDirectory() {
    return workingDirectory;
  }

  public ImmutableSortedMap<PathFragment, NinjaTarget> getTargetsMap() {
    return targetsMap;
  }

  public ImmutableSortedMap<PathFragment, PhonyTarget> getPhonyTargetsMap() {
    return phonyTargetsMap;
  }

  /** Output paths under output_root, that should be treated as symlink artifacts. */
  public ImmutableSet<PathFragment> getOutputRootSymlinks() {
    return outputRootSymlinks;
  }

  public ImmutableSet<PathFragment> getOutputRootInputsSymlinks() {
    return outputRootInputsSymlinks;
  }
}
