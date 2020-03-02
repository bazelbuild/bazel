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

import com.google.common.collect.ImmutableSortedMap;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.bazel.rules.ninja.parser.NinjaTarget;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.vfs.PathFragment;

/**
 * Provider for passing information between {@link NinjaGraphRule} and {@link NinjaBuildRule}.
 * Represents all usual and phony {@link NinjaTarget}s from the Ninja graph.
 */
@Immutable
public final class NinjaGraphProvider implements TransitiveInfoProvider {
  private final PathFragment outputRoot;
  private final PathFragment workingDirectory;
  private final ImmutableSortedMap<PathFragment, NinjaTarget> usualTargets;
  private final ImmutableSortedMap<PathFragment, PhonyTarget> phonyTargetsMap;

  public NinjaGraphProvider(
      PathFragment outputRoot,
      PathFragment workingDirectory,
      ImmutableSortedMap<PathFragment, NinjaTarget> usualTargets,
      ImmutableSortedMap<PathFragment, PhonyTarget> phonyTargetsMap) {
    this.outputRoot = outputRoot;
    this.workingDirectory = workingDirectory;
    this.usualTargets = usualTargets;
    this.phonyTargetsMap = phonyTargetsMap;
  }

  public PathFragment getOutputRoot() {
    return outputRoot;
  }

  public PathFragment getWorkingDirectory() {
    return workingDirectory;
  }

  public ImmutableSortedMap<PathFragment, NinjaTarget> getUsualTargets() {
    return usualTargets;
  }

  public ImmutableSortedMap<PathFragment, PhonyTarget> getPhonyTargetsMap() {
    return phonyTargetsMap;
  }
}
