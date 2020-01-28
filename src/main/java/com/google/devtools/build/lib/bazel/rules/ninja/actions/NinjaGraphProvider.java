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
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.vfs.PathFragment;
import javax.annotation.Nullable;

/**
 * Provider to pass the information between the parsing of Ninja graph in ninja_graph
 * ({@link NinjaGraphRule}) and the targets that will create the Bazel actions (ninja_build).
 */
public class NinjaGraphProvider implements TransitiveInfoProvider {
  private final String outputRoot;
  private final String workingDirectory;
  private final ImmutableSortedMap<PathFragment, NestedSet<PathFragment>> phonyTargets;

  NinjaGraphProvider(String outputRoot,
      String workingDirectory,
      ImmutableSortedMap<PathFragment, NestedSet<PathFragment>> phonyTargets) {
    this.outputRoot = outputRoot;
    this.workingDirectory = workingDirectory;
    this.phonyTargets = phonyTargets;
  }

  public String getOutputRoot() {
    return outputRoot;
  }

  public String getWorkingDirectory() {
    return workingDirectory;
  }

  @Nullable
  public NestedSet<PathFragment> getPhonyArtifacts(PathFragment fragment) {
    return phonyTargets.get(fragment);
  }
}
