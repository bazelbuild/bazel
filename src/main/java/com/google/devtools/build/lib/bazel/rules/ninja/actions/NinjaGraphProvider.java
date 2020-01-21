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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSortedMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.DerivedArtifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.bazel.rules.ninja.parser.NinjaScope;
import com.google.devtools.build.lib.bazel.rules.ninja.parser.NinjaTarget;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.List;

/**
 * Provider to pass the information between the parsing of Ninja graph in ninja_graph
 * ({@link NinjaGraphRule}) and the targets that will create the Bazel actions (ninja_build).
 */
public class NinjaGraphProvider implements TransitiveInfoProvider {
  private final String outputRoot;
  private final String workingDirectory;
  private final NinjaScope scope;
  private final ImmutableSortedMap<PathFragment, NinjaTarget> targets;
  private final ImmutableList<Artifact> symlinkedUnderOutputRoot;

  NinjaGraphProvider(String outputRoot,
      String workingDirectory,
      NinjaScope scope,
      ImmutableSortedMap<PathFragment, NinjaTarget> targets,
      ImmutableList<Artifact> symlinkedUnderOutputRoot) {
    this.outputRoot = outputRoot;
    this.workingDirectory = workingDirectory;
    this.scope = scope;
    this.targets = targets;
    this.symlinkedUnderOutputRoot = symlinkedUnderOutputRoot;
  }

  public String getOutputRoot() {
    return outputRoot;
  }

  public String getWorkingDirectory() {
    return workingDirectory;
  }

  public NinjaScope getScope() {
    return scope;
  }

  public ImmutableSortedMap<PathFragment, NinjaTarget> getTargets() {
    return targets;
  }

  public ImmutableList<Artifact> getSymlinkedUnderOutputRoot() {
    return symlinkedUnderOutputRoot;
  }
}
