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
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;

/**
 * ConfiguredTargets implementing this interface can provide command-specific
 * and unconditional extra artifacts to the build.
 */
@Immutable
public final class TopLevelArtifactProvider implements TransitiveInfoProvider {

  private final ImmutableList<String> commandsForExtraArtifacts;
  private final ImmutableList<Artifact> artifactsForCommand;
  private final ImmutableMap<String, NestedSet<Artifact>> outputGroups;

  public TopLevelArtifactProvider(ImmutableList<String> commandsForExtraArtifacts,
      ImmutableList<Artifact> artifactsForCommand) {
    this.commandsForExtraArtifacts = commandsForExtraArtifacts;
    this.artifactsForCommand = artifactsForCommand;
    this.outputGroups = ImmutableMap.<String, NestedSet<Artifact>>of();
  }

  public TopLevelArtifactProvider(String key, NestedSet<Artifact> artifactsToBuild) {
    this.commandsForExtraArtifacts = ImmutableList.of();
    this.artifactsForCommand = ImmutableList.of();
    this.outputGroups = ImmutableMap.<String, NestedSet<Artifact>>of(key, artifactsToBuild);
  }

  /** Returns the commands (in lowercase) that this provider should provide artifacts for. */
  public ImmutableList<String> getCommandsForExtraArtifacts() {
    return commandsForExtraArtifacts;
  }

  /** Returns the extra artifacts for the commands. */
  public ImmutableList<Artifact> getArtifactsForCommand() {
    return artifactsForCommand;
  }

  /** Returns artifacts that are to be built for every command. */
  public NestedSet<Artifact> getOutputGroup(String outputGroupName) {
    return outputGroups.get(outputGroupName);
  }
}
