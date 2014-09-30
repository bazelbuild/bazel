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
package com.google.devtools.build.lib.actions;

import com.google.common.collect.ImmutableSet;

import java.util.Set;

/**
 * Used to signal when the incremental builder has found the set of changed artifacts.
 */
public class ChangedArtifactsMessage {

  private final Set<Artifact> artifacts;

  public ChangedArtifactsMessage(Set<Artifact> changedArtifacts) {
    this.artifacts = ImmutableSet.copyOf(changedArtifacts);
  }

  public Set<Artifact> getChangedArtifacts() {
    return artifacts;
  }
}
