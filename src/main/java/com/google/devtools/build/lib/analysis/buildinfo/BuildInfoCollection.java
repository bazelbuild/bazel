// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis.buildinfo;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.Artifact;

import java.util.List;

/**
 * A collection of build-info files for both stamped and unstamped modes.
 */
public final class BuildInfoCollection {
  private final ImmutableList<ActionAnalysisMetadata> actions;
  private final ImmutableList<Artifact> stampedBuildInfo;
  private final ImmutableList<Artifact> redactedBuildInfo;

  public BuildInfoCollection(List<? extends ActionAnalysisMetadata> actions,
      List<Artifact> stampedBuildInfo, List<Artifact> redactedBuildInfo) {
    this.actions = ImmutableList.copyOf(actions);
    this.stampedBuildInfo = ImmutableList.copyOf(stampedBuildInfo);
    this.redactedBuildInfo = ImmutableList.copyOf(redactedBuildInfo);
  }

  public ImmutableList<ActionAnalysisMetadata> getActions() {
    return actions;
  }

  public ImmutableList<Artifact> getStampedBuildInfo() {
    return stampedBuildInfo;
  }

  public ImmutableList<Artifact> getRedactedBuildInfo() {
    return redactedBuildInfo;
  }
}
