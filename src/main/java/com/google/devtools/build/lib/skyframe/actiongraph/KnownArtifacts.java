// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe.actiongraph;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.AnalysisProtos;
import com.google.devtools.build.lib.analysis.AnalysisProtos.ActionGraphContainer;

/**
 * Cache for Artifacts in the action graph.
 */
public class KnownArtifacts extends BaseCache<Artifact, AnalysisProtos.Artifact> {

  KnownArtifacts(ActionGraphContainer.Builder actionGraphBuilder) {
    super(actionGraphBuilder);
  }

  @Override
  AnalysisProtos.Artifact createProto(Artifact artifact, String id) {
    return AnalysisProtos.Artifact.newBuilder()
            .setId(id)
            .setExecPath(artifact.getExecPathString())
            .setIsTreeArtifact(artifact.isTreeArtifact())
            .build();
  }

  @Override
  void addToActionGraphBuilder(AnalysisProtos.Artifact artifactProto) {
    actionGraphBuilder.addArtifacts(artifactProto);
  }
}
