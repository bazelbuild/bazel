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
package com.google.devtools.build.lib.skyframe.actiongraph.v2;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.AnalysisProtosV2;
import java.io.IOException;

/** Cache for Artifacts in the action graph. */
public class KnownArtifacts extends BaseCache<Artifact, AnalysisProtosV2.Artifact> {

  private final KnownPathFragments knownPathFragments;

  KnownArtifacts(AqueryOutputHandler aqueryOutputHandler) {
    super(aqueryOutputHandler);
    knownPathFragments = new KnownPathFragments(aqueryOutputHandler);
  }

  @Override
  AnalysisProtosV2.Artifact createProto(Artifact artifact, int id)
      throws IOException, InterruptedException {
    AnalysisProtosV2.Artifact.Builder artifactProtoBuilder =
        AnalysisProtosV2.Artifact.newBuilder()
            .setId(id)
            .setIsTreeArtifact(artifact.isTreeArtifact());

    int pathFragmentId = knownPathFragments.dataToIdAndStreamOutputProto(artifact.getExecPath());
    return artifactProtoBuilder.setPathFragmentId(pathFragmentId).build();
  }

  @Override
  void toOutput(AnalysisProtosV2.Artifact artifactProto) throws IOException {
    aqueryOutputHandler.outputArtifact(artifactProto);
  }
}
