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
import com.google.devtools.build.lib.analysis.AnalysisProtosV2.DepSetOfFiles;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import java.io.IOException;

/** Cache for NestedSets in the action graph. */
public class KnownNestedSets extends BaseCache<Object, DepSetOfFiles> {
  private final KnownArtifacts knownArtifacts;

  KnownNestedSets(AqueryOutputHandler aqueryOutputHandler, KnownArtifacts knownArtifacts) {
    super(aqueryOutputHandler);
    this.knownArtifacts = knownArtifacts;
  }

  @Override
  protected Object transformToKey(Object nestedSet) {
    return ((NestedSet) nestedSet).toNode();
  }

  @Override
  DepSetOfFiles createProto(Object nestedSetObject, int id)
      throws IOException, InterruptedException {
    NestedSet<?> nestedSet = (NestedSet) nestedSetObject;
    DepSetOfFiles.Builder depSetBuilder = DepSetOfFiles.newBuilder().setId(id);
    for (NestedSet<?> succ : nestedSet.getNonLeaves()) {
      depSetBuilder.addTransitiveDepSetIds(this.dataToIdAndStreamOutputProto(succ));
    }
    for (Object elem : nestedSet.getLeaves()) {
      depSetBuilder.addDirectArtifactIds(
          knownArtifacts.dataToIdAndStreamOutputProto((Artifact) elem));
    }
    return depSetBuilder.build();
  }

  @Override
  void toOutput(DepSetOfFiles depSetOfFilesProto) throws IOException {
    aqueryOutputHandler.outputDepSetOfFiles(depSetOfFilesProto);
  }
}
