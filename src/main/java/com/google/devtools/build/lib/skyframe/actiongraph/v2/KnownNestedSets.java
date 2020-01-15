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
import com.google.devtools.build.lib.collect.nestedset.NestedSetView;
import java.io.IOException;

/** Cache for NestedSets in the action graph. */
public class KnownNestedSets extends BaseCache<Object, DepSetOfFiles> {
  private final KnownArtifacts knownArtifacts;

  KnownNestedSets(AqueryOutputHandler aqueryOutputHandler, KnownArtifacts knownArtifacts) {
    super(aqueryOutputHandler);
    this.knownArtifacts = knownArtifacts;
  }

  @Override
  protected Object transformToKey(Object nestedSetViewObject) {
    NestedSetView<?> nestedSetView = (NestedSetView<?>) nestedSetViewObject;
    // The NestedSet is identified by their raw 'children' object since multiple NestedSetViews
    // can point to the same object.
    return nestedSetView.identifier();
  }

  @Override
  DepSetOfFiles createProto(Object nestedSetViewObject, int id) throws IOException {
    NestedSetView<?> nestedSetView = (NestedSetView) nestedSetViewObject;
    DepSetOfFiles.Builder depSetBuilder = DepSetOfFiles.newBuilder().setId(id);
    for (NestedSetView<?> transitiveNestedSet : nestedSetView.transitives()) {
      depSetBuilder.addTransitiveDepSetIds(this.dataToIdAndStreamOutputProto(transitiveNestedSet));
    }
    for (Object directArtifact : nestedSetView.directs()) {
      depSetBuilder.addDirectArtifactIds(
          knownArtifacts.dataToIdAndStreamOutputProto((Artifact) directArtifact));
    }
    return depSetBuilder.build();
  }

  @Override
  void toOutput(DepSetOfFiles depSetOfFilesProto) throws IOException {
    aqueryOutputHandler.outputDepSetOfFiles(depSetOfFilesProto);
  }
}
