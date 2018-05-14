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
import com.google.devtools.build.lib.collect.nestedset.NestedSetView;

/**
 * Cache for NestedSets in the action graph.
 */
public class KnownNestedSets extends BaseCache<Object, AnalysisProtos.DepSetOfFiles> {
  private final KnownArtifacts knownArtifacts;

  KnownNestedSets(ActionGraphContainer.Builder actionGraphBuilder, KnownArtifacts knownArtifacts) {
    super(actionGraphBuilder);
    this.knownArtifacts = knownArtifacts;
  }

  @Override
  protected Object transformToKey(Object nestedSetViewObject) {
    NestedSetView<Artifact> nestedSetView = (NestedSetView<Artifact>) nestedSetViewObject;
    // The NestedSet is identified by their raw 'children' object since multiple NestedSetViews
    // can point to the same object.
    return nestedSetView.identifier();
  }

  @Override
  AnalysisProtos.DepSetOfFiles createProto(Object nestedSetViewObject, String id) {
    NestedSetView<Artifact> nestedSetView = (NestedSetView<Artifact>) nestedSetViewObject;
    AnalysisProtos.DepSetOfFiles.Builder depSetBuilder = AnalysisProtos.DepSetOfFiles
        .newBuilder()
        .setId(id);
    for (NestedSetView<Artifact> transitiveNestedSet : nestedSetView.transitives()) {
      depSetBuilder.addTransitiveDepSetIds(this.dataToId(transitiveNestedSet));
    }
    for (Artifact directArtifact : nestedSetView.directs()) {
      depSetBuilder.addDirectArtifactIds(knownArtifacts.dataToId(directArtifact));
    }
    return depSetBuilder.build();
  }

  @Override
  void addToActionGraphBuilder(AnalysisProtos.DepSetOfFiles depSetOfFilesProto) {
    actionGraphBuilder.addDepSetOfFiles(depSetOfFilesProto);
  }
}
