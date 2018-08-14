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
package com.google.devtools.build.lib.skyframe;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.ActionInputMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Collection;
import java.util.Map;

class ActionInputMapHelper {

  // Adds a value obtained by an Artifact skyvalue lookup to the action input map
  static void addToMap(
      ActionInputMap inputMap,
      Map<Artifact, Collection<Artifact>> expandedArtifacts,
      Artifact key,
      SkyValue value) {
    if (value instanceof AggregatingArtifactValue) {
      AggregatingArtifactValue aggregatingValue = (AggregatingArtifactValue) value;
      for (Pair<Artifact, FileArtifactValue> entry : aggregatingValue.getFileArtifacts()) {
        inputMap.put(entry.first, entry.second);
      }
      for (Pair<Artifact, TreeArtifactValue> entry : aggregatingValue.getTreeArtifacts()) {
        expandTreeArtifactAndPopulateArtifactData(
            entry.getFirst(),
            Preconditions.checkNotNull(entry.getSecond()),
            expandedArtifacts,
            inputMap);
      }
      // We have to cache the "digest" of the aggregating value itself,
      // because the action cache checker may want it.
      inputMap.put(key, aggregatingValue.getSelfData());
      // While not obvious at all this code exists to ensure that we don't expand the
      // .runfiles/MANIFEST file into the inputs. The reason for that being that the MANIFEST
      // file contains absolute paths that don't work with remote execution.
      // Instead, the way the SpawnInputExpander expands runfiles is via the Runfiles class
      // which contains all artifacts in the runfiles tree minus the MANIFEST file.
      // TODO(buchgr): Clean this up and get rid of the RunfilesArtifactValue type.
      if (!(value instanceof RunfilesArtifactValue)) {
        ImmutableList.Builder<Artifact> expansionBuilder = ImmutableList.builder();
        for (Pair<Artifact, FileArtifactValue> pair : aggregatingValue.getFileArtifacts()) {
          expansionBuilder.add(Preconditions.checkNotNull(pair.getFirst()));
        }
        expandedArtifacts.put(key, expansionBuilder.build());
      }
    } else if (value instanceof TreeArtifactValue) {
      expandTreeArtifactAndPopulateArtifactData(
          key, (TreeArtifactValue) value, expandedArtifacts, inputMap);

    } else {
      Preconditions.checkState(value instanceof FileArtifactValue);
      inputMap.put(key, (FileArtifactValue) value);
    }
  }

  private static void expandTreeArtifactAndPopulateArtifactData(
      Artifact treeArtifact,
      TreeArtifactValue value,
      Map<Artifact, Collection<Artifact>> expandedArtifacts,
      ActionInputMap inputMap) {
    ImmutableSet.Builder<Artifact> children = ImmutableSet.builder();
    for (Map.Entry<Artifact.TreeFileArtifact, FileArtifactValue> child :
        value.getChildValues().entrySet()) {
      children.add(child.getKey());
      inputMap.put(child.getKey(), child.getValue());
    }
    expandedArtifacts.put(treeArtifact, children.build());
    // Again, we cache the "digest" of the value for cache checking.
    inputMap.put(treeArtifact, value.getSelfData());
  }
}
