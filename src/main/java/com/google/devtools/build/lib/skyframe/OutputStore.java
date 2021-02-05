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
package com.google.devtools.build.lib.skyframe;

import com.google.common.base.MoreObjects;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import javax.annotation.Nullable;

/**
 * Storage layer for data associated with outputs of an action.
 *
 * <p>Data stored in {@link #artifactData} and {@link #treeArtifactData} will be passed along to the
 * final {@link ActionExecutionValue}.
 */
@ThreadSafe
final class OutputStore {

  private final ConcurrentMap<Artifact, FileArtifactValue> artifactData = new ConcurrentHashMap<>();

  private final ConcurrentMap<SpecialArtifact, TreeArtifactValue> treeArtifactData =
      new ConcurrentHashMap<>();

  @Nullable
  FileArtifactValue getArtifactData(Artifact artifact) {
    return artifactData.get(artifact);
  }

  void putArtifactData(Artifact artifact, FileArtifactValue value) {
    Preconditions.checkArgument(
        !artifact.isTreeArtifact() && !artifact.isChildOfDeclaredDirectory(),
        "%s should be stored in a TreeArtifactValue",
        artifact);
    artifactData.put(artifact, value);
  }

  ImmutableMap<Artifact, FileArtifactValue> getAllArtifactData() {
    return ImmutableMap.copyOf(artifactData);
  }

  @Nullable
  TreeArtifactValue getTreeArtifactData(Artifact artifact) {
    return treeArtifactData.get(artifact);
  }

  void putTreeArtifactData(SpecialArtifact treeArtifact, TreeArtifactValue value) {
    Preconditions.checkArgument(treeArtifact.isTreeArtifact(), "%s is not a tree artifact");
    treeArtifactData.put(treeArtifact, value);
  }

  /**
   * Returns data for TreeArtifacts that was computed during execution. May contain copies of {@link
   * TreeArtifactValue#MISSING_TREE_ARTIFACT}.
   */
  ImmutableMap<Artifact, TreeArtifactValue> getAllTreeArtifactData() {
    return ImmutableMap.copyOf(treeArtifactData);
  }

  /** Clears all data in this store. */
  void clear() {
    artifactData.clear();
    treeArtifactData.clear();
  }

  /**
   * Clears data about a specific artifact from this store.
   *
   * <p>If a tree artifact parent is given, it will be cleared from {@link #treeArtifactData}. If a
   * tree artifact child is given, its enclosing tree artifact will not be removed.
   */
  void remove(Artifact artifact) {
    if (artifact.isTreeArtifact()) {
      treeArtifactData.remove(artifact);
    } else {
      artifactData.remove(artifact);
    }
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("artifactData", artifactData)
        .add("treeArtifactData", treeArtifactData)
        .toString();
  }
}
