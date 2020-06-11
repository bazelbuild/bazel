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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import javax.annotation.Nullable;

/**
 * Storage layer for data associated with outputs of an action.
 *
 * <p>Data stored in {@link #artifactData} and {@link #treeArtifactData} will be passed along to the
 * final {@link ActionExecutionValue}.
 *
 * <p>Tree file artifacts which should be stored in a {@link TreeArtifactValue} (according to {@link
 * Artifact#isChildOfDeclaredDirectory}) are temporarily cached in {@link #treeFileCache}, but it is
 * expected that the final {@link TreeArtifactValue} will eventually be added via {@link
 * #putTreeArtifactData}.
 *
 * <p>This implementation aggressively stores all data. Subclasses may override mutating methods to
 * avoid storing unnecessary data.
 */
@ThreadSafe
class OutputStore {

  private final ConcurrentMap<Artifact, FileArtifactValue> artifactData = new ConcurrentHashMap<>();

  private final ConcurrentMap<SpecialArtifact, TreeArtifactValue> treeArtifactData =
      new ConcurrentHashMap<>();

  // The keys in this map are all TreeFileArtifact, but the declared type is Artifact to make it
  // interchangeable with artifactData syntactically.
  private final ConcurrentMap<Artifact, FileArtifactValue> treeFileCache =
      new ConcurrentHashMap<>();

  private final Set<Artifact> injectedFiles = Sets.newConcurrentHashSet();

  @Nullable
  final FileArtifactValue getArtifactData(Artifact artifact) {
    return mapFor(artifact).get(artifact);
  }

  void putArtifactData(Artifact artifact, FileArtifactValue value) {
    mapFor(artifact).put(artifact, value);
  }

  final ImmutableMap<Artifact, FileArtifactValue> getAllArtifactData() {
    return ImmutableMap.copyOf(artifactData);
  }

  @Nullable
  final TreeArtifactValue getTreeArtifactData(Artifact artifact) {
    return treeArtifactData.get(artifact);
  }

  final void putTreeArtifactData(SpecialArtifact treeArtifact, TreeArtifactValue value) {
    Preconditions.checkArgument(treeArtifact.isTreeArtifact(), "%s is not a tree artifact");
    treeArtifactData.put(treeArtifact, value);
  }

  /**
   * Returns data for TreeArtifacts that was computed during execution. May contain copies of {@link
   * TreeArtifactValue#MISSING_TREE_ARTIFACT}.
   */
  final ImmutableMap<Artifact, TreeArtifactValue> getAllTreeArtifactData() {
    return ImmutableMap.copyOf(treeArtifactData);
  }

  final void injectOutputData(Artifact output, FileArtifactValue artifactValue) {
    injectedFiles.add(output);
    mapFor(output).put(output, artifactValue);
  }

  /** Returns a set that tracks which Artifacts have had metadata injected. */
  final Set<Artifact> injectedFiles() {
    return injectedFiles;
  }

  /** Clears all data in this store. */
  final void clear() {
    artifactData.clear();
    treeArtifactData.clear();
    treeFileCache.clear();
    injectedFiles.clear();
  }

  /**
   * Clears data about a specific artifact from this store.
   *
   * <p>If a tree artifact parent is given, it will be cleared from {@link #treeArtifactData} but
   * its children will remain in {@link #treeFileCache} if present. If a tree artifact child is
   * given, it will only be removed from {@link #treeFileCache}.
   */
  final void remove(Artifact artifact) {
    mapFor(artifact).remove(artifact);
    if (artifact.isTreeArtifact()) {
      treeArtifactData.remove(artifact);
    }
    injectedFiles.remove(artifact);
  }

  private Map<Artifact, FileArtifactValue> mapFor(Artifact artifact) {
    return artifact.isChildOfDeclaredDirectory() ? treeFileCache : artifactData;
  }
}
