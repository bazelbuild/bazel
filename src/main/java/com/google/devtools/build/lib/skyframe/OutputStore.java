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
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import javax.annotation.Nullable;

/**
 * Storage layer for data associated with outputs of an action.
 *
 * <p>Data is mainly stored in three maps, {@link #artifactData}, {@link #treeArtifactData} and
 * {@link #treeArtifactContents}, all of which are keyed on an {@link Artifact}. For each of these
 * maps, this class exposes standard methods such as {@code get}, {@code put}, {@code add}, and
 * {@code getAll}.
 *
 * <p>This implementation aggressively stores all data. Subclasses may override mutating methods to
 * avoid storing unnecessary data.
 */
@ThreadSafe
class OutputStore {

  private final ConcurrentMap<Artifact, FileArtifactValue> artifactData = new ConcurrentHashMap<>();

  private final ConcurrentMap<Artifact, TreeArtifactValue> treeArtifactData =
      new ConcurrentHashMap<>();

  private final ConcurrentMap<Artifact, Set<TreeFileArtifact>> treeArtifactContents =
      new ConcurrentHashMap<>();

  private final Set<Artifact> injectedFiles = Sets.newConcurrentHashSet();

  @Nullable
  final FileArtifactValue getArtifactData(Artifact artifact) {
    return artifactData.get(artifact);
  }

  void putArtifactData(Artifact artifact, FileArtifactValue value) {
    artifactData.put(artifact, value);
  }

  final ImmutableMap<Artifact, FileArtifactValue> getAllArtifactData() {
    return ImmutableMap.copyOf(artifactData);
  }

  @Nullable
  final TreeArtifactValue getTreeArtifactData(Artifact artifact) {
    return treeArtifactData.get(artifact);
  }

  void putTreeArtifactData(Artifact artifact, TreeArtifactValue value) {
    treeArtifactData.put(artifact, value);
  }

  /**
   * Returns data for TreeArtifacts that was computed during execution. May contain copies of {@link
   * TreeArtifactValue#MISSING_TREE_ARTIFACT}.
   */
  final ImmutableMap<Artifact, TreeArtifactValue> getAllTreeArtifactData() {
    return ImmutableMap.copyOf(treeArtifactData);
  }

  /**
   * Returns a set of the given tree artifact's contents.
   *
   * <p>If the return value is {@code null}, this means nothing was injected, and the output
   * TreeArtifact is to have its values read from disk instead.
   */
  @Nullable
  final Set<TreeFileArtifact> getTreeArtifactContents(Artifact artifact) {
    return treeArtifactContents.get(artifact);
  }

  void addTreeArtifactContents(Artifact artifact, TreeFileArtifact contents) {
    Preconditions.checkArgument(artifact.isTreeArtifact(), artifact);
    treeArtifactContents.computeIfAbsent(artifact, a -> Sets.newConcurrentHashSet()).add(contents);
  }

  void injectRemoteFile(
      Artifact output, byte[] digest, long size, int locationIndex, String actionId) {
    injectOutputData(
        output,
        new FileArtifactValue.RemoteFileArtifactValue(digest, size, locationIndex, actionId));
  }

  final void injectOutputData(Artifact output, FileArtifactValue artifactValue) {
    injectedFiles.add(output);
    artifactData.put(output, artifactValue);
  }

  /** Returns a set that tracks which Artifacts have had metadata injected. */
  final Set<Artifact> injectedFiles() {
    return injectedFiles;
  }

  /** Clears all data in this store. */
  final void clear() {
    artifactData.clear();
    treeArtifactData.clear();
    treeArtifactContents.clear();
    injectedFiles.clear();
  }

  /** Clears data about a specific Artifact from this store. */
  final void remove(Artifact artifact) {
    artifactData.remove(artifact);
    treeArtifactData.remove(artifact);
    treeArtifactContents.remove(artifact);
    injectedFiles.remove(artifact);
  }
}
