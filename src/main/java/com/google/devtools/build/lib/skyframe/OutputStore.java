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
import com.google.devtools.build.lib.actions.ArtifactFileMetadata;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import javax.annotation.Nullable;

/**
 * Storage layer for data associated with outputs of an action.
 *
 * <p>Data is mainly stored in four maps, {@link #artifactData}, {@link #treeArtifactData}, {@link
 * #additionalOutputData}, and {@link #treeArtifactContents}, all of which are keyed on an {@link
 * Artifact}. For each of these maps, this class exposes standard methods such as {@code get},
 * {@code put}, {@code add}, and {@code getAll}.
 *
 * <p>This implementation aggresively stores all data. Subclasses may override mutating methods to
 * avoid storing unnecessary data.
 */
@ThreadSafe
class OutputStore {

  private final ConcurrentMap<Artifact, ArtifactFileMetadata> artifactData =
      new ConcurrentHashMap<>();

  private final ConcurrentMap<Artifact, TreeArtifactValue> treeArtifactData =
      new ConcurrentHashMap<>();

  private final ConcurrentMap<Artifact, FileArtifactValue> additionalOutputData =
      new ConcurrentHashMap<>();

  private final ConcurrentMap<Artifact, Set<TreeFileArtifact>> treeArtifactContents =
      new ConcurrentHashMap<>();

  private final Set<Artifact> injectedFiles = Sets.newConcurrentHashSet();

  @Nullable
  final ArtifactFileMetadata getArtifactData(Artifact artifact) {
    return artifactData.get(artifact);
  }

  void putArtifactData(Artifact artifact, ArtifactFileMetadata value) {
    artifactData.put(artifact, value);
  }

  /**
   * Returns data for output files that was computed during execution.
   *
   * <p>The value is {@link ArtifactFileMetadata#PLACEHOLDER} if the artifact's metadata is not
   * fully captured in {@link #additionalOutputData}.
   */
  final ImmutableMap<Artifact, ArtifactFileMetadata> getAllArtifactData() {
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

  @Nullable
  final FileArtifactValue getAdditionalOutputData(Artifact artifact) {
    return additionalOutputData.get(artifact);
  }

  void putAdditionalOutputData(Artifact artifact, FileArtifactValue value) {
    additionalOutputData.put(artifact, value);
  }

  /**
   * Returns data for any output files whose metadata was not computable from the corresponding
   * entry in {@link #artifactData}.
   *
   * <p>There are two bits to consider: the filesystem possessing fast digests and the execution
   * service injecting metadata via {@link ActionMetadataHandler#injectRemoteFile} or {@link
   * ActionMetadataHandler#injectDigest}.
   *
   * <ol>
   *   <li>If the filesystem does not possess fast digests, then we will have additional output data
   *       for practically every artifact, since we will need to store their digests.
   *   <li>If we have a remote execution service injecting metadata, then we will just store that
   *       metadata here, and put {@link ArtifactFileMetadata#PLACEHOLDER} objects into {@link
   *       #outputArtifactData} if the filesystem supports fast digests, and the actual metadata if
   *       the filesystem does not support fast digests.
   *   <li>If the filesystem has fast digests <i>but</i> there is no remote execution injecting
   *       metadata, then we will not store additional metadata here.
   * </ol>
   *
   * <p>Note that this means that in the vastly common cases (Google-internal, where we have fast
   * digests and remote execution, and Bazel, where there is often neither), this map is always
   * populated. Locally executed actions are the exception to this rule inside Google.
   *
   * <p>Moreover, there are some artifacts that are always stored here. First, middleman artifacts
   * have no corresponding {@link ArtifactFileMetadata}. Second, directories' metadata contain their
   * mtimes, which the {@link ArtifactFileMetadata} does not expose, so that has to be stored
   * separately.
   *
   * <p>Note that for files that need digests, we can't easily inject the digest in the {@link
   * ArtifactFileMetadata} because it would complicate equality-checking on subsequent builds -- if
   * our filesystem doesn't do fast digests, the comparison value would not have a digest.
   */
  final ImmutableMap<Artifact, FileArtifactValue> getAllAdditionalOutputData() {
    return ImmutableMap.copyOf(additionalOutputData);
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

  void injectRemoteFile(Artifact output, byte[] digest, long size, int locationIndex) {
    // TODO(shahan): there are a couple of things that could reduce memory usage
    // 1. We might be able to skip creating an entry in `outputArtifactData` and only create
    // the `FileArtifactValue`, but there are likely downstream consumers that expect it that
    // would need to be cleaned up.
    // 2. Instead of creating an `additionalOutputData` entry, we could add the extra
    // `locationIndex` to `FileStateValue`.
    injectOutputData(
        output, new FileArtifactValue.RemoteFileArtifactValue(digest, size, locationIndex));
  }

  final void injectOutputData(Artifact output, FileArtifactValue artifactValue) {
    injectedFiles.add(output);

    // While `artifactValue` carries the important information, consumers also require an entry in
    // `artifactData` so a `PLACEHOLDER` is added to `artifactData`.
    artifactData.put(output, ArtifactFileMetadata.PLACEHOLDER);
    additionalOutputData.put(output, artifactValue);
  }

  /** Returns a set that tracks which Artifacts have had metadata injected. */
  final Set<Artifact> injectedFiles() {
    return injectedFiles;
  }

  /** Clears all data in this store. */
  final void clear() {
    artifactData.clear();
    treeArtifactData.clear();
    additionalOutputData.clear();
    treeArtifactContents.clear();
    injectedFiles.clear();
  }

  /** Clears data about a specific Artifact from this store. */
  final void remove(Artifact artifact) {
    artifactData.remove(artifact);
    treeArtifactData.remove(artifact);
    additionalOutputData.remove(artifact);
    treeArtifactContents.remove(artifact);
    injectedFiles.remove(artifact);
  }
}
