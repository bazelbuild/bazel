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
package com.google.devtools.build.lib.actions;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.base.MoreObjects;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.FileArtifactValue.ConstantMetadataValue;
import com.google.devtools.build.lib.skyframe.TreeArtifactValue;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.util.HashCodes;

/**
 * The artifacts behind a runfiles tree.
 *
 * <p>NB: since this class contains a nested set (through {@link RunfilesTree}), {@link
 * RunfilesTreeAction} needs to be special-cased in {@code
 * Actions.assignOwnersAndThrowIfConflictMaybeToleratingSharedActions}. The comment in that method
 * explains why.
 */
@AutoCodec
public final class RunfilesArtifactValue implements RichArtifactData {

  /** A callback for consuming artifacts in a runfiles tree. */
  @FunctionalInterface
  public interface RunfilesConsumer<T> {
    void accept(Artifact artifact, T metadata);
  }

  private final FileArtifactValue metadata;
  private final RunfilesTree runfilesTree;

  // Parallel lists.
  private final ImmutableList<Artifact> files;
  private final ImmutableList<FileArtifactValue> fileValues;

  // Parallel lists.
  private final ImmutableList<Artifact> trees;
  private final ImmutableList<TreeArtifactValue> treeValues;

  // Parallel lists
  private final ImmutableList<Artifact> filesets;
  private final ImmutableList<FilesetOutputTree> filesetValues;

  public RunfilesArtifactValue(
      RunfilesTree runfilesTree,
      ImmutableList<Artifact> files,
      ImmutableList<FileArtifactValue> fileValues,
      ImmutableList<Artifact> trees,
      ImmutableList<TreeArtifactValue> treeValues,
      ImmutableList<Artifact> filesets,
      ImmutableList<FilesetOutputTree> filesetValues) {
    this.runfilesTree = checkNotNull(runfilesTree);
    this.files = checkNotNull(files);
    this.fileValues = checkNotNull(fileValues);
    this.trees = checkNotNull(trees);
    this.treeValues = checkNotNull(treeValues);
    this.filesets = checkNotNull(filesets);
    this.filesetValues = checkNotNull(filesetValues);
    checkArgument(
        files.size() == fileValues.size()
            && trees.size() == treeValues.size()
            && filesets.size() == filesetValues.size(),
        "Size mismatch: %s",
        this);

    // Compute the digest of this runfiles tree by combining its layout and the digests of every
    // artifact it references.
    this.metadata = FileArtifactValue.createRunfilesProxy(computeDigest());
  }

  private byte[] computeDigest() {
    Fingerprint result = new Fingerprint();

    result.addInt(runfilesTree.getMapping().size());
    for (var entry : runfilesTree.getMapping().entrySet()) {
      result.addPath(entry.getKey());
      result.addBoolean(entry.getValue() != null);
      if (entry.getValue() != null) {
        result.addPath(entry.getValue().getExecPath());
      }
    }

    result.addInt(files.size());
    for (int i = 0; i < files.size(); i++) {
      FileArtifactValue value =
          files.get(i).isConstantMetadata() ? ConstantMetadataValue.INSTANCE : fileValues.get(i);
      value.addTo(result);
    }

    result.addInt(trees.size());
    for (int i = 0; i < trees.size(); i++) {
      result.addBytes(treeValues.get(i).getDigest());
    }

    for (int i = 0; i < filesets.size(); i++) {
      FilesetOutputTree fileset = filesetValues.get(i);
      fileset.addTo(result);
    }

    return result.digestAndReset();
  }

  public RunfilesArtifactValue withOverriddenRunfilesTree(RunfilesTree overrideTree) {
    return new RunfilesArtifactValue(
        overrideTree, files, fileValues, trees, treeValues, filesets, filesetValues);
  }

  /** Returns the data of the artifact for this value, as computed by the action cache checker. */
  public FileArtifactValue getMetadata() {
    return metadata;
  }

  /** Returns the runfiles tree this value represents. */
  public RunfilesTree getRunfilesTree() {
    return runfilesTree;
  }

  /**
   * Returns all artifacts in the runfiles tree this value represents. Tree artifacts and filesets
   * are included, but are not expanded.
   *
   * <p>This is similar to calling {@link RunfilesTree#getArtifacts} on the result of {@link
   * #getRunfilesTree}, except this method additionally includes manifest files.
   */
  public Iterable<Artifact> getAllArtifacts() {
    return Iterables.concat(files, trees, filesets);
  }

  /** Visits the file artifacts that this runfiles artifact expands to, together with their data. */
  public void forEachFile(RunfilesConsumer<FileArtifactValue> consumer) {
    for (int i = 0; i < files.size(); i++) {
      consumer.accept(files.get(i), fileValues.get(i));
    }
  }

  /** Visits the tree artifacts that this runfiles artifact expands to, together with their data. */
  public void forEachTree(RunfilesConsumer<TreeArtifactValue> consumer) {
    for (int i = 0; i < trees.size(); i++) {
      consumer.accept(trees.get(i), treeValues.get(i));
    }
  }

  /**
   * Visits the fileset artifacts that this runfiles artifact expands to, together with their data.
   */
  public void forEachFileset(RunfilesConsumer<FilesetOutputTree> consumer) {
    for (int i = 0; i < filesets.size(); i++) {
      consumer.accept(filesets.get(i), filesetValues.get(i));
    }
  }

  @Override
  public boolean equals(Object o) {
    // This method, seemingly erroneously, does not check whether the runfilesTree of the two
    // objects are equivalent. This is because it's unnecessary because the layout of the runfiles
    // tree is already factored into the equality decision in two ways:
    // - Through "metadata", which takes the layout into account (see computeDigest())
    // - Through the runfiles input manifest file, which is part of the runfiles tree, which
    //   contains the exact mapping and whose digest is in "fileValues"
    if (this == o) {
      return true;
    }
    if (!(o instanceof RunfilesArtifactValue that)) {
      return false;
    }
    return metadata.equals(that.metadata)
        && files.equals(that.files)
        && fileValues.equals(that.fileValues)
        && trees.equals(that.trees)
        && treeValues.equals(that.treeValues)
        && filesets.equals(that.filesets)
        && filesetValues.equals(that.filesetValues);
  }

  @Override
  public int hashCode() {
    return HashCodes.hashObjects(
        metadata, files, fileValues, trees, treeValues, filesets, filesetValues);
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("metadata", metadata)
        .add("files", files)
        .add("fileValues", fileValues)
        .add("trees", trees)
        .add("treeValues", treeValues)
        .add("filesets", filesets)
        .add("filesetValues", fileValues)
        .toString();
  }
}
