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
import com.google.devtools.build.lib.skyframe.TreeArtifactValue;
import com.google.devtools.build.lib.util.HashCodes;
import com.google.devtools.build.skyframe.SkyValue;

/** The artifacts behind a runfiles middleman. */
public final class RunfilesArtifactValue implements SkyValue {

  /** A callback for consuming artifacts in a runfiles tree. */
  @FunctionalInterface
  public interface RunfilesConsumer<T> {
    void accept(Artifact artifact, T metadata) throws InterruptedException;
  }

  private final FileArtifactValue metadata;
  private final RunfilesTree runfilesTree;

  // Parallel lists.
  private final ImmutableList<Artifact> files;
  private final ImmutableList<FileArtifactValue> fileValues;

  // Parallel lists.
  private final ImmutableList<Artifact> trees;
  private final ImmutableList<TreeArtifactValue> treeValues;

  public RunfilesArtifactValue(
      FileArtifactValue metadata,
      RunfilesTree runfilesTree,
      ImmutableList<Artifact> files,
      ImmutableList<FileArtifactValue> fileValues,
      ImmutableList<Artifact> trees,
      ImmutableList<TreeArtifactValue> treeValues) {
    this.metadata = checkNotNull(metadata);
    this.runfilesTree = checkNotNull(runfilesTree);
    this.files = checkNotNull(files);
    this.fileValues = checkNotNull(fileValues);
    this.trees = checkNotNull(trees);
    this.treeValues = checkNotNull(treeValues);
    checkArgument(
        files.size() == fileValues.size() && trees.size() == treeValues.size(),
        "Size mismatch: %s",
        this);
  }

  public RunfilesArtifactValue withOverriddenRunfilesTree(RunfilesTree overrideTree) {
    return new RunfilesArtifactValue(metadata, overrideTree, files, fileValues, trees, treeValues);
  }

  /** Returns the data of the artifact for this value, as computed by the action cache checker. */
  public FileArtifactValue getMetadata() {
    return metadata;
  }

  /** Returns the runfiles tree this value represents. */
  public RunfilesTree getRunfilesTree() {
    return runfilesTree;
  }

  /** Visits the file artifacts that this runfiles artifact expands to, together with their data. */
  public void forEachFile(RunfilesConsumer<FileArtifactValue> consumer)
      throws InterruptedException {
    for (int i = 0; i < files.size(); i++) {
      consumer.accept(files.get(i), fileValues.get(i));
    }
  }

  /** Visits the tree artifacts that this runfiles artifact expands to, together with their data. */
  public void forEachTree(RunfilesConsumer<TreeArtifactValue> consumer)
      throws InterruptedException {
    for (int i = 0; i < trees.size(); i++) {
      consumer.accept(trees.get(i), treeValues.get(i));
    }
  }

  @Override
  public boolean equals(Object o) {
    // This method, seemingly erroneously, does not check whether the runfilesTree of the two
    // objects is equivalent. This is because it's costly (it involves flattening nested sets and
    // even if one caches a fingerprint, it's still a fair amount of CPU) and because it's
    // currently not necessary: RunfilesArtifactValue is only ever created as the SkyValue of
    // runfiles middlemen and those are special-cased in ActionCacheChecker (see
    // ActionCacheChecker.checkMiddlemanAction()): the checksum of a middleman artifact is the
    // function of the checksum of all the artifacts on the inputs of the middleman action, which
    // includes both the artifacts the runfiles tree links to and the runfiles input manifest,
    // which in turn encodes the structure of the runfiles tree. The checksum of the middleman
    // artifact is here as the "metadata" field, which *is* compared here, so the
    // RunfilesArtifactValues of two runfiles middlemen will be equals iff they represent the same
    // runfiles tree.
    //
    // Eventually, if we ever do away with runfiles input manifests, it will be necessary to change
    // this (it's weird that one needs to do a round-trip to the file system to determine the
    // checksum of a runfiles tree), but that's not happening anytime soon.
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
        && treeValues.equals(that.treeValues);
  }

  @Override
  public int hashCode() {
    return HashCodes.hashObjects(metadata, files, fileValues, trees, treeValues);
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("metadata", metadata)
        .add("files", files)
        .add("fileValues", fileValues)
        .add("trees", trees)
        .add("treeValues", treeValues)
        .toString();
  }
}
