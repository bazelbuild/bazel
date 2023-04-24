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

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.base.MoreObjects;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.util.HashCodes;
import com.google.devtools.build.skyframe.SkyValue;

/** The artifacts behind a runfiles middleman. */
final class RunfilesArtifactValue implements SkyValue {

  @FunctionalInterface
  interface RunfilesConsumer<T> {
    void accept(Artifact artifact, T metadata) throws InterruptedException;
  }

  private final FileArtifactValue metadata;

  // Parallel lists.
  private final ImmutableList<Artifact> files;
  private final ImmutableList<FileArtifactValue> fileValues;

  // Parallel lists.
  private final ImmutableList<Artifact> trees;
  private final ImmutableList<TreeArtifactValue> treeValues;

  RunfilesArtifactValue(
      FileArtifactValue metadata,
      ImmutableList<Artifact> files,
      ImmutableList<FileArtifactValue> fileValues,
      ImmutableList<Artifact> trees,
      ImmutableList<TreeArtifactValue> treeValues) {
    this.metadata = checkNotNull(metadata);
    this.files = checkNotNull(files);
    this.fileValues = checkNotNull(fileValues);
    this.trees = checkNotNull(trees);
    this.treeValues = checkNotNull(treeValues);
    checkArgument(
        files.size() == fileValues.size() && trees.size() == treeValues.size(),
        "Size mismatch: %s",
        this);
  }

  /** Returns the data of the artifact for this value, as computed by the action cache checker. */
  FileArtifactValue getMetadata() {
    return metadata;
  }

  /** Visits the file artifacts that this runfiles artifact expands to, together with their data. */
  void forEachFile(RunfilesConsumer<FileArtifactValue> consumer) throws InterruptedException {
    for (int i = 0; i < files.size(); i++) {
      consumer.accept(files.get(i), fileValues.get(i));
    }
  }

  /** Visits the tree artifacts that this runfiles artifact expands to, together with their data. */
  void forEachTree(RunfilesConsumer<TreeArtifactValue> consumer) throws InterruptedException {
    for (int i = 0; i < trees.size(); i++) {
      consumer.accept(trees.get(i), treeValues.get(i));
    }
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (!(o instanceof RunfilesArtifactValue)) {
      return false;
    }
    RunfilesArtifactValue that = (RunfilesArtifactValue) o;
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
