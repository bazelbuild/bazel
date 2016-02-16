// Copyright 2016 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;

/** An ArtifactFile implementation for descendants of TreeArtifacts. */
final class TreeArtifactFile implements ArtifactFile {
  private final Artifact parent;
  private final PathFragment parentRelativePath;

  TreeArtifactFile(Artifact parent, PathFragment parentRelativePath) {
    Preconditions.checkArgument(parent.isTreeArtifact(), "%s must be a TreeArtifact", parent);
    this.parent = parent;
    this.parentRelativePath = parentRelativePath;
  }

  @Override
  public PathFragment getExecPath() {
    return parent.getExecPath().getRelative(parentRelativePath);
  }

  @Override
  public PathFragment getParentRelativePath() {
    return parentRelativePath;
  }

  @Override
  public Path getPath() {
    return parent.getPath().getRelative(parentRelativePath);
  }

  @Override
  public PathFragment getRootRelativePath() {
    return parent.getRootRelativePath().getRelative(parentRelativePath);
  }

  @Override
  public Root getRoot() {
    return parent.getRoot();
  }

  @Override
  public Artifact getParent() {
    return parent;
  }

  @Override
  public String prettyPrint() {
    return getRootRelativePath().toString();
  }

  @Override
  public String getExecPathString() {
    return getExecPath().toString();
  }

  @Override
  public String toString() {
    return "ArtifactFile:[" + parent.toDetailString() + "]" + parentRelativePath;
  }

  @Override
  public boolean equals(Object other) {
    if (!(other instanceof ArtifactFile)) {
      return false;
    }

    ArtifactFile that = (ArtifactFile) other;
    return this.getParent().equals(that.getParent())
        && this.getParentRelativePath().equals(that.getParentRelativePath());
  }

  @Override
  public int hashCode() {
    return getParent().hashCode() * 257 + getParentRelativePath().hashCode();
  }
}
