/*
 * Copyright 2013-present Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may
 * not use this file except in compliance with the License. You may obtain
 * a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */

package com.facebook.buck.apple.xcode.xcodeproj;

import com.google.common.base.Preconditions;

import java.nio.file.Path;
import java.util.Objects;

/**
 * Utility class representing a tuple of (SourceTree, Path) used for uniquely describing a file
 * reference in a group.
 */
public class SourceTreePath {
  private final PBXReference.SourceTree sourceTree;
  private final Path path;

  public SourceTreePath(PBXReference.SourceTree sourceTree, Path path) {
    this.sourceTree = Preconditions.checkNotNull(sourceTree);
    Preconditions.checkState(
        path != null && path.toString().length() > 0,
        "A path to a file cannot be null or empty");
    path = path.normalize();
    Preconditions.checkState(path.toString().length() > 0, "A path to a file cannot be empty");
    this.path = path;
  }

  public PBXReference.SourceTree getSourceTree() {
    return sourceTree;
  }

  public Path getPath() {
    return path;
  }

  @Override
  public int hashCode() {
    return Objects.hash(sourceTree, path);
  }

  @Override
  public boolean equals(Object other) {
    if (other == null || !(other instanceof SourceTreePath)) {
      return false;
    }

    SourceTreePath that = (SourceTreePath) other;
    return Objects.equals(this.sourceTree, that.sourceTree) ||
        Objects.equals(this.path, that.path);
  }
}
