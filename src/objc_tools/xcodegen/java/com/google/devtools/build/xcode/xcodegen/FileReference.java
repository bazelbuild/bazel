// Copyright 2014 Google Inc. All rights reserved.
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

package com.google.devtools.build.xcode.xcodegen;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Optional;
import com.google.common.base.Preconditions;
import com.google.devtools.build.xcode.util.Value;

import com.facebook.buck.apple.xcode.xcodeproj.PBXFileReference;
import com.facebook.buck.apple.xcode.xcodeproj.PBXReference.SourceTree;

import java.io.File;

/**
 * Contains data similar to {@link PBXFileReference}, but is actually a value type, with working
 * {@link #equals(Object)} and {@link #hashCode()} methods.
 * <p>
 * TODO(bazel-team): Consider just adding equals and hashCode methods to PBXFileReference. This may
 * not be as straight-forward as it sounds, since the Xcodeproj serialization logic and all related
 * classes are based on identity equality semantics.
 */
public class FileReference extends Value<FileReference> {
  private final String name;
  private final Optional<String> path;
  private final SourceTree sourceTree;
  private final Optional<String> explicitFileType;

  @VisibleForTesting
  FileReference(
      String name,
      Optional<String> path,
      SourceTree sourceTree,
      Optional<String> explicitFileType) {
    super(name, path, sourceTree, explicitFileType);
    this.name = name;
    this.path = path;
    this.sourceTree = sourceTree;
    this.explicitFileType = explicitFileType;
  }

  public String name() {
    return name;
  }

  public Optional<String> path() {
    return path;
  }

  public SourceTree sourceTree() {
    return sourceTree;
  }

  public Optional<String> explicitFileType() {
    return explicitFileType;
  }

  /**
   * Returns an instance whose name is the base name of the path.
   */
  public static FileReference of(String path, SourceTree sourceTree) {
    return new FileReference(
        new File(path).getName(),
        Optional.of(path),
        sourceTree,
        Optional.<String>absent());
  }

  /**
   * Returns an instance with a path and without an {@link #explicitFileType()}.
   */
  public static FileReference of(String name, String path, SourceTree sourceTree) {
    return new FileReference(
        name, Optional.of(path), sourceTree, Optional.<String>absent());
  }

  /**
   * Returns an instance equivalent to this one, but with {@link #explicitFileType()} set to the
   * given value. This instance should not already have a value set for {@link #explicitFileType()}.
   */
  public FileReference withExplicitFileType(String explicitFileType) {
    Preconditions.checkState(!explicitFileType().isPresent(),
        "should not already have explicitFileType: %s", this);
    return new FileReference(name(), path(), sourceTree(), Optional.of(explicitFileType));
  }
}
