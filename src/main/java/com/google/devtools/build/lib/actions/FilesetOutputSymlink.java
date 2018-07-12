// Copyright 2014 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package com.google.devtools.build.lib.actions;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Objects;
import javax.annotation.Nullable;

/** Definition of a symlink in the output tree of a Fileset rule. */
public abstract class FilesetOutputSymlink {
  private static final int STRIPPED_METADATA = -1;
  private final PathFragment name;
  private final PathFragment target;
  private final int metadataHash;

  FilesetOutputSymlink(PathFragment name, PathFragment target, int metadataHash) {
    this.name = Preconditions.checkNotNull(name);
    this.target = Preconditions.checkNotNull(target);
    this.metadataHash = metadataHash;
  }

  /** Final name of the symlink relative to the Fileset's output directory. */
  public PathFragment getName() {
    return name;
  }

  /**
   * Target of the symlink. This may be relative to the target's location if the target itself is a
   * relative symlink. We can override it by using FilesetEntry.symlinks = 'dereference'.
   */
  public PathFragment getTargetPath() {
    return target;
  }

  /** The hashcode of the target's file state. */
  public int getMetadataHash() {
    return metadataHash;
  }

  /** true if the target is a generated artifact */
  public abstract boolean isGeneratedTarget();

  /**
   * returns the target artifact if it's generated {@link #isGeneratedTarget() == true}, or null
   * otherwise.
   */
  @Nullable
  public abstract FileArtifactValue getTargetArtifactValue();

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (obj == null || !obj.getClass().equals(getClass())) {
      return false;
    }
    FilesetOutputSymlink o = (FilesetOutputSymlink) obj;
    return getName().equals(o.getName())
        && getTargetPath().equals(o.getTargetPath())
        && getMetadataHash() == o.getMetadataHash()
        && isGeneratedTarget() == o.isGeneratedTarget()
        && Objects.equals(getTargetArtifactValue(), o.getTargetArtifactValue());
  }

  @Override
  public int hashCode() {
    return Objects.hash(
        getName(),
        getTargetPath(),
        getMetadataHash(),
        isGeneratedTarget(),
        getTargetArtifactValue());
  }

  @Override
  public String toString() {
    if (getMetadataHash() == STRIPPED_METADATA) {
      return String.format(
          "FilesetOutputSymlink(%s -> %s)",
          getName().getPathString(), getTargetPath().getPathString());
    } else {
      return String.format(
          "FilesetOutputSymlink(%s -> %s | metadataHash=%s)",
          getName().getPathString(), getTargetPath().getPathString(), getMetadataHash());
    }
  }

  @VisibleForTesting
  public static FilesetOutputSymlink createForTesting(PathFragment name, PathFragment target) {
    return new SourceOutputSymlink(name, target, STRIPPED_METADATA);
  }

  /**
   * @param name relative path under the Fileset's output directory, including FilesetEntry.destdir
   *     with and FilesetEntry.strip_prefix applied (if applicable)
   * @param target relative or absolute value of the link
   * @param metadataHash hashcode of the target's file state
   */
  public static FilesetOutputSymlink createForSourceTarget(
      PathFragment name, PathFragment target, int metadataHash) {
    return new SourceOutputSymlink(name, target, metadataHash);
  }

  /**
   * @param name relative path under the Fileset's output directory, including FilesetEntry.destdir
   *     with and FilesetEntry.strip_prefix applied (if applicable)
   * @param target relative or absolute value of the link
   * @param metadataHash hashcode of the target's file state.
   * @param fileArtifactValue the {@link FileArtifactValue} corresponding to the target.
   */
  public static FilesetOutputSymlink createForDerivedTarget(
      PathFragment name,
      PathFragment target,
      int metadataHash,
      FileArtifactValue fileArtifactValue) {
    return new DerivedOutputSymlink(name, target, metadataHash, fileArtifactValue);
  }

  private static class DerivedOutputSymlink extends FilesetOutputSymlink {
    private final FileArtifactValue fileArtifactValue;

    DerivedOutputSymlink(
        PathFragment name,
        PathFragment target,
        int metadataHash,
        FileArtifactValue fileArtifactValue) {
      super(name, target, metadataHash);
      this.fileArtifactValue = fileArtifactValue;
    }

    @Override
    public boolean isGeneratedTarget() {
      return true;
    }

    @Override
    public FileArtifactValue getTargetArtifactValue() {
      return fileArtifactValue;
    }
  }

  private static class SourceOutputSymlink extends FilesetOutputSymlink {
    SourceOutputSymlink(PathFragment name, PathFragment target, int metadataHash) {
      super(name, target, metadataHash);
    }

    @Override
    public boolean isGeneratedTarget() {
      return false;
    }

    @Override
    public FileArtifactValue getTargetArtifactValue() {
      return null;
    }
  }
}
