// Copyright 2015 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.repository;

import com.google.common.base.Optional;
import com.google.devtools.build.lib.vfs.Path;

import java.util.Objects;

import javax.annotation.Nullable;

/**
 * Description of an archive to be decompressed for use in a SkyKey.
 * TODO(bazel-team): this should be an autovalue class.
 */
public class DecompressorDescriptor {
  private final String targetKind;
  private final String targetName;
  private final Path archivePath;
  private final Path repositoryPath;
  private final Optional<String> prefix;
  private final boolean executable;

  private DecompressorDescriptor(
      String targetKind, String targetName, Path archivePath, Path repositoryPath,
      @Nullable String prefix, boolean executable) {
    this.targetKind = targetKind;
    this.targetName = targetName;
    this.archivePath = archivePath;
    this.repositoryPath = repositoryPath;
    this.prefix = Optional.fromNullable(prefix);
    this.executable = executable;
  }

  public String targetKind() {
    return targetKind;
  }

  public String targetName() {
    return targetName;
  }

  public Path archivePath() {
    return archivePath;
  }

  public Path repositoryPath() {
    return repositoryPath;
  }

  public Optional<String> prefix() {
    return prefix;
  }

  public boolean executable() {
    return executable;
  }

  @Override
  public boolean equals(Object other) {
    if (this == other) {
      return true;
    }

    if (!(other instanceof DecompressorDescriptor)) {
      return false;
    }

    DecompressorDescriptor descriptor = (DecompressorDescriptor) other;
    return Objects.equals(targetKind, descriptor.targetKind)
        && Objects.equals(targetName, descriptor.targetName)
        && Objects.equals(archivePath, descriptor.archivePath)
        && Objects.equals(repositoryPath, descriptor.repositoryPath)
        && Objects.equals(prefix, descriptor.prefix)
        && Objects.equals(executable, descriptor.executable);
  }

  @Override
  public int hashCode() {
    return Objects.hash(targetKind, targetName, archivePath, repositoryPath, prefix);
  }

  public static Builder builder() {
    return new Builder();
  }

  /**
   * Builder for describing the file to be decompressed.  The fields set will depend on the type
   * of file.
   */
  public static class Builder {
    private String targetKind;
    private String targetName;
    private Path archivePath;
    private Path repositoryPath;
    private String prefix;
    private boolean executable;

    private Builder() {
    }

    public DecompressorDescriptor build() {
      return new DecompressorDescriptor(
          targetKind, targetName, archivePath, repositoryPath, prefix, executable);
    }

    public Builder setTargetKind(String targetKind) {
      this.targetKind = targetKind;
      return this;
    }

    public Builder setTargetName(String targetName) {
      this.targetName = targetName;
      return this;
    }

    public Builder setArchivePath(Path archivePath) {
      this.archivePath = archivePath;
      return this;
    }

    public Builder setRepositoryPath(Path repositoryPath) {
      this.repositoryPath = repositoryPath;
      return this;
    }

    public Builder setPrefix(String prefix) {
      this.prefix = prefix;
      return this;
    }

    public Builder setExecutable(boolean executable) {
      this.executable = executable;
      return this;
    }
  }
}
