// Copyright 2015 Google Inc. All rights reserved.
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

import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import java.io.IOException;
import java.util.Objects;

/**
 * The contents of decompressed archive.
 */
public class DecompressorValue implements SkyValue {

  private final Path directory;

  public DecompressorValue(Path repositoryPath) {
    directory = repositoryPath;
  }

  public Path getDirectory() {
    return directory;
  }

  @Override
  public boolean equals(Object other) {
    if (this == other) {
      return true;
    }

    if (!(other instanceof DecompressorValue)) {
      return false;
    }

    return directory.equals(((DecompressorValue) other).directory);
  }

  @Override
  public int hashCode() {
    return directory.hashCode();
  }

  public static SkyKey fileKey(
      String targetKind, String targetName, Path archivePath, Path repositoryPath) {
    return new SkyKey(
        FileFunction.NAME,
        new DecompressorDescriptor(targetKind, targetName, archivePath, repositoryPath));
  }

  public static SkyKey jarKey(
      String targetKind, String targetName, Path archivePath, Path repositoryPath) {
    return new SkyKey(JarFunction.NAME,
        new DecompressorDescriptor(targetKind, targetName, archivePath, repositoryPath));
  }

  public static SkyKey key(
      String targetKind, String targetName, Path archivePath, Path repositoryPath)
      throws IOException {
    String baseName = archivePath.getBaseName();

    DecompressorDescriptor descriptor =
        new DecompressorDescriptor(targetKind, targetName, archivePath, repositoryPath);

    if (baseName.endsWith(".zip") || baseName.endsWith(".jar") || baseName.endsWith(".war")) {
      return new SkyKey(ZipFunction.NAME, descriptor);
    } else if (baseName.endsWith(".tar.gz") || baseName.endsWith(".tgz")) {
      return new SkyKey(TarGzFunction.NAME, descriptor);
    } else {
      throw new IOException(
          String.format("Expected %s %s to create file with a .zip, .jar, .war, .tar.gz, or .tgz"
              + " suffix (got %s)", targetKind, targetName, archivePath));
    }
  }

  /**
   * Description of an archive to be decompressed for use in a SkyKey.
   * TODO(bazel-team): this should be an autovalue class.
   */
  public static class DecompressorDescriptor {
    private final String targetKind;
    private final String targetName;
    private final Path archivePath;
    private final Path repositoryPath;

    public DecompressorDescriptor(String targetKind, String targetName, Path archivePath,
        Path repositoryPath) {
      this.targetKind = targetKind;
      this.targetName = targetName;
      this.archivePath = archivePath;
      this.repositoryPath = repositoryPath;
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

    @Override
    public boolean equals(Object other) {
      if (this == other) {
        return true;
      }

      if (!(other instanceof DecompressorDescriptor)) {
        return false;
      }

      DecompressorDescriptor descriptor = (DecompressorDescriptor) other;
      return targetKind.equals(descriptor.targetKind)
          && targetName.equals(descriptor.targetName)
          && archivePath.equals(descriptor.archivePath)
          && repositoryPath.equals(descriptor.repositoryPath);
    }

    @Override
    public int hashCode() {
      return Objects.hash(targetKind, targetName, archivePath, repositoryPath);
    }
  }

  /**
   * Exceptions thrown when something goes wrong decompressing an archive.
   */
  static class DecompressorException extends Exception {
    public DecompressorException(String message) {
      super(message);
    }
  }
}
