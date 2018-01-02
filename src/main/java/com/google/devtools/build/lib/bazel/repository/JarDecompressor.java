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

import com.google.common.base.Joiner;
import com.google.common.base.Optional;
import com.google.devtools.build.lib.bazel.repository.DecompressorValue.Decompressor;
import com.google.devtools.build.lib.rules.repository.RepositoryFunction;
import com.google.devtools.build.lib.rules.repository.RepositoryFunction.RepositoryFunctionException;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import java.io.IOException;
import java.nio.charset.Charset;
import javax.annotation.Nullable;

/**
 * Creates a repository for a jar file.
 */
public class JarDecompressor implements Decompressor {
  public static final Decompressor INSTANCE = new JarDecompressor();

  protected JarDecompressor() {
  }

  @Override
  @Nullable
  public Path decompress(DecompressorDescriptor descriptor) throws RepositoryFunctionException {
    return decompressWithSrcjar(descriptor, Optional.absent());
  }

  /**
   * The jar (and srcjar if available) can be used compressed, so this just exposes them in a way
   * Bazel can use.
   *
   * <p>It moves the jar(s) from some-name/x/y/z/foo.jar to some-name/jar/foo.jar and creates a
   * BUILD.bazel file containing only the jar(s).
   */
  public Path decompressWithSrcjar(
      DecompressorDescriptor descriptor, Optional<DecompressorDescriptor> srcjarDescriptor)
      throws RepositoryFunctionException {
    try {
      // external/some-name/
      FileSystemUtils.createDirectoryAndParents(descriptor.repositoryPath());
      // external/some-name/WORKSPACE
      RepositoryFunction.createWorkspaceFile(
          descriptor.repositoryPath(), descriptor.targetKind(), descriptor.targetName());
      // external/some-name/jar/
      Path jarDirectory = descriptor.repositoryPath().getRelative(getPackageName());
      FileSystemUtils.createDirectoryAndParents(jarDirectory);
      // external/some-name/jar/BUILD.bazel defines the //jar target.
      Path buildFile = jarDirectory.getRelative("BUILD.bazel");

      // Example: get foo.jar from external/some-name/foo.jar.
      String baseName = descriptor.archivePath().getBaseName();
      makeSymlink(descriptor);
      String buildFileContents;
      if (srcjarDescriptor.isPresent()) {
        String srcjarBaseName = srcjarDescriptor.get().archivePath().getBaseName();
        makeSymlink(srcjarDescriptor.get());
        buildFileContents = createBuildFileWithSrcjar(baseName, srcjarBaseName);
      } else {
        buildFileContents = createBuildFile(baseName);
      }

      FileSystemUtils.writeLinesAs(
          buildFile,
          Charset.forName("UTF-8"),
          "# DO NOT EDIT: automatically generated BUILD.bazel file for "
              + descriptor.targetKind()
              + " rule "
              + descriptor.targetName(),
          buildFileContents);
      if (descriptor.executable()) {
        descriptor.archivePath().chmod(0755);
      }
    } catch (IOException e) {
      throw new RepositoryFunctionException(new IOException(
          "Error auto-creating jar repo structure: " + e.getMessage()), Transience.TRANSIENT);
    }
    return descriptor.repositoryPath();
  }

  /*
   * Links some-name/x/y/z/foo.jar to some-name/jar/foo.jar (represented by {@code descriptor}
   */
  protected void makeSymlink(DecompressorDescriptor descriptor) throws RepositoryFunctionException {
    try {
      Path jarDirectory = descriptor.repositoryPath().getRelative(getPackageName());
      // external/some-name/jar/foo.jar is a symbolic link to the jar in
      // external/some-name.
      Path jarSymlink = jarDirectory.getRelative(descriptor.archivePath().getBaseName());
      if (!jarSymlink.exists()) {
        jarSymlink.createSymbolicLink(descriptor.archivePath());
      }
    } catch (IOException e) {
      throw new RepositoryFunctionException(
          new IOException("Error auto-creating jar repo structure: " + e.getMessage()),
          Transience.TRANSIENT);
    }
  }

  protected String getPackageName() {
    return "jar";
  }

  protected String createBuildFile(String baseName) {
    return Joiner.on("\n")
        .join(
            "java_import(",
            "    name = 'jar',",
            "    jars = ['" + baseName + "'],",
            "    visibility = ['//visibility:public']",
            ")",
            "",
            "filegroup(",
            "    name = 'file',",
            "    srcs = ['" + baseName + "'],",
            "    visibility = ['//visibility:public']",
            ")");
  }

  protected String createBuildFileWithSrcjar(String baseName, String srcjarBaseName) {
    return Joiner.on("\n")
        .skipNulls()
        .join(
            "java_import(",
            "    name = 'jar',",
            "    jars = ['" + baseName + "'],",
            "    srcjar = '" + srcjarBaseName + "',",
            "    visibility = ['//visibility:public']",
            ")",
            "",
            "filegroup(",
            "    name = 'file',",
            "    srcs = [",
            "        '" + baseName + "',",
            baseName.equals(srcjarBaseName) ? null : "        '" + srcjarBaseName + "',",
            "    ],",
            "    visibility = ['//visibility:public']",
            ")");
  }
}
