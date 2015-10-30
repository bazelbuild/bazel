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
import com.google.devtools.build.lib.rules.repository.RepositoryFunction.RepositoryFunctionException;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import java.io.IOException;
import java.nio.charset.Charset;

import javax.annotation.Nullable;

/**
 * Creates a repository for a jar file.
 */
public class JarFunction implements SkyFunction {

  public static final SkyFunctionName NAME = SkyFunctionName.create("JAR_FUNCTION");

  /**
   * The .jar can be used compressed, so this just exposes it in a way Bazel can use.
   *
   * <p>It moves the jar from some-name/x/y/z/foo.jar to some-name/jar/foo.jar and creates a
   * BUILD file containing one entry: the .jar.</p>
   */
  @Override
  @Nullable
  public SkyValue compute(SkyKey skyKey, Environment env) throws RepositoryFunctionException {
    DecompressorDescriptor descriptor = (DecompressorDescriptor) skyKey.argument();
    // Example: archiveFile is external/some-name/foo.jar.
    String baseName = descriptor.archivePath().getBaseName();

    try {
      FileSystemUtils.createDirectoryAndParents(descriptor.repositoryPath());
      // external/some-name/WORKSPACE.
      Path workspaceFile = descriptor.repositoryPath().getRelative("WORKSPACE");
      FileSystemUtils.writeContent(workspaceFile, Charset.forName("UTF-8"), String.format(
          "# DO NOT EDIT: automatically generated WORKSPACE file for %s rule %s\n",
          descriptor.targetKind(), descriptor.targetName()));
      // external/some-name/jar.
      Path jarDirectory = descriptor.repositoryPath().getRelative(getPackageName());
      FileSystemUtils.createDirectoryAndParents(jarDirectory);
      // external/some-name/repository/jar/foo.jar is a symbolic link to the jar in
      // external/some-name.
      Path jarSymlink = jarDirectory.getRelative(baseName);
      if (!jarSymlink.exists()) {
        jarSymlink.createSymbolicLink(descriptor.archivePath());
      }
      // external/some-name/repository/jar/BUILD defines the //jar target.
      Path buildFile = jarDirectory.getRelative("BUILD");
      FileSystemUtils.writeLinesAs(
          buildFile,
          Charset.forName("UTF-8"),
          "# DO NOT EDIT: automatically generated BUILD file for "
              + descriptor.targetKind()
              + " rule "
              + descriptor.targetName(),
          createBuildFile(baseName));
      if (descriptor.executable()) {
        descriptor.archivePath().chmod(0755);
      }
    } catch (IOException e) {
      throw new RepositoryFunctionException(new IOException(
          "Error auto-creating jar repo structure: " + e.getMessage()), Transience.TRANSIENT);
    }
    return new DecompressorValue(descriptor.repositoryPath());
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
            ")");
  }

  @Override
  @Nullable
  public String extractTag(SkyKey skyKey) {
    return null;
  }

}
