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

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import java.io.IOException;

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

  public static SkyKey key(SkyFunctionName skyFunctionName, DecompressorDescriptor descriptor) {
    Preconditions.checkNotNull(descriptor.archivePath());
    return new SkyKey(skyFunctionName, descriptor);
  }

  public static SkyKey key(DecompressorDescriptor descriptor) throws IOException {
    Preconditions.checkNotNull(descriptor.archivePath());
    return key(getSkyFunctionName(descriptor.archivePath()), descriptor);
  }

  private static SkyFunctionName getSkyFunctionName(Path archivePath) throws IOException {
    String baseName = archivePath.getBaseName();
    if (baseName.endsWith(".zip") || baseName.endsWith(".jar") || baseName.endsWith(".war")) {
      return ZipFunction.NAME;
    } else if (baseName.endsWith(".tar.gz") || baseName.endsWith(".tgz")) {
      return TarGzFunction.NAME;
    } else {
      throw new IOException(String.format(
          "Expected a file with a .zip, .jar, .war, .tar.gz, or .tgz suffix (got %s)",
          archivePath));
    }
  }

}
