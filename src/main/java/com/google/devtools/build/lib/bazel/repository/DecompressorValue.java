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

import com.google.devtools.build.lib.rules.repository.RepositoryFunction.RepositoryFunctionException;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyValue;

/**
 * The contents of decompressed archive.
 */
public class DecompressorValue implements SkyValue {
  /** Implementation of a decompression algorithm. */
  public interface Decompressor {
    Path decompress(DecompressorDescriptor descriptor) throws RepositoryFunctionException;
  }

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

  static Decompressor getDecompressor(Path archivePath)
      throws RepositoryFunctionException {
    String baseName = archivePath.getBaseName();
    if (baseName.endsWith(".zip") || baseName.endsWith(".jar") || baseName.endsWith(".war")) {
      return ZipFunction.INSTANCE;
    } else if (baseName.endsWith(".tar.gz") || baseName.endsWith(".tgz")) {
      return TarGzFunction.INSTANCE;
    } else if (baseName.endsWith(".tar.xz")) {
      return TarXzFunction.INSTANCE;
    } else if (baseName.endsWith(".tar.bz2")) {
      return TarBz2Function.INSTANCE;
    } else {
      throw new RepositoryFunctionException(
          new EvalException(null, String.format(
              "Expected a file with a .zip, .jar, .war, .tar.gz, .tgz, .tar.xz, or .tar.bz2 "
              + "suffix (got %s)",
              archivePath)),
          Transience.PERSISTENT);
    }
  }

  public static Path decompress(DecompressorDescriptor descriptor)
      throws RepositoryFunctionException, InterruptedException {
    return descriptor.getDecompressor().decompress(descriptor);
  }
}
