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
package com.google.devtools.build.android;

import com.android.builder.dependency.SymbolFileProvider;
import com.google.common.base.Preconditions;
import java.io.File;
import java.nio.file.FileSystem;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Represents the R.txt symbol file and AndroidManifest (provides Java package) of libraries.
 */
class DependencySymbolFileProvider implements SymbolFileProvider {

  private final File symbolFile;
  private final File manifest;

  public DependencySymbolFileProvider(File symbolFile, File manifest) {
    this.symbolFile = symbolFile;
    this.manifest = manifest;
  }

  public static DependencySymbolFileProvider valueOf(String text) {
    return valueOf(text, FileSystems.getDefault());
  }

  @Override
  public File getSymbolFile() {
    return symbolFile;
  }

  @Override
  public boolean isOptional() {
    return false;
  }

  @Override
  public File getManifest() {
    return manifest;
  }

  private static DependencySymbolFileProvider valueOf(String text, FileSystem fileSystem) {
    int separatorIndex = text.indexOf(',');
    if (separatorIndex == -1) {
      // TODO(laszlocsomor): remove support for ":" after 2018-02-28 (about 6 months from now).
      // Everyone should have updated to newer Bazel versions by then.
      // ":" is supported for the sake of the deprecated --libraries flag whose format is
      // --libraries=key1:value1,key2:value2,keyN:valueN . The new flag is --library with
      // multi-value support and expected format: --library=key1,value1 --library=key2,value2 .
      separatorIndex = text.indexOf(':');
    }
    if (separatorIndex == -1) {
      throw new IllegalArgumentException(text + " is not in the format " + commandlineFormat(""));
    }
    return new DependencySymbolFileProvider(
        getFile(text.substring(0, separatorIndex), fileSystem),
        getFile(text.substring(separatorIndex + 1), fileSystem));
  }

  private static File getFile(String pathString, FileSystem fileSystem) {
    Preconditions.checkArgument(!pathString.trim().isEmpty());
    return exists(fileSystem.getPath(pathString)).toFile();
  }

  private static Path exists(Path path) {
    if (!Files.exists(path)) {
      throw new IllegalArgumentException(path + " does not exist");
    }
    return path;
  }

  public static String commandlineFormat(String libNum) {
    return String.format("lib%s/R.txt,lib%s/AndroidManifest.xml", libNum, libNum);
  }

  @Override
  public String toString() {
    return String.format("%s, %s", symbolFile, manifest);
  }

}
