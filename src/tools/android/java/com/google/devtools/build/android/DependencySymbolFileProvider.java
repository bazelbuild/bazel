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

import com.google.common.base.Preconditions;

import com.android.builder.dependency.SymbolFileProvider;

import java.io.File;
import java.nio.file.FileSystem;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.regex.Pattern;

/**
 * Represents the R.txt symbol file and AndroidManifest (provides Java package) of libraries.
 */
class DependencySymbolFileProvider implements SymbolFileProvider {

  private static final Pattern VALID_REGEX = Pattern.compile(".*:.*");

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
  public File getManifest() {
    return manifest;
  }

  private static DependencySymbolFileProvider valueOf(String text, FileSystem fileSystem) {
    if (!VALID_REGEX.matcher(text).find()) {
      throw new IllegalArgumentException(text + " is not in the format " + commandlineFormat(""));
    }
    String[] parts = text.split(File.pathSeparator);
    return new DependencySymbolFileProvider(getFile(parts[0], fileSystem),
        getFile(parts[1], fileSystem));
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
    return String.format("lib%s/R.txt:lib%s/AndroidManifest.xml", libNum, libNum);
  }

  @Override
  public String toString() {
    return String.format("%s, %s", symbolFile, manifest);
  }

}
