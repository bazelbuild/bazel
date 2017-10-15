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
import com.google.common.collect.ImmutableList;
import java.io.IOException;
import java.nio.file.FileSystem;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Objects;
import java.util.regex.Pattern;

/**
 * Android resource and assets that have been parsed ahead of time, and summarized by an {@link
 * AndroidDataSerializer}. This class drives reloading the data.
 */
public class SerializedAndroidData {

  private static final Pattern VALID_REGEX = Pattern.compile(".*;.*;.+;.*");

  public static final String EXPECTED_FORMAT =
      "resources[#resources];assets[#assets];label;symbols.bin";

  public static SerializedAndroidData valueOf(String text) {
    return valueOf(text, FileSystems.getDefault());
  }

  static SerializedAndroidData valueOf(String text, FileSystem fileSystem) {
    if (!VALID_REGEX.matcher(text).find()) {
      throw new IllegalArgumentException(text + " is not in the format '" + EXPECTED_FORMAT + "'");
    }
    String[] parts = text.split(";");
    return new SerializedAndroidData(
        splitPaths(parts[0], fileSystem),
        splitPaths(parts[1], fileSystem),
        parts[2],
        parts.length > 3 ? exists(fileSystem.getPath(parts[3])) : null);
  }

  protected static ImmutableList<Path> splitPaths(String pathsString, FileSystem fileSystem) {
    if (pathsString.trim().isEmpty()) {
      return ImmutableList.of();
    }
    ImmutableList.Builder<Path> paths = new ImmutableList.Builder<>();
    for (String pathString : pathsString.split("#")) {
      Preconditions.checkArgument(!pathString.trim().isEmpty());
      paths.add(exists(fileSystem.getPath(pathString)));
    }
    return paths.build();
  }

  protected static Path exists(Path path) {
    if (!Files.exists(path)) {
      throw new IllegalArgumentException(path + " does not exist");
    }
    return path;
  }

  protected final ImmutableList<Path> assetDirs;
  protected final ImmutableList<Path> resourceDirs;
  protected final String label;
  protected final Path symbols;

  public SerializedAndroidData(
      ImmutableList<Path> resourceDirs, ImmutableList<Path> assetDirs, String label, Path symbols) {
    this.resourceDirs = resourceDirs;
    this.assetDirs = assetDirs;
    this.label = label;
    this.symbols = symbols;
  }

  public void walk(final AndroidDataPathWalker pathWalker) throws IOException {
    for (Path path : resourceDirs) {
      pathWalker.walkResources(path);
    }
    for (Path path : assetDirs) {
      pathWalker.walkAssets(path);
    }
  }

  public void deserialize(AndroidDataDeserializer deserializer, KeyValueConsumers consumers)
      throws DeserializationException {
    // Missing symbols means the resources where provided via android_resources rules.
    if (symbols == null) {
      throw new DeserializationException(true);
    }
    deserializer.read(symbols, consumers);
  }

  public String getLabel() {
    return label;
  }

  @Override
  public String toString() {
    return String.format(
        "SerializedAndroidData(%s, %s, %s, %s)", resourceDirs, assetDirs, label, symbols);
  }

  @Override
  public int hashCode() {
    return Objects.hash(resourceDirs, assetDirs, label, symbols);
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (obj == null) {
      return false;
    }
    if (!(obj instanceof SerializedAndroidData)) {
      return false;
    }
    SerializedAndroidData other = (SerializedAndroidData) obj;
    return Objects.equals(other.resourceDirs, resourceDirs)
        && Objects.equals(other.assetDirs, assetDirs)
        && Objects.equals(other.symbols, symbols)
        && Objects.equals(other.label, label);
  }
}
