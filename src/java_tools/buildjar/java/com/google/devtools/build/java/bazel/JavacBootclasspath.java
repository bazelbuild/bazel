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

package com.google.devtools.build.java.bazel;

import static com.google.common.collect.ImmutableList.toImmutableList;

import com.google.common.collect.ImmutableList;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOError;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Utility class to provide java-level access to the blessed javac boot class path: {@code
 * //tools/defaults:javac_bootclasspath}.
 *
 * <p>This class is typically used only from a host build tool or in tests. When using this in
 * production, the bootclasspath is deployed as separate jar files within the runfiles directory.
 */
public class JavacBootclasspath {

  private static final List<File> AS_FILES;
  private static final String AS_STRING;

  private static String getRunfilesDir() {
    String dir = System.getenv("JAVA_RUNFILES");
    if (dir == null) {
      dir = System.getenv("TEST_SRCDIR");
    }
    if (dir == null) {
      throw new IllegalStateException(
          "Neither JAVA_RUNFILES nor TEST_SRCDIR environment variable was defined!");
    }
    return dir;
  }

  static {
    String[] locations = JavacBootclasspathLocations.BOOTCLASSPATH.split(":");
    String runfilesRoot = getRunfilesDir();
    List<File> files = new ArrayList<>(locations.length);
    StringBuilder str = new StringBuilder();
    for (String location : locations) {
      File file = new File(runfilesRoot, location);
      if (!file.isFile()) {
        throw new IOError(
            new FileNotFoundException("Can't find boot class path element: " + file.getPath()));
      }
      files.add(file);
      if (str.length() > 0) {
        str.append(':');
      }
      str.append(file.getAbsolutePath());
    }
    AS_FILES = files;
    AS_STRING = str.toString();
  }

  /**
   * Returns the blessed boot class path as a colon-separated string.
   *
   * <p>Suitable for passing as the value of a {@code -bootclasspath} flag. Valid while the current
   * build action or test is executing.
   */
  public static String asString() {
    return AS_STRING;
  }

  /**
   * Returns the blessed boot class path as a list of {@code File} objects.
   *
   * <p>Each {@code File} will represent a jar file that will exist while the current build action
   * or test is executing.
   */
  public static List<File> asFiles() {
    return new ArrayList<>(AS_FILES);
  }

  /** Returns the blessed boot class path as a list of {@code Path}s. */
  public static ImmutableList<Path> asPaths() {
    return asFiles().stream().map(File::toPath).map(Path::normalize).collect(toImmutableList());
  }
}
