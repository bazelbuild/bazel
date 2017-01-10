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

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOError;
import java.nio.file.AccessDeniedException;

/**
 * Utility class to provide java-level access to the blessed langtools jar path: {@code
 * //third_party/java/jdk:langtools}, as defined by bazel's --java_langtools flag.
 */
public class JavaLangtools {

  private static final File FILE;

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
    File file = new File(getRunfilesDir(), JavaLangtoolsLocation.FILE);
    if (!file.isFile()) {
      throw new IOError(new FileNotFoundException("Can't find langtools jar: " + file.getPath()));
    } else if (!file.canRead()) {
      throw new IOError(new AccessDeniedException("Can't read langtools jar: " + file.getPath()));
    }
    FILE = file;
  }

  /** Returns the blessed langtools jar path. */
  public static File file() {
    return FILE;
  }
}
