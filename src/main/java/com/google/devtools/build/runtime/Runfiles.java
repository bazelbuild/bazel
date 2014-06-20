// Copyright 2014 Google Inc. All rights reserved.
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

package com.google.devtools.build.runtime;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOError;

/**
 * Utilities related to google3 runfiles.
 */
public final class Runfiles {
  private static final String RUNFILES_DIR = runfilesDirFile().getAbsolutePath();

  private static final String GOOGLE3_DIR = RUNFILES_DIR + "/google3";

  private Runfiles() {}

  /**
   * Returns the runtime location of a data dependency, as a {@link File}.
   * This is analogous to the {@code $(location ...)} construct in
   * the google3 BUILD file language, except that it may also be used to
   * refer to directories which are ancestors of specific data dependency
   * files.
   */
  public static File location(String dataDependency) {
    checkRelativePath(dataDependency);
    File file = new File(GOOGLE3_DIR, dataDependency);
    if (!file.exists()) {
      throw new IOError(new FileNotFoundException(file.getPath()));
    }
    return file;
  }

  /**
   * Returns the runtime location of a data dependency, as a {@link File},
   * relative to a google3 package directory as specified by a google3 java
   * root directory and a java class defined in the google3 package.
   */
  public static File packageRelativeLocation(
      String javaPackageRoot,
      Class<?> clazz,
      String packageRelativePath) {
    checkRelativePath(packageRelativePath);
    String packagePath = clazz.getPackage().getName().replace('.', File.separatorChar);
    int len = javaPackageRoot.length() + packagePath.length() + packageRelativePath.length() + 2;
    StringBuilder sb = new StringBuilder(len)
        .append(javaPackageRoot)
        .append(File.separatorChar)
        .append(packagePath);
    if (!packageRelativePath.isEmpty()) {
      sb.append(File.separatorChar);
      sb.append(packageRelativePath);
    }
    return location(sb.toString());
  }

  private static void checkRelativePath(String path) {
    if (path.startsWith(File.separator)) {
      throw new IllegalArgumentException(
          String.format(
              "Relative path argument '%s' should not start with a file separator character",
              path));
    }
  }

  /**
   * Gets path of the runfiles directory for the current Java program.
   *
   * @return absolute path of the runfiles directory.  It will not contain
   * a trailing slash.
   */
  public static String getRunfilesDir() {
    return RUNFILES_DIR;
  }

  /** Extracts runfilesDir from environment. */
  private static File runfilesDirFile() {

    File runfilesDir;

    String runfilesDirStr = getRunfilesPathFromEnvironment();
    if (runfilesDirStr != null && runfilesDirStr.length() > 0) {
      runfilesDir = new File(runfilesDirStr);
    } else {
      // Goal is to find google3 directory, so we check current
      // directory, then keep backing up until we see google3.
      File dir = new File("");
      while (dir != null) {
        dir = dir.getAbsoluteFile();

        File google3 = new File(dir, "google3");
        if (google3.exists()) {
          return dir;
        }

        dir = dir.getParentFile();
      }

      runfilesDir = new File("").getAbsoluteFile().getParentFile();
    }

    return runfilesDir;
  }

  private static String getRunfilesPathFromEnvironment() {
    String propValue = System.getProperty("TEST_SRCDIR");
    return (propValue != null) ? propValue : System.getenv("TEST_SRCDIR");
  }
}
