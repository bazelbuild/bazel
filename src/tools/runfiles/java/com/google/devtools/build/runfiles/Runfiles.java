// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.runfiles;

import java.io.File;
import java.io.IOException;
import java.util.Map;

/**
 * Returns the runtime location of runfiles (data-dependencies of Bazel-built binaries and tests).
 *
 * <p>Usage:
 *
 * <pre>
 *   Runfiles runfiles = Runfiles.create();
 *   File p = new File(runfiles.rlocation("io_bazel/src/bazel"));
 * </pre>
 */
public abstract class Runfiles {

  // Package-private constructor, so only package-private classes may extend it.
  Runfiles() {}

  /**
   * Returns a new {@link Runfiles} instance.
   *
   * <p>This method passes the JVM's environment variable map to {@link #create(Map)}.
   */
  public static Runfiles create() throws IOException {
    return create(System.getenv());
  }

  /**
   * Returns a new {@link Runfiles} instance.
   *
   * <p>The returned object is either:
   *
   * <ul>
   *   <li>manifest-based, meaning it looks up runfile paths from a manifest file, or
   *   <li>directory-based, meaning it looks up runfile paths under a given directory path
   * </ul>
   *
   * <p>If {@code env} contains "RUNFILES_MANIFEST_ONLY" with value "1", this method returns a
   * manifest-based implementation. The manifest's path is defined by the "RUNFILES_MANIFEST_FILE"
   * key's value in {@code env}.
   *
   * <p>Otherwise this method returns a directory-based implementation. The directory's path is
   * defined by the value in {@code env} under the "RUNFILES_DIR" key, or if absent, then under the
   * "JAVA_RUNFILES" key.
   *
   * <p>Note about performance: the manifest-based implementation eagerly reads and caches the whole
   * manifest file upon instantiation.
   *
   * @throws IOException if RUNFILES_MANIFEST_ONLY=1 is in {@code env} but there's no
   *     "RUNFILES_MANIFEST_FILE", "RUNFILES_DIR", or "JAVA_RUNFILES" key in {@code env} or their
   *     values are empty, or some IO error occurs
   */
  public static Runfiles create(Map<String, String> env) throws IOException {
    if (isManifestOnly(env)) {
      // On Windows, Bazel sets RUNFILES_MANIFEST_ONLY=1.
      // On every platform, Bazel also sets RUNFILES_MANIFEST_FILE, but on Linux and macOS it's
      // faster to use RUNFILES_DIR.
      return new ManifestBased(getManifestPath(env));
    } else {
      return new DirectoryBased(getRunfilesDir(env));
    }
  }

  /**
   * Returns the runtime path of a runfile (a Bazel-built binary's/test's data-dependency).
   *
   * <p>The returned path may not be valid. The caller should check the path's validity and that the
   * path exists.
   *
   * <p>The function may return null. In that case the caller can be sure that the rule does not
   * know about this data-dependency.
   *
   * @param path runfiles-root-relative path of the runfile
   * @throws IllegalArgumentException if {@code path} fails validation, for example if it's null or
   *     empty, or contains uplevel references
   */
  public final String rlocation(String path) {
    Util.checkArgument(path != null);
    Util.checkArgument(!path.isEmpty());
    Util.checkArgument(!path.contains(".."), "path contains uplevel references: \"%s\"", path);
    if (new File(path).isAbsolute() || path.charAt(0) == File.separatorChar) {
      return path;
    }
    return rlocationChecked(path);
  }

  /** Returns true if the platform supports runfiles only via manifests. */
  private static boolean isManifestOnly(Map<String, String> env) {
    return "1".equals(env.get("RUNFILES_MANIFEST_ONLY"));
  }

  private static String getManifestPath(Map<String, String> env) throws IOException {
    String value = env.get("RUNFILES_MANIFEST_FILE");
    if (Util.isNullOrEmpty(value)) {
      throw new IOException(
          "Cannot load runfiles manifest: $RUNFILES_MANIFEST_ONLY is 1 but"
              + " $RUNFILES_MANIFEST_FILE is empty or undefined");
    }
    return value;
  }

  private static String getRunfilesDir(Map<String, String> env) throws IOException {
    String value = env.get("RUNFILES_DIR");
    if (Util.isNullOrEmpty(value)) {
      value = env.get("JAVA_RUNFILES");
    }
    if (Util.isNullOrEmpty(value)) {
      throw new IOException(
          "Cannot find runfiles: $RUNFILES_DIR and $JAVA_RUNFILES are both unset or empty");
    }
    return value;
  }

  abstract String rlocationChecked(String path);
}
