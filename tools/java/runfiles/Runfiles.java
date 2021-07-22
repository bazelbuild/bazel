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

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/**
 * Runfiles lookup library for Bazel-built Java binaries and tests.
 *
 * <p>USAGE:
 *
 * <p>1. Depend on this runfiles library from your build rule:
 *
 * <pre>
 *   java_binary(
 *       name = "my_binary",
 *       ...
 *       deps = ["@bazel_tools//tools/java/runfiles"],
 *   )
 * </pre>
 *
 * <p>2. Import the runfiles library.
 *
 * <pre>
 *   import com.google.devtools.build.runfiles.Runfiles;
 * </pre>
 *
 * <p>3. Create a Runfiles object and use rlocation to look up runfile paths:
 *
 * <pre>
 *   public void myFunction() {
 *     Runfiles runfiles = Runfiles.create();
 *     String path = runfiles.rlocation("my_workspace/path/to/my/data.txt");
 *     ...
 * </pre>
 *
 * <p>If you want to start subprocesses that also need runfiles, you need to set the right
 * environment variables for them:
 *
 * <pre>
 *   String path = r.rlocation("path/to/binary");
 *   ProcessBuilder pb = new ProcessBuilder(path);
 *   pb.environment().putAll(r.getEnvVars());
 *   ...
 *   Process p = pb.start();
 * </pre>
 */
public abstract class Runfiles {

  // Package-private constructor, so only package-private classes may extend it.
  private Runfiles() {}

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
   *     empty, or not normalized (contains "./", "../", or "//")
   */
  public final String rlocation(String path) {
    Util.checkArgument(path != null);
    Util.checkArgument(!path.isEmpty());
    Util.checkArgument(
        !path.startsWith("../")
            && !path.contains("/..")
            && !path.startsWith("./")
            && !path.contains("/./")
            && !path.endsWith("/.")
            && !path.contains("//"),
        "path is not normalized: \"%s\"",
        path);
    Util.checkArgument(
        !path.startsWith("\\"), "path is absolute without a drive letter: \"%s\"", path);
    if (new File(path).isAbsolute()) {
      return path;
    }
    return rlocationChecked(path);
  }

  /**
   * Returns environment variables for subprocesses.
   *
   * <p>The caller should add the returned key-value pairs to the environment of subprocesses in
   * case those subprocesses are also Bazel-built binaries that need to use runfiles.
   */
  public abstract Map<String, String> getEnvVars();

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

  /** {@link Runfiles} implementation that parses a runfiles-manifest file to look up runfiles. */
  private static final class ManifestBased extends Runfiles {
    private final Map<String, String> runfiles;
    private final String manifestPath;

    ManifestBased(String manifestPath) throws IOException {
      Util.checkArgument(manifestPath != null);
      Util.checkArgument(!manifestPath.isEmpty());
      this.manifestPath = manifestPath;
      this.runfiles = loadRunfiles(manifestPath);
    }

    private static Map<String, String> loadRunfiles(String path) throws IOException {
      HashMap<String, String> result = new HashMap<>();
      try (BufferedReader r =
          new BufferedReader(
              new InputStreamReader(new FileInputStream(path), StandardCharsets.UTF_8))) {
        String line = null;
        while ((line = r.readLine()) != null) {
          int index = line.indexOf(' ');
          String runfile = (index == -1) ? line : line.substring(0, index);
          String realPath = (index == -1) ? line : line.substring(index + 1);
          result.put(runfile, realPath);
        }
      }
      return Collections.unmodifiableMap(result);
    }

    private static String findRunfilesDir(String manifest) {
      if (manifest.endsWith("/MANIFEST")
          || manifest.endsWith("\\MANIFEST")
          || manifest.endsWith(".runfiles_manifest")) {
        String path = manifest.substring(0, manifest.length() - 9);
        if (new File(path).isDirectory()) {
          return path;
        }
      }
      return "";
    }

    @Override
    public String rlocationChecked(String path) {
      return runfiles.get(path);
    }

    @Override
    public Map<String, String> getEnvVars() {
      HashMap<String, String> result = new HashMap<>(4);
      result.put("RUNFILES_MANIFEST_ONLY", "1");
      result.put("RUNFILES_MANIFEST_FILE", manifestPath);
      String runfilesDir = findRunfilesDir(manifestPath);
      result.put("RUNFILES_DIR", runfilesDir);
      // TODO(laszlocsomor): remove JAVA_RUNFILES once the Java launcher can pick up RUNFILES_DIR.
      result.put("JAVA_RUNFILES", runfilesDir);
      return result;
    }
  }

  /** {@link Runfiles} implementation that appends runfiles paths to the runfiles root. */
  private static final class DirectoryBased extends Runfiles {
    private final String runfilesRoot;

    DirectoryBased(String runfilesDir) {
      Util.checkArgument(!Util.isNullOrEmpty(runfilesDir));
      Util.checkArgument(new File(runfilesDir).isDirectory());
      this.runfilesRoot = runfilesDir;
    }

    @Override
    String rlocationChecked(String path) {
      return runfilesRoot + "/" + path;
    }

    @Override
    public Map<String, String> getEnvVars() {
      HashMap<String, String> result = new HashMap<>(2);
      result.put("RUNFILES_DIR", runfilesRoot);
      // TODO(laszlocsomor): remove JAVA_RUNFILES once the Java launcher can pick up RUNFILES_DIR.
      result.put("JAVA_RUNFILES", runfilesRoot);
      return result;
    }
  }

  static Runfiles createManifestBasedForTesting(String manifestPath) throws IOException {
    return new ManifestBased(manifestPath);
  }

  static Runfiles createDirectoryBasedForTesting(String runfilesDir) {
    return new DirectoryBased(runfilesDir);
  }
}
