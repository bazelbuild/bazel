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
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Collectors;

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
 * <p>2. Import the runfiles library and the {@code AutoBazelRepository} annotation.
 *
 * <pre>
 *   import com.google.devtools.build.runfiles.AutoBazelRepository;
 *   import com.google.devtools.build.runfiles.Runfiles;
 * </pre>
 *
 * <p>3. Annotate the class in which a {@code Runfiles} object is created with
 * {@link AutoBazelRepository}:
 *
 * <pre>
 *   &#64;AutoBazelRepository
 *   public class MyClass {
 *     ...
 * </pre>
 *
 * <p>4. Create a Runfiles object and use rlocation to look up runfile paths:
 *
 * <pre>
 *   public void myFunction() {
 *     Runfiles runfiles = Runfiles.create(AutoBazelRepository_MyClass.BAZEL_REPOSITORY);
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
public final class Runfiles {

  public abstract static class Preloaded {

    /**
     * See {@link com.google.devtools.build.lib.analysis.RepoMappingManifestAction.Entry}.
     */
    static class RepoMappingKey {

      public final String sourceRepo;
      public final String targetRepoApparentName;

      public RepoMappingKey(String sourceRepo, String targetRepoApparentName) {
        this.sourceRepo = sourceRepo;
        this.targetRepoApparentName = targetRepoApparentName;
      }

      @Override
      public boolean equals(Object o) {
        if (this == o) {
          return true;
        }
        if (o == null || getClass() != o.getClass()) {
          return false;
        }
        RepoMappingKey that = (RepoMappingKey) o;
        return sourceRepo.equals(that.sourceRepo) && targetRepoApparentName.equals(
            that.targetRepoApparentName);
      }

      @Override
      public int hashCode() {
        return Objects.hash(sourceRepo, targetRepoApparentName);
      }
    }

    /**
     * Returns a new instance that uses the provided source repository as a default for all calls to
     * {@link #rlocation(String)}.
     *
     * <p>This is useful when receiving a {@link Runfiles} instance from a different Bazel
     * repository. In this case, while the runfiles manifest or directory encoded in the instance
     * should be used for runfiles lookups, the repository from which apparent repository names
     * should be resolved needs to change.
     *
     * @param sourceRepository the canonical name of the Bazel repository relative to which apparent
     *                         repository names should be resolved
     * @return a new {@link Runfiles} instance identical to this one, except that calls to
     * {@link #rlocation(String)} use the provided source repository.
     */
    public final Runfiles withSourceRepository(String sourceRepository) {
      Util.checkArgument(sourceRepository != null);
      return new Runfiles(this, sourceRepository);
    }

    public final Runfiles unmapped() {
      return new Runfiles(this, null);
    }

    protected abstract Map<String, String> getEnvVars();

    protected abstract String rlocationChecked(String path);

    protected abstract Map<RepoMappingKey, String> getRepoMapping();

    // Private constructor, so only nested classes may extend it.
    private Preloaded() {
    }
  }

  private static final String MAIN_REPOSITORY = "";

  private final Preloaded preloadedRunfiles;
  private final String sourceRepository;

  private Runfiles(Preloaded preloadedRunfiles, String sourceRepository) {
    this.preloadedRunfiles = preloadedRunfiles;
    this.sourceRepository = sourceRepository;
  }

  /**
   * Returns a new {@link Runfiles} instance.
   *
   * <p>This method passes the JVM's environment variable map to {@link #create(Map)}.
   *
   * @param sourceRepository the canonical name of the Bazel repository relative to which runfiles
   *                         lookups should be performed. This can be obtained using
   *                         {@link AutoBazelRepository} (see class documentation).
   */
  public static Preloaded preload() throws IOException {
    return preload(System.getenv());
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
   * @param sourceRepository the canonical name of the Bazel repository relative to which apparent
   *                         repository names in runfiles paths should be resolved. This can be
   *                         obtained using {@link AutoBazelRepository} (see class documentation).
   *
   * @throws IOException if RUNFILES_MANIFEST_ONLY=1 is in {@code env} but there's no
   *                     "RUNFILES_MANIFEST_FILE", "RUNFILES_DIR", or "JAVA_RUNFILES" key in
   *                     {@code env} or their values are empty, or some IO error occurs
   */
  public static Preloaded preload(Map<String, String> env) throws IOException {
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
   * Returns a new {@link Runfiles} instance.
   *
   * <p><strong>Deprecated: With {@code --enable_bzlmod}, this function can only resolve runfiles
   * correctly if called from the main repository. Use {@link #preload()}}
   * instead.</strong>
   *
   * <p>This method passes the JVM's environment variable map to {@link #create(Map)}.
   */
  @Deprecated
  public static Runfiles create() throws IOException {
    return preload().withSourceRepository(MAIN_REPOSITORY);
  }

  /**
   * Returns a new {@link Runfiles} instance.
   *
   * <p><strong>Deprecated: With {@code --enable_bzlmod}, this function can only resolve runfiles
   * correctly if called from the main repository. Use {@link #preload(Map)} instead.
   * </strong>
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
   *                     "RUNFILES_MANIFEST_FILE", "RUNFILES_DIR", or "JAVA_RUNFILES" key in
   *                     {@code env} or their values are empty, or some IO error occurs
   */
  @Deprecated
  public static Runfiles create(Map<String, String> env) throws IOException {
    return preload(env).withSourceRepository(MAIN_REPOSITORY);
  }

  /**
   * Returns the runtime path of a runfile (a Bazel-built binary's/test's data-dependency).
   *
   * <p>The returned path may not be valid. The caller should check the path's validity and that
   * the path exists.
   *
   * <p>The function may return null. In that case the caller can be sure that the rule does not
   * know about this data-dependency.
   *
   * @param path runfiles-root-relative path of the runfile
   * @throws IllegalArgumentException if {@code path} fails validation, for example if it's null or
   *                                  empty, or not normalized (contains "./", "../", or "//")
   */
  public String rlocation(String path) {
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

    if (sourceRepository == null) {
      return preloadedRunfiles.rlocationChecked(path);
    }
    String[] apparentTargetAndRemainder = path.split("/", 2);
    if (apparentTargetAndRemainder.length < 2) {
      return preloadedRunfiles.rlocationChecked(path);
    }
    String targetCanonical = preloadedRunfiles.getRepoMapping().getOrDefault(
        new Preloaded.RepoMappingKey(sourceRepository, apparentTargetAndRemainder[0]),
        apparentTargetAndRemainder[0]);
    return preloadedRunfiles.rlocationChecked(
        targetCanonical + "/" + apparentTargetAndRemainder[1]);
  }

  /**
   * Returns environment variables for subprocesses.
   *
   * <p>The caller should add the returned key-value pairs to the environment of subprocesses in
   * case those subprocesses are also Bazel-built binaries that need to use runfiles.
   */
  public Map<String, String> getEnvVars() {
    return preloadedRunfiles.getEnvVars();
  }

  /**
   * Returns true if the platform supports runfiles only via manifests.
   */
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

  private static Map<Preloaded.RepoMappingKey, String> loadRepositoryMapping(String path)
      throws IOException {
    if (path == null) {
      return Collections.emptyMap();
    }

    try (BufferedReader r = new BufferedReader(new FileReader(path, StandardCharsets.UTF_8))) {
      return Collections.unmodifiableMap(r.lines()
          .filter(line -> !line.isEmpty())
          .map(line -> {
            String[] split = line.split(",");
            if (split.length != 3) {
              throw new IllegalArgumentException(
                  "Invalid line in repository mapping: '" + line + "'");
            }
            return split;
          })
          .collect(Collectors.toMap(
              split -> new Preloaded.RepoMappingKey(split[0], split[1]),
              split -> split[2])));
    }
  }

  /**
   * {@link Runfiles} implementation that parses a runfiles-manifest file to look up runfiles.
   */
  private static final class ManifestBased extends Runfiles.Preloaded {

    private final Map<String, String> runfiles;
    private final String manifestPath;
    private final Map<RepoMappingKey, String> repoMapping;

    ManifestBased(String manifestPath) throws IOException {
      Util.checkArgument(manifestPath != null);
      Util.checkArgument(!manifestPath.isEmpty());
      this.manifestPath = manifestPath;
      this.runfiles = loadRunfiles(manifestPath);
      this.repoMapping = loadRepositoryMapping(findRepositoryMapping(manifestPath));
    }

    @Override
    protected String rlocationChecked(String path) {
      String exactMatch = runfiles.get(path);
      if (exactMatch != null) {
        return exactMatch;
      }
      // If path references a runfile that lies under a directory that itself is a runfile, then
      // only the directory is listed in the manifest. Look up all prefixes of path in the manifest
      // and append the relative path from the prefix if there is a match.
      int prefixEnd = path.length();
      while ((prefixEnd = path.lastIndexOf('/', prefixEnd - 1)) != -1) {
        String prefixMatch = runfiles.get(path.substring(0, prefixEnd));
        if (prefixMatch != null) {
          return prefixMatch + '/' + path.substring(prefixEnd + 1);
        }
      }
      return null;
    }

    @Override
    protected Map<String, String> getEnvVars() {
      HashMap<String, String> result = new HashMap<>(4);
      result.put("RUNFILES_MANIFEST_ONLY", "1");
      result.put("RUNFILES_MANIFEST_FILE", manifestPath);
      String runfilesDir = findRunfilesDir(manifestPath);
      result.put("RUNFILES_DIR", runfilesDir);
      // TODO(laszlocsomor): remove JAVA_RUNFILES once the Java launcher can pick up RUNFILES_DIR.
      result.put("JAVA_RUNFILES", runfilesDir);
      return result;
    }

    @Override
    protected Map<RepoMappingKey, String> getRepoMapping() {
      return repoMapping;
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

    private static String findRepositoryMapping(String manifestPath) {
      if (manifestPath != null && (manifestPath.endsWith(".runfiles/MANIFEST")
          || manifestPath.endsWith(".runfiles\\MANIFEST")
          || manifestPath.endsWith(".runfiles_manifest"))) {
        String path = manifestPath.substring(0, manifestPath.length() - 18) + ".repo_mapping";
        if (new File(path).isFile()) {
          return path;
        }
      }
      return null;
    }
  }

  /**
   * {@link Runfiles} implementation that appends runfiles paths to the runfiles root.
   */
  private static final class DirectoryBased extends Preloaded {

    private final String runfilesRoot;
    private final Map<RepoMappingKey, String> repoMapping;

    DirectoryBased(String runfilesDir) throws IOException {
      Util.checkArgument(!Util.isNullOrEmpty(runfilesDir));
      Util.checkArgument(new File(runfilesDir).isDirectory());
      this.runfilesRoot = runfilesDir;
      this.repoMapping = loadRepositoryMapping(findRepositoryMapping(runfilesRoot));
    }

    @Override
    protected String rlocationChecked(String path) {
      return runfilesRoot + "/" + path;
    }

    @Override
    protected Map<RepoMappingKey, String> getRepoMapping() {
      return repoMapping;
    }

    @Override
    protected Map<String, String> getEnvVars() {
      HashMap<String, String> result = new HashMap<>(2);
      result.put("RUNFILES_DIR", runfilesRoot);
      // TODO(laszlocsomor): remove JAVA_RUNFILES once the Java launcher can pick up RUNFILES_DIR.
      result.put("JAVA_RUNFILES", runfilesRoot);
      return result;
    }

    private static String findRepositoryMapping(String runfilesRoot) {
      if (runfilesRoot != null && runfilesRoot.endsWith(".runfiles")) {
        String path = runfilesRoot.substring(0, runfilesRoot.length() - 9) + ".repo_mapping";
        if (new File(path).isFile()) {
          return path;
        }
      }
      return null;
    }
  }

  static Preloaded createManifestBasedForTesting(String manifestPath) throws IOException {
    return new ManifestBased(manifestPath);
  }

  static Preloaded createDirectoryBasedForTesting(String runfilesDir) throws IOException {
    return new DirectoryBased(runfilesDir);
  }
}
