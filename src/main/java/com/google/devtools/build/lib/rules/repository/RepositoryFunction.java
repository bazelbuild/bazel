// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.repository;

import static com.google.common.base.Preconditions.checkState;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.io.BaseEncoding;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.LabelValidator;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.repository.ExternalPackageException;
import com.google.devtools.build.lib.repository.RepositoryFetchProgress;
import com.google.devtools.build.lib.skyframe.ActionEnvironmentFunction;
import com.google.devtools.build.lib.skyframe.AlreadyReportedException;
import com.google.devtools.build.lib.skyframe.PackageLookupFunction;
import com.google.devtools.build.lib.skyframe.PackageLookupValue;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import java.io.IOException;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Starlark;

/**
 * Implementation of fetching various external repository types.
 *
 * <p>These objects are called from {@link RepositoryDelegatorFunction}.
 *
 * <p>External repositories come in two flavors: local and non-local.
 *
 * <p>Local ones are those whose fetching does not require access to any external resources (e.g.
 * network). These are always re-fetched on Bazel server restarts. This operation is fast (usually
 * just a few symlinks and maybe writing a BUILD file). {@code --nofetch} does not apply to local
 * repositories.
 *
 * <p>The up-to-dateness of non-local repositories is checked using a marker file under the output
 * base. When such a repository is fetched, data from the rule in the WORKSPACE file is written to
 * the marker file which is consulted on next server startup. If the rule hasn't changed, the
 * repository is not re-fetched.
 *
 * <p>Fetching repositories can be disabled using the {@code --nofetch} command line option. If a
 * repository is on the file system, Bazel just tries to use it and hopes for the best. If the
 * repository has never been fetched, Bazel errors out for lack of a better option. This is
 * implemented using {@link
 * com.google.devtools.build.lib.bazel.BazelRepositoryModule#REPOSITORY_VALUE_CHECKER} and a flag in
 * {@link RepositoryDirectoryValue} that tells Bazel whether the value in Skyframe is stale
 * according to the value of {@code --nofetch} or not.
 *
 * <p>When a rule in the WORKSPACE file is changed, the corresponding {@link
 * RepositoryDirectoryValue} is invalidated using the usual Skyframe route.
 */
public abstract class RepositoryFunction {

  protected Map<String, String> clientEnvironment;

  /**
   * Exception thrown when something goes wrong accessing a remote repository.
   *
   * <p>This exception should be used by child classes to limit the types of exceptions {@link
   * RepositoryDelegatorFunction} has to know how to catch.
   */
  public static class RepositoryFunctionException extends SkyFunctionException {

    /** Error reading or writing to the filesystem. */
    public RepositoryFunctionException(IOException cause, Transience transience) {
      super(cause, transience);
    }

    /** For errors in WORKSPACE file rules (e.g., malformed paths or URLs). */
    public RepositoryFunctionException(EvalException cause, Transience transience) {
      super(cause, transience);
    }

    public RepositoryFunctionException(
        AlreadyReportedRepositoryAccessException cause, Transience transience) {
      super(cause, transience);
    }

    public RepositoryFunctionException(ExternalPackageException e) {
      super(e.getCause(), e.isTransient() ? Transience.TRANSIENT : Transience.PERSISTENT);
    }
  }

  /**
   * Encapsulates the exceptions that arise when accessing a repository. Error reporting should ONLY
   * be handled in {@link RepositoryDelegatorFunction#fetchRepository}.
   */
  public static class AlreadyReportedRepositoryAccessException extends AlreadyReportedException {
    public AlreadyReportedRepositoryAccessException(Exception e) {
      super(e.getMessage(), e.getCause());
      checkState(
          e instanceof NoSuchPackageException
              || e instanceof IOException
              || e instanceof EvalException
              || e instanceof ExternalPackageException,
          e);
    }
  }

  public static void setupRepoRoot(Path repoRoot) throws RepositoryFunctionException {
    try {
      repoRoot.deleteTree();
      Preconditions.checkNotNull(repoRoot.getParentDirectory()).createDirectoryAndParents();
    } catch (IOException e) {
      throw new RepositoryFunctionException(e, Transience.TRANSIENT);
    }
  }

  protected void setupRepoRootBeforeFetching(Path repoRoot) throws RepositoryFunctionException {
    setupRepoRoot(repoRoot);
  }

  public void reportSkyframeRestart(Environment env, RepositoryName repoName) {
    env.getListener().post(RepositoryFetchProgress.ongoing(repoName, "Restarting."));
  }

  /**
   * Fetch the remote repository represented by the given rule.
   *
   * <p>When this method is called, it has already been determined that the repository is stale and
   * that it needs to be re-fetched.
   *
   * <p>The {@code env} argument can be used to fetch Skyframe dependencies the repository
   * implementation needs on the following conditions:
   *
   * <ul>
   *   <li>When a Skyframe value is missing, fetching must be restarted, thus, in order to avoid
   *       doing duplicate work, it's better to first request the Skyframe dependencies you need and
   *       only then start doing anything costly.
   *   <li>The output directory must be populated from within this method (and not from within
   *       another SkyFunction). This is because if it was populated in another SkyFunction, the
   *       repository function would be restarted <b>after</b> that SkyFunction has been run, and it
   *       would wipe the output directory clean.
   * </ul>
   *
   * <p>The {@code markerData} argument can be mutated to augment the data to write to the
   * repository marker file. If any data in the {@code markerData} change between 2 execute of the
   * {@link RepositoryDelegatorFunction} then this should be a reason to invalidate the repository.
   * The {@link #verifyMarkerData} method is responsible for checking the value added to that map
   * when checking the content of a marker file.
   */
  @ThreadSafe
  @Nullable
  public abstract RepositoryDirectoryValue.Builder fetch(
      Rule rule,
      Path outputDirectory,
      BlazeDirectories directories,
      Environment env,
      Map<String, String> markerData,
      SkyKey key)
      throws InterruptedException, RepositoryFunctionException;

  @SuppressWarnings("unchecked")
  private static Collection<String> getEnviron(Rule rule) {
    if (rule.isAttrDefined("$environ", Type.STRING_LIST)) {
      return (Collection<String>) rule.getAttr("$environ");
    }
    return ImmutableList.of();
  }

  /**
   * Verify the data provided by the marker file to check if a refetch is needed. Returns true if
   * the data is up to date and no refetch is needed and false if the data is obsolete and a refetch
   * is needed.
   */
  public boolean verifyMarkerData(Rule rule, Map<String, String> markerData, Environment env)
      throws InterruptedException {
    return verifyEnvironMarkerData(markerData, env, getEnviron(rule))
        && verifyMarkerDataForFiles(rule, markerData, env)
        && verifySemanticsMarkerData(markerData, env);
  }

  protected boolean verifySemanticsMarkerData(Map<String, String> markerData, Environment env)
      throws InterruptedException {
    return true;
  }

  private static boolean verifyLabelMarkerData(Rule rule, String key, String value, Environment env)
      throws InterruptedException {
    Preconditions.checkArgument(key.startsWith("FILE:"));
    try {
      RootedPath rootedPath;
      String fileKey = key.substring(5);
      if (LabelValidator.isAbsolute(fileKey)) {
        rootedPath = getRootedPathFromLabel(Label.parseCanonical(fileKey), env);
      } else {
        // TODO(pcloudy): Removing checking absolute path, they should all be absolute label.
        PathFragment filePathFragment = PathFragment.create(fileKey);
        Path file = rule.getPackage().getPackageDirectory().getRelative(filePathFragment);
        rootedPath =
            RootedPath.toRootedPath(
                Root.fromPath(file.getParentDirectory()), PathFragment.create(file.getBaseName()));
      }

      SkyKey fileSkyKey = FileValue.key(rootedPath);
      FileValue fileValue = (FileValue) env.getValueOrThrow(fileSkyKey, IOException.class);

      if (fileValue == null || !fileValue.isFile() || fileValue.isSpecialFile()) {
        return false;
      }

      return Objects.equals(value, fileValueToMarkerValue(fileValue));
    } catch (LabelSyntaxException e) {
      throw new IllegalStateException(
          "Key " + key + " is not a correct file key (should be in form FILE:label)", e);
    } catch (IOException | EvalException e) {
      // Consider those exception to be a cause for invalidation
      return false;
    }
  }

  /**
   * Convert to a {@link com.google.devtools.build.lib.actions.FileValue} to a String appropriate
   * for placing in a repository marker file.
   *
   * @param fileValue The value to convert. It must correspond to a regular file.
   */
  public static String fileValueToMarkerValue(FileValue fileValue) throws IOException {
    Preconditions.checkArgument(fileValue.isFile() && !fileValue.isSpecialFile());
    // Return the file content digest in hex. fileValue may or may not have the digest available.
    byte[] digest = fileValue.realFileStateValue().getDigest();
    if (digest == null) {
      // Fast digest not available, or it would have been in the FileValue.
      digest = fileValue.realRootedPath().asPath().getDigest();
    }
    return BaseEncoding.base16().lowerCase().encode(digest);
  }

  static boolean verifyMarkerDataForFiles(
      Rule rule, Map<String, String> markerData, Environment env) throws InterruptedException {
    for (Map.Entry<String, String> entry : markerData.entrySet()) {
      if (entry.getKey().startsWith("FILE:")) {
        if (!verifyLabelMarkerData(rule, entry.getKey(), entry.getValue(), env)) {
          return false;
        }
      }
    }
    return true;
  }

  public static RootedPath getRootedPathFromLabel(Label label, Environment env)
      throws InterruptedException, EvalException {
    SkyKey pkgSkyKey = PackageLookupValue.key(label.getPackageIdentifier());
    PackageLookupValue pkgLookupValue = (PackageLookupValue) env.getValue(pkgSkyKey);
    if (pkgLookupValue == null) {
      throw new NeedsSkyframeRestartException();
    }
    if (!pkgLookupValue.packageExists()) {
      String message = pkgLookupValue.getErrorMsg();
      if (pkgLookupValue == PackageLookupValue.NO_BUILD_FILE_VALUE) {
        message = PackageLookupFunction.explainNoBuildFileValue(label.getPackageIdentifier(), env);
      }
      throw Starlark.errorf("Unable to load package for %s: %s", label, message);
    }

    // And now for the file
    Root packageRoot = pkgLookupValue.getRoot();
    return RootedPath.toRootedPath(packageRoot, label.toPathFragment());
  }

  /**
   * A method that can be called from a implementation of {@link #fetch(Rule, Path,
   * BlazeDirectories, Environment, Map, SkyKey)} to declare a list of Skyframe dependencies on
   * environment variable. It also add the information to the marker file. It returns the list of
   * environment variable on which the function depends, or null if the skyframe function needs to
   * be restarted.
   */
  @Nullable
  protected Map<String, String> declareEnvironmentDependencies(
      Map<String, String> markerData, Environment env, Iterable<String> keys)
      throws InterruptedException {
    ImmutableMap<String, String> envDep = getEnvVarValues(env, keys);
    if (envDep == null) {
      return null;
    }
    // Add the dependencies to the marker file
    keys.forEach(key -> markerData.put("ENV:" + key, envDep.get(key)));
    return envDep;
  }

  @Nullable
  public static ImmutableMap<String, String> getEnvVarValues(Environment env, Iterable<String> keys)
      throws InterruptedException {
    ImmutableMap<String, String> environ = ActionEnvironmentFunction.getEnvironmentView(env, keys);
    if (environ == null) {
      return null;
    }
    Map<String, String> repoEnvOverride = PrecomputedValue.REPO_ENV.get(env);
    if (repoEnvOverride == null) {
      return null;
    }

    // Only depend on --repo_env values that are specified in the "environ" attribute.
    ImmutableMap.Builder<String, String> repoEnv = ImmutableMap.builder();
    repoEnv.putAll(environ);
    for (String key : keys) {
      String value = repoEnvOverride.get(key);
      if (value != null) {
        repoEnv.put(key, value);
      }
    }
    return repoEnv.buildKeepingLast();
  }

  /**
   * Verify marker data previously saved by {@link #declareEnvironmentDependencies(Map, Environment,
   * Iterable)}. This function is to be called from a {@link #verifyMarkerData(Rule, Map,
   * Environment)} function to verify the values for environment variables.
   */
  protected boolean verifyEnvironMarkerData(
      Map<String, String> markerData, Environment env, Collection<String> keys)
      throws InterruptedException {
    ImmutableMap<String, String> environ = ActionEnvironmentFunction.getEnvironmentView(env, keys);
    if (env.valuesMissing()) {
      return false; // Returns false so caller knows to return immediately
    }

    Map<String, String> repoEnvOverride = PrecomputedValue.REPO_ENV.get(env);
    if (repoEnvOverride == null) {
      return false;
    }

    // Only depend on --repo_env values that are specified in the "environ" attribute.
    Map<String, String> repoEnv = new LinkedHashMap<>(environ);
    for (String key : keys) {
      String value = repoEnvOverride.get(key);
      if (value != null) {
        repoEnv.put(key, value);
      }
    }

    // Verify that all environment variable in the marker file are also in keys
    for (String key : markerData.keySet()) {
      if (key.startsWith("ENV:") && !keys.contains(key.substring(4))) {
        return false;
      }
    }

    // Now verify the values of the marker data
    for (String key : keys) {
      if (!markerData.containsKey("ENV:" + key)) {
        return false;
      }
      String markerValue = markerData.get("ENV:" + key);
      if (!Objects.equals(markerValue, repoEnv.get(key))) {
        return false;
      }
    }
    return true;
  }

  /**
   * Whether fetching is done using local operations only.
   *
   * <p>If this is false, Bazel may decide not to re-fetch the repository, for example when the
   * {@code --nofetch} command line option is used.
   */
  protected abstract boolean isLocal(Rule rule);

  /** Wheather the rule declares it inspects the local environment for configure purpose. */
  protected boolean isConfigure(Rule rule) {
    return false;
  }

  protected Path prepareLocalRepositorySymlinkTree(Rule rule, Path repositoryDirectory)
      throws RepositoryFunctionException {
    try {
      repositoryDirectory.createDirectoryAndParents();
    } catch (IOException e) {
      throw new RepositoryFunctionException(e, Transience.TRANSIENT);
    }

    // Add x/WORKSPACE.
    createWorkspaceFile(repositoryDirectory, rule.getTargetKind(), rule.getName());
    return repositoryDirectory;
  }

  public static void createWorkspaceFile(Path repositoryDirectory, String ruleKind, String ruleName)
      throws RepositoryFunctionException {
    try {
      Path workspaceFile = repositoryDirectory.getRelative(LabelConstants.WORKSPACE_FILE_NAME);
      FileSystemUtils.writeContent(
          workspaceFile,
          UTF_8,
          String.format(
              "# DO NOT EDIT: automatically generated WORKSPACE file for %s\n"
                  + "workspace(name = \"%s\")\n",
              ruleKind, ruleName));
    } catch (IOException e) {
      throw new RepositoryFunctionException(e, Transience.TRANSIENT);
    }
  }

  protected static RepositoryDirectoryValue.Builder writeFile(
      Path repositoryDirectory, String filename, String contents)
      throws RepositoryFunctionException {
    Path filePath = repositoryDirectory.getRelative(filename);
    try {
      // The repository could have an existing file that's either a regular file (for remote
      // repositories) or a symlink (for local repositories). Either way, we want to remove it and
      // write our own.
      if (filePath.exists(Symlinks.NOFOLLOW)) {
        filePath.delete();
      }
      FileSystemUtils.writeContentAsLatin1(filePath, contents);
    } catch (IOException e) {
      throw new RepositoryFunctionException(e, Transience.TRANSIENT);
    }

    return RepositoryDirectoryValue.builder().setPath(repositoryDirectory);
  }

  protected static RepositoryDirectoryValue.Builder writeBuildFile(
      Path repositoryDirectory, String contents) throws RepositoryFunctionException {
    return writeFile(repositoryDirectory, "BUILD.bazel", contents);
  }

  protected static String getPathAttr(Rule rule) throws RepositoryFunctionException {
    WorkspaceAttributeMapper mapper = WorkspaceAttributeMapper.of(rule);
    try {
      return mapper.get("path", Type.STRING);
    } catch (EvalException e) {
      throw new RepositoryFunctionException(e, Transience.PERSISTENT);
    }
  }

  @VisibleForTesting
  protected static PathFragment getTargetPath(String userDefinedPath, Path workspace) {
    PathFragment pathFragment = PathFragment.create(userDefinedPath);
    return workspace.getRelative(pathFragment).asFragment();
  }

  /**
   * Given a targetDirectory /some/path/to/y that contains files z, w, and v, create the following
   * directory structure:
   *
   * <pre>
   * .external-repository/
   *   x/
   *     WORKSPACE
   *     BUILD -> &lt;build_root&gt;/x.BUILD
   *     z -> /some/path/to/y/z
   *     w -> /some/path/to/y/w
   *     v -> /some/path/to/y/v
   * </pre>
   */
  public static boolean symlinkLocalRepositoryContents(
      Path repositoryDirectory, Path targetDirectory, String userDefinedPath)
      throws RepositoryFunctionException {
    try {
      repositoryDirectory.createDirectoryAndParents();
      for (Path target : targetDirectory.getDirectoryEntries()) {
        Path symlinkPath = repositoryDirectory.getRelative(target.getBaseName());
        createSymbolicLink(symlinkPath, target);
      }
    } catch (IOException e) {
      throw new RepositoryFunctionException(
          new IOException(
              String.format(
                  "The repository's path is \"%s\" (absolute: \"%s\") "
                      + "but a symlink could not be created for it, because: %s",
                  userDefinedPath, targetDirectory, e.getMessage())),
          Transience.TRANSIENT);
    }

    return true;
  }

  static void createSymbolicLink(Path from, Path to) throws RepositoryFunctionException {
    try {
      // Remove not-symlinks that are already there.
      if (from.exists()) {
        from.delete();
      }
      FileSystemUtils.ensureSymbolicLink(from, to);
    } catch (IOException e) {
      throw new RepositoryFunctionException(
          new IOException(
              String.format(
                  "Error creating symbolic link from %s to %s: %s", from, to, e.getMessage())),
          Transience.TRANSIENT);
    }
  }

  protected static Path getExternalRepositoryDirectory(BlazeDirectories directories) {
    return directories.getOutputBase().getRelative(LabelConstants.EXTERNAL_REPOSITORY_LOCATION);
  }

  /**
   * For files that are under $OUTPUT_BASE/external, add a dependency on the corresponding rule so
   * that if the WORKSPACE file changes, the File/DirectoryStateValue will be re-evaluated.
   *
   * <p>Note that: - We don't add a dependency on the parent directory at the package root boundary,
   * so the only transitive dependencies from files inside the package roots to external files are
   * through symlinks. So the upwards transitive closure of external files is small. - The only way
   * other than external repositories for external source files to get into the skyframe graph in
   * the first place is through symlinks outside the package roots, which we neither want to
   * encourage nor optimize for since it is not common. So the set of external files is small.
   */
  public static void addExternalFilesDependencies(
      RootedPath rootedPath, BlazeDirectories directories, Environment env)
      throws InterruptedException {
    Path externalRepoDir = getExternalRepositoryDirectory(directories);
    PathFragment repositoryPath = rootedPath.asPath().relativeTo(externalRepoDir);
    if (repositoryPath.isEmpty()) {
      // We are the top of the repository path (<outputBase>/external), not in an actual external
      // repository path.
      return;
    }
    String repositoryName = repositoryPath.getSegment(0);
    env.getValue(RepositoryDirectoryValue.key(RepositoryName.createUnvalidated(repositoryName)));
  }

  /** Sets up a mapping of environment variables to use. */
  public void setClientEnvironment(Map<String, String> clientEnvironment) {
    this.clientEnvironment = clientEnvironment;
  }

  /** Returns the RuleDefinition class for this type of repository. */
  public abstract Class<? extends RuleDefinition> getRuleDefinition();
}
