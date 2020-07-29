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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.io.BaseEncoding;
import com.google.devtools.build.lib.actions.FileStateValue.RegularFileStateValue;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.LabelValidator;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.packages.BuildFileContainsErrorsException;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.repository.ExternalPackageException;
import com.google.devtools.build.lib.repository.ExternalPackageHelper;
import com.google.devtools.build.lib.repository.ExternalRuleNotFoundException;
import com.google.devtools.build.lib.skyframe.ActionEnvironmentFunction;
import com.google.devtools.build.lib.skyframe.PackageLookupFunction;
import com.google.devtools.build.lib.skyframe.PackageLookupValue;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Location;
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
import java.nio.charset.Charset;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import javax.annotation.Nullable;

/**
 * Implementation of fetching various external repository types.
 *
 * <p>These objects are called from {@link RepositoryDelegatorFunction}.
 *
 * <p>External repositories come in two flavors: local and non-local.
 *
 * <p>Local ones are those whose fetching does not require access to any external resources
 * (e.g. network). These are always re-fetched on Bazel server restarts. This operation is fast
 * (usually just a few symlinks and maybe writing a BUILD file). {@code --nofetch} does not apply
 * to local repositories.
 *
 * <p>The up-to-dateness of non-local repositories is checked using a marker file under the
 * output base. When such a repository is fetched, data from the rule in the WORKSPACE file is
 * written to the marker file which is consulted on next server startup. If the rule hasn't changed,
 * the repository is not re-fetched.
 *
 * <p>Fetching repositories can be disabled using the {@code --nofetch} command line option. If a
 * repository is on the file system, Bazel just tries to use it and hopes for the best. If the
 * repository has never been fetched, Bazel errors out for lack of a better option. This is
 * implemented using
 * {@link com.google.devtools.build.lib.bazel.BazelRepositoryModule#REPOSITORY_VALUE_CHECKER} and
 * a flag in {@link RepositoryDirectoryValue} that tells Bazel whether the value in Skyframe is
 * stale according to the value of {@code --nofetch} or not.
 *
 * <p>When a rule in the WORKSPACE file is changed, the corresponding
 * {@link RepositoryDirectoryValue} is invalidated using the usual Skyframe route.
 */
public abstract class RepositoryFunction {

  protected Map<String, String> clientEnvironment;

  /**
   * Exception thrown when something goes wrong accessing a remote repository.
   *
   * <p>This exception should be used by child classes to limit the types of exceptions
   * {@link RepositoryDelegatorFunction} has to know how to catch.</p>
   */
  public static class RepositoryFunctionException extends SkyFunctionException {

    public RepositoryFunctionException(NoSuchPackageException cause, Transience transience) {
      super(cause, transience);
    }
    /**
     * Error reading or writing to the filesystem.
     */
    public RepositoryFunctionException(IOException cause, Transience transience) {
      super(cause, transience);
    }

    /**
     * For errors in WORKSPACE file rules (e.g., malformed paths or URLs).
     */
    public RepositoryFunctionException(EvalException cause, Transience transience) {
      super(cause, transience);
    }
  }
  /**
   * Exception thrown when something a repository rule cannot be found.
   */
  public static final class RepositoryNotFoundException extends RepositoryFunctionException {

    public RepositoryNotFoundException(String repositoryName) {
      super(
          new BuildFileContainsErrorsException(
              LabelConstants.EXTERNAL_PACKAGE_IDENTIFIER,
              "The repository named '" + repositoryName + "' could not be resolved"),
          Transience.PERSISTENT);
    }
  }

  /**
   * An exception thrown when a dependency is missing to notify the SkyFunction from an evaluation.
   */
  protected static class RepositoryMissingDependencyException extends EvalException {
    RepositoryMissingDependencyException() {
      super(Location.BUILTIN, "Internal exception");
    }

    @Override
    protected boolean canBeAddedToStackTrace() {
      return false; // to avoid polluting the log with internal cause information
    }
  }

  /**
   * repository functions can throw the result of this function to notify the RepositoryFunction
   * that a dependency was missing and the evaluation of the function must be restarted.
   */
  public static EvalException restart() {
    return new RepositoryMissingDependencyException();
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
      throws SkyFunctionException, InterruptedException;

  @SuppressWarnings("unchecked")
  private static Iterable<String> getEnviron(Rule rule) {
    if (rule.isAttrDefined("$environ", Type.STRING_LIST)) {
      return (Iterable<String>) rule.getAttributeContainer().getAttr("$environ");
    }
    return ImmutableList.of();
  }

  /**
   * Verify the data provided by the marker file to check if a refetch is needed. Returns true if
   * the data is up to date and no refetch is needed and false if the data is obsolete and a refetch
   * is needed.
   */
  public boolean verifyMarkerData(Rule rule, Map<String, String> markerData, Environment env)
      throws InterruptedException, RepositoryFunctionException {
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
        rootedPath = getRootedPathFromLabel(Label.parseAbsolute(fileKey, ImmutableMap.of()), env);
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
   * Convert to a @{link com.google.devtools.build.lib.skyframe.FileValue} to a String appropriate
   * for placing in a repository marker file.
   *
   * @param fileValue The value to convert. It must correspond to a regular file.
   */
  public static String fileValueToMarkerValue(FileValue fileValue) throws IOException {
    Preconditions.checkArgument(fileValue.isFile() && !fileValue.isSpecialFile());
    // Return the file content digest in hex. fileValue may or may not have the digest available.
    byte[] digest = ((RegularFileStateValue) fileValue.realFileStateValue()).getDigest();
    if (digest == null) {
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
    // Look for package.
    if (label.getPackageIdentifier().getRepository().isDefault()) {
      try {
        label = Label.create(label.getPackageIdentifier().makeAbsolute(), label.getName());
      } catch (LabelSyntaxException e) {
        throw new AssertionError(e); // Can't happen because the input label is valid
      }
    }
    SkyKey pkgSkyKey = PackageLookupValue.key(label.getPackageIdentifier());
    PackageLookupValue pkgLookupValue = (PackageLookupValue) env.getValue(pkgSkyKey);
    if (pkgLookupValue == null) {
      throw RepositoryFunction.restart();
    }
    if (!pkgLookupValue.packageExists()) {
      String message = pkgLookupValue.getErrorMsg();
      if (pkgLookupValue == PackageLookupValue.NO_BUILD_FILE_VALUE) {
        message = PackageLookupFunction.explainNoBuildFileValue(label.getPackageIdentifier(), env);
      }
      throw new EvalException(
          Location.BUILTIN, "Unable to load package for " + label + ": " + message);
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
  protected Map<String, String> declareEnvironmentDependencies(
      Map<String, String> markerData, Environment env, Iterable<String> keys)
      throws InterruptedException {
    Map<String, String> environ = ActionEnvironmentFunction.getEnvironmentView(env, keys);

    // Returns true if there is a null value and we need to wait for some dependencies.
    if (environ == null) {
      return null;
    }

    Map<String, String> repoEnvOverride = PrecomputedValue.REPO_ENV.get(env);
    if (repoEnvOverride == null) {
      return null;
    }

    Map<String, String> repoEnv = new LinkedHashMap<String, String>(environ);
    for (Map.Entry<String, String> value : repoEnvOverride.entrySet()) {
      repoEnv.put(value.getKey(), value.getValue());
    }

    // Add the dependencies to the marker file
    for (Map.Entry<String, String> value : repoEnv.entrySet()) {
      markerData.put("ENV:" + value.getKey(), value.getValue());
    }

    return repoEnv;
  }

  /**
   * Verify marker data previously saved by
   * {@link #declareEnvironmentDependencies(Map, Environment, Iterable)}. This function is to be
   * called from a {@link #verifyMarkerData(Rule, Map, Environment)} function to verify the values
   * for environment variables.
   */
  protected boolean verifyEnvironMarkerData(Map<String, String> markerData, Environment env,
      Iterable<String> keys) throws InterruptedException {
    Map<String, String> environ = ActionEnvironmentFunction.getEnvironmentView(env, keys);
    if (env.valuesMissing()) {
      return false; // Returns false so caller knows to return immediately
    }

    Map<String, String> repoEnvOverride = PrecomputedValue.REPO_ENV.get(env);
    if (repoEnvOverride == null) {
      return false;
    }

    Map<String, String> repoEnv = new LinkedHashMap<>(environ);
    for (Map.Entry<String, String> value : repoEnvOverride.entrySet()) {
      repoEnv.put(value.getKey(), value.getValue());
    }

    // Verify that all environment variable in the marker file are also in keys
    for (String key : markerData.keySet()) {
      if (key.startsWith("ENV:") && !repoEnv.containsKey(key.substring(4))) {
        return false;
      }
    }
    // Now verify the values of the marker data
    for (Map.Entry<String, String> value : repoEnv.entrySet()) {
      if (!markerData.containsKey("ENV:" + value.getKey())) {
        return false;
      }
      String markerValue = markerData.get("ENV:" + value.getKey());
      if (!Objects.equals(markerValue, value.getValue())) {
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
      FileSystemUtils.createDirectoryAndParents(repositoryDirectory);
    } catch (IOException e) {
      throw new RepositoryFunctionException(e, Transience.TRANSIENT);
    }

    // Add x/WORKSPACE.
    createWorkspaceFile(repositoryDirectory, rule.getTargetKind(), rule.getName());
    return repositoryDirectory;
  }

  public static void createWorkspaceFile(
      Path repositoryDirectory, String ruleKind, String ruleName)
      throws RepositoryFunctionException {
    try {
      Path workspaceFile = repositoryDirectory.getRelative(LabelConstants.WORKSPACE_FILE_NAME);
      FileSystemUtils.writeContent(workspaceFile, Charset.forName("UTF-8"),
          String.format("# DO NOT EDIT: automatically generated WORKSPACE file for %s\n"
              + "workspace(name = \"%s\")\n", ruleKind, ruleName));
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
  protected static PathFragment getTargetPath(String userDefinedPath, Path workspace)
      throws RepositoryFunctionException {
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
      FileSystemUtils.createDirectoryAndParents(repositoryDirectory);
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

  static void createSymbolicLink(Path from, Path to)
      throws RepositoryFunctionException {
    try {
      // Remove not-symlinks that are already there.
      if (from.exists()) {
        from.delete();
      }
      FileSystemUtils.ensureSymbolicLink(from, to);
    } catch (IOException e) {
      throw new RepositoryFunctionException(
          new IOException(String.format("Error creating symbolic link from %s to %s: %s",
              from, to, e.getMessage())), Transience.TRANSIENT);
    }
  }

  /**
   * Adds the repository's directory to the graph and, if it's a symlink, resolves it to an actual
   * directory.
   */
  @Nullable
  protected static FileValue getRepositoryDirectory(Path repositoryDirectory, Environment env)
      throws RepositoryFunctionException, InterruptedException {
    SkyKey outputDirectoryKey =
        FileValue.key(
            RootedPath.toRootedPath(
                Root.fromPath(repositoryDirectory), PathFragment.EMPTY_FRAGMENT));
    FileValue value;
    try {
      value = (FileValue) env.getValueOrThrow(outputDirectoryKey, IOException.class);
    } catch (IOException e) {
      throw new RepositoryFunctionException(
          new IOException("Could not access " + repositoryDirectory + ": " + e.getMessage()),
          Transience.PERSISTENT);
    }
    return value;
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
      RootedPath rootedPath,
      boolean isDirectory,
      BlazeDirectories directories,
      Environment env,
      ExternalPackageHelper externalPackageHelper)
      throws InterruptedException {
    Path externalRepoDir = getExternalRepositoryDirectory(directories);
    PathFragment repositoryPath = rootedPath.asPath().relativeTo(externalRepoDir);
    if (repositoryPath.isEmpty()) {
      // We are the top of the repository path (<outputBase>/external), not in an actual external
      // repository path.
      return;
    }
    String repositoryName = repositoryPath.getSegment(0);

    try {
      // Add a dependency to the repository rule. RepositoryDirectoryValue does add this
      // dependency already but we want to catch RepositoryNotFoundException, so invoke
      // #getRuleByName
      // first.
      Rule rule = externalPackageHelper.getRuleByName(repositoryName, env);
      if (rule == null) {
        // Still an override might change the content of the repository.
        RepositoryDelegatorFunction.REPOSITORY_OVERRIDES.get(env);
        return;
      }

      if (isDirectory || repositoryPath.segmentCount() > 1) {
        if (!isDirectory
            && rule.getRuleClass().equals(LocalRepositoryRule.NAME)
            && WorkspaceFileHelper.endsWithWorkspaceFileName(repositoryPath)) {
          // Ignore this, there is a dependency from LocalRepositoryFunction->WORKSPACE file already
          return;
        }

        // For all files under the repository directory, depend on the actual RepositoryDirectory
        // function so we get invalidation when the repository is fetched.
        // For the repository directory itself, we cannot depends on the RepositoryDirectoryValue
        // (cycle).
        env.getValue(
            RepositoryDirectoryValue.key(
                RepositoryName.createFromValidStrippedName(repositoryName)));
      } else {
        // Invalidate external/<repo> if the repository overrides change.
        RepositoryDelegatorFunction.REPOSITORY_OVERRIDES.get(env);
      }
    } catch (ExternalRuleNotFoundException ex) {
      // The repository we are looking for does not exist so we should depend on the whole
      // WORKSPACE file. In that case, the call to RepositoryFunction#getRuleByName(String,
      // Environment)
      // already requested all repository functions from the WORKSPACE file from Skyframe as part
      // of the resolution.
      //
      // Alternatively, the repository might still be provided by an override. Therefore, in
      // any case, register the dependency on the repository overrides.
      RepositoryDelegatorFunction.REPOSITORY_OVERRIDES.get(env);
    } catch (ExternalPackageException ex) {
      // This should never happen.
      throw new IllegalStateException(
          "Repository " + repositoryName + " cannot be resolved for path " + rootedPath, ex);
    }
  }

  /**
   * For paths that are under managed directories, we require that the corresponding FileStateValue
   * or DirectoryListingStateValue is evaluated only after RepositoryDirectoryValue is evaluated.
   * This way we guarantee that the repository rule is given a chance to update the managed
   * directory before the files under the managed directory are accessed.
   *
   * <p>We do not need to require anything else (comparing to dependencies required for external
   * repositories files), as overriding external repositories with managed directories is currently
   * forbidden; also, we do not have do perform special checks for local_repository targets, since
   * such targets cannot have managed directories by definition.
   */
  public static void addManagedDirectoryDependencies(RepositoryName repositoryName, Environment env)
      throws InterruptedException {
    env.getValue(RepositoryDirectoryValue.key(repositoryName));
  }

  /**
   * Sets up a mapping of environment variables to use.
   */
  public void setClientEnvironment(Map<String, String> clientEnvironment) {
    this.clientEnvironment = clientEnvironment;
  }

  /**
   * Returns the RuleDefinition class for this type of repository.
   */
  public abstract Class<? extends RuleDefinition> getRuleDefinition();
}
