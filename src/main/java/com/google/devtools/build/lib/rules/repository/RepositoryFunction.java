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
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.packages.BuildFileContainsErrorsException;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.skyframe.DirectoryListingValue;
import com.google.devtools.build.lib.skyframe.FileSymlinkException;
import com.google.devtools.build.lib.skyframe.FileValue;
import com.google.devtools.build.lib.skyframe.InconsistentFilesystemException;
import com.google.devtools.build.lib.skyframe.PackageLookupValue;
import com.google.devtools.build.lib.skyframe.WorkspaceFileValue;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.Map;
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
              Label.EXTERNAL_PACKAGE_IDENTIFIER,
              "The repository named '" + repositoryName + "' could not be resolved"),
          Transience.PERSISTENT);
    }
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
   * <li>When a Skyframe value is missing, fetching must be restarted, thus, in order to avoid doing
   *     duplicate work, it's better to first request the Skyframe dependencies you need and only
   *     then start doing anything costly.
   * <li>The output directory must be populated from within this method (and not from within another
   *     SkyFunction). This is because if it was populated in another SkyFunction, the repository
   *     function would be restarted <b>after</b> that SkyFunction has been run, and it would wipe
   *     the output directory clean.
   * </ul>
   *
   * <p>The {@code markerData} argument can be mutated to augment the data to write to the
   * repository marker file. If any data in the {@code markerData} change between 2 execute of
   * the {@link RepositoryDelegatorFunction} then this should be a reason to invalidate the
   * repository. The {@link #verifyMarkerData(Map<String, String>)} method is responsible for
   * checking the value added to that map when checking the content of a marker file.
   */
  @ThreadSafe
  @Nullable
  public abstract RepositoryDirectoryValue.Builder fetch(Rule rule, Path outputDirectory,
      BlazeDirectories directories, Environment env, Map<String, String> markerData)
      throws SkyFunctionException, InterruptedException;

  /**
   * Verify the data provided by the marker file to check if a refetch is needed. Returns true if
   * the data is up to date and no refetch is needed and false if the data is obsolete and a refetch
   * is needed.
   */
  @Nullable
  public boolean verifyMarkerData(Rule rule, Map<String, String> markerData, Environment env)
      throws InterruptedException {
    return true;
  }

  /**
   * Whether fetching is done using local operations only.
   *
   * <p>If this is false, Bazel may decide not to re-fetch the repository, for example when the
   * {@code --nofetch} command line option is used.
   */
  protected abstract boolean isLocal(Rule rule);

  /**
   * Returns a block of data that must be equal for two Rules for them to be considered the same.
   *
   * <p>This is used for the up-to-dateness check of fetched directory trees. The only reason for
   * this to exist is the {@code maven_server} rule (which should go away, but until then, we need
   * to keep it working somehow)
   */
  protected byte[] getRuleSpecificMarkerData(Rule rule, Environment env)
      throws RepositoryFunctionException, InterruptedException {
    return new byte[] {};
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
      Path workspaceFile = repositoryDirectory.getRelative("WORKSPACE");
      FileSystemUtils.writeContent(workspaceFile, Charset.forName("UTF-8"),
          String.format("# DO NOT EDIT: automatically generated WORKSPACE file for %s\n"
              + "workspace(name = \"%s\")\n", ruleKind, ruleName));
    } catch (IOException e) {
      throw new RepositoryFunctionException(e, Transience.TRANSIENT);
    }
  }

  protected static RepositoryDirectoryValue.Builder writeBuildFile(
      Path repositoryDirectory, String contents) throws RepositoryFunctionException {
    Path buildFilePath = repositoryDirectory.getRelative("BUILD.bazel");
    try {
      // The repository could have an existing BUILD file that's either a regular file (for remote
      // repositories) or a symlink (for local repositories). Either way, we want to remove it and
      // write our own.
      if (buildFilePath.exists(Symlinks.NOFOLLOW)) {
        buildFilePath.delete();
      }
      FileSystemUtils.writeContentAsLatin1(buildFilePath, contents);
    } catch (IOException e) {
      throw new RepositoryFunctionException(e, Transience.TRANSIENT);
    }

    return RepositoryDirectoryValue.builder().setPath(repositoryDirectory);
  }

  @VisibleForTesting
  protected static PathFragment getTargetPath(Rule rule, Path workspace)
      throws RepositoryFunctionException {
    WorkspaceAttributeMapper mapper = WorkspaceAttributeMapper.of(rule);
    String path;
    try {
      path = mapper.get("path", Type.STRING);
    } catch (EvalException e) {
      throw new RepositoryFunctionException(e, Transience.PERSISTENT);
    }
    PathFragment pathFragment = new PathFragment(path);
    return workspace.getRelative(pathFragment).asFragment();
  }

  /**
   * Given a targetDirectory /some/path/to/y that contains files z, w, and v, create the following
   * directory structure:
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
      Path repositoryDirectory, Path targetDirectory)
      throws RepositoryFunctionException {
    try {
      FileSystemUtils.createDirectoryAndParents(repositoryDirectory);
      for (Path target : targetDirectory.getDirectoryEntries()) {
        Path symlinkPath = repositoryDirectory.getRelative(target.getBaseName());
        createSymbolicLink(symlinkPath, target);
      }
    } catch (IOException e) {
      throw new RepositoryFunctionException(e, Transience.TRANSIENT);
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
   * Uses a remote repository name to fetch the corresponding Rule describing how to get it.
   *
   * <p>This should be the unique entry point for resolving a remote repository function.
   */
  @Nullable
  public static Rule getRule(String repository, Environment env)
      throws RepositoryFunctionException, InterruptedException {

    SkyKey packageLookupKey = PackageLookupValue.key(Label.EXTERNAL_PACKAGE_IDENTIFIER);
    PackageLookupValue packageLookupValue = (PackageLookupValue) env.getValue(packageLookupKey);
    if (packageLookupValue == null) {
      return null;
    }
    RootedPath workspacePath = packageLookupValue.getRootedPath(Label.EXTERNAL_PACKAGE_IDENTIFIER);

    SkyKey workspaceKey = WorkspaceFileValue.key(workspacePath);
    do {
      WorkspaceFileValue value = (WorkspaceFileValue) env.getValue(workspaceKey);
      if (value == null) {
        return null;
      }
      Package externalPackage = value.getPackage();
      if (externalPackage.containsErrors()) {
        Event.replayEventsOn(env.getListener(), externalPackage.getEvents());
        throw new RepositoryFunctionException(
            new BuildFileContainsErrorsException(
                Label.EXTERNAL_PACKAGE_IDENTIFIER, "Could not load //external package"),
            Transience.PERSISTENT);
      }
      Rule rule = externalPackage.getRule(repository);
      if (rule != null) {
        return rule;
      }
      workspaceKey = value.next();
    } while (workspaceKey != null);
    throw new RepositoryNotFoundException(repository);
  }

  @Nullable
  public static Rule getRule(String ruleName, @Nullable String ruleClassName, Environment env)
      throws RepositoryFunctionException, InterruptedException {
    try {
      return getRule(RepositoryName.create("@" + ruleName), ruleClassName, env);
    } catch (LabelSyntaxException e) {
      throw new RepositoryFunctionException(
          new IOException("Invalid rule name " + ruleName), Transience.PERSISTENT);
    }
  }

  /**
   * Uses a remote repository name to fetch the corresponding Rule describing how to get it. This
   * should be called from {@link SkyFunction#compute} functions, which should return null if this
   * returns null. If {@code ruleClassName} is set, the rule found must have a matching rule class
   * name.
   */
  @Nullable
  public static Rule getRule(
      RepositoryName repositoryName, @Nullable String ruleClassName, Environment env)
      throws RepositoryFunctionException, InterruptedException {
    Rule rule = getRule(repositoryName.strippedName(), env);
    Preconditions.checkState(
        rule == null || ruleClassName == null || rule.getRuleClass().equals(ruleClassName),
        "Got %s, was expecting a %s", rule, ruleClassName);
    return rule;
  }

  /**
   * Adds the repository's directory to the graph and, if it's a symlink, resolves it to an actual
   * directory.
   */
  @Nullable
  protected static FileValue getRepositoryDirectory(Path repositoryDirectory, Environment env)
      throws RepositoryFunctionException, InterruptedException {
    SkyKey outputDirectoryKey = FileValue.key(RootedPath.toRootedPath(
        repositoryDirectory, PathFragment.EMPTY_FRAGMENT));
    FileValue value;
    try {
      value = (FileValue) env.getValueOrThrow(outputDirectoryKey, IOException.class,
          FileSymlinkException.class, InconsistentFilesystemException.class);
    } catch (IOException | FileSymlinkException | InconsistentFilesystemException e) {
      throw new RepositoryFunctionException(
          new IOException("Could not access " + repositoryDirectory + ": " + e.getMessage()),
          Transience.PERSISTENT);
    }
    return value;
  }

  protected static Path getExternalRepositoryDirectory(BlazeDirectories directories) {
    return directories.getOutputBase().getRelative(Label.EXTERNAL_PACKAGE_NAME);
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
      throws IOException, InterruptedException {
    Path externalRepoDir = getExternalRepositoryDirectory(directories);
    PathFragment repositoryPath = rootedPath.asPath().relativeTo(externalRepoDir);
    if (repositoryPath.segmentCount() == 0) {
      // We are the top of the repository path (<outputBase>/external), not in an actual external
      // repository path.
      return;
    }
    String repositoryName = repositoryPath.getSegment(0);

    Rule repositoryRule;
    try {
      repositoryRule = RepositoryFunction.getRule(repositoryName, env);
    } catch (RepositoryFunction.RepositoryNotFoundException ex) {
      // The repository we are looking for does not exist so we should depend on the whole
      // WORKSPACE file. In that case, the call to RepositoryFunction#getRule(String, Environment)
      // already requested all repository functions from the WORKSPACE file from Skyframe as part
      // of the resolution. Therefore we are safe to ignore that Exception.
      return;
    } catch (RepositoryFunction.RepositoryFunctionException ex) {
      // This should never happen.
      throw new IllegalStateException(
          "Repository " + repositoryName + " cannot be resolved for path " + rootedPath, ex);
    }
    if (repositoryRule == null) {
      return;
    }

    // new_local_repository needs a dependency on the directory that `path` points to, as the
    // external/repo-name DirStateValue has a logical dependency on that directory that is not
    // reflected in the SkyFrame tree, since it's not symlinked to it or anything.
    // new_local_repository is responsible for verifying that the path exists and is a directory.
    if (repositoryRule.getRuleClass().equals(NewLocalRepositoryRule.NAME)
        && repositoryPath.segmentCount() == 1) {
      PathFragment pathDir;
      try {
        pathDir = RepositoryFunction.getTargetPath(
            repositoryRule, directories.getWorkspace());
      } catch (RepositoryFunctionException e) {
        throw new IOException(e.getMessage());
      }
      FileSystem fs = directories.getWorkspace().getFileSystem();
      SkyKey dirKey = DirectoryListingValue.key(
          RootedPath.toRootedPath(fs.getRootDirectory(), fs.getPath(pathDir)));
      try {
        env.getValueOrThrow(
            dirKey, IOException.class, FileSymlinkException.class,
            InconsistentFilesystemException.class);
      } catch (FileSymlinkException | InconsistentFilesystemException e) {
        throw new IOException(e.getMessage());
      }
    }
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
