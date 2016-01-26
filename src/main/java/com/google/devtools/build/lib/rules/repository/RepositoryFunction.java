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
import com.google.devtools.build.lib.packages.AggregatingAttributeMapper;
import com.google.devtools.build.lib.packages.BuildFileContainsErrorsException;
import com.google.devtools.build.lib.packages.BuildFileNotFoundException;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleSerializer;
import com.google.devtools.build.lib.skyframe.FileSymlinkException;
import com.google.devtools.build.lib.skyframe.FileValue;
import com.google.devtools.build.lib.skyframe.InconsistentFilesystemException;
import com.google.devtools.build.lib.skyframe.PackageValue;
import com.google.devtools.build.lib.skyframe.RepositoryValue;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import java.io.IOException;
import java.nio.charset.Charset;
import java.util.Arrays;

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
 * a flag in {@link RepositoryValue} that tells Bazel whether the value in Skyframe is stale
 * according to the value of {@code --nofetch} or not.
 *
 * <p>When a rule in the WORKSPACE file is changed, the corresponding {@link RepositoryValue} is
 * invalidated using the usual Skyframe route.
 */
public abstract class RepositoryFunction {
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

  private BlazeDirectories directories;

  private byte[] computeRuleKey(Rule rule, byte[] ruleSpecificData) {
    return new Fingerprint()
        .addBytes(RuleSerializer.serializeRule(rule).build().toByteArray())
        .addBytes(ruleSpecificData)
        .digestAndReset();
  }

  /**
   * Fetch the remote repository represented by the given rule.
   *
   * <p>When this method is called, it has already been determined that the repository is stale and
   * that it needs to be re-fetched.
   *
   * <p>The {@code env} argument can be used to fetch Skyframe dependencies the repository
   * implementation needs on the following conditions:
   * <ul>
   *   <li>When a Skyframe value is missing, fetching must be restarted, thus, in order to avoid
   *     doing duplicate work, it's better to first request the Skyframe dependencies you need and
   *     only then start doing anything costly.
   *   <li>The output directory must be populated from within this method (and not from within
   *     another SkyFunction). This is because if it was populated in another SkyFunction, the
   *     repository function would be restarted <b>after</b> that SkyFunction has been run, and
   *     it would wipe the output directory clean.
   * </ul>
   */
  @ThreadSafe
  @Nullable
  public abstract SkyValue fetch(Rule rule, Path outputDirectory, Environment env)
      throws SkyFunctionException, InterruptedException;

  /**
   * Whether fetching is done using local operations only.
   *
   * <p>If this is false, Bazel may decide not to re-fetch the repository, for example when the
   * {@code --nofetch} command line option is used.
   */
  protected abstract boolean isLocal();

  /**
   * Returns a block of data that must be equal for two Rules for them to be considered the same.
   *
   * <p>This is used for the up-to-dateness check of fetched directory trees. The only reason for
   * this to exist is the {@code maven_server} rule (which should go away, but until then, we need
   * to keep it working somehow)
   */
  protected byte[] getRuleSpecificMarkerData(Rule rule, Environment env)
    throws RepositoryFunctionException {
    return new byte[] {};
  }

  private Path getMarkerPath(Rule rule) {
    return getExternalRepositoryDirectory().getChild("@" + rule.getName() + ".marker");
  }

  /**
   * Checks if the state of the repository in the file system is consistent with the rule in the
   * WORKSPACE file.
   *
   * <p>Deletes the marker file if not so that no matter what happens after, the state of the file
   * system stays consistent.
   */
  boolean isFilesystemUpToDate(Rule rule, byte[] ruleSpecificData)
      throws RepositoryFunctionException {
    try {
      Path markerPath = getMarkerPath(rule);
      if (!markerPath.exists()) {
        return false;
      }

      boolean result = Arrays.equals(
          computeRuleKey(rule, ruleSpecificData),
          FileSystemUtils.readContent(markerPath));
      if (!result) {
        // So that we are in a consistent state if something happens while fetching the repository
        markerPath.delete();
      }

      return result;

    } catch (IOException e) {
      throw new RepositoryFunctionException(e, Transience.TRANSIENT);
    }
  }

  void writeMarkerFile(Rule rule, byte[] ruleSpecificData)
      throws RepositoryFunctionException {
    try {
      FileSystemUtils.writeContent(getMarkerPath(rule), computeRuleKey(rule, ruleSpecificData));
    } catch (IOException e) {
      throw new RepositoryFunctionException(e, Transience.TRANSIENT);
    }
  }


  protected Path prepareLocalRepositorySymlinkTree(Rule rule, Path repositoryDirectory)
      throws RepositoryFunctionException {
    try {
      FileSystemUtils.createDirectoryAndParents(repositoryDirectory);
    } catch (IOException e) {
      throw new RepositoryFunctionException(e, Transience.TRANSIENT);
    }

    // Add x/WORKSPACE.
    createWorkspaceFile(repositoryDirectory, rule);
    return repositoryDirectory;
  }

  protected void createWorkspaceFile(Path repositoryDirectory, Rule rule)
      throws RepositoryFunctionException {
    try {
      Path workspaceFile = repositoryDirectory.getRelative("WORKSPACE");
      FileSystemUtils.writeContent(workspaceFile, Charset.forName("UTF-8"),
          String.format("# DO NOT EDIT: automatically generated WORKSPACE file for %s\n"
              + "workspace(name = \"%s\")", rule, rule.getName()));
    } catch (IOException e) {
      throw new RepositoryFunctionException(e, Transience.TRANSIENT);
    }
  }

  protected RepositoryValue writeBuildFile(Path repositoryDirectory, String contents)
      throws RepositoryFunctionException {
    Path buildFilePath = repositoryDirectory.getRelative("BUILD");
    try {
      FileSystemUtils.writeContentAsLatin1(buildFilePath, contents);
    } catch (IOException e) {
      throw new RepositoryFunctionException(e, Transience.TRANSIENT);
    }

    return RepositoryValue.create(repositoryDirectory);
  }

  protected FileValue getBuildFileValue(Rule rule, Environment env)
      throws RepositoryFunctionException {
    AggregatingAttributeMapper mapper = AggregatingAttributeMapper.of(rule);
    PathFragment buildFile = new PathFragment(mapper.get("build_file", Type.STRING));
    Path buildFileTarget = directories.getWorkspace().getRelative(buildFile);
    if (!buildFileTarget.exists()) {
      throw new RepositoryFunctionException(
          new EvalException(rule.getLocation(),
              String.format("In %s the 'build_file' attribute does not specify an existing file "
                  + "(%s does not exist)", rule, buildFileTarget)),
          Transience.PERSISTENT);
    }

    RootedPath rootedBuild;
    if (buildFile.isAbsolute()) {
      rootedBuild = RootedPath.toRootedPath(
          buildFileTarget.getParentDirectory(), new PathFragment(buildFileTarget.getBaseName()));
    } else {
      rootedBuild = RootedPath.toRootedPath(directories.getWorkspace(), buildFile);
    }
    SkyKey buildFileKey = FileValue.key(rootedBuild);
    FileValue buildFileValue;
    try {
      // Note that this dependency is, strictly speaking, not necessary: the symlink could simply
      // point to this FileValue and the symlink chasing could be done while loading the package
      // but this results in a nicer error message and it's correct as long as RepositoryFunctions
      // don't write to things in the file system this FileValue depends on. In theory, the latter
      // is possible if the file referenced by build_file is a symlink to somewhere under the
      // external/ directory, but if you do that, you are really asking for trouble.
      buildFileValue = (FileValue) env.getValueOrThrow(buildFileKey, IOException.class,
          FileSymlinkException.class, InconsistentFilesystemException.class);
      if (buildFileValue == null) {
        return null;
      }
    } catch (IOException | FileSymlinkException | InconsistentFilesystemException e) {
      throw new RepositoryFunctionException(
          new IOException("Cannot lookup " + buildFile + ": " + e.getMessage()),
          Transience.TRANSIENT);
    }

    return buildFileValue;
  }

  /**
   * Symlinks a BUILD file from the local filesystem into the external repository's root.
   * @param buildFileValue {@link FileValue} representing the BUILD file to be linked in
   * @param outputDirectory the directory of the remote repository
   * @return the file value of the symlink created.
   * @throws RepositoryFunctionException if the BUILD file specified does not exist or cannot be
   *         linked.
   */
  protected RepositoryValue symlinkBuildFile(FileValue buildFileValue, Path outputDirectory)
      throws RepositoryFunctionException {
    Path buildFilePath = outputDirectory.getRelative("BUILD");
    createSymbolicLink(buildFilePath, buildFileValue.realRootedPath().asPath());
    return RepositoryValue.create(outputDirectory);
  }

  @VisibleForTesting
  protected static PathFragment getTargetPath(Rule rule, Path workspace)
      throws RepositoryFunctionException {
    AggregatingAttributeMapper mapper = AggregatingAttributeMapper.of(rule);
    String path = mapper.get("path", Type.STRING);
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
      FileSystem fs = repositoryDirectory.getFileSystem();
      if (repositoryDirectory.getFileSystem().supportsSymbolicLinksNatively()) {
        for (Path target : targetDirectory.getDirectoryEntries()) {
          Path symlinkPath =
              repositoryDirectory.getRelative(target.getBaseName());
          createSymbolicLink(symlinkPath, target);
        }
      } else {
        FileSystemUtils.copyTreesBelow(targetDirectory, repositoryDirectory);
      }
    } catch (IOException e) {
      throw new RepositoryFunctionException(e, Transience.TRANSIENT);
    }

    return true;
  }

  private static void createSymbolicLink(Path from, Path to)
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
   * This should be the unique entry point for resolving a remote repository function.
   */
  @Nullable
  public static Rule getRule(String repository, Environment env)
      throws RepositoryFunctionException {
    SkyKey packageKey = PackageValue.key(Label.EXTERNAL_PACKAGE_IDENTIFIER);
    PackageValue packageValue;
    try {
      packageValue = (PackageValue) env.getValueOrThrow(packageKey,
          NoSuchPackageException.class);
    } catch (NoSuchPackageException e) {
      throw new RepositoryFunctionException(
          new BuildFileNotFoundException(
              Label.EXTERNAL_PACKAGE_IDENTIFIER, "Could not load //external package"),
          Transience.PERSISTENT);
    }
    if (packageValue == null) {
      return null;
    }

    Package externalPackage = packageValue.getPackage();
    if (externalPackage.containsErrors()) {
      throw new RepositoryFunctionException(
          new BuildFileContainsErrorsException(
              Label.EXTERNAL_PACKAGE_IDENTIFIER, "Could not load //external package"),
          Transience.PERSISTENT);
    }
    Rule rule = externalPackage.getRule(repository);
    if (rule == null) {
      throw new RepositoryNotFoundException(repository);
    }
    return rule;
  }

  @Nullable
  public static Rule getRule(
      String ruleName, @Nullable String ruleClassName, Environment env)
      throws RepositoryFunctionException {
    try {
      return getRule(RepositoryName.create("@" + ruleName), ruleClassName, env);
    } catch (LabelSyntaxException e) {
      throw new RepositoryFunctionException(
          new IOException("Invalid rule name " + ruleName), Transience.PERSISTENT);
    }
  }

  /**
   * Uses a remote repository name to fetch the corresponding Rule describing how to get it.
   * This should be called from {@link SkyFunction#compute} functions, which should return null if
   * this returns null. If {@code ruleClassName} is set, the rule found must have a matching rule
   * class name.
   */
  @Nullable
  public static Rule getRule(
      RepositoryName repositoryName, @Nullable String ruleClassName, Environment env)
      throws RepositoryFunctionException {
    Rule rule = getRule(repositoryName.strippedName(), env);
    Preconditions.checkState(
        rule == null || ruleClassName == null || rule.getRuleClass().equals(ruleClassName),
        "Got %s, was expecting a %s", rule, ruleClassName);
    return rule;
  }

  /**
   * Adds the repository's directory to the graph and, if it's a symlink, resolves it to an
   * actual directory.
   */
  @Nullable
  public static FileValue getRepositoryDirectory(Path repositoryDirectory, Environment env)
      throws RepositoryFunctionException {
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

  /**
   * Sets up output path information.
   */
  public void setDirectories(BlazeDirectories directories) {
    this.directories = directories;
  }

  protected Path getExternalRepositoryDirectory() {
    return RepositoryFunction.getExternalRepositoryDirectory(directories);
  }

  public static Path getExternalRepositoryDirectory(BlazeDirectories directories) {
    return directories
        .getOutputBase()
        .getRelative(Label.EXTERNAL_PATH_PREFIX);
  }

  /**
   * Gets the base directory repositories should be stored in locally.
   */
  protected Path getOutputBase() {
    return directories.getOutputBase();
  }

  /**
   * Gets the directory the WORKSPACE file for the build is in.
   */
  protected Path getWorkspace() {
    return directories.getWorkspace();
  }


  /**
   * Returns the RuleDefinition class for this type of repository.
   */
  public abstract Class<? extends RuleDefinition> getRuleDefinition();
}
