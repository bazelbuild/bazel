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

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.PackageIdentifier.RepositoryName;
import com.google.devtools.build.lib.packages.AggregatingAttributeMapper;
import com.google.devtools.build.lib.packages.BuildFileContainsErrorsException;
import com.google.devtools.build.lib.packages.BuildFileNotFoundException;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.skyframe.FileSymlinkException;
import com.google.devtools.build.lib.skyframe.FileValue;
import com.google.devtools.build.lib.skyframe.InconsistentFilesystemException;
import com.google.devtools.build.lib.skyframe.PackageValue;
import com.google.devtools.build.lib.skyframe.RepositoryValue;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;

import java.io.IOException;
import java.nio.charset.Charset;

import javax.annotation.Nullable;

/**
 * Parent class for repository-related Skyframe functions.
 */
public abstract class RepositoryFunction implements SkyFunction {
  /**
   * Exception thrown when something goes wrong accessing a remote repository.
   *
   * <p>This exception should be used by child classes to limit the types of exceptions
   * {@link RepositoryDelegatorFunction} has to know how to catch.</p>
   */
  public static final class RepositoryFunctionException extends SkyFunctionException {
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

  private BlazeDirectories directories;

  protected FileValue prepareLocalRepositorySymlinkTree(Rule rule, Environment env)
      throws RepositoryFunctionException {
    Path repositoryDirectory = getExternalRepositoryDirectory().getRelative(rule.getName());
    try {
      FileSystemUtils.deleteTree(repositoryDirectory);
      FileSystemUtils.createDirectoryAndParents(repositoryDirectory);
    } catch (IOException e) {
      throw new RepositoryFunctionException(e, Transience.TRANSIENT);
    }
    FileValue directoryValue = getRepositoryDirectory(repositoryDirectory, env);

    if (directoryValue == null) {
      return null;
    }

    // Add x/WORKSPACE.
    createWorkspaceFile(repositoryDirectory, rule);
    return directoryValue;
  }

  protected void createWorkspaceFile(Path repositoryDirectory, Rule rule)
      throws RepositoryFunctionException {
    try {
      Path workspaceFile = repositoryDirectory.getRelative("WORKSPACE");
      FileSystemUtils.writeContent(workspaceFile, Charset.forName("UTF-8"),
          String.format("# DO NOT EDIT: automatically generated WORKSPACE file for %s\n", rule));
    } catch (IOException e) {
      throw new RepositoryFunctionException(e, Transience.TRANSIENT);
    }
  }

  protected RepositoryValue writeBuildFile(FileValue directoryValue, String contents)
      throws RepositoryFunctionException {
    Path buildFilePath = directoryValue.realRootedPath().asPath().getRelative("BUILD");
    try {
      FileSystemUtils.writeContentAsLatin1(buildFilePath, contents);
    } catch (IOException e) {
      throw new RepositoryFunctionException(e, Transience.TRANSIENT);
    }

    return RepositoryValue.create(directoryValue);
  }

  /**
   * Symlinks a BUILD file from the local filesystem into the external repository's root.
   * @param rule the rule that declares the build_file path.
   * @param workspaceDirectory the workspace root for the build.
   * @param directoryValue the FileValue corresponding to the external repository's root directory.
   * @param env the Skyframe environment.
   * @return the file value of the symlink created.
   * @throws RepositoryFunctionException if the BUILD file specified does not exist or cannot be
   *         linked.
   */
  protected RepositoryValue symlinkBuildFile(
      Rule rule, Path workspaceDirectory, FileValue directoryValue, Environment env)
      throws RepositoryFunctionException {
    AggregatingAttributeMapper mapper = AggregatingAttributeMapper.of(rule);
    PathFragment buildFile = new PathFragment(mapper.get("build_file", Type.STRING));
    Path buildFileTarget = workspaceDirectory.getRelative(buildFile);
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
      rootedBuild = RootedPath.toRootedPath(workspaceDirectory, buildFile);
    }
    SkyKey buildFileKey = FileValue.key(rootedBuild);
    FileValue buildFileValue;
    try {
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

    Path buildFilePath = directoryValue.realRootedPath().asPath().getRelative("BUILD");
    if (createSymbolicLink(buildFilePath, buildFileTarget, env) == null) {
      return null;
    }
    return RepositoryValue.createNew(directoryValue, buildFileValue);
  }

  protected static PathFragment getTargetPath(Rule rule) throws RepositoryFunctionException {
    AggregatingAttributeMapper mapper = AggregatingAttributeMapper.of(rule);
    String path = mapper.get("path", Type.STRING);
    PathFragment pathFragment = new PathFragment(path);
    if (!pathFragment.isAbsolute()) {
      throw new RepositoryFunctionException(
          new EvalException(
              rule.getLocation(),
              "In " + rule + " the 'path' attribute must specify an absolute path"),
          Transience.PERSISTENT);
    }

    return pathFragment;
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
      Path repositoryDirectory, Path targetDirectory, Environment env)
      throws RepositoryFunctionException {
    try {
      for (Path target : targetDirectory.getDirectoryEntries()) {
        Path symlinkPath =
            repositoryDirectory.getRelative(target.getBaseName());
        if (createSymbolicLink(symlinkPath, target, env) == null) {
          return false;
        }
      }
    } catch (IOException e) {
      throw new RepositoryFunctionException(e, Transience.TRANSIENT);
    }

    return true;
  }

  private static FileValue createSymbolicLink(Path from, Path to, Environment env)
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

    SkyKey outputDirectoryKey = FileValue.key(RootedPath.toRootedPath(
        from, PathFragment.EMPTY_FRAGMENT));
    try {
      return (FileValue) env.getValueOrThrow(outputDirectoryKey, IOException.class,
          FileSymlinkException.class, InconsistentFilesystemException.class);
    } catch (IOException | FileSymlinkException | InconsistentFilesystemException e) {
      throw new RepositoryFunctionException(
          new IOException(String.format("Could not access %s: %s", from, e.getMessage())),
          Transience.PERSISTENT);
    }
  }

  @Nullable
  public static Package getExternalPackage(Environment env)
      throws RepositoryFunctionException {
    SkyKey packageKey = PackageValue.key(Package.EXTERNAL_PACKAGE_IDENTIFIER);
    PackageValue packageValue;
    try {
      packageValue = (PackageValue) env.getValueOrThrow(packageKey,
          NoSuchPackageException.class);
    } catch (NoSuchPackageException e) {
      throw new RepositoryFunctionException(
          new BuildFileNotFoundException(
              Package.EXTERNAL_PACKAGE_IDENTIFIER, "Could not load //external package"),
          Transience.PERSISTENT);
    }
    if (packageValue == null) {
      return null;
    }

    Package externalPackage = packageValue.getPackage();
    if (externalPackage.containsErrors()) {
      throw new RepositoryFunctionException(
          new BuildFileContainsErrorsException(
              Package.EXTERNAL_PACKAGE_IDENTIFIER, "Could not load //external package"),
          Transience.PERSISTENT);
    }
    return externalPackage;
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
    Package externalPackage = getExternalPackage(env);
    if (externalPackage == null) {
      return null;
    }

    Rule rule = externalPackage.getRule(repositoryName.strippedName());
    if (rule == null) {
      throw new RepositoryFunctionException(
          new BuildFileContainsErrorsException(
              Package.EXTERNAL_PACKAGE_IDENTIFIER,
              "The repository named '" + repositoryName + "' could not be resolved"),
          Transience.PERSISTENT);
    }
    Preconditions.checkState(ruleClassName == null || rule.getRuleClass().equals(ruleClassName),
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

  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }

  /**
   * Gets Skyframe's name for this.
   */
  public abstract SkyFunctionName getSkyFunctionName();

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
        .getRelative(Package.EXTERNAL_PACKAGE_IDENTIFIER.getPackageFragment());
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
