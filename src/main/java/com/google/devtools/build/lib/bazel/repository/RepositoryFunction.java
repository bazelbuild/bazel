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

package com.google.devtools.build.lib.bazel.repository;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.packages.BuildFileContainsErrorsException;
import com.google.devtools.build.lib.packages.BuildFileNotFoundException;
import com.google.devtools.build.lib.packages.ExternalPackage;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.PackageIdentifier;
import com.google.devtools.build.lib.packages.PackageIdentifier.RepositoryName;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.skyframe.FileSymlinkCycleException;
import com.google.devtools.build.lib.skyframe.FileValue;
import com.google.devtools.build.lib.skyframe.InconsistentFilesystemException;
import com.google.devtools.build.lib.skyframe.PackageFunction;
import com.google.devtools.build.lib.skyframe.PackageValue;
import com.google.devtools.build.lib.syntax.EvalException;
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

import javax.annotation.Nullable;

/**
 * Parent class for repository-related Skyframe functions.
 */
public abstract class RepositoryFunction implements SkyFunction {
  private static final String EXTERNAL_REPOSITORY_DIRECTORY = ".external-repository";
  private BlazeDirectories directories;

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
    return directories.getOutputBase().getRelative(EXTERNAL_REPOSITORY_DIRECTORY);
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
    SkyKey packageKey = PackageValue.key(
        PackageIdentifier.createInDefaultRepo(PackageFunction.EXTERNAL_PACKAGE_NAME));
    PackageValue packageValue;
    try {
      packageValue = (PackageValue) env.getValueOrThrow(packageKey,
          NoSuchPackageException.class);
    } catch (NoSuchPackageException e) {
      throw new RepositoryFunctionException(
          new BuildFileNotFoundException(
              PackageFunction.EXTERNAL_PACKAGE_NAME, "Could not load //external package"),
          Transience.PERSISTENT);
    }
    if (packageValue == null) {
      return null;
    }
    ExternalPackage externalPackage = (ExternalPackage) packageValue.getPackage();
    Rule rule = externalPackage.getRepositoryInfo(repositoryName);
    if (rule == null) {
      throw new RepositoryFunctionException(
          new BuildFileContainsErrorsException(
              PackageFunction.EXTERNAL_PACKAGE_NAME,
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
   *
   * <p>Also creates a symlink from x/external/x to x, where x is the directory containing a
   * WORKSPACE file. This is used in the execution root.</p>
   */
  @Nullable
  protected static FileValue getRepositoryDirectory(Path repositoryDirectory, Environment env)
      throws RepositoryFunctionException {
    SkyKey outputDirectoryKey = FileValue.key(RootedPath.toRootedPath(
        repositoryDirectory, PathFragment.EMPTY_FRAGMENT));
    FileValue value;
    try {
      value = (FileValue) env.getValueOrThrow(outputDirectoryKey, IOException.class,
          FileSymlinkCycleException.class, InconsistentFilesystemException.class);
    } catch (IOException | FileSymlinkCycleException | InconsistentFilesystemException e) {
      throw new RepositoryFunctionException(
          new IOException("Could not access " + repositoryDirectory + ": " + e.getMessage()),
          Transience.PERSISTENT);
    }

    String targetName = repositoryDirectory.getBaseName();
    try {
      Path backlink = repositoryDirectory.getRelative("external").getRelative(targetName);
      FileSystemUtils.createDirectoryAndParents(backlink.getParentDirectory());
      if (backlink.exists()) {
        backlink.delete();
      }
      backlink.createSymbolicLink(repositoryDirectory);
    } catch (IOException e) {
      throw new RepositoryFunctionException(new IOException(
          "Error creating execution root symlink for " + targetName + ": " + e.getMessage()),
          Transience.TRANSIENT);
    }
    return value;
  }

  /**
   * Exception thrown when something goes wrong accessing a remote repository.
   *
   * <p>This exception should be used by child classes to limit the types of exceptions
   * {@link RepositoryDelegatorFunction} has to know how to catch.</p>
   */
  static final class RepositoryFunctionException extends SkyFunctionException {
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
}
