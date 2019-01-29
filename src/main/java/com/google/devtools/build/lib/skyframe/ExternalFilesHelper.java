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
package com.google.devtools.build.lib.skyframe;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadCompatible;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.rules.repository.RepositoryFunction;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction;
import java.io.IOException;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.logging.Logger;

/** Common utilities for dealing with paths outside the package roots. */
public class ExternalFilesHelper {
  private static final boolean IN_TEST = System.getenv("TEST_TMPDIR") != null;

  private static final Logger logger = Logger.getLogger(ExternalFilesHelper.class.getName());

  private final AtomicReference<PathPackageLocator> pkgLocator;
  private final ExternalFileAction externalFileAction;
  private final BlazeDirectories directories;
  private final int maxNumExternalFilesToLog;
  private final AtomicInteger numExternalFilesLogged = new AtomicInteger(0);

  // These variables are set to true from multiple threads, but only read in the main thread.
  // So volatility or an AtomicBoolean is not needed.
  private boolean anyOutputFilesSeen = false;
  private boolean anyNonOutputExternalFilesSeen = false;

  private ExternalFilesHelper(
      AtomicReference<PathPackageLocator> pkgLocator,
      ExternalFileAction externalFileAction,
      BlazeDirectories directories,
      int maxNumExternalFilesToLog) {
    this.pkgLocator = pkgLocator;
    this.externalFileAction = externalFileAction;
    this.directories = directories;
    this.maxNumExternalFilesToLog = maxNumExternalFilesToLog;
  }

  public static ExternalFilesHelper create(
      AtomicReference<PathPackageLocator> pkgLocator,
      ExternalFileAction externalFileAction,
      BlazeDirectories directories) {
    return IN_TEST
        ? createForTesting(pkgLocator, externalFileAction, directories)
        : new ExternalFilesHelper(
            pkgLocator,
            externalFileAction,
            directories,
            /*maxNumExternalFilesToLog=*/ 100);
  }

  public static ExternalFilesHelper createForTesting(
      AtomicReference<PathPackageLocator> pkgLocator,
      ExternalFileAction externalFileAction,
      BlazeDirectories directories) {
    return new ExternalFilesHelper(
        pkgLocator,
        externalFileAction,
        directories,
        // These log lines are mostly spam during unit and integration tests.
        /*maxNumExternalFilesToLog=*/ 0);
  }


  /**
   * The action to take when an external path is encountered. See {@link FileType} for the
   * definition of "external".
   */
  public enum ExternalFileAction {
    /**
     * For paths of type {@link FileType#EXTERNAL_REPO}, introduce a Skyframe dependency on the
     * 'external' package.
     *
     * <p>This is the default for Bazel, since it's required for correctness of the external
     * repositories feature.
     */
    DEPEND_ON_EXTERNAL_PKG_FOR_EXTERNAL_REPO_PATHS,

    /**
     * For paths of type {@link FileType#EXTERNAL} or {@link FileType#OUTPUT}, assume the path does
     * not exist and will never exist.
     */
    ASSUME_NON_EXISTENT_AND_IMMUTABLE_FOR_EXTERNAL_PATHS,
  }

  /** Classification of a path encountered by Bazel. */
  public enum FileType {
    /** A path inside the package roots or in an external repository. */
    INTERNAL,

    /**
     * A non {@link #EXTERNAL_REPO} path outside the package roots about which we may make no other
     * assumptions.
     */
    EXTERNAL,

    /**
     * A path in Bazel's output tree that's a proper output of an action (*not* a source file in an
     * external repository). Such files are theoretically mutable, but certain Bazel flags may tell
     * Bazel to assume these paths are immutable.
     *
     * <p>Note that {@link ExternalFilesHelper#maybeHandleExternalFile} is only used for {@link
     * FileStateValue} and {@link DirectoryListingStateValue}, and also note that output files do
     * not normally have corresponding {@link FileValue} instances (and thus also {@link
     * FileStateValue} instances) in the Skyframe graph ({@link ArtifactFunction} only uses {@link
     * FileValue}s for source files). But {@link FileStateValue}s for output files can still make
     * their way into the Skyframe graph if e.g. a source file is a symlink to an output file.
     */
    // TODO(nharmata): Consider an alternative design where we have an OutputFileDiffAwareness. This
    // could work but would first require that we clean up all RootedPath usage.
    OUTPUT,

    /**
     * A path in the part of Bazel's output tree that contains (/ symlinks to) to external
     * repositories.
     */
    EXTERNAL_REPO,
  }

  /**
   * Thrown by {@link #maybeHandleExternalFile} when an applicable path is processed (see
   * {@link ExternalFileAction#ASSUME_NON_EXISTENT_AND_IMMUTABLE_FOR_EXTERNAL_PATHS}.
   */
  static class NonexistentImmutableExternalFileException extends Exception {
  }

  static class ExternalFilesKnowledge {
    final boolean anyOutputFilesSeen;
    final boolean anyNonOutputExternalFilesSeen;

    private ExternalFilesKnowledge(boolean anyOutputFilesSeen,
        boolean anyNonOutputExternalFilesSeen) {
      this.anyOutputFilesSeen = anyOutputFilesSeen;
      this.anyNonOutputExternalFilesSeen = anyNonOutputExternalFilesSeen;
    }
  }

  @ThreadCompatible
  ExternalFilesKnowledge getExternalFilesKnowledge() {
    return new ExternalFilesKnowledge(anyOutputFilesSeen, anyNonOutputExternalFilesSeen);
  }

  @ThreadCompatible
  void setExternalFilesKnowledge(ExternalFilesKnowledge externalFilesKnowledge) {
    anyOutputFilesSeen = externalFilesKnowledge.anyOutputFilesSeen;
    anyNonOutputExternalFilesSeen = externalFilesKnowledge.anyNonOutputExternalFilesSeen;
  }

  ExternalFilesHelper cloneWithFreshExternalFilesKnowledge() {
    return new ExternalFilesHelper(
        pkgLocator, externalFileAction, directories, maxNumExternalFilesToLog);
  }

  FileType getAndNoteFileType(RootedPath rootedPath) {
    PathPackageLocator packageLocator = pkgLocator.get();
    if (packageLocator.getPathEntries().contains(rootedPath.getRoot())) {
      return FileType.INTERNAL;
    }
    // The outputBase may be null if we're not actually running a build.
    Path outputBase = packageLocator.getOutputBase();
    if (outputBase == null) {
      anyNonOutputExternalFilesSeen = true;
      return FileType.EXTERNAL;
    }
    if (rootedPath.asPath().startsWith(outputBase)) {
      Path externalRepoDir = outputBase.getRelative(LabelConstants.EXTERNAL_PACKAGE_NAME);
      if (rootedPath.asPath().startsWith(externalRepoDir)) {
        anyNonOutputExternalFilesSeen = true;
        return FileType.EXTERNAL_REPO;
      } else {
        anyOutputFilesSeen = true;
        return FileType.OUTPUT;
      }
    }
    anyNonOutputExternalFilesSeen = true;
    return FileType.EXTERNAL;
  }

  /**
   * If this instance is configured with
   * {@link ExternalFileAction#DEPEND_ON_EXTERNAL_PKG_FOR_EXTERNAL_REPO_PATHS} and
   * {@code rootedPath} isn't under a package root then this adds a dependency on the //external
   * package. If the action is
   * {@link ExternalFileAction#ASSUME_NON_EXISTENT_AND_IMMUTABLE_FOR_EXTERNAL_PATHS}, it will throw
   * a {@link NonexistentImmutableExternalFileException} instead.
   */
  @ThreadSafe
  void maybeHandleExternalFile(RootedPath rootedPath, SkyFunction.Environment env)
      throws NonexistentImmutableExternalFileException, IOException, InterruptedException {
    FileType fileType = getAndNoteFileType(rootedPath);
    if (fileType == FileType.INTERNAL) {
      return;
    }
    if (fileType == FileType.OUTPUT || fileType == FileType.EXTERNAL) {
      if (externalFileAction
          == ExternalFileAction.ASSUME_NON_EXISTENT_AND_IMMUTABLE_FOR_EXTERNAL_PATHS) {
        throw new NonexistentImmutableExternalFileException();
      }
      if (fileType == FileType.EXTERNAL
          && numExternalFilesLogged.incrementAndGet() < maxNumExternalFilesToLog) {
        logger.info("Encountered an external path " + rootedPath);
      }
      return;
    }
    Preconditions.checkState(
        externalFileAction == ExternalFileAction.DEPEND_ON_EXTERNAL_PKG_FOR_EXTERNAL_REPO_PATHS,
        externalFileAction);
    RepositoryFunction.addExternalFilesDependencies(rootedPath, directories, env);
  }
}
