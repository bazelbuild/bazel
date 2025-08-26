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
import com.google.common.collect.Sets;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider.BundledFileSystem;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadCompatible;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.rules.repository.RepositoryDirectoryValue;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.TestType;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import java.io.IOException;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

/** Common utilities for dealing with paths outside the package roots. */
public class ExternalFilesHelper {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private final AtomicReference<PathPackageLocator> pkgLocator;
  private final ExternalFileAction externalFileAction;
  private final BlazeDirectories directories;
  private final int maxNumExternalFilesToLog;
  private final AtomicInteger numExternalFilesLogged = new AtomicInteger(0);
  private static final int MAX_EXTERNAL_FILES_TO_TRACK = 2500;

  // These variables are set to true from multiple threads, but only read in the main thread.
  // So volatility or an AtomicBoolean is not needed.
  private boolean anyOutputFilesSeen = false;
  private boolean tooManyExternalOtherFilesSeen = false;
  private boolean anyFilesInExternalReposSeen = false;

  // This is the set of EXTERNAL_OTHER files.
  private Set<RootedPath> externalOtherFilesSeen = Sets.newConcurrentHashSet();

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
    return TestType.isInTest()
        ? createForTesting(pkgLocator, externalFileAction, directories)
        : new ExternalFilesHelper(
            pkgLocator, externalFileAction, directories, /*maxNumExternalFilesToLog=*/ 100);
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
     * For paths of type {@link FileType#EXTERNAL_OTHER} or {@link FileType#OUTPUT}, assume the path
     * does not exist and will never exist.
     */
    ASSUME_NON_EXISTENT_AND_IMMUTABLE_FOR_EXTERNAL_PATHS,
  }

  /** Classification of a path encountered by Bazel. */
  public enum FileType {
    /**
     * A path to a file located inside of Bazel, e.g. the bundled builtins_bzl root (for {@code
     * --experimental_builtins_bzl_root=%bundled%}) that live in an InMemoryFileSystem. These files
     * must not change throughout the lifetime of the server.
     */
    BUNDLED,

    /** A path inside the package roots. */
    INTERNAL,

    /**
     * A path in Bazel's output tree that's a proper output of an action (*not* a source file in an
     * external repository). Such files are theoretically mutable, but certain Bazel flags may tell
     * Bazel to assume these paths are immutable.
     *
     * <p>Note that {@link ExternalFilesHelper#maybeHandleExternalFile} is only used for {@link
     * com.google.devtools.build.lib.actions.FileStateValue} and {@link DirectoryListingStateValue},
     * and also note that output files do not normally have corresponding {@link
     * com.google.devtools.build.lib.actions.FileValue} instances (and thus also {@link
     * com.google.devtools.build.lib.actions.FileStateValue} instances) in the Skyframe graph
     * ({@link ArtifactFunction} only uses {@link com.google.devtools.build.lib.actions.FileValue}s
     * for source files). But {@link com.google.devtools.build.lib.actions.FileStateValue}s for
     * output files can still make their way into the Skyframe graph if e.g. a source file is a
     * symlink to an output file.
     */
    // TODO(nharmata): Consider an alternative design where we have an OutputFileDiffAwareness. This
    // could work but would first require that we clean up all RootedPath usage.
    OUTPUT,

    /**
     * A path in the part of Bazel's output tree that contains (/ symlinks to) to external
     * repositories. Every such path under the external repository is generated by the execution of
     * the corresponding repository rule, so these paths should not be cached by Skyframe before the
     * RepositoryDirectoryValue is computed.
     */
    EXTERNAL_REPO,

    /**
     * None of the above. We encounter these paths when outputs, source files or external repos
     * symlink to files outside aforementioned Bazel-managed directories. For example, C compilation
     * by the host compiler may depend on /usr/bin/gcc. Bazel makes a best-effort attempt to detect
     * changes in such files.
     */
    EXTERNAL_OTHER,
  }

  /**
   * Thrown by {@link #maybeHandleExternalFile} when an applicable path is processed (see
   * {@link ExternalFileAction#ASSUME_NON_EXISTENT_AND_IMMUTABLE_FOR_EXTERNAL_PATHS}.
   */
  static class NonexistentImmutableExternalFileException extends Exception {
  }

  static class ExternalFilesKnowledge {
    final boolean anyOutputFilesSeen;
    final Set<RootedPath> externalOtherFilesSeen;
    final boolean anyFilesInExternalReposSeen;
    final boolean tooManyExternalOtherFilesSeen;

    private ExternalFilesKnowledge(
        boolean anyOutputFilesSeen,
        Set<RootedPath> externalOtherFilesSeen,
        boolean anyFilesInExternalReposSeen,
        boolean tooManyExternalOtherFilesSeen) {
      this.anyOutputFilesSeen = anyOutputFilesSeen;
      this.externalOtherFilesSeen = externalOtherFilesSeen;
      this.anyFilesInExternalReposSeen = anyFilesInExternalReposSeen;
      this.tooManyExternalOtherFilesSeen = tooManyExternalOtherFilesSeen;
    }
  }

  @ThreadCompatible
  ExternalFilesKnowledge getExternalFilesKnowledge() {
    return new ExternalFilesKnowledge(
        anyOutputFilesSeen,
        externalOtherFilesSeen,
        anyFilesInExternalReposSeen,
        tooManyExternalOtherFilesSeen);
  }

  @ThreadCompatible
  void setExternalFilesKnowledge(ExternalFilesKnowledge externalFilesKnowledge) {
    anyOutputFilesSeen = externalFilesKnowledge.anyOutputFilesSeen;
    externalOtherFilesSeen = externalFilesKnowledge.externalOtherFilesSeen;
    anyFilesInExternalReposSeen = externalFilesKnowledge.anyFilesInExternalReposSeen;
    tooManyExternalOtherFilesSeen = externalFilesKnowledge.tooManyExternalOtherFilesSeen;
  }

  ExternalFilesHelper cloneWithFreshExternalFilesKnowledge() {
    return new ExternalFilesHelper(
        pkgLocator, externalFileAction, directories, maxNumExternalFilesToLog);
  }

  public FileType getAndNoteFileType(RootedPath rootedPath) {
    return getFileTypeAndRepository(rootedPath).getFirst();
  }

  private Pair<FileType, RepositoryName> getFileTypeAndRepository(RootedPath rootedPath) {
    FileType fileType = detectFileType(rootedPath);
    if (fileType == FileType.EXTERNAL_OTHER) {
      if (externalOtherFilesSeen.size() >= MAX_EXTERNAL_FILES_TO_TRACK) {
        tooManyExternalOtherFilesSeen = true;
      } else {
        externalOtherFilesSeen.add(rootedPath);
      }
    }
    if (FileType.EXTERNAL_REPO == fileType) {
      anyFilesInExternalReposSeen = true;
    }
    if (FileType.OUTPUT == fileType) {
      anyOutputFilesSeen = true;
    }
    return Pair.of(fileType, null);
  }

  /**
   * Returns the classification of a path, and updates this instance's state regarding what kinds of
   * paths have been seen.
   */
  private FileType detectFileType(RootedPath rootedPath) {
    if (rootedPath.getRoot().getFileSystem() instanceof BundledFileSystem) {
      return FileType.BUNDLED;
    }
    PathPackageLocator packageLocator = pkgLocator.get();
    if (packageLocator.getPathEntries().contains(rootedPath.getRoot())) {
      return FileType.INTERNAL;
    }
    // The outputBase may be null if we're not actually running a build.
    Path outputBase = packageLocator.getOutputBase();
    if (outputBase == null) {
      return FileType.EXTERNAL_OTHER;
    }
    if (rootedPath.asPath().startsWith(outputBase)) {
      Path externalRepoDir = outputBase.getRelative(LabelConstants.EXTERNAL_REPOSITORY_LOCATION);
      if (rootedPath.asPath().startsWith(externalRepoDir)) {
        return FileType.EXTERNAL_REPO;
      } else {
        return FileType.OUTPUT;
      }
    }
    return FileType.EXTERNAL_OTHER;
  }

  /**
   * If this instance is configured with {@link
   * ExternalFileAction#DEPEND_ON_EXTERNAL_PKG_FOR_EXTERNAL_REPO_PATHS} and {@code rootedPath} isn't
   * under a package root then this adds a dependency on the //external package. If the action is
   * {@link ExternalFileAction#ASSUME_NON_EXISTENT_AND_IMMUTABLE_FOR_EXTERNAL_PATHS}, it will throw
   * a {@link NonexistentImmutableExternalFileException} instead.
   */
  @ThreadSafe
  FileType maybeHandleExternalFile(RootedPath rootedPath, SkyFunction.Environment env)
      throws NonexistentImmutableExternalFileException, IOException, InterruptedException {
    Pair<FileType, RepositoryName> pair = getFileTypeAndRepository(rootedPath);

    FileType fileType = Preconditions.checkNotNull(pair.getFirst());
    switch (fileType) {
      case BUNDLED:
      case INTERNAL:
        break;
      case EXTERNAL_OTHER:
        if (numExternalFilesLogged.incrementAndGet() < maxNumExternalFilesToLog) {
          logger.atInfo().log("Encountered an external path %s", rootedPath);
        }
        // fall through
      case OUTPUT:
        if (externalFileAction
            == ExternalFileAction.ASSUME_NON_EXISTENT_AND_IMMUTABLE_FOR_EXTERNAL_PATHS) {
          throw new NonexistentImmutableExternalFileException();
        }
        break;
      case EXTERNAL_REPO:
        Preconditions.checkState(
            externalFileAction == ExternalFileAction.DEPEND_ON_EXTERNAL_PKG_FOR_EXTERNAL_REPO_PATHS,
            externalFileAction);
        addExternalFilesDependencies(rootedPath, directories, env);
        break;
    }
    return fileType;
  }

  /**
   * For files that are under $OUTPUT_BASE/external, add a dependency on the corresponding repo so
   * that if the repo definition changes, the File/DirectoryStateValue will be re-evaluated.
   *
   * <p>Note that: - We don't add a dependency on the parent directory at the package root boundary,
   * so the only transitive dependencies from files inside the package roots to external files are
   * through symlinks. So the upwards transitive closure of external files is small. - The only way
   * other than external repositories for external source files to get into the skyframe graph in
   * the first place is through symlinks outside the package roots, which we neither want to
   * encourage nor optimize for since it is not common. So the set of external files is small.
   */
  private static void addExternalFilesDependencies(
      RootedPath rootedPath, BlazeDirectories directories, Environment env)
      throws InterruptedException {
    Path externalRepoDir =
        directories.getOutputBase().getRelative(LabelConstants.EXTERNAL_REPOSITORY_LOCATION);
    PathFragment repositoryPath = rootedPath.asPath().relativeTo(externalRepoDir);
    if (repositoryPath.isEmpty()) {
      // We are the top of the repository path (<outputBase>/external), not in an actual external
      // repository path.
      return;
    }
    RepositoryName repositoryName;
    try {
      repositoryName = RepositoryName.create(repositoryPath.getSegment(0));
    } catch (LabelSyntaxException ignored) {
      // The directory of a repository with an invalid name can never exist, so we don't need to
      // add a dependency on it.
      return;
    }
    env.getValue(RepositoryDirectoryValue.key(repositoryName));
  }
}
