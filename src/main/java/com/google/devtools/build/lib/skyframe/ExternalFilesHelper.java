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
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Sets;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.actions.FileStateValue;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider.BundledFileSystem;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadCompatible;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.rules.repository.RepositoryDirectoryValue;
import com.google.devtools.build.lib.util.TestType;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyKey;
import java.io.IOException;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Supplier;
import javax.annotation.Nullable;

/** Common utilities for dealing with paths outside the package roots. */
public class ExternalFilesHelper {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private final AtomicReference<PathPackageLocator> pkgLocator;
  private final ExternalFileAction externalFileAction;
  private final BlazeDirectories directories;
  private final Supplier<Path> repoContentsCachePathSupplier;
  private final int maxNumExternalFilesToLog;
  private final AtomicInteger numExternalFilesLogged = new AtomicInteger(0);
  private static final int MAX_EXTERNAL_FILES_TO_TRACK = 2500;

  // These variables are set to true from multiple threads, but only read in the main thread.
  // So volatility or an AtomicBoolean is not needed.
  private boolean anyOutputFilesSeen = false;
  private boolean tooManyExternalOtherFilesSeen = false;
  private boolean anyFilesInExternalReposSeen = false;
  private boolean anyRepoCacheEntriesSeen = false;

  // This is the set of EXTERNAL_OTHER files.
  private Set<RootedPath> externalOtherFilesSeen = Sets.newConcurrentHashSet();

  private ExternalFilesHelper(
      AtomicReference<PathPackageLocator> pkgLocator,
      ExternalFileAction externalFileAction,
      BlazeDirectories directories,
      Supplier<Path> repoContentsCachePathSupplier,
      int maxNumExternalFilesToLog) {
    this.pkgLocator = pkgLocator;
    this.externalFileAction = externalFileAction;
    this.directories = directories;
    this.repoContentsCachePathSupplier = repoContentsCachePathSupplier;
    this.maxNumExternalFilesToLog = maxNumExternalFilesToLog;
  }

  public static ExternalFilesHelper create(
      AtomicReference<PathPackageLocator> pkgLocator,
      ExternalFileAction externalFileAction,
      BlazeDirectories directories,
      Supplier<Path> repoContentsCachePath) {
    return TestType.isInTest()
        ? createForTesting(pkgLocator, externalFileAction, directories, repoContentsCachePath)
        : new ExternalFilesHelper(
            pkgLocator,
            externalFileAction,
            directories,
            repoContentsCachePath,
            /* maxNumExternalFilesToLog= */ 100);
  }

  public static ExternalFilesHelper createForTesting(
      AtomicReference<PathPackageLocator> pkgLocator,
      ExternalFileAction externalFileAction,
      BlazeDirectories directories,
      Supplier<Path> repoContentsCachePath) {
    return new ExternalFilesHelper(
        pkgLocator,
        externalFileAction,
        directories,
        repoContentsCachePath,
        // These log lines are mostly spam during unit and integration tests.
        /* maxNumExternalFilesToLog= */ 0);
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
     * repositories. Every such path under the external repository is generated by the execution of
     * the corresponding repository rule, so these paths should not be cached by Skyframe before the
     * RepositoryDirectoryValue is computed.
     */
    EXTERNAL_REPO,

    /**
     * A top-level directory in the repo contents cache, i.e., either a directory corresponding to a
     * particular entry, predeclared input hash or the root of the repo contents cache itself. Bazel
     * may create these directories when they are found to be missing at the beginning of an
     * invocation and the GC idle task may delete them.
     */
    REPO_CONTENTS_CACHE_TOP_LEVEL_DIRECTORIES,

    /**
     * A file within an entry in the repo contents cache. Bazel creates entries with names that
     * include a fresh UUID and never modifies any files inside an entry directory after it has been
     * created. However, other Bazel instances may delete old entries as part of the GC idle task.
     */
    REPO_CONTENTS_CACHE_ENTRY,

    /**
     * None of the above. We encounter these paths when outputs, source files or external repos
     * symlink to files outside aforementioned Bazel-managed directories. For example, C compilation
     * by the host compiler may depend on /usr/bin/gcc. Bazel makes a best-effort attempt to detect
     * changes in such files.
     */
    EXTERNAL_OTHER;

    /** Whether Bazel may modify files of this type after they have first been tracked. */
    public boolean mayBeModifiedByBazel() {
      return switch (this) {
        // Output files are regularly modified during execution. External repos, including their
        // corresponding directories in the repo contents cache, may be refetched by Bazel if it
        // notices that they have been deleted or may be cleaned up by the GC idle task.
        case OUTPUT, EXTERNAL_REPO, REPO_CONTENTS_CACHE_TOP_LEVEL_DIRECTORIES -> true;
        // Other external files are not managed by Bazel. The contents of entries in the repo
        // contents cache are created under UUIDs and never modified after creation.
        case INTERNAL, BUNDLED, EXTERNAL_OTHER, REPO_CONTENTS_CACHE_ENTRY -> false;
      };
    }

    /**
     * Whether files of this type may belong to an external repository and thus can be passed to
     * {@link ExternalFilesHelper#getRepositoryName(RootedPath)}.
     */
    public boolean mayBelongToExternalRepository() {
      return switch (this) {
        case EXTERNAL_REPO, REPO_CONTENTS_CACHE_TOP_LEVEL_DIRECTORIES, REPO_CONTENTS_CACHE_ENTRY ->
            true;
        case BUNDLED, INTERNAL, OUTPUT, EXTERNAL_OTHER -> false;
      };
    }
  }

  /**
   * Thrown by {@link #maybeHandleExternalFile} when an applicable path is processed (see {@link
   * ExternalFileAction#ASSUME_NON_EXISTENT_AND_IMMUTABLE_FOR_EXTERNAL_PATHS}.
   */
  static class NonexistentImmutableExternalFileException extends Exception {}

  record ExternalFilesKnowledge(
      boolean anyOutputFilesSeen,
      Set<RootedPath> externalOtherFilesSeen,
      boolean anyFilesInExternalReposSeen,
      boolean anyRepoCacheEntriesSeen,
      boolean tooManyExternalOtherFilesSeen) {}

  @ThreadCompatible
  ExternalFilesKnowledge getExternalFilesKnowledge() {
    return new ExternalFilesKnowledge(
        anyOutputFilesSeen,
        externalOtherFilesSeen,
        anyFilesInExternalReposSeen,
        anyRepoCacheEntriesSeen,
        tooManyExternalOtherFilesSeen);
  }

  @ThreadCompatible
  void setExternalFilesKnowledge(ExternalFilesKnowledge externalFilesKnowledge) {
    anyOutputFilesSeen = externalFilesKnowledge.anyOutputFilesSeen;
    externalOtherFilesSeen = externalFilesKnowledge.externalOtherFilesSeen;
    anyFilesInExternalReposSeen = externalFilesKnowledge.anyFilesInExternalReposSeen;
    anyRepoCacheEntriesSeen = externalFilesKnowledge.anyRepoCacheEntriesSeen;
    tooManyExternalOtherFilesSeen = externalFilesKnowledge.tooManyExternalOtherFilesSeen;
  }

  ExternalFilesHelper cloneWithFreshExternalFilesKnowledge() {
    return new ExternalFilesHelper(
        pkgLocator,
        externalFileAction,
        directories,
        repoContentsCachePathSupplier,
        maxNumExternalFilesToLog);
  }

  public FileType getAndNoteFileType(RootedPath rootedPath) {
    FileType fileType = detectFileType(rootedPath);
    switch (fileType) {
      case EXTERNAL_OTHER -> {
        if (externalOtherFilesSeen.size() >= MAX_EXTERNAL_FILES_TO_TRACK) {
          tooManyExternalOtherFilesSeen = true;
        } else {
          externalOtherFilesSeen.add(rootedPath);
        }
      }
      case EXTERNAL_REPO, REPO_CONTENTS_CACHE_ENTRY -> anyFilesInExternalReposSeen = true;
      case REPO_CONTENTS_CACHE_TOP_LEVEL_DIRECTORIES -> anyRepoCacheEntriesSeen = true;
      case OUTPUT -> anyOutputFilesSeen = true;
      case BUNDLED, INTERNAL -> {}
    }
    return fileType;
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
    var repoContentsCachePath = repoContentsCachePathSupplier.get();
    if (repoContentsCachePath != null && rootedPath.asPath().startsWith(repoContentsCachePath)) {
      // This condition covers the following directories and files:
      // <repo contents cache>
      // <repo contents cache>/<repo name>-<predeclared-input-hash>
      // <repo contents cache>/<repo name>-<predeclared-input-hash>/<uuid>
      // <repo contents cache>/<repo name>-<predeclared-input-hash>/<uuid>.recorded_inputs
      return rootedPath.asPath().asFragment().segmentCount()
              <= repoContentsCachePath.asFragment().segmentCount() + 2
          ? FileType.REPO_CONTENTS_CACHE_TOP_LEVEL_DIRECTORIES
          : FileType.REPO_CONTENTS_CACHE_ENTRY;
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
  FileType maybeHandleExternalFile(RootedPath rootedPath, Environment env)
      throws NonexistentImmutableExternalFileException, IOException, InterruptedException {
    FileType fileType = Preconditions.checkNotNull(getAndNoteFileType(rootedPath));
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
        addExternalFilesDependencies(rootedPath, env);
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
  private void addExternalFilesDependencies(RootedPath rootedPath, Environment env)
      throws InterruptedException {
    var repositoryName = getRepositoryName(rootedPath);
    if (repositoryName != null) {
      env.getValue(RepositoryDirectoryValue.key(repositoryName));
    }
  }

  /**
   * For a file of type {@link FileType#EXTERNAL_REPO}, returns the name of the external repository
   * it is in or null if the path is not in a valid external repository.
   */
  @Nullable
  RepositoryName getRepositoryName(RootedPath rootedPath) {
    String repoName;
    if (rootedPath.asPath().startsWith(getExternalDirectory())) {
      PathFragment repositoryPath = rootedPath.asPath().relativeTo(getExternalDirectory());
      if (repositoryPath.isEmpty()) {
        // We are the top of the repository path (<outputBase>/external), not in an actual external
        // repository path.
        return null;
      }
      repoName = repositoryPath.getSegment(0);
    } else {
      PathFragment repoCachePath =
          rootedPath.asPath().relativeTo(repoContentsCachePathSupplier.get());
      if (repoCachePath.isEmpty()) {
        // We are the top of the repo contents cache.
        return null;
      }
      var firstSegment = repoCachePath.getSegment(0);
      var lastDash = firstSegment.lastIndexOf('-');
      if (lastDash <= 0) {
        // The directory of a repository with an invalid name isn't referenced.
        return null;
      }
      repoName = firstSegment.substring(0, lastDash);
    }
    try {
      return RepositoryName.create(repoName);
    } catch (LabelSyntaxException ignored) {
      // The directory of a repository with an invalid name can never exist.
      return null;
    }
  }

  Iterable<SkyKey> getExtraKeysToInvalidate(
      ImmutableMap<RepositoryName, RootedPath> dirtyExternalRepos,
      ExtendedEventHandler eventHandler) {
    dirtyExternalRepos.forEach(
        (repoName, file) -> {
          var fileType = getAndNoteFileType(file);
          if (fileType == FileType.EXTERNAL_REPO
              || fileType == FileType.REPO_CONTENTS_CACHE_ENTRY) {
            eventHandler.handle(
                Event.warn(
                    """
                  Repository '%s' will be fetched again since the file '%s' has been modified \
                  externally. External modifications can lead to incorrect builds.\
                  """
                        .formatted(repoName, file.getRootRelativePath())));
          }
          switch (fileType) {
            case EXTERNAL_REPO -> {
              // Delete the marker file so that invalidating the RepositoryDirectoryValue actually
              // causes a re-fetch of the repository.
              try {
                getExternalDirectory().getRelative(repoName.getMarkerFileName()).delete();
              } catch (IOException e) {
                // Any failure to delete the file should also make it non-readable, so we still
                // achieve our goal of forcing a re-fetch of the repository.
              }
            }
            case REPO_CONTENTS_CACHE_ENTRY -> {
              var cacheRelativePath = file.asPath().relativeTo(repoContentsCachePathSupplier.get());
              if (cacheRelativePath.segmentCount() > 2) {
                var candidateDir =
                    repoContentsCachePathSupplier
                        .get()
                        .getRelative(cacheRelativePath.subFragment(0, 2));
                var recordedInputsFile =
                    candidateDir.replaceName(candidateDir.getBaseName() + ".recorded_inputs");
                try {
                  // TODO: GC the stale directory.
                  recordedInputsFile.delete();
                } catch (IOException e) {
                  // Any failure to delete the file should also make it non-readable, so we still
                  // achieve our goal of forcing a re-fetch of the repository.
                }
              }
            }
          }
        });
    return Iterables.transform(dirtyExternalRepos.keySet(), RepositoryDirectoryValue::key);
  }

  private Path getExternalDirectory() {
    return directories.getOutputBase().getRelative(LabelConstants.EXTERNAL_REPOSITORY_LOCATION);
  }
}
