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

import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadCompatible;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.rules.repository.RepositoryFunction;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction;

import java.util.concurrent.atomic.AtomicReference;

/** Common utilities for dealing with files outside the package roots. */
public class ExternalFilesHelper {
  private final AtomicReference<PathPackageLocator> pkgLocator;
  private final ExternalFileAction externalFileAction;

  // These variables are set to true from multiple threads, but only read in the main thread.
  // So volatility or an AtomicBoolean is not needed.
  private boolean anyOutputFilesSeen = false;
  private boolean anyNonOutputExternalFilesSeen = false;

  /**
   * @param pkgLocator an {@link AtomicReference} to a {@link PathPackageLocator} used to
   *    determine what files are internal.
   * @param errorOnExternalFiles If files outside of package paths should be allowed.
   */
  public ExternalFilesHelper(
      AtomicReference<PathPackageLocator> pkgLocator, boolean errorOnExternalFiles) {
    this(
        pkgLocator,
        errorOnExternalFiles
            ? ExternalFileAction.ERROR_OUT
            : ExternalFileAction.DEPEND_ON_EXTERNAL_PKG_FOR_EXTERNAL_REPO_FILES);
  }

  private ExternalFilesHelper(AtomicReference<PathPackageLocator> pkgLocator,
      ExternalFileAction externalFileAction) {
    this.pkgLocator = pkgLocator;
    this.externalFileAction = externalFileAction;
  }

  private enum ExternalFileAction {
    /** Re-check the files when the WORKSPACE file changes. */
    DEPEND_ON_EXTERNAL_PKG_FOR_EXTERNAL_REPO_FILES,

    /** Throw an exception if there is an external file. */
    ERROR_OUT,
  }

  enum FileType {
    /** A file inside the package roots or in an external repository. */
    INTERNAL,

    /** A file outside the package roots about which we may make no other assumptions. */
    EXTERNAL_MUTABLE,

    /**
     * A file in Bazel's output tree that's a proper output of an action (*not* a source file in an
     * external repository). Such files are theoretically mutable, but certain Blaze flags may tell
     * Blaze to assume these files are immutable.
     *
     * Note that {@link ExternalFilesHelper#maybeHandleExternalFile} is only used for
     * {@link FileStateValue} and {@link DirectoryStateValue}, and also note that output files do
     * not normally have corresponding {@link FileValue} instances (and thus also
     * {@link FileStateValue} instances) in the Skyframe graph ({@link ArtifactFunction} only uses
     * {@link FileValue}s for source files). But {@link FileStateValue}s for output files can still
     * make their way into the Skyframe graph if e.g. a source file is a symlink to an output file.
     */
    // TODO(nharmata): Consider an alternative design where we have an OutputFileDiffAwareness. This
    // could work but would first require that we clean up all RootedPath usage.
    OUTPUT,

    /**
     * A file in the part of Bazel's output tree that contains (/ symlinks to) to external
     * repositories.
     */
    EXTERNAL_REPO,
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
    return new ExternalFilesHelper(pkgLocator, externalFileAction);
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
      return FileType.EXTERNAL_MUTABLE;
    }
    if (rootedPath.asPath().startsWith(outputBase)) {
      Path externalRepoDir = outputBase.getRelative(Label.EXTERNAL_PATH_PREFIX);
      if (rootedPath.asPath().startsWith(externalRepoDir)) {
        anyNonOutputExternalFilesSeen = true;
        return FileType.EXTERNAL_REPO;
      } else {
        anyOutputFilesSeen = true;
        return FileType.OUTPUT;
      }
    }
    anyNonOutputExternalFilesSeen = true;
    return FileType.EXTERNAL_MUTABLE;
  }

  /**
   * If this instance is configured with DEPEND_ON_EXTERNAL_PKG and rootedPath is a file that isn't
   * under a package root then this adds a dependency on the //external package. If the action is
   * ERROR_OUT, it will throw an error instead.
   */
  @ThreadSafe
  public void maybeHandleExternalFile(RootedPath rootedPath, SkyFunction.Environment env)
      throws FileOutsidePackageRootsException {
    FileType fileType = getAndNoteFileType(rootedPath);
    if (fileType == FileType.INTERNAL) {
      return;
    }
    if (fileType == FileType.OUTPUT || fileType == FileType.EXTERNAL_MUTABLE) {
      if (externalFileAction == ExternalFileAction.ERROR_OUT) {
        throw new FileOutsidePackageRootsException(rootedPath);
      }
      return;
    }
    Preconditions.checkState(
        externalFileAction == ExternalFileAction.DEPEND_ON_EXTERNAL_PKG_FOR_EXTERNAL_REPO_FILES,
        externalFileAction);

    // For files that are under $OUTPUT_BASE/external, add a dependency on the corresponding rule
    // so that if the WORKSPACE file changes, the File/DirectoryStateValue will be re-evaluated.
    //
    // Note that:
    // - We don't add a dependency on the parent directory at the package root boundary, so
    // the only transitive dependencies from files inside the package roots to external files
    // are through symlinks. So the upwards transitive closure of external files is small.
    // - The only way other than external repositories for external source files to get into the
    // skyframe graph in the first place is through symlinks outside the package roots, which we
    // neither want to encourage nor optimize for since it is not common. So the set of external
    // files is small.

    Path externalRepoDir = pkgLocator.get().getOutputBase().getRelative(Label.EXTERNAL_PATH_PREFIX);
    PathFragment repositoryPath = rootedPath.asPath().relativeTo(externalRepoDir);
    if (repositoryPath.segmentCount() == 0) {
      // We are the top of the repository path (<outputBase>/external), not in an actual external
      // repository path.
      return;
    }
    String repositoryName = repositoryPath.getSegment(0);

    try {
      RepositoryFunction.getRule(repositoryName, env);
    } catch (RepositoryFunction.RepositoryNotFoundException ex) {
      // The repository we are looking for does not exist so we should depend on the whole
      // WORKSPACE file. In that case, the call to RepositoryFunction#getRule(String, Environment)
      // already requested all repository functions from the WORKSPACE file from Skyframe as part of
      // the resolution. Therefore we are safe to ignore that Exception.
    } catch (RepositoryFunction.RepositoryFunctionException ex) {
      // This should never happen.
      throw new IllegalStateException(
          "Repository " + repositoryName + " cannot be resolved for path " + rootedPath, ex);
    }
  }
}
