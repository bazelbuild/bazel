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
package com.google.devtools.build.lib.actions;

import com.github.benmanes.caffeine.cache.Caffeine;
import com.github.benmanes.caffeine.cache.CaffeineSpec;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.Lists;
import com.google.common.util.concurrent.Striped;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Symlinks;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.locks.Lock;

/**
 * Helper class to create directories for {@linkplain com.google.devtools.build.lib.actions.Action
 * action} outputs.
 */
public final class ActionOutputDirectoryHelper {

  // Used to prevent check-then-act races in #createOutputDirectories. See the comment there for
  // more detail.
  private static final Striped<Lock> outputDirectoryDeletionLock = Striped.lock(64);

  // Directories which are known to be created as regular directories within this invocation. This
  // implies parent directories are also regular directories.
  private final Map<PathFragment, DirectoryState> knownDirectories;

  private enum DirectoryState {
    FOUND,
    CREATED
  }

  public ActionOutputDirectoryHelper(CaffeineSpec cacheBuilderSpec) {
    knownDirectories =
        Caffeine.from(cacheBuilderSpec)
            .initialCapacity(Runtime.getRuntime().availableProcessors())
            .<PathFragment, DirectoryState>build()
            .asMap();
  }

  @VisibleForTesting
  public static ActionOutputDirectoryHelper createForTesting() {
    // Matches the --directory_creation_cache default.
    return new ActionOutputDirectoryHelper(CaffeineSpec.parse("maximumSize=10000"));
  }

  /**
   * Creates output directories, including missing ancestor directories, for the given set of action
   * outputs.
   *
   * <p>This method should only be used with an action filesystem ({@link
   * com.google.devtools.build.lib.vfs.OutputService.ActionFileSystemType}). Otherwise, please use
   * call {@link #createOutputDirectories} to avoid recreating output directories shared by multiple
   * actions.
   *
   * @throws CreateOutputDirectoryException if one of the output directories or one of its ancestor
   *     directories fails to be created
   */
  public void createActionFsOutputDirectories(
      Collection<Artifact> actionOutputs, ArtifactPathResolver artifactPathResolver)
      throws CreateOutputDirectoryException {
    Set<Path> done = new HashSet<>(); // avoid redundant calls for the same directory.
    for (Artifact outputFile : actionOutputs) {
      Path outputDir;
      if (outputFile.isTreeArtifact()) {
        outputDir = artifactPathResolver.toPath(outputFile);
      } else {
        outputDir = artifactPathResolver.toPath(outputFile).getParentDirectory();
      }

      if (done.add(outputDir)) {
        try {
          outputDir.createDirectoryAndParents();
          outputDir.setWritable(true);
          continue;
        } catch (IOException e) {
          /* Fall through to plan B. */
        }

        Path rootPath = artifactPathResolver.convertPath(outputFile.getRoot().getRoot().asPath());
        forceCreateDirectoryAndParents(outputDir, rootPath);
      }
    }
  }

  /**
   * Invalidates the cached creation of tree artifact directories when an action is going to be
   * rewound.
   *
   * <p>We use {@link #knownDirectories} to only create an output directory once per build. With
   * rewinding, actions that output tree artifacts need to recreate the directories because they are
   * deleted as part of the {@link com.google.devtools.build.lib.actions.Action#prepare} step.
   *
   * <p>Note that this does not need to be called if using an in-memory action file system ({@link
   * com.google.devtools.build.lib.vfs.OutputService.ActionFileSystemType#inMemoryFileSystem}).
   */
  public void invalidateTreeArtifactDirectoryCreation(Collection<Artifact> actionOutputs) {
    for (Artifact output : actionOutputs) {
      if (output.isTreeArtifact()) {
        knownDirectories.remove(output.getPath().asFragment());
      }
    }
  }

  /**
   * Creates output directories, including missing ancestor directories, for the given set of action
   * outputs.
   *
   * <p>For a non-tree output, the parent directory is created; for a tree output, the root
   * directory for the tree is created.
   *
   * <p>If a path to be created already exists but is not a directory, it is recursively deleted and
   * an empty directory is created in its place. If the path exists but is a non-writable directory,
   * it is made writable.
   *
   * <p>Already created directories are recorded in {@link #knownDirectories} to avoid recreating
   * them; calling this method a second time for the same directory is a no-op. For this reason,
   * this method should not be used with an action file system ({@link
   * com.google.devtools.build.lib.vfs.OutputService.ActionFileSystemType}), as an output directory
   * shared by multiple actions would only be created in the action filesystem for one of them.
   * Please use {@link #createActionFsOutputDirectories} instead.
   *
   * @throws CreateOutputDirectoryException if one of the output directories or one of its ancestor
   *     directories fails to be created
   */
  public void createOutputDirectories(Collection<Artifact> actionOutputs)
      throws CreateOutputDirectoryException {
    Set<Path> done = new HashSet<>(); // avoid redundant calls for the same directory.
    for (Artifact outputFile : actionOutputs) {
      Path outputDir;
      // Given we know that we are not using action file system, we can get safely get paths
      // directly from the artifacts.
      if (outputFile.isTreeArtifact()) {
        outputDir = outputFile.getPath();
      } else {
        outputDir = outputFile.getPath().getParentDirectory();
      }

      if (done.add(outputDir)) {
        createOutputDirectory(outputDir, outputFile.getRoot().getRoot().asPath());
      }
    }
  }

  /**
   * Creates a writable output directory, including missing ancestor directories.
   *
   * <p>If a path to be created already exists but is not a directory, it is recursively deleted and
   * an empty directory is created in its place. If the path exists but is a non-writable directory,
   * it is made writable.
   *
   * <p>Already created directories are recorded in {@link #knownDirectories} to avoid recreating
   * them; calling this method a second time for the same directory is a no-op. For this reason,
   * this method should not be used with an action file system, as an output directory shared across
   * actions would only be created in the action filesystem for one of them.
   *
   * @throws CreateOutputDirectoryException if the output directory or one of its ancestor
   *     directories fails to be created
   */
  public void createOutputDirectory(Path outputDir, Path rootPath)
      throws CreateOutputDirectoryException {
    try {
      createAndCheckForSymlinks(outputDir, rootPath);
    } catch (IOException e) {
      /* Fall through to plan B. */
      forceCreateDirectoryAndParents(outputDir, rootPath);
    }
  }

  private void forceCreateDirectoryAndParents(Path outputDir, Path rootPath)
      throws CreateOutputDirectoryException {
    // Possibly some direct ancestors are not directories.  In that case, we traverse the
    // ancestors downward, deleting any non-directories. This handles the case where a file
    // becomes a directory. The traversal is done downward because otherwise we may delete
    // files through a symlink in a parent directory. Since Blaze never creates such
    // directories within a build, we have no idea where on disk we're actually deleting.
    //
    // Symlinks should not be followed so in order to clean up symlinks pointing to Fileset
    // outputs from previous builds. See bug [incremental build of Fileset fails if
    // Fileset.out was changed to be a subdirectory of the old value].
    try {
      Path p = rootPath;
      for (String segment : outputDir.relativeTo(p).segments()) {
        p = p.getRelative(segment);

        // This lock ensures that the only thread that observes a filesystem transition in
        // which the path p first exists and then does not is the thread that calls
        // p.delete() and causes the transition.
        //
        // If it were otherwise, then some thread A could test p.exists(), see that it does,
        // then test p.isDirectory(), see that p isn't a directory (because, say, thread
        // B deleted it), and then call p.delete(). That could result in two different kinds
        // of failures:
        //
        // 1) In the time between when thread A sees that p is not a directory and when thread
        // A calls p.delete(), thread B may reach the call to createDirectoryAndParents
        // and create a directory at p, which thread A then deletes. Thread B would then try
        // adding outputs to the directory it thought was there, and fail.
        //
        // 2) In the time between when thread A sees that p is not a directory and when thread
        // A calls p.delete(), thread B may create a directory at p, and then either create a
        // subdirectory beneath it or add outputs to it. Then when thread A tries to delete p,
        // it would fail.
        Lock lock = outputDirectoryDeletionLock.get(p);
        lock.lock();
        try {
          FileStatus stat = p.statIfFound(Symlinks.NOFOLLOW);
          if (stat == null) {
            // Missing entry: Break out and create expected directories.
            break;
          }
          if (stat.isDirectory()) {
            // If this directory used to be a tree artifact it won't be writable.
            p.setWritable(true);
            knownDirectories.put(p.asFragment(), DirectoryState.FOUND);
          } else {
            // p may be a file or symlink (possibly from a Fileset in a previous build).
            p.delete(); // throws IOException
            break;
          }
        } finally {
          lock.unlock();
        }
      }
      outputDir.createDirectoryAndParents();
    } catch (IOException e) {
      throw new CreateOutputDirectoryException(outputDir.asFragment(), e);
    }
  }

  /**
   * Create an output directory and ensure that no symlinks exists between the output root and the
   * output file. These are all expected to be regular directories. Violations of this expectations
   * can only come from state left behind by previous invocations or external filesystem mutation.
   */
  private void createAndCheckForSymlinks(Path dir, Path rootPath) throws IOException {
    PathFragment root = rootPath.asFragment();

    // If the output root has not been created yet, do so now.
    if (!knownDirectories.containsKey(root)) {
      FileStatus stat = rootPath.statNullable(Symlinks.NOFOLLOW);
      if (stat == null) {
        rootPath.createDirectoryAndParents();
        knownDirectories.put(root, DirectoryState.CREATED);
      } else {
        knownDirectories.put(root, DirectoryState.FOUND);
      }
    }

    // Walk up until the first known directory is found (must be root or below).
    List<Path> checkDirs = new ArrayList<>();
    while (!dir.equals(rootPath) && !knownDirectories.containsKey(dir.asFragment())) {
      checkDirs.add(dir);
      dir = dir.getParentDirectory();
    }

    // Check in reverse order (parent directory first):
    // - If symlink -> Exception.
    // - If non-existent -> Create directory and all children.
    boolean parentCreated = knownDirectories.get(dir.asFragment()) == DirectoryState.CREATED;
    for (Path path : Lists.reverse(checkDirs)) {
      if (parentCreated) {
        // If we have created this directory's parent, we know that it doesn't exist or else we
        // would know about it already. Even if a parallel thread has created it in the meantime,
        // createDirectory() will return normally and we can assume that a regular directory exists
        // afterwards.
        path.createDirectory();
        knownDirectories.put(path.asFragment(), DirectoryState.CREATED);
        continue;
      }
      boolean createdNew = path.createWritableDirectory();
      knownDirectories.put(
          path.asFragment(), createdNew ? DirectoryState.CREATED : DirectoryState.FOUND);
    }
  }

  /** An exception that occurred while attempting to create an output directory. */
  public static final class CreateOutputDirectoryException extends IOException {
    private final PathFragment directoryPath;

    private CreateOutputDirectoryException(PathFragment directoryPath, IOException cause) {
      super(cause.getMessage(), cause);
      this.directoryPath = directoryPath;
    }

    /** Returns the path to the output directory for which the exception occurred. */
    public PathFragment getDirectoryPath() {
      return directoryPath;
    }
  }
}
