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

import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheBuilderSpec;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Lists;
import com.google.common.util.concurrent.Striped;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactPathResolver;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Symlinks;
import java.io.IOException;
import java.util.ArrayList;
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

  ActionOutputDirectoryHelper(CacheBuilderSpec cacheBuilderSpec) {
    knownDirectories =
        CacheBuilder.from(cacheBuilderSpec)
            .concurrencyLevel(Runtime.getRuntime().availableProcessors())
            .<PathFragment, DirectoryState>build()
            .asMap();
  }

  /**
   * Create output directories for an ActionFS. The action-local filesystem starts empty, so we
   * expect the output directory creation to always succeed. There can be no interference from state
   * left behind by prior builds or other actions intra-build.
   */
  void createActionFsOutputDirectories(
      ImmutableSet<Artifact> actionOutputs, ArtifactPathResolver artifactPathResolver)
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
        } catch (IOException e) {
          throw new CreateOutputDirectoryException(outputDir.asFragment(), e);
        }
      }
    }
  }

  /**
   * Creates output directories, with ancestors, for all action outputs.
   *
   * <p>For regular file outputs, creates the parent directories, for tree outputs creates the tree
   * directory (with ancestors in both cases).
   *
   * <p>It is only valid to use this method when not using an action file system, in which case
   * please use {@link #createActionFsOutputDirectories} instead. In particular, {@linkplain
   * #knownDirectories the cache of directories}, shared across actions will cause common
   * directories to be created in action file system for only one of the actions as opposed to all
   * of them.
   */
  void createOutputDirectories(ImmutableSet<Artifact> actionOutputs)
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
        try {
          createAndCheckForSymlinks(outputDir, outputFile);
          continue;
        } catch (IOException e) {
          /* Fall through to plan B. */
        }

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
          Path p = outputFile.getRoot().getRoot().asPath();
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
    }
  }

  /**
   * Create an output directory and ensure that no symlinks exists between the output root and the
   * output file. These are all expected to be regular directories. Violations of this expectations
   * can only come from state left behind by previous invocations or external filesystem mutation.
   */
  private void createAndCheckForSymlinks(Path dir, Artifact outputFile) throws IOException {
    Path rootPath = outputFile.getRoot().getRoot().asPath();
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
      FileStatus stat = path.statNullable(Symlinks.NOFOLLOW);
      if (stat != null && !stat.isDirectory()) {
        throw new IOException(dir + " is not a regular directory");
      }
      if (stat == null) {
        parentCreated = true;
        path.createDirectory();
        knownDirectories.put(path.asFragment(), DirectoryState.CREATED);
      } else {
        knownDirectories.put(path.asFragment(), DirectoryState.FOUND);
      }
    }
  }

  static final class CreateOutputDirectoryException extends IOException {
    final PathFragment directoryPath;

    private CreateOutputDirectoryException(PathFragment directoryPath, IOException cause) {
      super(cause.getMessage(), cause);
      this.directoryPath = directoryPath;
    }
  }
}
