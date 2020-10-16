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

package com.google.devtools.build.lib.vfs;

import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionInputMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactPathResolver;
import com.google.devtools.build.lib.actions.BuildFailedException;
import com.google.devtools.build.lib.actions.EnvironmentalExecException;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.FilesetOutputSymlink;
import com.google.devtools.build.lib.actions.LostInputsActionExecutionException;
import com.google.devtools.build.lib.actions.cache.MetadataHandler;
import com.google.devtools.build.lib.actions.cache.MetadataInjector;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import java.io.IOException;
import java.util.Map;
import java.util.UUID;
import javax.annotation.Nullable;

/**
 * An OutputService retains control over the Blaze output tree, and provides a higher level of
 * abstraction compared to the VFS layer.
 *
 * <p>Higher-level facilities include batch statting, cleaning the output tree, creating symlink
 * trees, and out-of-band insertion of metadata into the tree.
 */
public interface OutputService {

  /** Properties of the action file system implementation provided by this output service. */
  enum ActionFileSystemType {

    /** Action file system is disabled */
    DISABLED,

    /**
     * The action file system implementation does not take over the output base but complements the
     * file system by being able to stage remote outputs accessed as inputs by local actions, as
     * used by Bazel.
     */
    STAGE_REMOTE_FILES,

    /**
     * The action file system implementation is fully featured in-memory file system implementation
     * and takes full control of the output base, as used by Blaze.
     */
    IN_MEMORY_FILE_SYSTEM;

    public boolean inMemoryFileSystem() {
      return this == IN_MEMORY_FILE_SYSTEM;
    }

    public boolean isEnabled() {
      return this != DISABLED;
    }
  }

  /**
   * @return the name of filesystem, akin to what you might see in /proc/mounts
   */
  String getFilesSystemName();

  /**
   * Returns true if Bazel should trust (and not verify) build artifacts that were last seen
   * remotely and do not exist locally.
   */
  public default boolean shouldTrustRemoteArtifacts() {
    return true;
  }

  /**
   * Start the build.
   *
   * @param buildId the UUID build identifier
   * @param finalizeActions whether this build is finalizing actions so that the output service can
   *     track output tree modifications
   * @return a ModifiedFileSet of changed output files.
   * @throws BuildFailedException if build preparation failed
   * @throws InterruptedException
   */
  ModifiedFileSet startBuild(EventHandler eventHandler, UUID buildId, boolean finalizeActions)
      throws BuildFailedException, AbruptExitException, InterruptedException;

  /**
   * Finish the build.
   *
   * @param buildSuccessful iff build was successful
   * @throws BuildFailedException on failure
   */
  void finalizeBuild(boolean buildSuccessful)
      throws BuildFailedException, AbruptExitException, InterruptedException;

  /** Notify the output service of a completed action. */
  void finalizeAction(Action action, MetadataHandler metadataHandler)
      throws IOException, EnvironmentalExecException, InterruptedException;

  /**
   * @return the BatchStat instance or null.
   */
  BatchStat getBatchStatter();

  /**
   * @return true iff createSymlinkTree() is available.
   */
  boolean canCreateSymlinkTree();

  /**
   * Creates the symlink tree
   *
   * @param symlinks the symlinks to create
   * @param symlinkTreeRoot the symlink tree root, relative to the execRoot
   * @throws ExecException on failure
   * @throws InterruptedException
   */
  void createSymlinkTree(Map<PathFragment, PathFragment> symlinks, PathFragment symlinkTreeRoot)
      throws ExecException, InterruptedException;

  /**
   * Cleans the entire output tree.
   *
   * @throws ExecException on failure
   * @throws InterruptedException
   */
  void clean() throws ExecException, InterruptedException;

  /** @return true iff the file actually lives on a remote server */
  boolean isRemoteFile(Artifact file);

  default ActionFileSystemType actionFileSystemType() {
    return ActionFileSystemType.DISABLED;
  }

  /**
   * @param sourceDelegate filesystem for reading source files (excludes output files)
   * @param execRootFragment absolute path fragment pointing to the execution root
   * @param relativeOutputPath execution root relative path to output
   * @param sourceRoots list of directories on the package path (from {@link
   *     com.google.devtools.build.lib.pkgcache.PathPackageLocator})
   * @param inputArtifactData information about required inputs to the action
   * @param outputArtifacts required outputs of the action
   * @param trackFailedRemoteReads whether to track failed remote reads to make LostInput exceptions
   * @return an action-scoped filesystem if {@link #supportsActionFileSystem} is not {@code NONE}
   */
  @Nullable
  default FileSystem createActionFileSystem(
      FileSystem sourceDelegate,
      PathFragment execRootFragment,
      String relativeOutputPath,
      ImmutableList<Root> sourceRoots,
      ActionInputMap inputArtifactData,
      Iterable<Artifact> outputArtifacts,
      boolean trackFailedRemoteReads) {
    return null;
  }

  /**
   * Updates the context used by the filesystem returned by {@link #createActionFileSystem}.
   *
   * <p>Should be called as context changes throughout action execution.
   *
   * @param actionFileSystem must be a filesystem returned by {@link #createActionFileSystem}.
   * @param filesets The Fileset symlinks known for this action.
   */
  default void updateActionFileSystemContext(
      FileSystem actionFileSystem,
      Environment env,
      MetadataInjector injector,
      ImmutableMap<Artifact, ImmutableList<FilesetOutputSymlink>> filesets)
      throws IOException {}

  /**
   * Checks the filesystem returned by {@link #createActionFileSystem} for errors attributable to
   * lost inputs.
   */
  default void checkActionFileSystemForLostInputs(FileSystem actionFileSystem, Action action)
      throws LostInputsActionExecutionException {}

  default boolean supportsPathResolverForArtifactValues() {
    return false;
  }

  default ArtifactPathResolver createPathResolverForArtifactValues(
      PathFragment execRoot,
      String relativeOutputPath,
      FileSystem fileSystem,
      ImmutableList<Root> pathEntries,
      ActionInputMap actionInputMap,
      Map<Artifact, ImmutableCollection<Artifact>> expandedArtifacts,
      Map<Artifact, ImmutableList<FilesetOutputSymlink>> filesets)
      throws IOException {
    throw new IllegalStateException("Path resolver not supported by this class");
  }

  @Nullable
  default BulkDeleter bulkDeleter() {
    return null;
  }
}
