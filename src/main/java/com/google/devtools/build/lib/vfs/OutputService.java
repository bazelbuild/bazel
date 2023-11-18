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
import com.google.devtools.build.lib.actions.ActionExecutionMetadata;
import com.google.devtools.build.lib.actions.ActionInputMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactPathResolver;
import com.google.devtools.build.lib.actions.BuildFailedException;
import com.google.devtools.build.lib.actions.EnvironmentalExecException;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.FilesetOutputSymlink;
import com.google.devtools.build.lib.actions.LostInputsActionExecutionException;
import com.google.devtools.build.lib.actions.RemoteArtifactChecker;
import com.google.devtools.build.lib.actions.cache.MetadataInjector;
import com.google.devtools.build.lib.actions.cache.OutputMetadataStore;
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

    /** The action file system is disabled. */
    DISABLED,

    /**
     * The action file system implementation is purely in-memory, taking full control of the output
     * base. It's not able to stage remote outputs accessed as inputs by local actions, but is able
     * to do input discovery. Used by Blaze.
     */
    IN_MEMORY_ONLY_FILE_SYSTEM,

    /**
     * The action file system implementation mixes an in-memory and a local file system. It uses the
     * in-memory filesystem for in-process and remote actions, but is also aware of outputs from
     * local actions. It's able to stage remote outputs accessed as inputs by local actions and to
     * do input discovery. Used by Blaze.
     */
    STAGE_REMOTE_FILES_FILE_SYSTEM,

    /**
     * The action file system implementation mixes an in-memory and a local file system. It uses the
     * in-memory filesystem for in-process and remote actions, but is also aware of outputs from
     * local actions. It's able to stage remote outputs accessed as inputs by local actions, but
     * unable to do input discovery. Used by Bazel.
     */
    REMOTE_FILE_SYSTEM;

    public boolean inMemoryFileSystem() {
      return this != DISABLED;
    }

    public boolean supportsLocalActions() {
      return this != IN_MEMORY_ONLY_FILE_SYSTEM;
    }

    public boolean supportsInputDiscovery() {
      return this != REMOTE_FILE_SYSTEM;
    }

    public boolean isEnabled() {
      return this != DISABLED;
    }
  }

  /**
   * @return the name of filesystem, akin to what you might see in /proc/mounts
   */
  String getFilesSystemName();

  /** Returns true if remote output metadata should be stored in action cache. */
  default boolean shouldStoreRemoteOutputMetadataInActionCache() {
    return false;
  }

  default RemoteArtifactChecker getRemoteArtifactChecker() {
    return RemoteArtifactChecker.TRUST_ALL;
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

  /** Flush and wait for in-progress downloads. */
  default void flushOutputTree() throws InterruptedException {}

  /**
   * Finish the build.
   *
   * @param buildSuccessful iff build was successful
   * @throws BuildFailedException on failure
   */
  void finalizeBuild(boolean buildSuccessful)
      throws BuildFailedException, AbruptExitException, InterruptedException;

  /** Notify the output service of a completed action. */
  void finalizeAction(Action action, OutputMetadataStore outputMetadataStore)
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

  default ActionFileSystemType actionFileSystemType() {
    return ActionFileSystemType.DISABLED;
  }

  /**
   * Returns an action-scoped filesystem if {@link #actionFileSystemType} is enabled.
   *
   * @param delegateFileSystem the actual underlying filesystem
   * @param execRootFragment absolute path fragment pointing to the execution root
   * @param relativeOutputPath execution root relative path to output
   * @param sourceRoots list of directories on the package path (from {@link
   *     com.google.devtools.build.lib.pkgcache.PathPackageLocator})
   * @param inputArtifactData information about required inputs to the action
   * @param outputArtifacts required outputs of the action
   * @param rewindingEnabled whether to track failed remote reads to enable action rewinding
   */
  @Nullable
  default FileSystem createActionFileSystem(
      FileSystem delegateFileSystem,
      PathFragment execRootFragment,
      String relativeOutputPath,
      ImmutableList<Root> sourceRoots,
      ActionInputMap inputArtifactData,
      Iterable<Artifact> outputArtifacts,
      boolean rewindingEnabled) {
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
      ActionExecutionMetadata action,
      FileSystem actionFileSystem,
      Environment env,
      MetadataInjector injector,
      ImmutableMap<Artifact, ImmutableList<FilesetOutputSymlink>> filesets) {}

  /**
   * Checks the filesystem returned by {@link #createActionFileSystem} for errors attributable to
   * lost inputs.
   */
  default void checkActionFileSystemForLostInputs(FileSystem actionFileSystem, Action action)
      throws LostInputsActionExecutionException {}

  /**
   * Flush the internal state of filesystem returned by {@link #createActionFileSystem} after action
   * execution, before skyframe checking the action outputs.
   */
  default void flushActionFileSystem(FileSystem actionFileSystem)
      throws IOException, InterruptedException {}

  default boolean supportsPathResolverForArtifactValues() {
    return false;
  }

  default ArtifactPathResolver createPathResolverForArtifactValues(
      PathFragment execRoot,
      String relativeOutputPath,
      FileSystem fileSystem,
      ImmutableList<Root> pathEntries,
      ActionInputMap actionInputMap,
      Map<Artifact, ImmutableCollection<? extends Artifact>> expandedArtifacts,
      Map<Artifact, ImmutableList<FilesetOutputSymlink>> filesets) {
    throw new IllegalStateException("Path resolver not supported by this class");
  }

  @Nullable
  default BulkDeleter bulkDeleter() {
    return null;
  }
}
