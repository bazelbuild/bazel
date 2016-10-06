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

package com.google.devtools.build.lib.exec;

import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.BuildFailedException;
import com.google.devtools.build.lib.actions.EnvironmentalExecException;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.cache.MetadataHandler;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.vfs.BatchStat;
import com.google.devtools.build.lib.vfs.ModifiedFileSet;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.io.IOException;
import java.util.UUID;

/**
 * An OutputService retains control over the Blaze output tree, and provides a higher level of
 * abstraction compared to the VFS layer.
 *
 * <p>Higher-level facilities include batch statting, cleaning the output tree, creating symlink
 * trees, and out-of-band insertion of metadata into the tree.
 */
public interface OutputService {

  /**
   * @return the name of filesystem, akin to what you might see in /proc/mounts
   */
  String getFilesSystemName();

  /**
   * @return true if the output service uses FUSE
   */
  boolean usesFuse();

  /**
   * @return a human-readable, one word name for the service
   */
  String getName();

  /**
   * Start the build.
   *
   * @param buildId the UUID build identifier
   * @param finalizeActions whether this build is finalizing actions so that the output service
   *                        can track output tree modifications
   * @return a ModifiedFileSet of changed output files.
   * @throws BuildFailedException if build preparation failed
   * @throws InterruptedException
   */
  ModifiedFileSet startBuild(UUID buildId, boolean finalizeActions)
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
      throws IOException, EnvironmentalExecException;

  /**
   * Stages the given tool from the package path, possibly copying it to local disk.
   *
   * @param tool target representing the tool to stage
   * @return a Path pointing to the staged target
   */
  Path stageTool(Target tool) throws IOException;

  /**
   * @return the name of the workspace this output service controls.
   */
  String getWorkspace();

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
   * @param inputPath the input manifest
   * @param outputPath the output manifest
   * @param filesetTree is true iff we're constructing a Fileset
   * @param symlinkTreeRoot the symlink tree root, relative to the execRoot
   * @throws ExecException on failure
   * @throws InterruptedException
   */
  void createSymlinkTree(Path inputPath, Path outputPath, boolean filesetTree,
      PathFragment symlinkTreeRoot) throws ExecException, InterruptedException;

  /**
   * Cleans the entire output tree.
   *
   * @throws ExecException on failure
   * @throws InterruptedException
   */
  void clean() throws ExecException, InterruptedException;

  /**
   * @param file the File
   * @return true iff the file actually lives on a remote server
   */
  boolean isRemoteFile(Path file);

  /**
   * @param path a fully-resolved path
   * @return true iff path is under this output service's control
   */
  boolean resolvedPathUnderTree(Path path);
}
