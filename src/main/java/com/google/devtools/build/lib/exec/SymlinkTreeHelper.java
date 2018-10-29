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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionMetadata;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.ExecutionRequirements;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.actions.SimpleSpawn;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.build.lib.shell.Command;
import com.google.devtools.build.lib.shell.CommandException;
import com.google.devtools.build.lib.util.CommandBuilder;
import com.google.devtools.build.lib.util.CommandUtils;
import com.google.devtools.build.lib.util.OsUtils;
import com.google.devtools.build.lib.util.io.OutErr;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.util.List;

/**
 * Helper class responsible for the symlink tree creation. Used to generate runfiles and fileset
 * symlink farms.
 */
public final class SymlinkTreeHelper {
  @VisibleForTesting
  public static final String BUILD_RUNFILES = "build-runfiles" + OsUtils.executableExtension();

  private final Path inputManifest;
  private final Path symlinkTreeRoot;
  private final boolean filesetTree;

  /**
   * Creates SymlinkTreeHelper instance. Can be used independently of SymlinkTreeAction.
   *
   * @param inputManifest exec path to the input runfiles manifest
   * @param symlinkTreeRoot the root of the symlink tree to be created
   * @param filesetTree true if this is fileset symlink tree, false if this is a runfiles symlink
   *     tree.
   */
  public SymlinkTreeHelper(Path inputManifest, Path symlinkTreeRoot, boolean filesetTree) {
    this.inputManifest = inputManifest;
    this.symlinkTreeRoot = symlinkTreeRoot;
    this.filesetTree = filesetTree;
  }

  public Path getOutputManifest() {
    return symlinkTreeRoot;
  }

  /**
   * Creates a symlink tree using a CommandBuilder. This means that the symlink tree will always be
   * present on the developer's workstation. Useful when running commands locally.
   *
   * <p>Warning: this method REALLY executes the command on the box Bazel is running on, without any
   * kind of synchronization, locking, or anything else.
   *
   * @param config the configuration that is used for creating the symlink tree.
   * @throws CommandException
   */
  public void createSymlinksUsingCommand(
      Path execRoot,
      BinTools binTools,
      ImmutableMap<String, String> shellEnvironment,
      OutErr outErr)
      throws CommandException {
    List<String> argv = getSpawnArgumentList(execRoot,
        binTools.getEmbeddedPath(BUILD_RUNFILES).asFragment());
    Preconditions.checkNotNull(shellEnvironment);
    Command command =
        new CommandBuilder().addArgs(argv).setWorkingDir(execRoot).setEnv(shellEnvironment).build();
    if (outErr != null) {
      command.execute(outErr.getOutputStream(), outErr.getErrorStream());
    } else {
      command.execute();
    }
  }

  /**
   * Creates symlink tree using appropriate method. At this time tree always created using
   * build-runfiles helper application.
   *
   * @param owner action instance that requested symlink tree creation
   * @param actionExecutionContext Services that are in the scope of the action.
   * @param enableRunfiles
   * @return a list of SpawnResults created during symlink creation, if any
   */
  public void createSymlinks(
      ActionExecutionContext actionExecutionContext,
      BinTools binTools,
      ImmutableMap<String, String> shellEnvironment,
      boolean enableRunfiles)
      throws ExecException {
    if (enableRunfiles) {
      try {
        createSymlinksUsingCommand(
            actionExecutionContext.getExecRoot(),
            binTools,
            shellEnvironment,
            actionExecutionContext.getFileOutErr());
      } catch (CommandException e) {
        throw new UserExecException(CommandUtils.describeCommandFailure(true, e), e);
      }
    } else {
      // Pretend we created the runfiles tree by copying the manifest
      try {
        symlinkTreeRoot.createDirectoryAndParents();
        FileSystemUtils.copyFile(inputManifest, symlinkTreeRoot.getChild("MANIFEST"));
      } catch (IOException e) {
        throw new UserExecException(e.getMessage(), e);
      }
    }
  }

  @VisibleForTesting
  Spawn createSpawn(
      ActionExecutionMetadata owner,
      Path execRoot,
      BinTools binTools,
      ImmutableMap<String, String> environment,
      ActionInput inputManifestArtifact) {
    ActionInput buildRunfiles = binTools.getActionInput(BUILD_RUNFILES);
    return new SimpleSpawn(
        owner,
        getSpawnArgumentList(execRoot, buildRunfiles.getExecPath()),
        environment,
        ImmutableMap.of(
            ExecutionRequirements.LOCAL, "",
            ExecutionRequirements.NO_CACHE, "",
            ExecutionRequirements.NO_SANDBOX, ""),
        ImmutableList.of(inputManifestArtifact, buildRunfiles),
        /*outputs=*/ ImmutableList.of(),
        ResourceSet.ZERO);
  }

  /** Returns the complete argument list build-runfiles has to be called with. */
  private ImmutableList<String> getSpawnArgumentList(Path execRoot, PathFragment buildRunfiles) {
    List<String> args = Lists.newArrayList();
    args.add(buildRunfiles.getPathString());

    if (filesetTree) {
      args.add("--allow_relative");
      args.add("--use_metadata");
    }

    args.add(inputManifest.relativeTo(execRoot).getPathString());
    args.add(symlinkTreeRoot.relativeTo(execRoot).getPathString());

    return ImmutableList.copyOf(args);
  }
}
