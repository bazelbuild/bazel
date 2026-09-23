// Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.remote;

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.eventbus.Subscribe;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionExecutionMetadata;
import com.google.devtools.build.lib.actions.ActionInputMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactPathResolver;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.InputMetadataProvider;
import com.google.devtools.build.lib.actions.LostInputsActionExecutionException;
import com.google.devtools.build.lib.actions.OutputChecker;
import com.google.devtools.build.lib.actions.TopLevelOutputException;
import com.google.devtools.build.lib.actions.cache.OutputMetadataStore;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.buildtool.buildevent.ExecutionPhaseCompleteEvent;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.exec.RunfilesTreeUpdater;
import com.google.devtools.build.lib.remote.options.RemoteOutputsMode;
import com.google.devtools.build.lib.server.FailureDetails.Execution;
import com.google.devtools.build.lib.server.FailureDetails.Execution.Code;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.vfs.BatchStat;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.ModifiedFileSet;
import com.google.devtools.build.lib.vfs.OutputService;
import com.google.devtools.build.lib.vfs.OutputService.SymlinkTreeCreationResult;
import com.google.devtools.build.lib.vfs.OutputService.SymlinkTreeType;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.skyframe.WalkableGraph;
import java.io.IOException;
import java.util.Map;
import java.util.UUID;
import java.util.function.Supplier;
import javax.annotation.Nullable;

/** Output service implementation for the remote build without local output service daemon. */
public class RemoteOutputService implements OutputService {

  private final BlazeDirectories directories;
  private final boolean rewindLostInputs;
  private final RemoteOutputsMode outputsMode;
  private final RunfilesTreeUpdater runfilesTreeUpdater;

  private RewoundActionSynchronizer rewoundActionSynchronizer = RewoundActionSynchronizer.NOOP;

  @Nullable private RemoteOutputChecker remoteOutputChecker;
  @Nullable private RemoteActionInputFetcher actionInputFetcher;
  @Nullable private LeaseService leaseService;

  RemoteOutputService(
      BlazeDirectories directories,
      boolean rewindLostInputs,
      RemoteOutputsMode outputsMode,
      RunfilesTreeUpdater runfilesTreeUpdater) {
    this.directories = checkNotNull(directories);
    this.rewindLostInputs = rewindLostInputs;
    this.outputsMode = checkNotNull(outputsMode);
    this.runfilesTreeUpdater = checkNotNull(runfilesTreeUpdater);
    if (outputsMode != RemoteOutputsMode.ALL) {
      runfilesTreeUpdater.setMaterializeBuiltRunfilesTrees();
    }
  }

  void setRemoteOutputChecker(RemoteOutputChecker remoteOutputChecker) {
    this.remoteOutputChecker = remoteOutputChecker;
  }

  void setActionInputFetcher(RemoteActionInputFetcher actionInputFetcher, WalkableGraph graph) {
    this.actionInputFetcher = checkNotNull(actionInputFetcher, "actionInputFetcher");
    if (rewindLostInputs) {
      this.rewoundActionSynchronizer =
          new RemoteRewoundActionSynchronizer(actionInputFetcher, graph);
    }
  }

  void setLeaseService(LeaseService leaseService) {
    this.leaseService = leaseService;
  }

  @Override
  public ActionFileSystemType actionFileSystemType() {
    return actionInputFetcher != null
        ? ActionFileSystemType.REMOTE_FILE_SYSTEM
        : ActionFileSystemType.DISABLED;
  }

  @Nullable
  @Override
  public FileSystem createActionFileSystem(
      FileSystem delegateFileSystem,
      PathFragment execRootFragment,
      String relativeOutputPath,
      ImmutableList<Root> sourceRoots,
      InputMetadataProvider inputArtifactData,
      Iterable<Artifact> outputArtifacts,
      boolean rewindingEnabled) {
    checkNotNull(actionInputFetcher, "actionInputFetcher");
    return new RemoteActionFileSystem(
        delegateFileSystem,
        execRootFragment,
        relativeOutputPath,
        inputArtifactData,
        actionInputFetcher);
  }

  @Override
  public void updateActionFileSystemContext(
      ActionExecutionMetadata action,
      FileSystem actionFileSystem,
      InputMetadataProvider inputMetadataProvider,
      OutputMetadataStore outputMetadataStore) {
    ((RemoteActionFileSystem) actionFileSystem).updateContext(action);
  }

  @Override
  public String getFileSystemName(String outputBaseFileSystemName) {
    return "remoteActionFS";
  }

  @Override
  public ModifiedFileSet startBuild(
      UUID buildId, String workspaceName, EventHandler eventHandler, boolean finalizeActions)
      throws AbruptExitException {
    // One of the responsibilities of OutputService.startBuild() is that it ensures the output path
    // is valid. If the previous OutputService redirected the output path to a remote location, we
    // must undo this.
    Path outputPath = directories.getOutputPath(workspaceName);
    try {
      if (outputPath.isSymbolicLink()) {
        outputPath.delete();
      }
    } catch (IOException e) {
      throw new AbruptExitException(
          DetailedExitCode.of(
              FailureDetail.newBuilder()
                  .setMessage(
                      String.format("Couldn't remove output path symlink: %s", e.getMessage()))
                  .setExecution(
                      Execution.newBuilder().setCode(Code.LOCAL_OUTPUT_DIRECTORY_SYMLINK_FAILURE))
                  .build()),
          e);
    }
    return ModifiedFileSet.EVERYTHING_MODIFIED;
  }

  @Override
  public void flushOutputTree() throws InterruptedException {
    if (actionInputFetcher != null) {
      actionInputFetcher.flushOutputTree();
    }
  }

  @Override
  public void finalizeBuild(boolean buildSuccessful) {
    // Intentionally left empty.
  }

  @Subscribe
  public void onExecutionPhaseCompleteEvent(ExecutionPhaseCompleteEvent event) {
    if (leaseService != null) {
      leaseService.finalizeExecution();
    }
  }

  @Override
  public void finalizeAction(Action action, OutputMetadataStore outputMetadataStore)
      throws IOException, InterruptedException {
    if (actionInputFetcher != null) {
      actionInputFetcher.finalizeAction(action, outputMetadataStore);
    }

    if (leaseService != null) {
      leaseService.finalizeAction();
    }
  }

  @Override
  public void finalizeTopLevelOutputs(InputMetadataProvider metadataProvider)
      throws TopLevelOutputException, InterruptedException {
    if (outputsMode == RemoteOutputsMode.ALL) {
      return;
    }
    RemoteOutputChecker checker =
        checkNotNull(remoteOutputChecker, "remoteOutputChecker must not be null");
    try {
      runfilesTreeUpdater.updateRunfiles(
          Iterables.filter(
              metadataProvider.getRunfilesTrees(),
              tree -> checker.shouldCreateRunfilesTree(tree.getExecPath())));
    } catch (ExecException | IOException e) {
      String message = "Failed to create runfiles symlinks: " + e.getMessage();
      throw new TopLevelOutputException(
          message,
          DetailedExitCode.of(
              FailureDetail.newBuilder()
                  .setMessage(message)
                  .setExecution(
                      Execution.newBuilder()
                          .setCode(Execution.Code.SYMLINK_TREE_CREATION_IO_EXCEPTION))
                  .build()));
    }
  }

  @Override
  public boolean shouldStoreRemoteOutputMetadataInActionCache() {
    return true;
  }

  @Override
  public OutputChecker getOutputChecker() {
    return checkNotNull(remoteOutputChecker, "remoteOutputChecker must not be null");
  }

  @Nullable
  @Override
  public BatchStat getBatchStatter() {
    return null;
  }

  @Override
  public SymlinkTreeCreationResult createSymlinkTree(
      SymlinkTreeType type,
      Supplier<Map<PathFragment, PathFragment>> symlinks,
      PathFragment symlinkTreeRoot) {
    // When building without the bytes, only create the runfiles trees that are actually needed:
    // those of top-level targets, which SymlinkTreeAction creates just like their outputs are
    // downloaded, and those required by local actions or the run command, which RunfilesTreeUpdater
    // creates on demand. Targets that only become top-level after their SymlinkTreeAction has run,
    // e.g. because they were previously only built as a dependency, are handled by
    // finalizeTopLevelOutputs at target completion.
    if (type == SymlinkTreeType.FILESET
        || outputsMode == RemoteOutputsMode.ALL
        || (remoteOutputChecker != null
            && remoteOutputChecker.shouldCreateRunfilesTree(symlinkTreeRoot))) {
      return SymlinkTreeCreationResult.NOT_HANDLED;
    }
    return SymlinkTreeCreationResult.DEFERRED;
  }

  @Override
  public void clean() {
    // Intentionally left empty.
  }

  @Override
  public boolean supportsPathResolverForArtifactValues() {
    return actionFileSystemType() != ActionFileSystemType.DISABLED;
  }

  @Override
  public ArtifactPathResolver createPathResolverForArtifactValues(
      PathFragment execRoot,
      String relativeOutputPath,
      FileSystem fileSystem,
      ImmutableList<Root> pathEntries,
      ActionInputMap actionInputMap) {
    FileSystem remoteFileSystem =
        new RemoteActionFileSystem(
            fileSystem, execRoot, relativeOutputPath, actionInputMap, actionInputFetcher);
    return ArtifactPathResolver.createPathResolver(remoteFileSystem, fileSystem.getPath(execRoot));
  }

  @Override
  public void checkActionFileSystemForLostInputs(FileSystem actionFileSystem, Action action)
      throws LostInputsActionExecutionException {
    if (actionFileSystem instanceof RemoteActionFileSystem remoteFileSystem) {
      remoteFileSystem.checkForLostInputs(action);
    }
  }

  @Override
  public RewoundActionSynchronizer getRewoundActionSynchronizer() {
    return rewoundActionSynchronizer;
  }
}
