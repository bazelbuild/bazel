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

import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.eventbus.Subscribe;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionExecutionMetadata;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactPathResolver;
import com.google.devtools.build.lib.actions.FilesetOutputSymlink;
import com.google.devtools.build.lib.actions.InputMetadataProvider;
import com.google.devtools.build.lib.actions.RemoteArtifactChecker;
import com.google.devtools.build.lib.actions.cache.MetadataInjector;
import com.google.devtools.build.lib.actions.cache.OutputMetadataStore;
import com.google.devtools.build.lib.buildtool.buildevent.ExecutionPhaseCompleteEvent;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.vfs.BatchStat;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.ModifiedFileSet;
import com.google.devtools.build.lib.vfs.OutputService;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import java.io.IOException;
import java.util.Map;
import java.util.UUID;
import java.util.function.Supplier;
import javax.annotation.Nullable;

/** Output service implementation for the remote module */
public class RemoteOutputService implements OutputService {

  @Nullable private RemoteOutputChecker remoteOutputChecker;
  @Nullable private RemoteActionInputFetcher actionInputFetcher;
  @Nullable private LeaseService leaseService;
  @Nullable private Supplier<InputMetadataProvider> fileCacheSupplier;

  void setRemoteOutputChecker(RemoteOutputChecker remoteOutputChecker) {
    this.remoteOutputChecker = remoteOutputChecker;
  }

  void setActionInputFetcher(RemoteActionInputFetcher actionInputFetcher) {
    this.actionInputFetcher = checkNotNull(actionInputFetcher, "actionInputFetcher");
  }

  void setLeaseService(LeaseService leaseService) {
    this.leaseService = leaseService;
  }

  void setFileCacheSupplier(Supplier<InputMetadataProvider> fileCacheSupplier) {
    this.fileCacheSupplier = fileCacheSupplier;
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
      ActionInputMap inputArtifactData,
      Iterable<Artifact> outputArtifacts,
      boolean rewindingEnabled) {
    checkNotNull(actionInputFetcher, "actionInputFetcher");
    return new RemoteActionFileSystem(
        delegateFileSystem,
        execRootFragment,
        relativeOutputPath,
        inputArtifactData,
        outputArtifacts,
        fileCacheSupplier.get(),
        actionInputFetcher);
  }

  @Override
  public void updateActionFileSystemContext(
      ActionExecutionMetadata action,
      FileSystem actionFileSystem,
      Environment env,
      MetadataInjector injector,
      ImmutableMap<Artifact, ImmutableList<FilesetOutputSymlink>> filesets) {
    ((RemoteActionFileSystem) actionFileSystem).updateContext(action, injector);
  }

  @Override
  public String getFilesSystemName() {
    return "remoteActionFS";
  }

  @Override
  public ModifiedFileSet startBuild(
      EventHandler eventHandler, UUID buildId, boolean finalizeActions) throws AbruptExitException {
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
      var missingActionInputs = ImmutableSet.<ActionInput>of();
      if (actionInputFetcher != null) {
        missingActionInputs = actionInputFetcher.getMissingActionInputs();
      }
      leaseService.finalizeExecution(missingActionInputs);
    }
  }

  @Override
  public void flushActionFileSystem(FileSystem actionFileSystem)
      throws InterruptedException, IOException {
    ((RemoteActionFileSystem) actionFileSystem).flush();
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
  public boolean shouldStoreRemoteOutputMetadataInActionCache() {
    return true;
  }

  @Override
  public RemoteArtifactChecker getRemoteArtifactChecker() {
    return checkNotNull(remoteOutputChecker, "remoteOutputChecker must not be null");
  }

  @Nullable
  @Override
  public BatchStat getBatchStatter() {
    return null;
  }

  @Override
  public boolean canCreateSymlinkTree() {
    /* TODO(buchgr): Optimize symlink creation for remote execution */
    return false;
  }

  @Override
  public void createSymlinkTree(
      Map<PathFragment, PathFragment> symlinks, PathFragment symlinkTreeRoot) {
    throw new UnsupportedOperationException();
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
      ActionInputMap actionInputMap,
      Map<Artifact, ImmutableCollection<? extends Artifact>> expandedArtifacts,
      Map<Artifact, ImmutableList<FilesetOutputSymlink>> filesets) {
    FileSystem remoteFileSystem =
        new RemoteActionFileSystem(
            fileSystem,
            execRoot,
            relativeOutputPath,
            actionInputMap,
            ImmutableList.of(),
            fileCacheSupplier.get(),
            actionInputFetcher);
    return ArtifactPathResolver.createPathResolver(remoteFileSystem, fileSystem.getPath(execRoot));
  }
}
