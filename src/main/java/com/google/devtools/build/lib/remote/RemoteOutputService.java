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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionInputMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactPathResolver;
import com.google.devtools.build.lib.actions.FilesetOutputSymlink;
import com.google.devtools.build.lib.actions.cache.MetadataHandler;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.vfs.BatchStat;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.ModifiedFileSet;
import com.google.devtools.build.lib.vfs.OutputService;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import java.util.Collection;
import java.util.Map;
import java.util.UUID;
import javax.annotation.Nullable;

/** Output service implementation for the remote module */
public class RemoteOutputService implements OutputService {

  @Nullable private RemoteActionInputFetcher actionInputFetcher;

  void setActionInputFetcher(RemoteActionInputFetcher actionInputFetcher) {
    this.actionInputFetcher = Preconditions.checkNotNull(actionInputFetcher, "actionInputFetcher");
  }

  @Override
  public ActionFileSystemType actionFileSystemType() {
    return actionInputFetcher != null
        ? ActionFileSystemType.STAGE_REMOTE_FILES
        : ActionFileSystemType.DISABLED;
  }

  @Nullable
  @Override
  public FileSystem createActionFileSystem(
      FileSystem sourceDelegate,
      PathFragment execRootFragment,
      String relativeOutputPath,
      ImmutableList<Root> sourceRoots,
      ActionInputMap inputArtifactData,
      Iterable<Artifact> outputArtifacts,
      boolean trackFailedRemoteReads) {
    Preconditions.checkNotNull(actionInputFetcher, "actionInputFetcher");
    return new RemoteActionFileSystem(
        sourceDelegate,
        execRootFragment,
        relativeOutputPath,
        inputArtifactData,
        actionInputFetcher);
  }

  @Override
  public String getFilesSystemName() {
    return "remoteActionFS";
  }

  @Override
  public ModifiedFileSet startBuild(
      EventHandler eventHandler, UUID buildId, boolean finalizeActions) {
    return ModifiedFileSet.EVERYTHING_MODIFIED;
  }

  @Override
  public void finalizeBuild(boolean buildSuccessful) {
    // Intentionally left empty.
  }

  @Override
  public void finalizeAction(Action action, MetadataHandler metadataHandler) {
    // Intentionally left empty.
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
  public boolean isRemoteFile(Artifact artifact) {
    Path path = artifact.getPath();
    return path.getFileSystem() instanceof RemoteActionFileSystem
        && ((RemoteActionFileSystem) path.getFileSystem()).isRemote(path);
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
      Map<Artifact, Collection<Artifact>> expandedArtifacts,
      Map<Artifact, ImmutableList<FilesetOutputSymlink>> filesets) {
    FileSystem remoteFileSystem =
        new RemoteActionFileSystem(
            fileSystem, execRoot, relativeOutputPath, actionInputMap, actionInputFetcher);
    return ArtifactPathResolver.createPathResolver(remoteFileSystem, fileSystem.getPath(execRoot));
  }
}
