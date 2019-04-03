package com.google.devtools.build.lib.remote;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionInputMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SourceArtifact;
import com.google.devtools.build.lib.actions.cache.MetadataHandler;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.vfs.BatchStat;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.ModifiedFileSet;
import com.google.devtools.build.lib.vfs.OutputService;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import java.util.UUID;
import java.util.function.Function;
import javax.annotation.Nullable;

public class RemoteOutputService implements OutputService {

  @Nullable
  private RemoteActionInputFetcher actionInputFetcher;

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
  public FileSystem createActionFileSystem(FileSystem sourceDelegate, PathFragment execRootFragment,
      String relativeOutputPath, ImmutableList<Root> sourceRoots, ActionInputMap inputArtifactData,
      Iterable<Artifact> outputArtifacts,
      Function<PathFragment, SourceArtifact> sourceArtifactFactory) {
    Preconditions.checkNotNull(actionInputFetcher, "actionInputFetcher");
    return new RemoteActionFileSystem(sourceDelegate, execRootFragment, relativeOutputPath,
        inputArtifactData, actionInputFetcher);
  }

  @Override
  public String getFilesSystemName() {
    return "remoteActionFS";
  }

  @Override
  public ModifiedFileSet startBuild(EventHandler eventHandler, UUID buildId,
      boolean finalizeActions) {
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
  public void createSymlinkTree(Path inputManifest, Path outputManifest, boolean filesetTree,
      PathFragment symlinkTreeRoot) {
    throw new UnsupportedOperationException();
  }

  @Override
  public void clean() {
    // Intentionally left empty.
  }

  @Override
  public boolean isRemoteFile(Artifact file) {
    return false;
  }
}
