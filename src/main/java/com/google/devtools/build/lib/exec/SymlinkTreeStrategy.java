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

import static com.google.common.collect.ImmutableList.toImmutableList;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Function;
import com.google.common.base.Throwables;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Maps;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionInputPrefetcher.Priority;
import com.google.devtools.build.lib.actions.ActionInputPrefetcher.Reason;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.EnvironmentalExecException;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.FilesetOutputSymlink;
import com.google.devtools.build.lib.actions.RunningActionEvent;
import com.google.devtools.build.lib.analysis.actions.SymlinkTreeAction;
import com.google.devtools.build.lib.analysis.actions.SymlinkTreeActionContext;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue.RunfileSymlinksMode;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.server.FailureDetails.Execution.Code;
import com.google.devtools.build.lib.vfs.OutputService;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.util.Collection;
import java.util.Map;
import java.util.concurrent.ExecutionException;

/**
 * Implements SymlinkTreeAction by using the output service or by running an embedded script to
 * create the symlink tree.
 */
public final class SymlinkTreeStrategy implements SymlinkTreeActionContext {
  @VisibleForTesting
  static final Function<Artifact, PathFragment> TO_PATH =
      (artifact) -> artifact == null ? null : artifact.getPath().asFragment();

  private final OutputService outputService;
  private final String workspaceName;

  public SymlinkTreeStrategy(OutputService outputService, String workspaceName) {
    this.outputService = outputService;
    this.workspaceName = workspaceName;
  }

  @Override
  public void createSymlinks(
      SymlinkTreeAction action, ActionExecutionContext actionExecutionContext)
      throws ActionExecutionException, InterruptedException {
    actionExecutionContext.getEventHandler().post(new RunningActionEvent(action, "local"));
    try (var _ = Profiler.instance().profile("SymlinkTreeStrategy.createSymlinks")) {
      SymlinkTreeHelper helper = createSymlinkTreeHelper(action, actionExecutionContext);
      // TODO(tjgq): Respect RunfileSymlinksMode.SKIP even in the presence of an OutputService.
      try {
        // Note that the output manifest must always be created last, as its presence ascertains
        // that the runfiles tree has been updated (only the output manifest is an action output,
        // so Skyframe cannot invalidate the symlink tree).
        if (outputService.canCreateSymlinkTree()) {
          Map<PathFragment, PathFragment> symlinks;
          if (action.isFilesetTree()) {
            symlinks = getFilesetMap(action, actionExecutionContext);
          } else {
            // TODO(tjgq): This produces an incorrect path for unresolved symlinks, which should be
            // created textually.
            symlinks = Maps.transformValues(getRunfilesMap(action), TO_PATH);
          }
          outputService.createSymlinkTree(
              symlinks, action.getOutputManifest().getExecPath().getParentDirectory());
          helper.linkManifest();
        } else if (action.getRunfileSymlinksMode() == RunfileSymlinksMode.SKIP) {
          // Clear the runfiles directory, then create just the output manifest and the workspace
          // subdirectory. This is required because only the output manifest is considered an action
          // output, so if the previous invocation created a symlink tree, Skyframe will not clear
          // it for us.
          helper.createMinimalRunfilesDirectory();
        } else {
          if (action.isFilesetTree()) {
            helper.createFilesetSymlinks(getFilesetMap(action, actionExecutionContext));
          } else {
            Map<PathFragment, Artifact> runfilesMap = getRunfilesMap(action);
            if (helper.requiresExistingFileTargets()) {
              prefetchRunfiles(action, runfilesMap.values(), actionExecutionContext);
            }
            helper.createRunfilesSymlinks(runfilesMap);
          }
          helper.linkManifest();
        }
      } catch (ExecException e) {
        throw ActionExecutionException.fromExecException(e, action);
      }
    }
  }

  /**
   * Ensures that the runfiles that are regular files are present on disk.
   *
   * <p>Outputs of remotely cached or executed actions are generally only downloaded when a local
   * spawn needs them as inputs. Since the symlink tree is created in-process, its inputs (which
   * include the runfiles on Windows, see {@link SymlinkTreeAction}) aren't prefetched
   * automatically. This is only an issue on file systems that emulate symlinks to files with copies
   * and would otherwise create a junction to the not yet existing file, which can't be used to
   * access it even after it has been downloaded.
   *
   * <p>Tree artifacts don't need to exist as they are linked via junctions, which are allowed to
   * dangle, and unresolved symlinks are created textually from their metadata.
   */
  private static void prefetchRunfiles(
      SymlinkTreeAction action,
      Collection<Artifact> runfiles,
      ActionExecutionContext actionExecutionContext)
      throws ExecException, InterruptedException {
    ImmutableList<Artifact> files =
        runfiles.stream()
            .filter(
                artifact ->
                    artifact != null
                        && !artifact.isTreeArtifact()
                        && !artifact.isSymlink()
                        && !artifact.isFileset()
                        && !artifact.isRunfilesTree())
            .collect(toImmutableList());
    if (files.isEmpty()) {
      return;
    }
    ListenableFuture<Void> prefetch =
        actionExecutionContext
            .getActionInputPrefetcher()
            .prefetchFiles(
                action,
                /* spawn= */ null,
                () -> files,
                actionExecutionContext.getInputMetadataProvider(),
                Priority.CRITICAL,
                Reason.INPUTS);
    try {
      prefetch.get();
    } catch (ExecutionException e) {
      Throwable cause = e.getCause();
      if (cause instanceof IOException ioException) {
        throw new EnvironmentalExecException(
            ioException, Code.SYMLINK_TREE_CREATION_IO_EXCEPTION);
      }
      if (cause instanceof InterruptedException) {
        throw new InterruptedException(cause.getMessage());
      }
      Throwables.throwIfUnchecked(cause);
      throw new IllegalStateException(cause);
    }
  }

  private static ImmutableMap<PathFragment, PathFragment> getFilesetMap(
      SymlinkTreeAction action, ActionExecutionContext actionExecutionContext) {
    ImmutableList<FilesetOutputSymlink> filesetLinks =
        actionExecutionContext
            .getInputMetadataProvider()
            .getFileset(action.getInputManifest())
            .symlinks();
    return SymlinkTreeHelper.processFilesetLinks(filesetLinks, action.getWorkspaceNameForFileset());
  }

  private static Map<PathFragment, Artifact> getRunfilesMap(SymlinkTreeAction action) {
    // This call outputs warnings about overlapping symlinks. However, since this has already been
    // called by the SourceManifestAction, we silence the warnings here.
    return action
        .getRunfiles()
        .getRunfilesInputs(
            action.getRepoMappingManifest(),
            action.isPreferTargetConfigurationRunfiles()
                ? action.getOutputManifest().getRoot()
                : null);
  }

  private SymlinkTreeHelper createSymlinkTreeHelper(
      SymlinkTreeAction action, ActionExecutionContext actionExecutionContext) {
    return new SymlinkTreeHelper(
        actionExecutionContext.getInputPath(action.getInputManifest()),
        actionExecutionContext.getInputPath(action.getOutputManifest()),
        actionExecutionContext.getInputPath(action.getOutputManifest()).getParentDirectory(),
        workspaceName);
  }
}
