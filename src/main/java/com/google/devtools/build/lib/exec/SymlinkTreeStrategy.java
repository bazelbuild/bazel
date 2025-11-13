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
import com.google.common.base.Function;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.FilesetOutputSymlink;
import com.google.devtools.build.lib.actions.RunningActionEvent;
import com.google.devtools.build.lib.analysis.actions.SymlinkTreeAction;
import com.google.devtools.build.lib.analysis.actions.SymlinkTreeActionContext;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue.RunfileSymlinksMode;
import com.google.devtools.build.lib.profiler.AutoProfiler;
import com.google.devtools.build.lib.profiler.GoogleAutoProfilerUtils;
import com.google.devtools.build.lib.vfs.OutputService;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.time.Duration;
import java.util.Map;

/**
 * Implements SymlinkTreeAction by using the output service or by running an embedded script to
 * create the symlink tree.
 */
public final class SymlinkTreeStrategy implements SymlinkTreeActionContext {
  private static final Duration MIN_LOGGING = Duration.ofMillis(100);

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
    try (AutoProfiler p =
        GoogleAutoProfilerUtils.logged("running " + action.prettyPrint(), MIN_LOGGING)) {
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
            helper.createRunfilesSymlinks(getRunfilesMap(action));
          }
          helper.linkManifest();
        }
      } catch (ExecException e) {
        throw ActionExecutionException.fromExecException(e, action);
      }
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
    return action.getRunfiles().getRunfilesInputs(action.getRepoMappingManifest());
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
