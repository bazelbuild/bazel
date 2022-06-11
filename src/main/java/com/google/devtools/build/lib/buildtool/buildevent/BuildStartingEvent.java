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

package com.google.devtools.build.lib.buildtool.buildevent;

import com.google.auto.value.AutoValue;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.BlazeVersionInfo;
import com.google.devtools.build.lib.buildeventstream.BuildEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEventContext;
import com.google.devtools.build.lib.buildeventstream.BuildEventIdUtil;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId;
import com.google.devtools.build.lib.buildeventstream.GenericBuildEvent;
import com.google.devtools.build.lib.buildeventstream.ProgressEvent;
import com.google.devtools.build.lib.buildtool.BuildRequest;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.CommandLineEvent;
import com.google.protobuf.util.Timestamps;
import javax.annotation.Nullable;

/**
 * This event is fired from BuildTool#startRequest(). At this point, the set of target patters are
 * known, but have yet to be parsed.
 */
@AutoValue
public abstract class BuildStartingEvent implements BuildEvent {
  BuildStartingEvent() {}

  /** Returns the name of output file system. */
  public abstract String outputFileSystem();

  /**
   * Returns whether the build uses in-memory {@link
   * com.google.devtools.build.lib.vfs.OutputService.ActionFileSystemType#inMemoryFileSystem()}.
   */
  public abstract boolean usesInMemoryFileSystem();

  /** Returns the active BuildRequest. */
  public abstract BuildRequest request();

  @Nullable
  abstract String workspace();

  abstract String pwd();

  /**
   * Construct the BuildStartingEvent
   *
   * @param request the build request.
   * @param env the environment of the request invocation.
   */
  public static BuildStartingEvent create(CommandEnvironment env, BuildRequest request) {
    return create(
        env.determineOutputFileSystem(),
        env.getOutputService() != null
            && env.getOutputService().actionFileSystemType().inMemoryFileSystem(),
        request,
        env.getDirectories().getWorkspace() != null
            ? env.getDirectories().getWorkspace().toString()
            : null,
        env.getWorkingDirectory().toString());
  }

  @VisibleForTesting
  public static BuildStartingEvent create(
      String outputFileSystem,
      boolean usesInMemoryFileSystem,
      BuildRequest request,
      @Nullable String workspace,
      String pwd) {
    return new AutoValue_BuildStartingEvent(
        outputFileSystem, usesInMemoryFileSystem, request, workspace, pwd);
  }

  @Override
  public final BuildEventId getEventId() {
    return BuildEventIdUtil.buildStartedId();
  }

  @Override
  public final ImmutableList<BuildEventId> getChildrenEvents() {
    return ImmutableList.of(
        ProgressEvent.INITIAL_PROGRESS_UPDATE,
        BuildEventIdUtil.unstructuredCommandlineId(),
        BuildEventIdUtil.structuredCommandlineId(CommandLineEvent.OriginalCommandLineEvent.LABEL),
        BuildEventIdUtil.structuredCommandlineId(CommandLineEvent.CanonicalCommandLineEvent.LABEL),
        BuildEventIdUtil.structuredCommandlineId(CommandLineEvent.ToolCommandLineEvent.LABEL),
        BuildEventIdUtil.buildMetadataId(),
        BuildEventIdUtil.optionsParsedId(),
        BuildEventIdUtil.workspaceStatusId(),
        BuildEventIdUtil.targetPatternExpanded(request().getTargets()),
        BuildEventIdUtil.buildFinished());
  }

  @Override
  public final BuildEventStreamProtos.BuildEvent asStreamProto(BuildEventContext converters) {
    BuildEventStreamProtos.BuildStarted.Builder started =
        BuildEventStreamProtos.BuildStarted.newBuilder()
            .setUuid(request().getId().toString())
            .setStartTime(Timestamps.fromMillis(request().getStartTime()))
            .setStartTimeMillis(request().getStartTime())
            .setBuildToolVersion(BlazeVersionInfo.instance().getVersion())
            .setOptionsDescription(request().getOptionsDescription())
            .setCommand(request().getCommandName())
            .setServerPid(ProcessHandle.current().pid())
            .setWorkingDirectory(pwd());
    if (workspace() != null) {
      started.setWorkspaceDirectory(workspace());
    }
    return GenericBuildEvent.protoChaining(this).setStarted(started.build()).build();
  }
}
