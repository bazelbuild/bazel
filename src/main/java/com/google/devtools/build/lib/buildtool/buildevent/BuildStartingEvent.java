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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.BlazeVersionInfo;
import com.google.devtools.build.lib.buildeventstream.BuildEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEventContext;
import com.google.devtools.build.lib.buildeventstream.BuildEventId;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos;
import com.google.devtools.build.lib.buildeventstream.GenericBuildEvent;
import com.google.devtools.build.lib.buildeventstream.ProgressEvent;
import com.google.devtools.build.lib.buildtool.BuildRequest;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.CommandLineEvent;
import com.google.devtools.build.lib.util.ProcessUtils;
import java.util.Collection;

/**
 * This event is fired from BuildTool#startRequest(). At this point, the set of target patters are
 * known, but have yet to be parsed.
 */
public final class BuildStartingEvent implements BuildEvent {
  private final String outputFileSystem;
  private final BuildRequest request;
  private final String workspace;
  private final String pwd;
  private final ImmutableList<BuildEventId> additionalChildEvents;

  /**
   * Construct the BuildStartingEvent
   *
   * @param request the build request.
   * @param env the environment of the request invocation.
   */
  public BuildStartingEvent(
      CommandEnvironment env,
      BuildRequest request,
      ImmutableList<BuildEventId> additionalChildEvents) {
    this.request = request;
    if (env != null) {
      this.outputFileSystem = env.determineOutputFileSystem();
      if (env.getDirectories().getWorkspace() != null) {
        this.workspace = env.getDirectories().getWorkspace().toString();
      } else {
        this.workspace = null;
      }
      this.pwd = env.getWorkingDirectory().toString();
    } else {
      this.workspace = null;
      this.pwd = null;
      this.outputFileSystem = null;
    }
    this.additionalChildEvents = additionalChildEvents;
  }

  /**
   * @return the output file system.
   */
  public String getOutputFileSystem() {
    return outputFileSystem;
  }

  /**
   * @return the active BuildRequest.
   */
  public BuildRequest getRequest() {
    return request;
  }

  @Override
  public BuildEventId getEventId() {
    return BuildEventId.buildStartedId();
  }

  @Override
  public Collection<BuildEventId> getChildrenEvents() {
    return ImmutableList.<BuildEventId>builder()
        .add(ProgressEvent.INITIAL_PROGRESS_UPDATE)
        .add(BuildEventId.unstructuredCommandlineId())
        .add(BuildEventId.structuredCommandlineId(CommandLineEvent.OriginalCommandLineEvent.LABEL))
        .add(BuildEventId.structuredCommandlineId(CommandLineEvent.CanonicalCommandLineEvent.LABEL))
        .add(BuildEventId.structuredCommandlineId(CommandLineEvent.ToolCommandLineEvent.LABEL))
        .add(BuildEventId.optionsParsedId())
        .add(BuildEventId.workspaceStatusId())
        .add(BuildEventId.targetPatternExpanded(request.getTargets()))
        .add(BuildEventId.buildFinished())
        .addAll(additionalChildEvents)
        .build();
  }

  @Override
  public BuildEventStreamProtos.BuildEvent asStreamProto(BuildEventContext converters) {
    BuildEventStreamProtos.BuildStarted.Builder started =
        BuildEventStreamProtos.BuildStarted.newBuilder()
            .setUuid(request.getId().toString())
            .setStartTimeMillis(request.getStartTime())
            .setBuildToolVersion(BlazeVersionInfo.instance().getVersion())
            .setOptionsDescription(request.getOptionsDescription())
            .setCommand(request.getCommandName())
            .setServerPid(ProcessUtils.getpid());
    if (pwd != null) {
      started.setWorkingDirectory(pwd);
    }
    if (workspace != null) {
      started.setWorkspaceDirectory(workspace);
    }
    return GenericBuildEvent.protoChaining(this).setStarted(started.build()).build();
  }
}
