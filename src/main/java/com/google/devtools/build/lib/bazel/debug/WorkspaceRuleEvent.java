// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.bazel.debug;

import com.google.devtools.build.lib.bazel.debug.proto.WorkspaceLogProtos;
import com.google.devtools.build.lib.events.ExtendedEventHandler.ProgressLike;
import com.google.devtools.build.lib.events.Location;
import java.util.Map;

/** An event to record events happening during workspace rule resolution */
public final class WorkspaceRuleEvent implements ProgressLike {
  WorkspaceLogProtos.WorkspaceEvent event;

  public WorkspaceLogProtos.WorkspaceEvent getLogEvent() {
    return event;
  }

  private WorkspaceRuleEvent(WorkspaceLogProtos.WorkspaceEvent event) {
    this.event = event;
  }

  /**
   * Creates a new WorkspaceRuleEvent for an execution event.
   *
   * <p>Note: we will add more granular information as needed.
   */
  public static WorkspaceRuleEvent newExecuteEvent(
      Iterable<Object> args,
      Integer timeout,
      Map<String, String> commonEnvironment,
      Map<String, String> customEnvironment,
      String outputDirectory,
      boolean quiet,
      String ruleLabel,
      Location location) {

    WorkspaceLogProtos.ExecuteEvent.Builder e =
        WorkspaceLogProtos.ExecuteEvent.newBuilder()
            .setTimeoutSeconds(timeout.intValue())
            .setOutputDirectory(outputDirectory)
            .setQuiet(quiet);
    if (commonEnvironment != null) {
      e = e.putAllEnvironment(commonEnvironment);
    }
    if (customEnvironment != null) {
      e = e.putAllEnvironment(customEnvironment);
    }

    for (Object a : args) {
      e.addArguments(a.toString());
    }

    WorkspaceLogProtos.WorkspaceEvent.Builder result =
        WorkspaceLogProtos.WorkspaceEvent.newBuilder();
    result = result.setExecuteEvent(e.build());
    if (location != null) {
      result = result.setLocation(location.print());
    }
    if (ruleLabel != null) {
      result = result.setRule(ruleLabel);
    }
    return new WorkspaceRuleEvent(result.build());
  }

  /*
   * @return a message to log for this event
   */
  public String logMessage() {
    return event.toString();
  }
}
