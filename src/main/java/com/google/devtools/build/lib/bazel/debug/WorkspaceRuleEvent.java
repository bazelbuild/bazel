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
import com.google.devtools.build.lib.bazel.debug.proto.WorkspaceLogProtos.FileEvent;
import com.google.devtools.build.lib.bazel.debug.proto.WorkspaceLogProtos.OsEvent;
import com.google.devtools.build.lib.bazel.debug.proto.WorkspaceLogProtos.SymlinkEvent;
import com.google.devtools.build.lib.bazel.debug.proto.WorkspaceLogProtos.TemplateEvent;
import com.google.devtools.build.lib.bazel.debug.proto.WorkspaceLogProtos.WhichEvent;
import com.google.devtools.build.lib.events.ExtendedEventHandler.ProgressLike;
import com.google.devtools.build.lib.events.Location;
import java.net.URL;
import java.util.List;
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

  /** Creates a new WorkspaceRuleEvent for a download event. */
  public static WorkspaceRuleEvent newDownloadEvent(
      List<URL> urls,
      String output,
      String sha256,
      Boolean executable,
      String ruleLabel,
      Location location) {
    WorkspaceLogProtos.DownloadEvent.Builder e =
        WorkspaceLogProtos.DownloadEvent.newBuilder()
            .setOutput(output)
            .setSha256(sha256)
            .setExecutable(executable);
    for (URL u : urls) {
      e.addUrl(u.toString());
    }

    WorkspaceLogProtos.WorkspaceEvent.Builder result =
        WorkspaceLogProtos.WorkspaceEvent.newBuilder();
    result = result.setDownloadEvent(e.build());
    if (location != null) {
      result = result.setLocation(location.print());
    }
    if (ruleLabel != null) {
      result = result.setRule(ruleLabel);
    }
    return new WorkspaceRuleEvent(result.build());
  }

  /** Creates a new WorkspaceRuleEvent for an extract event. */
  public static WorkspaceRuleEvent newExtractEvent(
      String archive,
      String output,
      String stripPrefix,
      String ruleLabel,
      Location location) {
    WorkspaceLogProtos.ExtractEvent.Builder e =
        WorkspaceLogProtos.ExtractEvent.newBuilder()
            .setArchive(archive)
            .setOutput(output)
            .setStripPrefix(stripPrefix);

    WorkspaceLogProtos.WorkspaceEvent.Builder result =
        WorkspaceLogProtos.WorkspaceEvent.newBuilder();
    result = result.setExtractEvent(e.build());
    if (location != null) {
      result = result.setLocation(location.print());
    }
    if (ruleLabel != null) {
      result = result.setRule(ruleLabel);
    }
    return new WorkspaceRuleEvent(result.build());
  }

  /** Creates a new WorkspaceRuleEvent for a download and extract event. */
  public static WorkspaceRuleEvent newDownloadAndExtractEvent(
      List<URL> urls,
      String output,
      String sha256,
      String type,
      String stripPrefix,
      String ruleLabel,
      Location location) {
    WorkspaceLogProtos.DownloadAndExtractEvent.Builder e =
        WorkspaceLogProtos.DownloadAndExtractEvent.newBuilder()
            .setOutput(output)
            .setSha256(sha256)
            .setType(type)
            .setStripPrefix(stripPrefix);
    for (URL u : urls) {
      e.addUrl(u.toString());
    }

    WorkspaceLogProtos.WorkspaceEvent.Builder result =
        WorkspaceLogProtos.WorkspaceEvent.newBuilder();
    result = result.setDownloadAndExtractEvent(e.build());
    if (location != null) {
      result = result.setLocation(location.print());
    }
    if (ruleLabel != null) {
      result = result.setRule(ruleLabel);
    }
    return new WorkspaceRuleEvent(result.build());
  }

  /** Creates a new WorkspaceRuleEvent for a file event. */
  public static WorkspaceRuleEvent newFileEvent(
      String path, String content, boolean executable, String ruleLabel, Location location) {
    FileEvent e =
        WorkspaceLogProtos.FileEvent.newBuilder()
            .setPath(path)
            .setContent(content)
            .setExecutable(executable)
            .build();

    WorkspaceLogProtos.WorkspaceEvent.Builder result =
        WorkspaceLogProtos.WorkspaceEvent.newBuilder();
    result = result.setFileEvent(e);
    if (location != null) {
      result = result.setLocation(location.print());
    }
    if (ruleLabel != null) {
      result = result.setRule(ruleLabel);
    }
    return new WorkspaceRuleEvent(result.build());
  }

  /** Creates a new WorkspaceRuleEvent for an os event. */
  public static WorkspaceRuleEvent newOsEvent(String ruleLabel, Location location) {
    OsEvent e = WorkspaceLogProtos.OsEvent.getDefaultInstance();

    WorkspaceLogProtos.WorkspaceEvent.Builder result =
        WorkspaceLogProtos.WorkspaceEvent.newBuilder();
    result = result.setOsEvent(e);
    if (location != null) {
      result = result.setLocation(location.print());
    }
    if (ruleLabel != null) {
      result = result.setRule(ruleLabel);
    }
    return new WorkspaceRuleEvent(result.build());
  }

  /** Creates a new WorkspaceRuleEvent for a symlink event. */
  public static WorkspaceRuleEvent newSymlinkEvent(
      String from, String to, String ruleLabel, Location location) {
    SymlinkEvent e =
        WorkspaceLogProtos.SymlinkEvent.newBuilder().setTarget(from).setPath(to).build();

    WorkspaceLogProtos.WorkspaceEvent.Builder result =
        WorkspaceLogProtos.WorkspaceEvent.newBuilder();
    result = result.setSymlinkEvent(e);
    if (location != null) {
      result = result.setLocation(location.print());
    }
    if (ruleLabel != null) {
      result = result.setRule(ruleLabel);
    }
    return new WorkspaceRuleEvent(result.build());
  }

  /** Creates a new WorkspaceRuleEvent for a template event. */
  public static WorkspaceRuleEvent newTemplateEvent(
      String path,
      String template,
      Map<String, String> substitutions,
      boolean executable,
      String ruleLabel,
      Location location) {
    TemplateEvent e =
        WorkspaceLogProtos.TemplateEvent.newBuilder()
            .setPath(path)
            .setTemplate(template)
            .putAllSubstitutions(substitutions)
            .setExecutable(executable)
            .build();

    WorkspaceLogProtos.WorkspaceEvent.Builder result =
        WorkspaceLogProtos.WorkspaceEvent.newBuilder();
    result = result.setTemplateEvent(e);
    if (location != null) {
      result = result.setLocation(location.print());
    }
    if (ruleLabel != null) {
      result = result.setRule(ruleLabel);
    }
    return new WorkspaceRuleEvent(result.build());
  }

  /** Creates a new WorkspaceRuleEvent for a which event. */
  public static WorkspaceRuleEvent newWhichEvent(
      String program, String ruleLabel, Location location) {
    WhichEvent e = WorkspaceLogProtos.WhichEvent.newBuilder().setProgram(program).build();

    WorkspaceLogProtos.WorkspaceEvent.Builder result =
        WorkspaceLogProtos.WorkspaceEvent.newBuilder();
    result = result.setWhichEvent(e);
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
