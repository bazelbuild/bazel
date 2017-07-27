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

package com.google.devtools.build.lib.actions;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.buildeventstream.BuildEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEventConverters;
import com.google.devtools.build.lib.buildeventstream.BuildEventId;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos;
import com.google.devtools.build.lib.buildeventstream.BuildEventWithConfiguration;
import com.google.devtools.build.lib.buildeventstream.GenericBuildEvent;
import com.google.devtools.build.lib.buildeventstream.NullConfiguration;
import com.google.devtools.build.lib.buildeventstream.PathConverter;
import com.google.devtools.build.lib.vfs.Path;
import java.util.Collection;

/**
 * This event is fired during the build, when an action is executed. It contains information about
 * the action: the Action itself, and the output file names its stdout and stderr are recorded in.
 */
public class ActionExecutedEvent implements BuildEventWithConfiguration {
  private final Action action;
  private final ActionExecutionException exception;
  private final Path stdout;
  private final Path stderr;

  public ActionExecutedEvent(Action action,
      ActionExecutionException exception, Path stdout, Path stderr) {
    this.action = action;
    this.exception = exception;
    this.stdout = stdout;
    this.stderr = stderr;
  }

  public Action getAction() {
    return action;
  }

  // null if action succeeded
  public ActionExecutionException getException() {
    return exception;
  }

  public String getStdout() {
    if (stdout == null) {
      return null;
    }
    return stdout.toString();
  }

  public String getStderr() {
    if (stderr == null) {
      return null;
    }
    return stderr.toString();
  }

  @Override
  public BuildEventId getEventId() {
    if (action.getOwner() == null) {
      return BuildEventId.actionCompleted(action.getPrimaryOutput().getPath());
    } else {
      return BuildEventId.actionCompleted(
          action.getPrimaryOutput().getPath(),
          action.getOwner().getLabel(),
          action.getOwner().getConfigurationChecksum());
    }
  }

  @Override
  public Collection<BuildEventId> getChildrenEvents() {
    return ImmutableList.<BuildEventId>of();
  }

  @Override
  public Collection<BuildEvent> getConfigurations() {
    if (action.getOwner() != null) {
      BuildEvent configuration = action.getOwner().getConfiguration();
      if (configuration == null) {
        configuration = new NullConfiguration();
      }
      return ImmutableList.of(configuration);
    } else {
      return ImmutableList.<BuildEvent>of();
    }
  }

  @Override
  public BuildEventStreamProtos.BuildEvent asStreamProto(BuildEventConverters converters) {
    PathConverter pathConverter = converters.pathConverter();
    BuildEventStreamProtos.ActionExecuted.Builder actionBuilder =
        BuildEventStreamProtos.ActionExecuted.newBuilder().setSuccess(getException() == null);
    if (exception != null && exception.getExitCode() != null) {
      actionBuilder.setExitCode(exception.getExitCode().getNumericExitCode());
    }
    if (stdout != null) {
      actionBuilder.setStdout(
          BuildEventStreamProtos.File.newBuilder()
          .setName("stdout")
          .setUri(pathConverter.apply(stdout))
          .build());
    }
    if (stderr != null) {
      actionBuilder.setStderr(
          BuildEventStreamProtos.File.newBuilder()
          .setName("stderr")
          .setUri(pathConverter.apply(stderr))
          .build());
    }
    if (action.getOwner() != null && action.getOwner().getLabel() != null) {
      actionBuilder.setLabel(action.getOwner().getLabel().toString());
    }
    if (action.getOwner() != null) {
      BuildEvent configuration = action.getOwner().getConfiguration();
      if (configuration == null) {
        configuration = new NullConfiguration();
      }
      actionBuilder.setConfiguration(configuration.getEventId().asStreamProto().getConfiguration());
    }
    if (exception == null) {
      actionBuilder.setPrimaryOutput(
          BuildEventStreamProtos.File.newBuilder()
              .setUri(pathConverter.apply(action.getPrimaryOutput().getPath()))
              .build());
    }
    return GenericBuildEvent.protoChaining(this).setAction(actionBuilder.build()).build();
  }
}
