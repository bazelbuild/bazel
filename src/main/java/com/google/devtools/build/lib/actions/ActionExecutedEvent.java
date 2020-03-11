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

import com.google.common.base.MoreObjects;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.SpawnResult.MetadataLog;
import com.google.devtools.build.lib.buildeventstream.BuildEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEvent.LocalFile.LocalFileType;
import com.google.devtools.build.lib.buildeventstream.BuildEventContext;
import com.google.devtools.build.lib.buildeventstream.BuildEventId;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos;
import com.google.devtools.build.lib.buildeventstream.BuildEventWithConfiguration;
import com.google.devtools.build.lib.buildeventstream.GenericBuildEvent;
import com.google.devtools.build.lib.buildeventstream.NullConfiguration;
import com.google.devtools.build.lib.buildeventstream.PathConverter;
import com.google.devtools.build.lib.events.ExtendedEventHandler.ProgressLike;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Collection;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * This event is fired during the build, when an action is executed. It contains information about
 * the action: the Action itself, and the output file names its stdout and stderr are recorded in.
 */
public class ActionExecutedEvent implements BuildEventWithConfiguration, ProgressLike {
  private static final Logger logger = Logger.getLogger(ActionExecutedEvent.class.getName());

  private final PathFragment actionId;
  private final Action action;
  private final ActionExecutionException exception;
  private final Path primaryOutput;
  private final Path stdout;
  private final Path stderr;
  private final ImmutableList<MetadataLog> actionMetadataLogs;
  private final boolean isInMemoryFs;
  private final ErrorTiming timing;

  public ActionExecutedEvent(
      PathFragment actionId,
      Action action,
      ActionExecutionException exception,
      Path primaryOutput,
      Path stdout,
      Path stderr,
      ImmutableList<MetadataLog> actionMetadataLogs,
      ErrorTiming timing,
      boolean isInMemoryFs) {
    this.actionId = actionId;
    this.action = action;
    this.exception = exception;
    this.primaryOutput = primaryOutput;
    this.stdout = stdout;
    this.stderr = stderr;
    this.timing = timing;
    this.actionMetadataLogs = actionMetadataLogs;
    this.isInMemoryFs = isInMemoryFs;
    Preconditions.checkNotNull(this.actionMetadataLogs, this);
    Preconditions.checkState(
        (this.exception == null) == (this.timing == ErrorTiming.NO_ERROR), this);
  }

  public Action getAction() {
    return action;
  }

  // null if action succeeded
  public ActionExecutionException getException() {
    return exception;
  }

  public ErrorTiming errorTiming() {
    return timing;
  }

  public boolean hasInMemoryFs() {
    return isInMemoryFs;
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

  public ImmutableList<MetadataLog> getActionMetadataLogs() {
    return actionMetadataLogs;
  }

  @Override
  public BuildEventId getEventId() {
    if (action.getOwner() == null) {
      return BuildEventId.actionCompleted(actionId);
    } else {
      return BuildEventId.actionCompleted(
          actionId,
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
  public Collection<LocalFile> referencedLocalFiles() {
    ImmutableList.Builder<LocalFile> localFiles = ImmutableList.builder();
    if (stdout != null) {
      localFiles.add(new LocalFile(stdout, LocalFileType.STDOUT));
    }
    if (stderr != null) {
      localFiles.add(new LocalFile(stderr, LocalFileType.STDERR));
    }
    for (MetadataLog actionMetadataLog : actionMetadataLogs) {
      localFiles.add(new LocalFile(actionMetadataLog.getFilePath(), LocalFileType.LOG));
    }
    if (exception == null) {
      localFiles.add(new LocalFile(primaryOutput, LocalFileType.OUTPUT));
    }
    return localFiles.build();
  }

  @Override
  public BuildEventStreamProtos.BuildEvent asStreamProto(BuildEventContext converters) {
    PathConverter pathConverter = converters.pathConverter();
    BuildEventStreamProtos.ActionExecuted.Builder actionBuilder =
        BuildEventStreamProtos.ActionExecuted.newBuilder()
            .setSuccess(getException() == null)
            .setType(action.getMnemonic());

    if (exception != null && exception.getExitCode() != null) {
      // TODO(b/150405553): This statement seems to be confused. The exit_code field of
      //  ActionExecuted is documented as "The exit code of the action, if it is available."
      //  However, the value returned by exception.getExitCode().getNumericExitCode() is intended as
      //  an exit code that this Bazel invocation might return to the user.
      actionBuilder.setExitCode(exception.getExitCode().getNumericExitCode());
    }
    if (stdout != null) {
      String uri = pathConverter.apply(stdout);
      if (uri != null) {
        actionBuilder.setStdout(
            BuildEventStreamProtos.File.newBuilder().setName("stdout").setUri(uri).build());
      }
    }
    if (stderr != null) {
      String uri = pathConverter.apply(stderr);
      if (uri != null) {
        actionBuilder.setStderr(
            BuildEventStreamProtos.File.newBuilder().setName("stderr").setUri(uri).build());
      }
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
    for (MetadataLog actionMetadataLog : actionMetadataLogs) {
      String uri = pathConverter.apply(actionMetadataLog.getFilePath());
      if (uri != null) {
        actionBuilder.addActionMetadataLogs(
            BuildEventStreamProtos.File.newBuilder()
                .setName(actionMetadataLog.getName())
                .setUri(uri)
                .build());
      }
    }
    if (exception == null) {
      String uri = pathConverter.apply(primaryOutput);
      if (uri != null) {
        actionBuilder.setPrimaryOutput(
            BuildEventStreamProtos.File.newBuilder().setUri(uri).build());
      }
    }
    try {
      if (action instanceof CommandAction) {
        actionBuilder.addAllCommandLine(((CommandAction) action).getArguments());
      }
    } catch (CommandLineExpansionException e) {
      // Command-line not available, so just not report it
      logger.log(Level.INFO, "Could no compute commandline of reported action", e);
    }
    return GenericBuildEvent.protoChaining(this).setAction(actionBuilder.build()).build();
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("exception", exception)
        .add("timing", timing)
        .add("stdout", stdout)
        .add("stderr", stderr)
        .add("action", action)
        .toString();
  }

  /** When an error occurred that aborted action execution, if any. */
  public enum ErrorTiming {
    NO_ERROR,
    BEFORE_EXECUTION,
    AFTER_EXECUTION
  }
}
