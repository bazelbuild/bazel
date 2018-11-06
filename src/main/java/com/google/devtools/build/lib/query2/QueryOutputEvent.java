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
package com.google.devtools.build.lib.query2;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.buildeventstream.BuildEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEvent.LocalFile.LocalFileType;
import com.google.devtools.build.lib.buildeventstream.BuildEventContext;
import com.google.devtools.build.lib.buildeventstream.BuildEventId;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos;
import com.google.devtools.build.lib.buildeventstream.GenericBuildEvent;
import com.google.devtools.build.lib.vfs.Path;
import java.util.Collection;

/**
 * A {@link BuildEvent} reporting the location of output from `query`, `cquery`, and `aquery`
 * commands.
 */
public abstract class QueryOutputEvent implements BuildEvent {
  public static final BuildEventId BUILD_EVENT_ID = BuildEventId.queryOutput();

  private QueryOutputEvent() {
  }

  public static QueryOutputEvent forOutputWrittenToStdout() {
    return new QueryOutputPrintedToConsoleEvent();
  }

  public static QueryOutputEvent forOutputWrittenToFile(
      String commandName, Path queryOutputFilePath) {
    return new QueryOutputWrittenToFileEvent(commandName, queryOutputFilePath);
  }

  @Override
  public BuildEventId getEventId() {
    return BUILD_EVENT_ID;
  }

  @Override
  public Collection<BuildEventId> getChildrenEvents() {
    return ImmutableList.of();
  }

  @VisibleForTesting
  public abstract boolean equalsForTesting(QueryOutputEvent other);

  private static class QueryOutputPrintedToConsoleEvent extends QueryOutputEvent {
    @Override
    public BuildEventStreamProtos.BuildEvent asStreamProto(BuildEventContext context) {
      BuildEventStreamProtos.QueryOutput queryOutput =
          BuildEventStreamProtos.QueryOutput.newBuilder()
              .setOutputPrintedToConsole(
                  BuildEventStreamProtos.QueryOutput.OutputPrintedToConsole.getDefaultInstance())
              .build();
      return GenericBuildEvent.protoChaining(this).setQueryOutput(queryOutput).build();
    }

    @Override
    public boolean equalsForTesting(QueryOutputEvent other) {
      return other instanceof QueryOutputPrintedToConsoleEvent;
    }
  }

  private static class QueryOutputWrittenToFileEvent extends QueryOutputEvent {
    private final String commandName;
    private final Path queryOutputFilePath;

    private QueryOutputWrittenToFileEvent(String commandName, Path queryOutputFilePath) {
      this.commandName = commandName;
      this.queryOutputFilePath = queryOutputFilePath;
    }

    @Override
    public BuildEventStreamProtos.BuildEvent asStreamProto(BuildEventContext context) {
      String uri = context.pathConverter().apply(queryOutputFilePath);
      BuildEventStreamProtos.QueryOutput.Builder queryOutputBuilder =
          BuildEventStreamProtos.QueryOutput.newBuilder();
      if (uri != null) {
        queryOutputBuilder.setQueryOutputFile(
            BuildEventStreamProtos.File.newBuilder()
                .setName(commandName + " output")
                .setUri(uri)
                .build());
      } else {
        queryOutputBuilder.setUploadFailed(
            BuildEventStreamProtos.QueryOutput.UploadFailed.getDefaultInstance());
      }

      return GenericBuildEvent.protoChaining(this)
          .setQueryOutput(queryOutputBuilder.build())
          .build();
    }

    @Override
    public Collection<LocalFile> referencedLocalFiles() {
      // TODO(nharmata): Introduce a new LocalFileType that has a more appropriate retention policy.
      // TODO(nharmata): Introduce a feature to LocalFile that causes the file gets deleted after
      // the upload succeeds.
      return ImmutableList.of(new LocalFile(queryOutputFilePath, LocalFileType.OUTPUT));
    }

    @Override
    public boolean equalsForTesting(QueryOutputEvent other) {
      if (!(other instanceof QueryOutputWrittenToFileEvent)) {
        return false;
      }
      QueryOutputWrittenToFileEvent otherQueryOutputWrittenToFileEvent =
          (QueryOutputWrittenToFileEvent) other;
      return this.commandName.equals(otherQueryOutputWrittenToFileEvent.commandName)
          && this.queryOutputFilePath.equals(
              otherQueryOutputWrittenToFileEvent.queryOutputFilePath);
    }
  }
}
