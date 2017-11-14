// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.buildeventstream;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.Path;
import com.google.protobuf.ByteString;
import java.util.Collection;

/** Event reporting on statistics about the build. */
public class BuildToolLogs implements BuildEventWithOrderConstraint {
  private final Collection<Pair<String, ByteString>> directValues;
  private final Collection<Pair<String, Path>> logFiles;

  public BuildToolLogs(
      Collection<Pair<String, ByteString>> directValues, Collection<Pair<String, Path>> logFiles) {
    this.directValues = directValues;
    this.logFiles = logFiles;
  }

  @Override
  public BuildEventId getEventId() {
    return BuildEventId.buildToolLogs();
  }

  @Override
  public Collection<BuildEventId> getChildrenEvents() {
    return ImmutableList.<BuildEventId>of();
  }

  @Override
  public BuildEventStreamProtos.BuildEvent asStreamProto(BuildEventConverters converters) {
    BuildEventStreamProtos.BuildToolLogs.Builder toolLogs =
        BuildEventStreamProtos.BuildToolLogs.newBuilder();
    for (Pair<String, ByteString> direct : directValues) {
      toolLogs.addLog(
          BuildEventStreamProtos.File.newBuilder()
              .setName(direct.getFirst())
              .setContents(direct.getSecond())
              .build());
    }
    for (Pair<String, Path> logFile : logFiles) {
      toolLogs.addLog(
          BuildEventStreamProtos.File.newBuilder()
              .setName(logFile.getFirst())
              .setUri(converters.pathConverter().apply(logFile.getSecond()))
              .build());
    }
    return GenericBuildEvent.protoChaining(this).setBuildToolLogs(toolLogs.build()).build();
  }

  @Override
  public Collection<BuildEventId> postedAfter() {
    return ImmutableList.of(BuildEventId.buildFinished());
  }
}
