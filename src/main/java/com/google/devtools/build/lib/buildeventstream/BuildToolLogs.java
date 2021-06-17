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

import com.google.common.base.MoreObjects;
import com.google.common.collect.ImmutableList;
import com.google.common.flogger.GoogleLogger;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.buildeventstream.BuildEvent.LocalFile.LocalFileCompression;
import com.google.devtools.build.lib.buildeventstream.BuildEvent.LocalFile.LocalFileType;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.Path;
import com.google.protobuf.ByteString;
import java.util.Collection;
import java.util.concurrent.ExecutionException;

/** Event reporting on statistics about the build. */
public class BuildToolLogs implements BuildEventWithOrderConstraint {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  /** These values are posted as byte strings to the BEP. */
  private final Collection<Pair<String, ByteString>> directValues;
  /** These values are posted as Future URIs to the BEP. */
  private final Collection<Pair<String, ListenableFuture<String>>> futureUris;
  /**
   * These values are local files that are uploaded if required, and turned into URIs as part of the
   * process.
   */
  private final Collection<LogFileEntry> logFiles;

  public BuildToolLogs(
      Collection<Pair<String, ByteString>> directValues,
      Collection<Pair<String, ListenableFuture<String>>> futureUris,
      Collection<LogFileEntry> logFiles) {
    this.directValues = directValues;
    this.futureUris = futureUris;
    this.logFiles = logFiles;
  }

  @Override
  public BuildEventId getEventId() {
    return BuildEventIdUtil.buildToolLogs();
  }

  @Override
  public Collection<BuildEventId> getChildrenEvents() {
    return ImmutableList.of();
  }

  @Override
  public Collection<LocalFile> referencedLocalFiles() {
    ImmutableList.Builder<LocalFile> localFiles = ImmutableList.builder();
    for (LogFileEntry logFile : logFiles) {
      localFiles.add(logFile.toLocalFile());
    }
    return localFiles.build();
  }

  @Override
  public Collection<ListenableFuture<String>> remoteUploads() {
    ImmutableList.Builder<ListenableFuture<String>> remoteUploads = ImmutableList.builder();
    for (Pair<String, ListenableFuture<String>> uploadPair : futureUris) {
      remoteUploads.add(uploadPair.getSecond());
    }
    return remoteUploads.build();
  }

  @Override
  public BuildEventStreamProtos.BuildEvent asStreamProto(BuildEventContext converters) {
    BuildEventStreamProtos.BuildToolLogs.Builder toolLogs =
        BuildEventStreamProtos.BuildToolLogs.newBuilder();
    for (Pair<String, ByteString> direct : directValues) {
      toolLogs.addLog(
          BuildEventStreamProtos.File.newBuilder()
              .setName(direct.getFirst())
              .setContents(direct.getSecond())
              .build());
    }
    for (Pair<String, ListenableFuture<String>> directFuturePair : futureUris) {
      String name = directFuturePair.getFirst();
      ListenableFuture<String> directFuture = directFuturePair.getSecond();
      try {
        String uri =
            directFuture.isDone() && !directFuture.isCancelled()
                ? Futures.getDone(directFuture)
                : null;
        if (uri != null) {
          toolLogs.addLog(
              BuildEventStreamProtos.File.newBuilder().setName(name).setUri(uri).build());
        } else {
          logger.atInfo().log("Dropped unfinished upload: %s (%s)", name, directFuture);
        }
      } catch (ExecutionException e) {
        logger.atWarning().withCause(e).log("Skipping build tool log upload %s", name);
      }
    }
    for (LogFileEntry logFile : logFiles) {
      String uri = converters.pathConverter().apply(logFile.localPath);
      if (uri != null) {
        toolLogs.addLog(
            BuildEventStreamProtos.File.newBuilder().setName(logFile.name).setUri(uri).build());
      }
    }
    return GenericBuildEvent.protoChaining(this).setBuildToolLogs(toolLogs.build()).build();
  }

  @Override
  public Collection<BuildEventId> postedAfter() {
    return ImmutableList.of(BuildEventIdUtil.buildFinished());
  }

  /** A local log file. */
  public static class LogFileEntry {
    private final String name;
    private final Path localPath;
    private final LocalFileType fileType;
    private final LocalFileCompression compression;

    public LogFileEntry(
        String name, Path localPath, LocalFileType fileType, LocalFileCompression compression) {
      this.name = name;
      this.localPath = localPath;
      this.fileType = fileType;
      this.compression = compression;
    }

    public String getName() {
      return name;
    }

    public Path getLocalPath() {
      return localPath;
    }

    public LocalFileCompression getCompression() {
      return compression;
    }

    LocalFile toLocalFile() {
      return new LocalFile(localPath, fileType, compression);
    }

    @Override
    public String toString() {
      return MoreObjects.toStringHelper(this)
          .add("name", name)
          .add("localPath", localPath)
          .add("fileType", fileType)
          .toString();
    }
  }
}
