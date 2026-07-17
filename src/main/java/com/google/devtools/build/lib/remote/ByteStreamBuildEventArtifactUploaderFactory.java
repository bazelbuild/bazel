// Copyright 2018 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package com.google.devtools.build.lib.remote;

import static com.google.common.base.Preconditions.checkState;

import com.google.devtools.build.lib.buildeventstream.BuildEventArtifactUploader;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.remote.options.RemoteBuildEventUploadMode;
import com.google.devtools.build.lib.runtime.BuildEventArtifactUploaderFactory;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import java.util.concurrent.Executor;
import javax.annotation.Nullable;

/** A factory for {@link ByteStreamBuildEventArtifactUploader}. */
class ByteStreamBuildEventArtifactUploaderFactory implements BuildEventArtifactUploaderFactory {

  private final Executor executor;
  private final ExtendedEventHandler reporter;
  private final boolean verboseFailures;
  private final CombinedCache combinedCache;
  private final String remoteInstanceName;
  private final String remoteBytestreamUriPrefix;
  private final String buildRequestId;
  private final String commandId;
  private final RemoteBuildEventUploadMode remoteBuildEventUploadMode;

  @Nullable private ByteStreamBuildEventArtifactUploader uploader;

  ByteStreamBuildEventArtifactUploaderFactory(
      Executor executor,
      ExtendedEventHandler reporter,
      boolean verboseFailures,
      CombinedCache combinedCache,
      String remoteInstanceName,
      String remoteBytestreamUriPrefix,
      String buildRequestId,
      String commandId,
      RemoteBuildEventUploadMode remoteBuildEventUploadMode) {
    this.executor = executor;
    this.reporter = reporter;
    this.verboseFailures = verboseFailures;
    this.combinedCache = combinedCache;
    this.remoteInstanceName = remoteInstanceName;
    this.remoteBytestreamUriPrefix = remoteBytestreamUriPrefix;
    this.buildRequestId = buildRequestId;
    this.commandId = commandId;
    this.remoteBuildEventUploadMode = remoteBuildEventUploadMode;
  }

  @Override
  public BuildEventArtifactUploader create(CommandEnvironment env) {
    checkState(uploader == null, "Already created");
    uploader =
        new ByteStreamBuildEventArtifactUploader(
            executor,
            reporter,
            verboseFailures,
            combinedCache.retain(),
            remoteInstanceName,
            remoteBytestreamUriPrefix,
            buildRequestId,
            commandId,
            env.getXattrProvider(),
            remoteBuildEventUploadMode);
    env.getEventBus().register(uploader);
    return uploader;
  }

  @Nullable
  public ByteStreamBuildEventArtifactUploader get() {
    return uploader;
  }
}
