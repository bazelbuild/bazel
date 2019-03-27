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

import com.google.devtools.build.lib.buildeventstream.BuildEventArtifactUploader;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.runtime.BuildEventArtifactUploaderFactory;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import io.grpc.Context;
import javax.annotation.Nullable;

/**
 * A factory for {@link ByteStreamBuildEventArtifactUploader}.
 */
class ByteStreamBuildEventArtifactUploaderFactory implements
    BuildEventArtifactUploaderFactory {

  private final ByteStreamUploader uploader;
  private final String remoteServerName;
  private final Context ctx;
  private final @Nullable String remoteInstanceName;

  ByteStreamBuildEventArtifactUploaderFactory(
      ByteStreamUploader uploader, String remoteServerName, Context ctx,
      @Nullable String remoteInstanceName) {
    this.uploader = uploader;
    this.remoteServerName = remoteServerName;
    this.ctx = ctx;
    this.remoteInstanceName = remoteInstanceName;
  }

  @Override
  public BuildEventArtifactUploader create(CommandEnvironment env) {
    return new ByteStreamBuildEventArtifactUploader(
        uploader.retain(),
        remoteServerName,
        ctx,
        remoteInstanceName,
        env.getOptions().getOptions(RemoteOptions.class).buildEventUploadMaxThreads);
  }
}
