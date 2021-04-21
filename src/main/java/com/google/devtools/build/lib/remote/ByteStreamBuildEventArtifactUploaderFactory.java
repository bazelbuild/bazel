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
import com.google.devtools.build.lib.remote.common.MissingDigestsFinder;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.runtime.BuildEventArtifactUploaderFactory;
import com.google.devtools.build.lib.runtime.CommandEnvironment;

/**
 * A factory for {@link ByteStreamBuildEventArtifactUploader}.
 */
class ByteStreamBuildEventArtifactUploaderFactory implements
    BuildEventArtifactUploaderFactory {

  private final ByteStreamUploader uploader;
  private final String remoteServerInstanceName;
  private final String buildRequestId;
  private final String commandId;
  private final MissingDigestsFinder missingDigestsFinder;

  ByteStreamBuildEventArtifactUploaderFactory(
      ByteStreamUploader uploader,
      MissingDigestsFinder missingDigestsFinder,
      String remoteServerInstanceName,
      String buildRequestId,
      String commandId) {
    this.uploader = uploader;
    this.missingDigestsFinder = missingDigestsFinder;
    this.remoteServerInstanceName = remoteServerInstanceName;
    this.buildRequestId = buildRequestId;
    this.commandId = commandId;
  }

  @Override
  public BuildEventArtifactUploader create(CommandEnvironment env) {
    return new ByteStreamBuildEventArtifactUploader(
        uploader.retain(),
        missingDigestsFinder,
        remoteServerInstanceName,
        buildRequestId,
        commandId,
        env.getOptions().getOptions(RemoteOptions.class).buildEventUploadMaxThreads);
  }
}
