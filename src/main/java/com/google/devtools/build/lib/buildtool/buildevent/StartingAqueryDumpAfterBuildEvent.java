// Copyright 2020 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.buildtool.buildevent;

import com.google.devtools.build.lib.buildeventstream.BuildEventArtifactUploader.UploadContext;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.vfs.Path;
import javax.annotation.Nullable;

/** Fired before the start of the aquery dump after a build. */
public class StartingAqueryDumpAfterBuildEvent implements ExtendedEventHandler.Postable {

  @Nullable private UploadContext streamingContext;
  @Nullable private Path localAqueryDumpPath;
  private final String aqueryDumpName;

  public StartingAqueryDumpAfterBuildEvent(UploadContext streamingContext, String aqueryDumpName) {
    this.streamingContext = streamingContext;
    this.aqueryDumpName = aqueryDumpName;
  }

  public StartingAqueryDumpAfterBuildEvent(Path localAqueryDumpPath, String aqueryDumpName) {
    this.localAqueryDumpPath = localAqueryDumpPath;
    this.aqueryDumpName = aqueryDumpName;
  }

  @Nullable
  public UploadContext getStreamingContext() {
    return streamingContext;
  }

  @Nullable
  public Path getLocalAqueryDumpPath() {
    return localAqueryDumpPath;
  }

  public String getAqueryDumpName() {
    return aqueryDumpName;
  }
}
