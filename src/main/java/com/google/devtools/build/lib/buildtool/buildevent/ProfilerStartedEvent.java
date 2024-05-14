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
package com.google.devtools.build.lib.buildtool.buildevent;

import com.google.devtools.build.lib.buildeventstream.BuildEventArtifactUploader.UploadContext;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.vfs.Path;
import javax.annotation.Nullable;

/**
 * This event is fired when the profiler is started.
 */
public class ProfilerStartedEvent implements ExtendedEventHandler.Postable {
  @Nullable private final Path profilePath;
  @Nullable private final UploadContext streamingContext;
  private final Profiler.Format format;
  @Nullable private final String name;

  public ProfilerStartedEvent(
      String name, Path profilePath, Profiler.Format format, UploadContext streamingContext) {
    this.profilePath = profilePath;
    this.streamingContext = streamingContext;
    this.format = format;
    this.name = name;
  }

  public Path getProfilePath() {
    return profilePath;
  }

  public UploadContext getStreamingContext() {
    return streamingContext;
  }

  public Profiler.Format getFormat() {
    return format;
  }

  public String getName() {
    return name;
  }
}
