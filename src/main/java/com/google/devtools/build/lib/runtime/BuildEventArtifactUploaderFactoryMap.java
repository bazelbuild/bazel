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
package com.google.devtools.build.lib.runtime;

import com.google.common.collect.ImmutableMap;
import java.util.SortedMap;
import java.util.TreeMap;
import javax.annotation.Nullable;
import javax.annotation.concurrent.ThreadSafe;

/** Selects between multiple available upload strategies. */
@ThreadSafe
public class BuildEventArtifactUploaderFactoryMap {
  private final ImmutableMap<String, BuildEventArtifactUploaderFactory> uploaders;

  private BuildEventArtifactUploaderFactoryMap(
      ImmutableMap<String, BuildEventArtifactUploaderFactory> uploaders) {
    this.uploaders = uploaders;
  }

  public BuildEventArtifactUploaderFactory select(@Nullable String name) {
    if (name == null && !uploaders.values().isEmpty()) {
      // TODO(b/110235226): We currently choose the strategy with alphabetically first strategy,
      // which happens to be backwards-compatible; we need to set
      // experimental_build_event_upload_strategy to appropriate default values instead, and then
      // make it an error to pass null.
      return uploaders.values().iterator().next();
    }
    return uploaders.getOrDefault(
        name, BuildEventArtifactUploaderFactory.LOCAL_FILES_UPLOADER_FACTORY);
  }

  /** Builder class for {@link BuildEventArtifactUploaderFactoryMap}. */
  public static class Builder {
    private final SortedMap<String, BuildEventArtifactUploaderFactory> uploaders = new TreeMap<>();

    public Builder() {}

    public Builder add(String name, BuildEventArtifactUploaderFactory uploader) {
      uploaders.put(name, uploader);
      return this;
    }

    public BuildEventArtifactUploaderFactoryMap build() {
      return new BuildEventArtifactUploaderFactoryMap(ImmutableMap.copyOf(uploaders));
    }
  }
}
