// Copyright 2025 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.buildtool.buildevent;

import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId;

/**
 * Posts a pending {@link ReplaceableBuildEvent} immediately. It no longer waits for either the
 * build to end or a non-replaceable version to replace it.
 */
public final class ReleaseReplaceableBuildEvent {
  private final BuildEventId eventId;

  public ReleaseReplaceableBuildEvent(BuildEventId eventId) {
    this.eventId = eventId;
  }

  public BuildEventId getEventId() {
    return eventId;
  }
}
