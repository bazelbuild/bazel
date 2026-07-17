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

package com.google.devtools.build.lib.buildeventstream;

/**
 * A {@link BuildEvent} that buffers until either a non-replaceable version (with the same {@link
 * BuildEventId}) is posted to replace it or the build ends without the other version happening.
 *
 * <p>If the non-replaceable version posts later or was posted before this one, this one is
 * discarded without posting.
 *
 * <p>If the non-replaceable version isn't posted for any reason - including build completion, build
 * error, or a crash - this version posts.
 *
 * <p>This lets builds guarantee exactly once instance of an event gets posted. This is useful for
 * single-instance events that might be updated later in the build.
 */
public interface ReplaceableBuildEvent extends BuildEvent {
  /**
   * Is this event replaceable? If so, a non-replaceable version with the same {@link BuildEventId}
   * can replace it.
   */
  boolean replaceable();
}
