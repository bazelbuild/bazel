// Copyright 2016 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableList;
import java.util.Collection;

/**
 * A {@link BuildEvent} reporting about progress.
 *
 * <p>Events of this type are used to report updates on the progress of the build. They are also
 * used to chain in failure events where the canonical parents (e.g., test suites) can only be
 * reported later.
 */
public final class ProgressEvent extends GenericBuildEvent {

  /** The {@link BuildEventId} of the first progress event to be reported. */
  public static BuildEventId INITIAL_PROGRESS_UPDATE = BuildEventId.progressId(0);

  private ProgressEvent(BuildEventId id, Collection<BuildEventId> children) {
    super(id, children);
  }

  @Override
  public BuildEventStreamProtos.BuildEvent asStreamProto(BuildEventConverters converters) {
    return GenericBuildEvent.protoChaining(this)
        .setProgress(BuildEventStreamProtos.Progress.newBuilder().build())
        .build();
  }

  /** Create a regular progress update with the given running number. */
  public static BuildEvent progressUpdate(int number) {
    BuildEventId id = BuildEventId.progressId(number);
    BuildEventId next = BuildEventId.progressId(number + 1);
    return new ProgressEvent(id, ImmutableList.of(next));
  }

  /** Create a progress update event also chaining in a given id. */
  public static BuildEvent progressChainIn(int number, BuildEventId chainIn) {
    BuildEventId id = BuildEventId.progressId(number);
    BuildEventId next = BuildEventId.progressId(number + 1);
    return new ProgressEvent(id, ImmutableList.of(next, chainIn));
  }

  /**
   * A progress update event with a given id, that has no children (and hence usually is the last
   * progress event in the stream).
   */
  public static BuildEvent finalProgressUpdate(int number) {
    BuildEventId id = BuildEventId.progressId(number);
    return new ProgressEvent(id, ImmutableList.<BuildEventId>of());
  }
}
