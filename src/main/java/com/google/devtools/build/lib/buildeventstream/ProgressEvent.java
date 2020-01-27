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
import javax.annotation.Nullable;

/**
 * A {@link BuildEvent} reporting about progress.
 *
 * <p>Events of this type are used to report updates on the progress of the build. They are also
 * used to chain in failure events where the canonical parents (e.g., test suites) can only be
 * reported later.
 */
public final class ProgressEvent extends GenericBuildEvent {

  @Nullable private final String out;
  @Nullable private final String err;

  /** The {@link BuildEventId} of the first progress event to be reported. */
  public static final BuildEventId INITIAL_PROGRESS_UPDATE = BuildEventId.progressId(0);

  private ProgressEvent(
      BuildEventId id, Collection<BuildEventId> children, String out, String err) {
    super(id, children);
    this.out = out;
    this.err = err;
  }

  @Override
  public BuildEventStreamProtos.BuildEvent asStreamProto(BuildEventContext converters) {
    BuildEventStreamProtos.Progress.Builder builder = BuildEventStreamProtos.Progress.newBuilder();
    if (out != null) {
      builder.setStdout(out);
    }
    if (err != null) {
      builder.setStderr(err);
    }
    return GenericBuildEvent.protoChaining(this).setProgress(builder.build()).build();
  }

  /** Create a regular progress update with the given running number. */
  public static BuildEvent progressUpdate(int number, String out, String err) {
    BuildEventId id = BuildEventId.progressId(number);
    BuildEventId next = BuildEventId.progressId(number + 1);
    return new ProgressEvent(id, ImmutableList.of(next), out, err);
  }

  public static BuildEvent progressUpdate(int number) {
    return progressUpdate(number, null, null);
  }

  /** Create a progress update event also chaining in a given id. */
  public static BuildEvent progressChainIn(
      int number, BuildEventId chainIn, String out, String err) {
    BuildEventId id = BuildEventId.progressId(number);
    BuildEventId next = BuildEventId.progressId(number + 1);
    return new ProgressEvent(id, ImmutableList.of(next, chainIn), out, err);
  }

  public static BuildEvent progressChainIn(int number, BuildEventId chainIn) {
    return progressChainIn(number, chainIn, null, null);
  }

  /**
   * A progress update event with a given id, that has no children (and hence usually is the last
   * progress event in the stream).
   */
  public static BuildEvent finalProgressUpdate(
      int number, @Nullable String out, @Nullable String err) {
    BuildEventId id = BuildEventId.progressId(number);
    return new ProgressEvent(id, ImmutableList.<BuildEventId>of(), out, err);
  }

  public static BuildEvent finalProgressUpdate(int number) {
    return finalProgressUpdate(number, null, null);
  }
}
