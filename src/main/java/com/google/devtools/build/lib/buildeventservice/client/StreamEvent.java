// Copyright 2024 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.buildeventservice.client;

import com.google.devtools.build.lib.skybridge.SkybridgeInterface;
import java.time.Instant;

/** An event sent over a {@link StreamContext}. */
@SkybridgeInterface
public interface StreamEvent {
  /** The time at which the event occurred. */
  Instant eventTime();

  /** The sequence number of the event. */
  long sequenceNumber();

  /** An event containing a {@link BuildEventStreamProtos.BuildEvent}. */
  interface BazelEvent extends StreamEvent {
    /** The {@link BuildEventStreamProtos.BuildEvent} in wire format. */
    byte[] payload();
  }

  /** An event signalling the end of the stream. */
  interface StreamFinished extends StreamEvent {}
}
