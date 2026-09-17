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

package com.google.devtools.build.lib.buildeventservice;

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.base.MoreObjects;
import com.google.devtools.build.lib.buildeventservice.client.StreamEvent;
import java.time.Instant;
import java.util.Arrays;
import java.util.Objects;

/** Implementation of {@link StreamEvent.BazelEvent}. */
@SuppressWarnings("ArrayRecordComponent")
public record BazelEventImpl(Instant eventTime, long sequenceNumber, byte[] payload)
    implements StreamEvent.BazelEvent {
  public BazelEventImpl {
    checkNotNull(eventTime, "eventTime");
    checkNotNull(payload, "payload");
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    return o instanceof BazelEventImpl that
        && sequenceNumber == that.sequenceNumber
        && Objects.equals(eventTime, that.eventTime)
        && Arrays.equals(payload, that.payload);
  }

  @Override
  public int hashCode() {
    return Objects.hash(eventTime, sequenceNumber, Arrays.hashCode(payload));
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("eventTime", eventTime)
        .add("sequenceNumber", sequenceNumber)
        .add("payload", Arrays.toString(payload))
        .toString();
  }
}
