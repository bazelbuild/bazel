// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe.serialization.analysis;

import com.google.devtools.build.lib.skybridge.SkybridgeInterface;
import java.util.Arrays;
import java.util.Objects;

/**
 * The result of a remote analysis cache lookup.
 *
 * @param value The serialized SkyValue, or empty if the lookup missed.
 * @param missReason Corresponds to
 *     com.google.devtools.build.lib.skyframe.serialization.analysis.proto.MissReason. We use an int
 *     instead of the proto to keep the SkybridgeInterface simple. Since older LCs may not know
 *     about the new enum values, consumers must check for possible version skews and map the value
 *     to MISS_REASON_UNSPECIFIED.
 */
@SuppressWarnings("ArrayRecordComponent") // To keep the SkybridgeInterface simple.
@SkybridgeInterface
public record LookupResult(byte[] value, int missReason) {
  public LookupResult(byte[] value) {
    this(value, 0); // 0 corresponds to MISS_REASON_UNSPECIFIED
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (!(o instanceof LookupResult that)) {
      return false;
    }
    return missReason == that.missReason && Arrays.equals(value, that.value);
  }

  @Override
  public int hashCode() {
    return Objects.hash(missReason, Arrays.hashCode(value));
  }
}
