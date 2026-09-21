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

import java.util.Arrays;
import java.util.Objects;
import javax.annotation.Nullable;

/** Concrete record implementation of {@link LookupResult}. */
@SuppressWarnings("ArrayRecordComponent")
public record LookupResultImpl(
    byte[] value, @Nullable byte[] invalidationFingerprint, int missReason)
    implements LookupResult {
  public LookupResultImpl(byte[] value) {
    this(value, null, 0); // 0 corresponds to MISS_REASON_UNSPECIFIED
  }

  public LookupResultImpl(byte[] value, int missReason) {
    this(value, null, missReason);
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (!(o instanceof LookupResult that)) {
      return false;
    }
    return missReason == that.missReason()
        && Arrays.equals(value, that.value())
        && Arrays.equals(invalidationFingerprint, that.invalidationFingerprint());
  }

  @Override
  public int hashCode() {
    return Objects.hash(
        missReason, Arrays.hashCode(value), Arrays.hashCode(invalidationFingerprint));
  }
}
