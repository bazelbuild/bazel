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
import javax.annotation.Nullable;

/** The result of a remote analysis cache lookup. */
@SuppressWarnings("ArrayRecordComponent") // To keep the SkybridgeInterface simple.
@SkybridgeInterface
public interface LookupResult {
  /** The serialized SkyValue, or empty if the lookup missed. */
  byte[] value();

  /** The invalidation fingerprint of the node, or null if missing. */
  @Nullable
  byte[] invalidationFingerprint();

  /**
   * Corresponds to com.google.devtools.build.lib.skyframe.serialization.analysis.proto.MissReason.
   * We use an int instead of the proto to keep the SkybridgeInterface simple. Since older LCs may
   * not know about the new enum values, consumers must check for possible version skews and map the
   * value to MISS_REASON_UNSPECIFIED.
   */
  int missReason();
}
