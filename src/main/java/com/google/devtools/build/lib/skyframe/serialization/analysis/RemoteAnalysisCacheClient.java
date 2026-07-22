// Copyright 2025 The Bazel Authors. All rights reserved.
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

import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.skybridge.SkybridgeInterface;
import com.google.devtools.build.lib.util.Bucket;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeoutException;

/** Interface to the remote analysis cache. */
@SkybridgeInterface
public interface RemoteAnalysisCacheClient {

  /** Timeout when accessing the future in order to shutdown the client. */
  int SHUTDOWN_TIMEOUT_IN_SECONDS = 5;

  /** The key for memoizing top-level targets lookup results. */
  @SkybridgeInterface
  interface TopLevelTargetsCacheKey {
    long evaluatingVersion();

    String configurationHash();

    boolean useFakeStampData();

    String blazeVersion();
  }

  /** Usage statistics. */
  @SkybridgeInterface
  interface Stats {
    long bytesSent();

    long bytesReceived();

    long requestsSent();

    long batches();

    List<Bucket> latencyMicros();

    List<Bucket> batchLatencyMicros();

    /**
     * Maps to
     * com.google.devtools.build.lib.skyframe.serialization.analysis.proto.TopLevelTargetsMatchStatus.
     * See {@link LookupTopLevelTargetsResult} for more details.
     */
    int matchStatus();
  }

  @SuppressWarnings("JdkImmutableCollections") // Keep the SkybridgeInterface simple.
  Stats EMPTY_STATS =
      new Stats() {
        @Override
        public long bytesSent() {
          return 0;
        }

        @Override
        public long bytesReceived() {
          return 0;
        }

        @Override
        public long requestsSent() {
          return 0;
        }

        @Override
        public long batches() {
          return 0;
        }

        @Override
        public List<Bucket> latencyMicros() {
          return List.of();
        }

        @Override
        public List<Bucket> batchLatencyMicros() {
          return List.of();
        }

        @Override
        public int matchStatus() {
          return 0; // TopLevelTargetsMatchStatus.MATCH_STATUS_UNSPECIFIED
        }
      };

  /** Looks up an entry in the remote analysis cache based on a serialized key. */
  ListenableFuture<LookupResult> lookup(byte[] key);

  /** Returns the usage statistics. */
  Stats getStats();

  /** Looks up the targets in the metadata table */
  @CanIgnoreReturnValue
  LookupTopLevelTargetsResult lookupTopLevelTargets(
      long evaluatingVersion,
      String configurationHash,
      boolean useFakeStampData,
      String bazelVersion)
      throws ExecutionException, TimeoutException, InterruptedException;

  /**
   * Sets the status of the metadata result to MATCH_STATUS_MISSING_FINGERPRINT. This signals that
   * the build bailed out due to a missing fingerprint during deserialization. This can happen after
   * having started in Skycache mode and having confirmed with metadata that cache hits were
   * possible.
   */
  void bailOutDueToMissingFingerprint();
}
