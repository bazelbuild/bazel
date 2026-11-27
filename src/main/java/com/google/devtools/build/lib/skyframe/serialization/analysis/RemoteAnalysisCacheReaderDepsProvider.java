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

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.devtools.build.lib.skyframe.serialization.SkyValueRetriever;
import com.google.devtools.build.lib.skyframe.serialization.SkyValueRetriever.RetrievalPhase;
import com.google.devtools.build.lib.skyframe.serialization.SkyValueRetriever.RetrievalResult;
import com.google.devtools.build.skyframe.SkyKey;
import javax.annotation.Nullable;

/** Functionality needed to retrieve values from the remote cache. */
public interface RemoteAnalysisCacheReaderDepsProvider extends SkycacheCommonDepsProvider {
  @Nullable
  RemoteAnalysisCacheClient getAnalysisCacheClient() throws InterruptedException;

  @Nullable
  SkyValueRetriever getSkyValueRetriever() throws InterruptedException;

  @Nullable
  SkycacheUploadClient getSkycacheUploadClient() throws InterruptedException;

  void recordRetrievalResult(
      RetrievalResult retrievalResult,
      SkyKey key,
      ImmutableMap<RetrievalPhase, Long> phaseDurationMicros);

  void recordSerializationException(
      SerializationException e, SkyKey key, ImmutableMap<RetrievalPhase, Long> phaseDurationMicros);

  /** Returns true if there was at least one remote cache hit during this invocation. */
  boolean sawCacheHit();

  /** Returns true if bailing out on the first missing fingerprint is enabled. */
  boolean shouldBailOutOnMissingFingerprint();
}
