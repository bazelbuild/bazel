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

package com.google.devtools.build.lib.remote.chunking;

import build.bazel.remote.execution.v2.CacheCapabilities;
import build.bazel.remote.execution.v2.ChunkingFunction;
import build.bazel.remote.execution.v2.ServerCapabilities;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import javax.annotation.Nullable;

/**
 * Configuration for content-defined chunking. All sizes are in bytes.
 *
 * <p>Each implementation corresponds to one of the chunking functions defined by the Remote
 * Execution API and carries the parameters negotiated from the server capabilities.
 */
public sealed interface ChunkingConfig permits FastCdcChunkingConfig, RepMaxCdcChunkingConfig {

  /** The chunking function this configuration applies to. */
  ChunkingFunction.Value chunkingFunction();

  /** The minimum size of a chunk. Only the last chunk of a blob may be smaller. */
  int minChunkSize();

  /** The maximum size of a chunk. */
  int maxChunkSize();

  /**
   * Blobs larger than this should be chunked. Always equal to {@link #maxChunkSize()}: anything
   * above it splits into at least two chunks, while anything at or below it would come back as a
   * single chunk, making chunking pointless. Implementations must not override this.
   */
  default long chunkingThreshold() {
    return maxChunkSize();
  }

  /** Creates a chunker that splits blobs according to this configuration. */
  ContentDefinedChunker newChunker(DigestUtil digestUtil);

  /**
   * Returns a configuration negotiated from the server capabilities: FastCDC 2020 if the server
   * advertises it, otherwise RepMaxCDC, or {@code null} if the server advertises neither.
   *
   * <p>FastCDC 2020 is preferred for backward compatibility, so that a fleet of clients with
   * mixed versions keeps producing identical chunks against the same server.
   */
  @Nullable
  static ChunkingConfig fromServerCapabilities(ServerCapabilities capabilities) {
    ChunkingConfig config =
        fromServerCapabilities(capabilities, ChunkingFunction.Value.FAST_CDC_2020);
    if (config == null) {
      config = fromServerCapabilities(capabilities, ChunkingFunction.Value.REP_MAX_CDC);
    }
    return config;
  }

  /**
   * Returns the configuration for the given chunking function based on the parameters advertised
   * by the server, or {@code null} if the server does not support the function.
   *
   * <p>Advertised parameters that are out of the expected range are replaced with defaults.
   */
  @Nullable
  static ChunkingConfig fromServerCapabilities(
      ServerCapabilities capabilities, ChunkingFunction.Value chunkingFunction) {
    if (!capabilities.hasCacheCapabilities()) {
      return null;
    }
    CacheCapabilities cacheCap = capabilities.getCacheCapabilities();

    switch (chunkingFunction) {
      case FAST_CDC_2020 -> {
        if (cacheCap.hasFastCdc2020Params()) {
          return FastCdcChunkingConfig.fromParams(cacheCap.getFastCdc2020Params());
        }
      }
      case REP_MAX_CDC -> {
        if (cacheCap.hasRepMaxCdcParams()) {
          return RepMaxCdcChunkingConfig.fromParams(cacheCap.getRepMaxCdcParams());
        }
      }
      default -> {}
    }
    return null;
  }
}
