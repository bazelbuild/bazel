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

import build.bazel.remote.execution.v2.ChunkingFunction;
import build.bazel.remote.execution.v2.RepMaxCdcParams;
import com.google.devtools.build.lib.remote.util.DigestUtil;

/**
 * Configuration for the RepMaxCDC chunking function. All sizes are in bytes.
 *
 * <p>RepMaxCDC produces chunks within {@code [minChunkSize, 2 * minChunkSize)} in size. The horizon
 * size controls the lookahead window used to find optimal cutting points; larger values improve
 * deduplication quality with diminishing returns, and a value of zero produces uniform chunks of
 * {@code minChunkSize}.
 */
public record RepMaxCdcChunkingConfig(int minChunkSize, int horizonSize) implements ChunkingConfig {

  public static final int DEFAULT_MIN_CHUNK_SIZE = 256 * 1024;

  /** The default horizon size, expressed as a multiple of the minimum chunk size. */
  public static final int DEFAULT_HORIZON_SIZE_FACTOR = 8;

  // Keep server-provided parameters bounded because each chunking session allocates a buffer of
  // twice (2 * minChunkSize + horizonSize) bytes. These maxima cap that buffer at 40 MiB.
  private static final int MAX_MIN_CHUNK_SIZE = 2 * 1024 * 1024;
  private static final int MAX_HORIZON_SIZE =
      DEFAULT_HORIZON_SIZE_FACTOR * MAX_MIN_CHUNK_SIZE;

  @Override
  public ChunkingFunction.Value chunkingFunction() {
    return ChunkingFunction.Value.REP_MAX_CDC;
  }

  @Override
  public int maxChunkSize() {
    // Chunks are strictly smaller than twice the minimum chunk size.
    return 2 * minChunkSize - 1;
  }

  @Override
  public ContentDefinedChunker newChunker(DigestUtil digestUtil) {
    return new RepMaxCdcChunker(this, digestUtil);
  }

  public static RepMaxCdcChunkingConfig defaults() {
    return new RepMaxCdcChunkingConfig(
        DEFAULT_MIN_CHUNK_SIZE, DEFAULT_HORIZON_SIZE_FACTOR * DEFAULT_MIN_CHUNK_SIZE);
  }

  /**
   * Creates a configuration from the parameters advertised by the server, replacing values outside
   * the expected range with defaults.
   */
  static RepMaxCdcChunkingConfig fromParams(RepMaxCdcParams params) {
    int minSize = DEFAULT_MIN_CHUNK_SIZE;
    long configMinSize = params.getMinChunkSizeBytes();
    if (configMinSize >= GearTable.GEAR_HASH_WINDOW_SIZE
        && configMinSize <= MAX_MIN_CHUNK_SIZE) {
      minSize = (int) configMinSize;
    }

    int horizonSize = Math.min(DEFAULT_HORIZON_SIZE_FACTOR * minSize, MAX_HORIZON_SIZE);
    long configHorizonSize = params.getHorizonSizeBytes();
    if (configHorizonSize >= 0 && configHorizonSize <= MAX_HORIZON_SIZE) {
      horizonSize = (int) configHorizonSize;
    }

    return new RepMaxCdcChunkingConfig(minSize, horizonSize);
  }
}
