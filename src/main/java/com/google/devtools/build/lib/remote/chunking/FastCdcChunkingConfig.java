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
import build.bazel.remote.execution.v2.FastCdc2020Params;
import com.google.devtools.build.lib.remote.util.DigestUtil;

/** Configuration for the FastCDC 2020 chunking function. All sizes are in bytes. */
public record FastCdcChunkingConfig(int avgChunkSize, int normalizationLevel, int seed)
    implements ChunkingConfig {

  public static final int DEFAULT_AVG_CHUNK_SIZE = 512 * 1024;
  public static final int DEFAULT_NORMALIZATION_LEVEL = 2;
  public static final int DEFAULT_SEED = 0;

  @Override
  public ChunkingFunction.Value chunkingFunction() {
    return ChunkingFunction.Value.FAST_CDC_2020;
  }

  @Override
  public int minChunkSize() {
    return avgChunkSize / 4;
  }

  @Override
  public int maxChunkSize() {
    return avgChunkSize * 4;
  }

  @Override
  public ContentDefinedChunker newChunker(DigestUtil digestUtil) {
    return new FastCdcChunker(this, digestUtil);
  }

  public static FastCdcChunkingConfig defaults() {
    return new FastCdcChunkingConfig(
        DEFAULT_AVG_CHUNK_SIZE, DEFAULT_NORMALIZATION_LEVEL, DEFAULT_SEED);
  }

  /**
   * Creates a configuration from the parameters advertised by the server, replacing values
   * outside the expected range with defaults.
   */
  static FastCdcChunkingConfig fromParams(FastCdc2020Params params) {
    int avgSize = DEFAULT_AVG_CHUNK_SIZE;
    long configAvgSize = params.getAvgChunkSizeBytes();
    if (configAvgSize >= 1024
        && configAvgSize <= 1024 * 1024
        && (configAvgSize & (configAvgSize - 1)) == 0) {
      avgSize = (int) configAvgSize;
    }
    int seed = params.getSeed();

    return new FastCdcChunkingConfig(avgSize, DEFAULT_NORMALIZATION_LEVEL, seed);
  }
}
