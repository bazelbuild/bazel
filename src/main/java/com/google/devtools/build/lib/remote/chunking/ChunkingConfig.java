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
import build.bazel.remote.execution.v2.FastCdc2020Params;
import build.bazel.remote.execution.v2.ServerCapabilities;

/** Configuration for content-defined chunking. All sizes are in bytes. */
public record ChunkingConfig(int avgChunkSize, int normalizationLevel, int seed) {

  public static final int DEFAULT_AVG_CHUNK_SIZE = 512 * 1024;
  public static final int DEFAULT_NORMALIZATION_LEVEL = 2;
  public static final int DEFAULT_SEED = 0;

  public int minChunkSize() {
    return avgChunkSize / 4;
  }

  public int maxChunkSize() {
    return avgChunkSize * 4;
  }

  /** Blobs larger than this should be chunked. Equal to maxChunkSize(). */
  public long chunkingThreshold() {
    return maxChunkSize();
  }

  public static ChunkingConfig defaults() {
    return new ChunkingConfig(
        DEFAULT_AVG_CHUNK_SIZE,
        DEFAULT_NORMALIZATION_LEVEL,
        DEFAULT_SEED);
  }

  public static ChunkingConfig fromServerCapabilities(ServerCapabilities capabilities) {
    if (!capabilities.hasCacheCapabilities()) {
      return null;
    }
    CacheCapabilities cacheCap = capabilities.getCacheCapabilities();

    if (!cacheCap.hasFastCdc2020Params()) {
      return null;
    }

    FastCdc2020Params params = cacheCap.getFastCdc2020Params();
    int avgSize = DEFAULT_AVG_CHUNK_SIZE;
    int seed = DEFAULT_SEED;

    long configAvgSize = params.getAvgChunkSizeBytes();
    if (configAvgSize >= 1024
        && configAvgSize <= 1024 * 1024
        && (configAvgSize & (configAvgSize - 1)) == 0) {
      avgSize = (int) configAvgSize;
    }
    seed = params.getSeed();

    return new ChunkingConfig(avgSize, DEFAULT_NORMALIZATION_LEVEL, seed);
  }
}
