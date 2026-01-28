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
// import build.bazel.remote.execution.v2.ChunkingConfiguration;
// import build.bazel.remote.execution.v2.ChunkingConfiguration.FastCDCParams;
// import build.bazel.remote.execution.v2.ChunkingFunction;
import build.bazel.remote.execution.v2.ServerCapabilities;

/** Configuration for content-defined chunking. All sizes are in bytes. */
public record ChunkingConfig(
    long chunkingThreshold, int avgChunkSize, int normalizationLevel, int seed) {

  public static final int DEFAULT_AVG_CHUNK_SIZE = 512 * 1024;
  public static final int DEFAULT_NORMALIZATION_LEVEL = 2;
  public static final int DEFAULT_SEED = 0;
  public static final long DEFAULT_CHUNKING_THRESHOLD = 2 * 1024 * 1024;

  public int minChunkSize() {
    return avgChunkSize / 4;
  }

  public int maxChunkSize() {
    return avgChunkSize * 4;
  }

  public static ChunkingConfig defaults() {
    return new ChunkingConfig(
        DEFAULT_CHUNKING_THRESHOLD,
        DEFAULT_AVG_CHUNK_SIZE,
        DEFAULT_NORMALIZATION_LEVEL,
        DEFAULT_SEED);
  }

  public static ChunkingConfig fromServerCapabilities(ServerCapabilities capabilities) {
    if (!capabilities.hasCacheCapabilities()) {
      return null;
    }
    CacheCapabilities cacheCap = capabilities.getCacheCapabilities();
    return defaults();

    // TODO(https://github.com/bazelbuild/remote-apis/pull/357): Enable once servers
    // advertise threshold and ChunkingConfiguration with FASTCDC_2020.

    // if (!cacheCap.hasChunkingConfiguration()) {
    //   return defaults();
    // }

    // ChunkingConfiguration config = cacheCap.getChunkingConfiguration();
    // if (!config.getSupportedChunkingAlgorithmsList().contains(ChunkingFunction.Value.FASTCDC_2020)) {
    //   return null;
    // }

    // long threshold = config.getChunkingThresholdBytes();
    // long threshold = threshold = DEFAULT_CHUNKING_THRESHOLD;
    // int avgSize = DEFAULT_AVG_CHUNK_SIZE;
    // int normalization = DEFAULT_NORMALIZATION_LEVEL;
    // int seed = DEFAULT_SEED;

    // if (config.hasFastcdcParams()) {
    //   FastCDCParams params = config.getFastcdcParams();
    //   long configAvgSize = params.getAvgChunkSizeBytes();
    //   if (configAvgSize >= 1024 && configAvgSize <= 1024 * 1024) {
    //     avgSize = (int) configAvgSize;
    //   }
    //   int configNorm = params.getNormalizationLevel();
    //   if (configNorm >= 0 && configNorm <= 3) {
    //     normalization = configNorm;
    //   }
    //   seed = params.getSeed();
    // }
    // return new ChunkingConfig(threshold, avgSize, normalization, seed);
  }
}
