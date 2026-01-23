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

import static com.google.common.truth.Truth.assertThat;

import build.bazel.remote.execution.v2.CacheCapabilities;
import build.bazel.remote.execution.v2.FastCdc2020Params;
import build.bazel.remote.execution.v2.ServerCapabilities;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link ChunkingConfig}. */
@RunWith(JUnit4.class)
public class ChunkingConfigTest {

  @Test
  public void defaults_returnsExpectedValues() {
    ChunkingConfig config = ChunkingConfig.defaults();

    assertThat(config.avgChunkSize()).isEqualTo(512 * 1024);
    assertThat(config.normalizationLevel()).isEqualTo(2);
    assertThat(config.seed()).isEqualTo(0);
    assertThat(config.chunkingThreshold()).isEqualTo(512 * 1024 * 4);
  }

  @Test
  public void minChunkSize_returnsQuarterOfAvg() {
    ChunkingConfig config = new ChunkingConfig(1024, 2, 0);

    assertThat(config.minChunkSize()).isEqualTo(256);
  }

  @Test
  public void maxChunkSize_returnsFourTimesAvg() {
    ChunkingConfig config = new ChunkingConfig(1024, 2, 0);

    assertThat(config.maxChunkSize()).isEqualTo(4096);
  }

  @Test
  public void chunkingThreshold_equalsMaxChunkSize() {
    ChunkingConfig config = new ChunkingConfig(1024, 2, 0);

    assertThat(config.chunkingThreshold()).isEqualTo(config.maxChunkSize());
  }

  @Test
  public void minAndMaxChunkSize_withDefaultConfig() {
    ChunkingConfig config = ChunkingConfig.defaults();

    assertThat(config.minChunkSize()).isEqualTo(128 * 1024);
    assertThat(config.maxChunkSize()).isEqualTo(2048 * 1024);
  }

  @Test
  public void fromServerCapabilities_withoutCacheCapabilities_returnsNull() {
    ServerCapabilities capabilities = ServerCapabilities.getDefaultInstance();

    ChunkingConfig config = ChunkingConfig.fromServerCapabilities(capabilities);

    assertThat(config).isNull();
  }

  @Test
  public void fromServerCapabilities_withoutFastCdcParams_returnsNull() {
    ServerCapabilities capabilities = ServerCapabilities.newBuilder()
        .setCacheCapabilities(CacheCapabilities.getDefaultInstance())
        .build();

    ChunkingConfig config = ChunkingConfig.fromServerCapabilities(capabilities);

    assertThat(config).isNull();
  }

  @Test
  public void fromServerCapabilities_withFastCdcParams_returnsConfig() {
    ServerCapabilities capabilities = ServerCapabilities.newBuilder()
        .setCacheCapabilities(CacheCapabilities.newBuilder()
            .setFastCdc2020Params(FastCdc2020Params.newBuilder()
                .setAvgChunkSizeBytes(256 * 1024)
                .setSeed(42)
                .build())
            .build())
        .build();

    ChunkingConfig config = ChunkingConfig.fromServerCapabilities(capabilities);

    assertThat(config).isNotNull();
    assertThat(config.avgChunkSize()).isEqualTo(256 * 1024);
    assertThat(config.seed()).isEqualTo(42);
    assertThat(config.chunkingThreshold()).isEqualTo(256 * 1024 * 4);
  }

  @Test
  public void fromServerCapabilities_withDefaultFastCdcParams_returnsDefaults() {
    ServerCapabilities capabilities = ServerCapabilities.newBuilder()
        .setCacheCapabilities(CacheCapabilities.newBuilder()
            .setFastCdc2020Params(FastCdc2020Params.newBuilder()
                .setAvgChunkSizeBytes(512 * 1024)
                .setSeed(0)
                .build())
            .build())
        .build();

    ChunkingConfig config = ChunkingConfig.fromServerCapabilities(capabilities);

    assertThat(config).isEqualTo(ChunkingConfig.defaults());
  }

  @Test
  public void fromServerCapabilities_nonPowerOfTwoAvgSize_fallsBackToDefault() {
    ServerCapabilities capabilities = ServerCapabilities.newBuilder()
        .setCacheCapabilities(CacheCapabilities.newBuilder()
            .setFastCdc2020Params(FastCdc2020Params.newBuilder()
                .setAvgChunkSizeBytes(300 * 1024)
                .build())
            .build())
        .build();

    ChunkingConfig config = ChunkingConfig.fromServerCapabilities(capabilities);

    assertThat(config).isNotNull();
    assertThat(config.avgChunkSize()).isEqualTo(ChunkingConfig.DEFAULT_AVG_CHUNK_SIZE);
  }
}
