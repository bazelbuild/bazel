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
import build.bazel.remote.execution.v2.ChunkingFunction;
import build.bazel.remote.execution.v2.FastCdc2020Params;
import build.bazel.remote.execution.v2.RepMaxCdcParams;
import build.bazel.remote.execution.v2.ServerCapabilities;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link ChunkingConfig}. */
@RunWith(JUnit4.class)
public class ChunkingConfigTest {

  private static ServerCapabilities capabilitiesWithFastCdcParams(FastCdc2020Params params) {
    return ServerCapabilities.newBuilder()
        .setCacheCapabilities(CacheCapabilities.newBuilder().setFastCdc2020Params(params).build())
        .build();
  }

  private static ServerCapabilities capabilitiesWithRepMaxCdcParams(RepMaxCdcParams params) {
    return ServerCapabilities.newBuilder()
        .setCacheCapabilities(CacheCapabilities.newBuilder().setRepMaxCdcParams(params).build())
        .build();
  }

  @Test
  public void fastCdcDefaults_returnsExpectedValues() {
    FastCdcChunkingConfig config = FastCdcChunkingConfig.defaults();

    assertThat(config.chunkingFunction()).isEqualTo(ChunkingFunction.Value.FAST_CDC_2020);
    assertThat(config.avgChunkSize()).isEqualTo(512 * 1024);
    assertThat(config.normalizationLevel()).isEqualTo(2);
    assertThat(config.seed()).isEqualTo(0);
    assertThat(config.chunkingThreshold()).isEqualTo(512 * 1024 * 4);
  }

  @Test
  public void fastCdcMinChunkSize_returnsQuarterOfAvg() {
    FastCdcChunkingConfig config = new FastCdcChunkingConfig(1024, 2, 0);

    assertThat(config.minChunkSize()).isEqualTo(256);
  }

  @Test
  public void fastCdcMaxChunkSize_returnsFourTimesAvg() {
    FastCdcChunkingConfig config = new FastCdcChunkingConfig(1024, 2, 0);

    assertThat(config.maxChunkSize()).isEqualTo(4096);
  }

  @Test
  public void fastCdcChunkingThreshold_equalsMaxChunkSize() {
    FastCdcChunkingConfig config = new FastCdcChunkingConfig(1024, 2, 0);

    assertThat(config.chunkingThreshold()).isEqualTo(config.maxChunkSize());
  }

  @Test
  public void fastCdcMinAndMaxChunkSize_withDefaultConfig() {
    FastCdcChunkingConfig config = FastCdcChunkingConfig.defaults();

    assertThat(config.minChunkSize()).isEqualTo(128 * 1024);
    assertThat(config.maxChunkSize()).isEqualTo(2048 * 1024);
  }

  @Test
  public void fastCdcNewChunker_returnsFastCdcChunker() {
    FastCdcChunkingConfig config = FastCdcChunkingConfig.defaults();

    assertThat(config.newChunker(/* digestUtil= */ null)).isInstanceOf(FastCdcChunker.class);
  }

  @Test
  public void repMaxCdcDefaults_returnsExpectedValues() {
    RepMaxCdcChunkingConfig config = RepMaxCdcChunkingConfig.defaults();

    assertThat(config.chunkingFunction()).isEqualTo(ChunkingFunction.Value.REP_MAX_CDC);
    assertThat(config.minChunkSize()).isEqualTo(256 * 1024);
    assertThat(config.horizonSize()).isEqualTo(8 * 256 * 1024);
    assertThat(config.chunkingThreshold()).isEqualTo(2 * 256 * 1024 - 1);
  }

  @Test
  public void repMaxCdcMaxChunkSize_isBelowTwiceMinChunkSize() {
    RepMaxCdcChunkingConfig config = new RepMaxCdcChunkingConfig(1024, 8 * 1024);

    assertThat(config.maxChunkSize()).isEqualTo(2047);
  }

  @Test
  public void repMaxCdcChunkingThreshold_equalsMaxChunkSize() {
    RepMaxCdcChunkingConfig config = new RepMaxCdcChunkingConfig(1024, 8 * 1024);

    assertThat(config.chunkingThreshold()).isEqualTo(config.maxChunkSize());
  }

  @Test
  public void repMaxCdcNewChunker_returnsRepMaxCdcChunker() {
    RepMaxCdcChunkingConfig config = RepMaxCdcChunkingConfig.defaults();

    assertThat(config.newChunker(/* digestUtil= */ null)).isInstanceOf(RepMaxCdcChunker.class);
  }

  @Test
  public void fromServerCapabilities_withoutCacheCapabilities_returnsNull() {
    ServerCapabilities capabilities = ServerCapabilities.getDefaultInstance();

    assertThat(
            ChunkingConfig.fromServerCapabilities(
                capabilities, ChunkingFunction.Value.FAST_CDC_2020))
        .isNull();
    assertThat(
            ChunkingConfig.fromServerCapabilities(capabilities, ChunkingFunction.Value.REP_MAX_CDC))
        .isNull();
  }

  @Test
  public void fromServerCapabilities_withoutParamsForRequestedFunction_returnsNull() {
    ServerCapabilities capabilities =
        ServerCapabilities.newBuilder()
            .setCacheCapabilities(CacheCapabilities.getDefaultInstance())
            .build();

    assertThat(
            ChunkingConfig.fromServerCapabilities(
                capabilities, ChunkingFunction.Value.FAST_CDC_2020))
        .isNull();
    assertThat(
            ChunkingConfig.fromServerCapabilities(capabilities, ChunkingFunction.Value.REP_MAX_CDC))
        .isNull();
  }

  @Test
  public void fromServerCapabilities_fastCdcRequestedButOnlyRepMaxCdcAdvertised_returnsNull() {
    ServerCapabilities capabilities =
        capabilitiesWithRepMaxCdcParams(
            RepMaxCdcParams.newBuilder().setMinChunkSizeBytes(256 * 1024).build());

    ChunkingConfig config =
        ChunkingConfig.fromServerCapabilities(capabilities, ChunkingFunction.Value.FAST_CDC_2020);

    assertThat(config).isNull();
  }

  @Test
  public void fromServerCapabilities_repMaxCdcRequestedButOnlyFastCdcAdvertised_returnsNull() {
    ServerCapabilities capabilities =
        capabilitiesWithFastCdcParams(
            FastCdc2020Params.newBuilder().setAvgChunkSizeBytes(512 * 1024).build());

    ChunkingConfig config =
        ChunkingConfig.fromServerCapabilities(capabilities, ChunkingFunction.Value.REP_MAX_CDC);

    assertThat(config).isNull();
  }

  @Test
  public void fromServerCapabilities_withFastCdcParams_returnsConfig() {
    ServerCapabilities capabilities =
        capabilitiesWithFastCdcParams(
            FastCdc2020Params.newBuilder().setAvgChunkSizeBytes(256 * 1024).setSeed(42).build());

    ChunkingConfig config =
        ChunkingConfig.fromServerCapabilities(capabilities, ChunkingFunction.Value.FAST_CDC_2020);

    assertThat(config).isInstanceOf(FastCdcChunkingConfig.class);
    FastCdcChunkingConfig fastCdcConfig = (FastCdcChunkingConfig) config;
    assertThat(fastCdcConfig.avgChunkSize()).isEqualTo(256 * 1024);
    assertThat(fastCdcConfig.seed()).isEqualTo(42);
    assertThat(fastCdcConfig.chunkingThreshold()).isEqualTo(256 * 1024 * 4);
  }

  @Test
  public void fromServerCapabilities_withDefaultFastCdcParams_returnsDefaults() {
    ServerCapabilities capabilities =
        capabilitiesWithFastCdcParams(
            FastCdc2020Params.newBuilder().setAvgChunkSizeBytes(512 * 1024).setSeed(0).build());

    ChunkingConfig config =
        ChunkingConfig.fromServerCapabilities(capabilities, ChunkingFunction.Value.FAST_CDC_2020);

    assertThat(config).isEqualTo(FastCdcChunkingConfig.defaults());
  }

  @Test
  public void fromServerCapabilities_nonPowerOfTwoAvgSize_fallsBackToDefault() {
    ServerCapabilities capabilities =
        capabilitiesWithFastCdcParams(
            FastCdc2020Params.newBuilder().setAvgChunkSizeBytes(300 * 1024).build());

    ChunkingConfig config =
        ChunkingConfig.fromServerCapabilities(capabilities, ChunkingFunction.Value.FAST_CDC_2020);

    assertThat(config).isInstanceOf(FastCdcChunkingConfig.class);
    assertThat(((FastCdcChunkingConfig) config).avgChunkSize())
        .isEqualTo(FastCdcChunkingConfig.DEFAULT_AVG_CHUNK_SIZE);
  }

  @Test
  public void fromServerCapabilities_withRepMaxCdcParams_returnsConfig() {
    ServerCapabilities capabilities =
        capabilitiesWithRepMaxCdcParams(
            RepMaxCdcParams.newBuilder()
                .setMinChunkSizeBytes(128 * 1024)
                .setHorizonSizeBytes(1024 * 1024)
                .build());

    ChunkingConfig config =
        ChunkingConfig.fromServerCapabilities(capabilities, ChunkingFunction.Value.REP_MAX_CDC);

    assertThat(config).isInstanceOf(RepMaxCdcChunkingConfig.class);
    RepMaxCdcChunkingConfig repMaxConfig = (RepMaxCdcChunkingConfig) config;
    assertThat(repMaxConfig.minChunkSize()).isEqualTo(128 * 1024);
    assertThat(repMaxConfig.horizonSize()).isEqualTo(1024 * 1024);
    assertThat(repMaxConfig.chunkingThreshold()).isEqualTo(2 * 128 * 1024 - 1);
  }

  @Test
  public void fromServerCapabilities_repMaxCdcMinSizeOutOfRange_fallsBackToDefault() {
    ServerCapabilities capabilities =
        capabilitiesWithRepMaxCdcParams(
            RepMaxCdcParams.newBuilder()
                .setMinChunkSizeBytes(64)
                .setHorizonSizeBytes(1024 * 1024)
                .build());

    ChunkingConfig config =
        ChunkingConfig.fromServerCapabilities(capabilities, ChunkingFunction.Value.REP_MAX_CDC);

    assertThat(config).isInstanceOf(RepMaxCdcChunkingConfig.class);
    RepMaxCdcChunkingConfig repMaxConfig = (RepMaxCdcChunkingConfig) config;
    assertThat(repMaxConfig.minChunkSize())
        .isEqualTo(RepMaxCdcChunkingConfig.DEFAULT_MIN_CHUNK_SIZE);
    assertThat(repMaxConfig.horizonSize()).isEqualTo(1024 * 1024);
  }

  @Test
  public void fromServerCapabilities_repMaxCdcHorizonSizeOutOfRange_fallsBackToDefault() {
    ServerCapabilities capabilities =
        capabilitiesWithRepMaxCdcParams(
            RepMaxCdcParams.newBuilder()
                .setMinChunkSizeBytes(128 * 1024)
                .setHorizonSizeBytes(1024L * 1024 * 1024)
                .build());

    ChunkingConfig config =
        ChunkingConfig.fromServerCapabilities(capabilities, ChunkingFunction.Value.REP_MAX_CDC);

    assertThat(config).isInstanceOf(RepMaxCdcChunkingConfig.class);
    RepMaxCdcChunkingConfig repMaxConfig = (RepMaxCdcChunkingConfig) config;
    assertThat(repMaxConfig.minChunkSize()).isEqualTo(128 * 1024);
    assertThat(repMaxConfig.horizonSize())
        .isEqualTo(RepMaxCdcChunkingConfig.DEFAULT_HORIZON_SIZE_FACTOR * 128 * 1024);
  }

  @Test
  public void fromServerCapabilities_repMaxCdcZeroHorizonSize_isAccepted() {
    ServerCapabilities capabilities =
        capabilitiesWithRepMaxCdcParams(
            RepMaxCdcParams.newBuilder()
                .setMinChunkSizeBytes(128 * 1024)
                .setHorizonSizeBytes(0)
                .build());

    ChunkingConfig config =
        ChunkingConfig.fromServerCapabilities(capabilities, ChunkingFunction.Value.REP_MAX_CDC);

    assertThat(config).isInstanceOf(RepMaxCdcChunkingConfig.class);
    assertThat(((RepMaxCdcChunkingConfig) config).horizonSize()).isEqualTo(0);
  }
}
