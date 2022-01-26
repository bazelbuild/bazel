// Copyright 2020 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.actions;

import static com.google.common.truth.Truth.assertThat;

import java.time.Duration;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link SpawnMetrics}. */
@RunWith(JUnit4.class)
public final class SpawnMetricsTest {

  @Test
  public void builder_addDurationsNonDurations() throws Exception {
    SpawnMetrics metrics1 =
        SpawnMetrics.Builder.forRemoteExec()
            .setTotalTime(Duration.ofSeconds(1))
            .setExecutionWallTime(Duration.ofSeconds(2))
            .setInputBytes(10)
            .setInputFiles(20)
            .setMemoryEstimateBytes(30)
            .setInputBytesLimit(20)
            .setInputFilesLimit(40)
            .setOutputBytesLimit(50)
            .setOutputFilesLimit(60)
            .setMemoryBytesLimit(70)
            .setTimeLimit(Duration.ofSeconds(80))
            .build();
    SpawnMetrics metrics2 =
        SpawnMetrics.Builder.forRemoteExec()
            .setTotalTime(Duration.ofSeconds(10))
            .setExecutionWallTime(Duration.ofSeconds(20))
            .setInputBytes(100)
            .setInputFiles(200)
            .setMemoryEstimateBytes(300)
            .setInputBytesLimit(200)
            .setInputFilesLimit(400)
            .setOutputBytesLimit(500)
            .setOutputFilesLimit(600)
            .setMemoryBytesLimit(700)
            .setTimeLimit(Duration.ofSeconds(800))
            .build();

    SpawnMetrics result =
        SpawnMetrics.Builder.forRemoteExec()
            .addDurations(metrics1)
            .addDurations(metrics2)
            .addNonDurations(metrics1)
            .addNonDurations(metrics2)
            .build();

    assertThat(result.totalTime()).isEqualTo(Duration.ofSeconds(11));
    assertThat(result.executionWallTime()).isEqualTo(Duration.ofSeconds(22));
    assertThat(result.inputBytes()).isEqualTo(110);
    assertThat(result.inputFiles()).isEqualTo(220);
    assertThat(result.memoryEstimate()).isEqualTo(330);
    assertThat(result.inputBytesLimit()).isEqualTo(220);
    assertThat(result.inputFilesLimit()).isEqualTo(440);
    assertThat(result.outputBytesLimit()).isEqualTo(550);
    assertThat(result.outputFilesLimit()).isEqualTo(660);
    assertThat(result.memoryLimit()).isEqualTo(770);
    assertThat(result.timeLimit()).isEqualTo(Duration.ofSeconds(880));
  }

  @Test
  public void builder_addDurationsMaxNonDurations() throws Exception {
    SpawnMetrics metrics1 =
        SpawnMetrics.Builder.forRemoteExec()
            .setTotalTime(Duration.ofSeconds(1))
            .setExecutionWallTime(Duration.ofSeconds(2))
            .setInputBytes(10)
            .setInputFiles(20)
            .setMemoryEstimateBytes(30)
            .setInputBytesLimit(20)
            .setInputFilesLimit(40)
            .setOutputBytesLimit(50)
            .setOutputFilesLimit(60)
            .setMemoryBytesLimit(70)
            .setTimeLimit(Duration.ofSeconds(80))
            .build();
    SpawnMetrics metrics2 =
        SpawnMetrics.Builder.forRemoteExec()
            .setTotalTime(Duration.ofSeconds(10))
            .setExecutionWallTime(Duration.ofSeconds(20))
            .setInputBytes(100)
            .setInputFiles(200)
            .setMemoryEstimateBytes(300)
            .setInputBytesLimit(200)
            .setInputFilesLimit(400)
            .setOutputBytesLimit(500)
            .setOutputFilesLimit(600)
            .setMemoryBytesLimit(700)
            .setTimeLimit(Duration.ofSeconds(800))
            .build();

    SpawnMetrics result =
        SpawnMetrics.Builder.forRemoteExec()
            .addDurations(metrics1)
            .addDurations(metrics2)
            .maxNonDurations(metrics1)
            .maxNonDurations(metrics2)
            .build();

    assertThat(result.totalTime()).isEqualTo(Duration.ofSeconds(11));
    assertThat(result.executionWallTime()).isEqualTo(Duration.ofSeconds(22));
    assertThat(result.inputBytes()).isEqualTo(100);
    assertThat(result.inputFiles()).isEqualTo(200);
    assertThat(result.memoryEstimate()).isEqualTo(300);
    assertThat(result.inputBytesLimit()).isEqualTo(200);
    assertThat(result.inputFilesLimit()).isEqualTo(400);
    assertThat(result.outputBytesLimit()).isEqualTo(500);
    assertThat(result.outputFilesLimit()).isEqualTo(600);
    assertThat(result.memoryLimit()).isEqualTo(700);
    assertThat(result.timeLimit()).isEqualTo(Duration.ofSeconds(800));
  }
}
