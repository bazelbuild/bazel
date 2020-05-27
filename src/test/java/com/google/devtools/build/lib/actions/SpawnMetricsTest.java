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

import com.google.common.collect.ImmutableList;
import java.time.Duration;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link SpawnMetrics}. */
@RunWith(JUnit4.class)
public final class SpawnMetricsTest {

  @Test
  public void sumAllMetrics() throws Exception {
    SpawnMetrics metrics1 =
        new SpawnMetrics.Builder()
            .setTotalTime(Duration.ofSeconds(1))
            .setExecutionWallTime(Duration.ofSeconds(2))
            .setInputBytes(10)
            .setInputFiles(20)
            .setMemoryEstimateBytes(30)
            .build();
    SpawnMetrics metrics2 =
        new SpawnMetrics.Builder()
            .setTotalTime(Duration.ofSeconds(10))
            .setExecutionWallTime(Duration.ofSeconds(20))
            .setInputBytes(100)
            .setInputFiles(200)
            .setMemoryEstimateBytes(300)
            .build();

    SpawnMetrics result = SpawnMetrics.sumAllMetrics(ImmutableList.of(metrics1, metrics2));

    assertThat(result.totalTime()).isEqualTo(Duration.ofSeconds(11));
    assertThat(result.executionWallTime()).isEqualTo(Duration.ofSeconds(22));
    assertThat(result.inputBytes()).isEqualTo(110);
    assertThat(result.inputFiles()).isEqualTo(220);
    assertThat(result.memoryEstimate()).isEqualTo(330);
  }

  @Test
  public void sumDurationMetricsMaxOther() throws Exception {
    SpawnMetrics metrics1 =
        new SpawnMetrics.Builder()
            .setTotalTime(Duration.ofSeconds(1))
            .setExecutionWallTime(Duration.ofSeconds(2))
            .setInputBytes(10)
            .setInputFiles(20)
            .setMemoryEstimateBytes(30)
            .build();
    SpawnMetrics metrics2 =
        new SpawnMetrics.Builder()
            .setTotalTime(Duration.ofSeconds(10))
            .setExecutionWallTime(Duration.ofSeconds(20))
            .setInputBytes(100)
            .setInputFiles(200)
            .setMemoryEstimateBytes(300)
            .build();

    SpawnMetrics result = SpawnMetrics.sumDurationsMaxOther(ImmutableList.of(metrics1, metrics2));

    assertThat(result.totalTime()).isEqualTo(Duration.ofSeconds(11));
    assertThat(result.executionWallTime()).isEqualTo(Duration.ofSeconds(22));
    assertThat(result.inputBytes()).isEqualTo(100);
    assertThat(result.inputFiles()).isEqualTo(200);
    assertThat(result.memoryEstimate()).isEqualTo(300);
  }

  @Test
  public void aggregatingMetrics_defaultsToRemote() throws Exception {
    SpawnMetrics metrics1 = new SpawnMetrics.Builder().setTotalTime(Duration.ofSeconds(1)).build();
    SpawnMetrics metrics2 =
        new SpawnMetrics.Builder()
            .setExecKind(SpawnMetrics.ExecKind.LOCAL)
            .setTotalTime(Duration.ofSeconds(5))
            .build();

    SpawnMetrics result = SpawnMetrics.sumAllMetrics(ImmutableList.of(metrics1, metrics2));

    assertThat(result.execKind()).isEqualTo(SpawnMetrics.ExecKind.REMOTE);
  }
}
