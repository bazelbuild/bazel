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

/** Tests for {@link AggregatedSpawnMetrics}. */
@RunWith(JUnit4.class)
public final class AggregatedSpawnMetricsTest {

  @Test
  public void sumAllMetrics() throws Exception {
    SpawnMetrics metrics1 =
        SpawnMetrics.Builder.forRemoteExec()
            .setTotalTime(Duration.ofSeconds(1))
            .setExecutionWallTime(Duration.ofSeconds(2))
            .setInputBytes(10)
            .setInputFiles(20)
            .setMemoryEstimateBytes(30)
            .build();
    SpawnMetrics metrics2 =
        SpawnMetrics.Builder.forRemoteExec()
            .setTotalTime(Duration.ofSeconds(10))
            .setExecutionWallTime(Duration.ofSeconds(20))
            .setInputBytes(100)
            .setInputFiles(200)
            .setMemoryEstimateBytes(300)
            .build();

    AggregatedSpawnMetrics aggregated = AggregatedSpawnMetrics.EMPTY;
    aggregated = aggregated.sumAllMetrics(metrics1);
    aggregated = aggregated.sumAllMetrics(metrics2);

    assertThat(aggregated.getRemoteMetrics().totalTime()).isEqualTo(Duration.ofSeconds(11));
    assertThat(aggregated.getRemoteMetrics().executionWallTime()).isEqualTo(Duration.ofSeconds(22));
    assertThat(aggregated.getRemoteMetrics().inputBytes()).isEqualTo(110);
    assertThat(aggregated.getRemoteMetrics().inputFiles()).isEqualTo(220);
    assertThat(aggregated.getRemoteMetrics().memoryEstimate()).isEqualTo(330);
  }

  @Test
  public void sumDurationMetricsMaxOther() throws Exception {
    SpawnMetrics metrics1 =
        SpawnMetrics.Builder.forRemoteExec()
            .setTotalTime(Duration.ofSeconds(1))
            .setExecutionWallTime(Duration.ofSeconds(2))
            .setInputBytes(10)
            .setInputFiles(20)
            .setMemoryEstimateBytes(30)
            .build();
    SpawnMetrics metrics2 =
        SpawnMetrics.Builder.forRemoteExec()
            .setTotalTime(Duration.ofSeconds(10))
            .setExecutionWallTime(Duration.ofSeconds(20))
            .setInputBytes(100)
            .setInputFiles(200)
            .setMemoryEstimateBytes(300)
            .build();

    AggregatedSpawnMetrics aggregated = AggregatedSpawnMetrics.EMPTY;
    aggregated = aggregated.sumDurationsMaxOther(metrics1);
    aggregated = aggregated.sumDurationsMaxOther(metrics2);

    assertThat(aggregated.getRemoteMetrics().totalTime()).isEqualTo(Duration.ofSeconds(11));
    assertThat(aggregated.getRemoteMetrics().executionWallTime()).isEqualTo(Duration.ofSeconds(22));
    assertThat(aggregated.getRemoteMetrics().inputBytes()).isEqualTo(100);
    assertThat(aggregated.getRemoteMetrics().inputFiles()).isEqualTo(200);
    assertThat(aggregated.getRemoteMetrics().memoryEstimate()).isEqualTo(300);
  }

  @Test
  public void aggregatingMetrics_preservesExecKind() throws Exception {
    SpawnMetrics metrics1 =
        SpawnMetrics.Builder.forLocalExec().setTotalTime(Duration.ofSeconds(1)).build();
    SpawnMetrics metrics2 =
        SpawnMetrics.Builder.forRemoteExec().setTotalTime(Duration.ofSeconds(2)).build();
    SpawnMetrics metrics3 =
        SpawnMetrics.Builder.forWorkerExec().setTotalTime(Duration.ofSeconds(3)).build();

    AggregatedSpawnMetrics aggregated = AggregatedSpawnMetrics.EMPTY;
    aggregated = aggregated.sumAllMetrics(metrics1);
    aggregated = aggregated.sumDurationsMaxOther(metrics2);
    aggregated = aggregated.sumDurationsMaxOther(metrics3);

    assertThat(aggregated.getMetrics(SpawnMetrics.ExecKind.LOCAL).totalTime())
        .isEqualTo(Duration.ofSeconds(1));
    assertThat(aggregated.getMetrics(SpawnMetrics.ExecKind.REMOTE).totalTime())
        .isEqualTo(Duration.ofSeconds(2));
    assertThat(aggregated.getMetrics(SpawnMetrics.ExecKind.WORKER).totalTime())
        .isEqualTo(Duration.ofSeconds(3));
  }

  @Test
  public void toString_printsOnlyRemote() throws Exception {
    SpawnMetrics metrics1 =
        SpawnMetrics.Builder.forLocalExec()
            .setTotalTime(Duration.ofSeconds(1))
            .setExecutionWallTime(Duration.ofSeconds(1))
            .build();
    SpawnMetrics metrics2 =
        SpawnMetrics.Builder.forRemoteExec()
            .setTotalTime(Duration.ofSeconds(2))
            .setNetworkTime(Duration.ofSeconds(1))
            .setExecutionWallTime(Duration.ofSeconds(1))
            .build();
    SpawnMetrics metrics3 =
        SpawnMetrics.Builder.forWorkerExec()
            .setTotalTime(Duration.ofSeconds(3))
            .setQueueTime(Duration.ofSeconds(1))
            .setExecutionWallTime(Duration.ofSeconds(2))
            .build();

    AggregatedSpawnMetrics aggregated =
        new AggregatedSpawnMetrics.Builder()
            .addDurations(metrics1)
            .addNonDurations(metrics1)
            .addDurations(metrics2)
            .addNonDurations(metrics2)
            .addDurations(metrics3)
            .addNonDurations(metrics3)
            .build();

    assertThat(aggregated.toString(Duration.ofSeconds(6), true))
        .isEqualTo(
            "Remote (33.33% of the time): "
                + "[queue: 0.00%, network: 16.67%, setup: 0.00%, process: 16.67%]");
  }
}
