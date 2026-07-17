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
  public void sumDurationMetricsMaxOther() throws Exception {
    SpawnMetrics metrics1 =
        SpawnMetrics.Builder.forRemoteExec()
            .setTotalTimeInMs(1 * 1000)
            .setExecutionWallTimeInMs(2 * 1000)
            .setInputBytes(10)
            .setInputFiles(20)
            .setMemoryEstimateBytes(30)
            .build();
    SpawnMetrics metrics2 =
        SpawnMetrics.Builder.forRemoteExec()
            .setTotalTimeInMs(10 * 1000)
            .setExecutionWallTimeInMs(20 * 1000)
            .setInputBytes(100)
            .setInputFiles(200)
            .setMemoryEstimateBytes(300)
            .build();

    AggregatedSpawnMetrics aggregated = AggregatedSpawnMetrics.EMPTY;
    aggregated = aggregated.sumDurationsMaxOther(metrics1);
    aggregated = aggregated.sumDurationsMaxOther(metrics2);

    assertThat(aggregated.getRemoteMetrics().totalTimeInMs()).isEqualTo(11 * 1000);
    assertThat(aggregated.getRemoteMetrics().executionWallTimeInMs()).isEqualTo(22 * 1000);
    assertThat(aggregated.getRemoteMetrics().inputBytes()).isEqualTo(100);
    assertThat(aggregated.getRemoteMetrics().inputFiles()).isEqualTo(200);
    assertThat(aggregated.getRemoteMetrics().memoryEstimate()).isEqualTo(300);
  }

  @Test
  public void aggregatingMetrics_preservesExecKind() throws Exception {
    SpawnMetrics metrics1 = SpawnMetrics.Builder.forLocalExec().setTotalTimeInMs(1 * 1000).build();
    SpawnMetrics metrics2 = SpawnMetrics.Builder.forRemoteExec().setTotalTimeInMs(2 * 1000).build();
    SpawnMetrics metrics3 = SpawnMetrics.Builder.forWorkerExec().setTotalTimeInMs(3 * 1000).build();

    AggregatedSpawnMetrics aggregated = AggregatedSpawnMetrics.EMPTY;
    aggregated = aggregated.sumDurationsMaxOther(metrics1);
    aggregated = aggregated.sumDurationsMaxOther(metrics2);
    aggregated = aggregated.sumDurationsMaxOther(metrics3);

    assertThat(aggregated.getMetrics(SpawnMetrics.ExecKind.LOCAL).totalTimeInMs())
        .isEqualTo(1 * 1000L);
    assertThat(aggregated.getMetrics(SpawnMetrics.ExecKind.REMOTE).totalTimeInMs())
        .isEqualTo(2 * 1000L);
    assertThat(aggregated.getMetrics(SpawnMetrics.ExecKind.WORKER).totalTimeInMs())
        .isEqualTo(3 * 1000L);
  }

  @Test
  public void toString_printsOnlyRemote() throws Exception {
    SpawnMetrics metrics1 =
        SpawnMetrics.Builder.forLocalExec()
            .setTotalTimeInMs(1 * 1000)
            .setExecutionWallTimeInMs(1 * 1000)
            .build();
    SpawnMetrics metrics2 =
        SpawnMetrics.Builder.forRemoteExec()
            .setTotalTimeInMs(2 * 1000)
            .setNetworkTimeInMs(1 * 1000)
            .setExecutionWallTimeInMs(1 * 1000)
            .build();
    SpawnMetrics metrics3 =
        SpawnMetrics.Builder.forWorkerExec()
            .setTotalTimeInMs(3 * 1000)
            .setQueueTimeInMs(1 * 1000)
            .setExecutionWallTimeInMs(2 * 1000)
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
