// Copyright 2017 The Bazel Authors. All rights reserved.
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

import static com.google.common.truth.Truth8.assertThat;

import com.google.common.collect.ImmutableList;
import java.time.Duration;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link ActionResult}. */
@RunWith(JUnit4.class)
public final class ActionResultTest {

  @Test
  public void testCumulativeCommandExecutionTime_noSpawnResults() {
    List<SpawnResult> spawnResults = ImmutableList.of();
    ActionResult actionResult = ActionResult.create(spawnResults);
    assertThat(actionResult.cumulativeCommandExecutionWallTime()).isEmpty();
    assertThat(actionResult.cumulativeCommandExecutionCpuTime()).isEmpty();
    assertThat(actionResult.cumulativeCommandExecutionUserTime()).isEmpty();
    assertThat(actionResult.cumulativeCommandExecutionSystemTime()).isEmpty();
    assertThat(actionResult.cumulativeCommandExecutionBlockOutputOperations()).isEmpty();
    assertThat(actionResult.cumulativeCommandExecutionBlockInputOperations()).isEmpty();
    assertThat(actionResult.cumulativeCommandExecutionInvoluntaryContextSwitches()).isEmpty();
  }

  @Test
  public void testCumulativeCommandExecutionTime_oneSpawnResult() {
    SpawnResult spawnResult =
        new SpawnResult.Builder()
            .setWallTime(Duration.ofMillis(1984))
            .setUserTime(Duration.ofMillis(225))
            .setSystemTime(Duration.ofMillis(42))
            .setNumBlockOutputOperations(10)
            .setNumBlockInputOperations(20)
            .setNumInvoluntaryContextSwitches(30)
            .setStatus(SpawnResult.Status.SUCCESS)
            .setRunnerName("test")
            .build();
    List<SpawnResult> spawnResults = ImmutableList.of(spawnResult);
    ActionResult actionResult = ActionResult.create(spawnResults);
    assertThat(actionResult.cumulativeCommandExecutionWallTime()).hasValue(Duration.ofMillis(1984));
    assertThat(actionResult.cumulativeCommandExecutionCpuTime()).hasValue(Duration.ofMillis(267));
    assertThat(actionResult.cumulativeCommandExecutionUserTime()).hasValue(Duration.ofMillis(225));
    assertThat(actionResult.cumulativeCommandExecutionSystemTime()).hasValue(Duration.ofMillis(42));
    assertThat(actionResult.cumulativeCommandExecutionBlockOutputOperations()).hasValue(10L);
    assertThat(actionResult.cumulativeCommandExecutionBlockInputOperations()).hasValue(20L);
    assertThat(actionResult.cumulativeCommandExecutionInvoluntaryContextSwitches()).hasValue(30L);
  }

  @Test
  public void testCumulativeCommandExecutionTime_manySpawnResults() {
    SpawnResult spawnResult1 =
        new SpawnResult.Builder()
            .setWallTime(Duration.ofMillis(1979))
            .setUserTime(Duration.ofMillis(1))
            .setSystemTime(Duration.ofMillis(33))
            .setNumBlockOutputOperations(10)
            .setNumBlockInputOperations(20)
            .setNumInvoluntaryContextSwitches(30)
            .setStatus(SpawnResult.Status.SUCCESS)
            .setRunnerName("test")
            .build();
    SpawnResult spawnResult2 =
        new SpawnResult.Builder()
            .setWallTime(Duration.ofMillis(4))
            .setUserTime(Duration.ofMillis(1))
            .setSystemTime(Duration.ofMillis(7))
            .setNumBlockOutputOperations(100)
            .setNumBlockInputOperations(200)
            .setNumInvoluntaryContextSwitches(300)
            .setStatus(SpawnResult.Status.SUCCESS)
            .setRunnerName("test")
            .build();
    SpawnResult spawnResult3 =
        new SpawnResult.Builder()
            .setWallTime(Duration.ofMillis(1))
            .setUserTime(Duration.ofMillis(2))
            .setSystemTime(Duration.ofMillis(2))
            .setNumBlockOutputOperations(1000)
            .setNumBlockInputOperations(2000)
            .setNumInvoluntaryContextSwitches(3000)
            .setStatus(SpawnResult.Status.SUCCESS)
            .setRunnerName("test")
            .build();
    List<SpawnResult> spawnResults = ImmutableList.of(spawnResult1, spawnResult2, spawnResult3);
    ActionResult actionResult = ActionResult.create(spawnResults);
    assertThat(actionResult.cumulativeCommandExecutionWallTime()).hasValue(Duration.ofMillis(1984));
    assertThat(actionResult.cumulativeCommandExecutionCpuTime()).hasValue(Duration.ofMillis(46));
    assertThat(actionResult.cumulativeCommandExecutionUserTime()).hasValue(Duration.ofMillis(4));
    assertThat(actionResult.cumulativeCommandExecutionSystemTime()).hasValue(Duration.ofMillis(42));
    assertThat(actionResult.cumulativeCommandExecutionBlockOutputOperations()).hasValue(1110L);
    assertThat(actionResult.cumulativeCommandExecutionBlockInputOperations()).hasValue(2220L);
    assertThat(actionResult.cumulativeCommandExecutionInvoluntaryContextSwitches()).hasValue(3330L);
  }

  @Test
  public void testCumulativeCommandExecutionTime_manyEmptySpawnResults() {
    SpawnResult spawnResult1 =
        new SpawnResult.Builder()
            .setStatus(SpawnResult.Status.SUCCESS)
            .setRunnerName("test")
            .build();
    SpawnResult spawnResult2 =
        new SpawnResult.Builder()
            .setStatus(SpawnResult.Status.SUCCESS)
            .setRunnerName("test")
            .build();
    SpawnResult spawnResult3 =
        new SpawnResult.Builder()
            .setStatus(SpawnResult.Status.SUCCESS)
            .setRunnerName("test")
            .build();
    List<SpawnResult> spawnResults = ImmutableList.of(spawnResult1, spawnResult2, spawnResult3);
    ActionResult actionResult = ActionResult.create(spawnResults);
    assertThat(actionResult.cumulativeCommandExecutionWallTime()).isEmpty();
    assertThat(actionResult.cumulativeCommandExecutionCpuTime()).isEmpty();
    assertThat(actionResult.cumulativeCommandExecutionUserTime()).isEmpty();
    assertThat(actionResult.cumulativeCommandExecutionSystemTime()).isEmpty();
    assertThat(actionResult.cumulativeCommandExecutionBlockOutputOperations()).isEmpty();
    assertThat(actionResult.cumulativeCommandExecutionBlockInputOperations()).isEmpty();
    assertThat(actionResult.cumulativeCommandExecutionInvoluntaryContextSwitches()).isEmpty();
  }

  @Test
  public void testCumulativeCommandExecutionTime_manySpawnResults_butOnlyUserTime() {
    SpawnResult spawnResult1 =
        new SpawnResult.Builder()
            .setUserTime(Duration.ofMillis(2))
            .setStatus(SpawnResult.Status.SUCCESS)
            .setRunnerName("test")
            .build();
    SpawnResult spawnResult2 =
        new SpawnResult.Builder()
            .setUserTime(Duration.ofMillis(3))
            .setStatus(SpawnResult.Status.SUCCESS)
            .setRunnerName("test")
            .build();
    SpawnResult spawnResult3 =
        new SpawnResult.Builder()
            .setUserTime(Duration.ofMillis(4))
            .setStatus(SpawnResult.Status.SUCCESS)
            .setRunnerName("test")
            .build();
    List<SpawnResult> spawnResults = ImmutableList.of(spawnResult1, spawnResult2, spawnResult3);
    ActionResult actionResult = ActionResult.create(spawnResults);
    assertThat(actionResult.cumulativeCommandExecutionCpuTime()).hasValue(Duration.ofMillis(9));
    assertThat(actionResult.cumulativeCommandExecutionUserTime()).hasValue(Duration.ofMillis(9));
  }

  @Test
  public void testCumulativeCommandExecutionTime_manySpawnResults_butOnlySystemTime() {
    SpawnResult spawnResult1 =
        new SpawnResult.Builder()
            .setSystemTime(Duration.ofMillis(33))
            .setStatus(SpawnResult.Status.SUCCESS)
            .setRunnerName("test")
            .build();
    SpawnResult spawnResult2 =
        new SpawnResult.Builder()
            .setSystemTime(Duration.ofMillis(7))
            .setStatus(SpawnResult.Status.SUCCESS)
            .setRunnerName("test")
            .build();
    SpawnResult spawnResult3 =
        new SpawnResult.Builder()
            .setSystemTime(Duration.ofMillis(2))
            .setStatus(SpawnResult.Status.SUCCESS)
            .setRunnerName("test")
            .build();
    List<SpawnResult> spawnResults = ImmutableList.of(spawnResult1, spawnResult2, spawnResult3);
    ActionResult actionResult = ActionResult.create(spawnResults);
    assertThat(actionResult.cumulativeCommandExecutionCpuTime()).hasValue(Duration.ofMillis(42));
    assertThat(actionResult.cumulativeCommandExecutionSystemTime()).hasValue(Duration.ofMillis(42));
  }
}
