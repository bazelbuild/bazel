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
  public void testCumulativeCommandExecutionTime_NoSpawnResults() {
    List<SpawnResult> spawnResults = ImmutableList.of();
    ActionResult actionResult = ActionResult.create(spawnResults);
    assertThat(actionResult.cumulativeCommandExecutionWallTime()).isEmpty();
    assertThat(actionResult.cumulativeCommandExecutionUserTime()).isEmpty();
    assertThat(actionResult.cumulativeCommandExecutionSystemTime()).isEmpty();
  }

  @Test
  public void testCumulativeCommandExecutionTime_OneSpawnResult() {
    SpawnResult spawnResult =
        new SpawnResult.Builder()
            .setWallTime(Duration.ofMillis(1984))
            .setUserTime(Duration.ofMillis(225))
            .setSystemTime(Duration.ofMillis(42))
            .setStatus(SpawnResult.Status.SUCCESS)
            .build();
    List<SpawnResult> spawnResults = ImmutableList.of(spawnResult);
    ActionResult actionResult = ActionResult.create(spawnResults);
    assertThat(actionResult.cumulativeCommandExecutionWallTime()).isPresent();
    assertThat(actionResult.cumulativeCommandExecutionWallTime()).hasValue(Duration.ofMillis(1984));
    assertThat(actionResult.cumulativeCommandExecutionUserTime()).isPresent();
    assertThat(actionResult.cumulativeCommandExecutionUserTime()).hasValue(Duration.ofMillis(225));
    assertThat(actionResult.cumulativeCommandExecutionSystemTime()).isPresent();
    assertThat(actionResult.cumulativeCommandExecutionSystemTime()).hasValue(Duration.ofMillis(42));
  }

  @Test
  public void testCumulativeCommandExecutionTime_ManySpawnResults() {
    SpawnResult spawnResult1 =
        new SpawnResult.Builder()
            .setWallTime(Duration.ofMillis(1979))
            .setUserTime(Duration.ofMillis(1))
            .setSystemTime(Duration.ofMillis(33))
            .setStatus(SpawnResult.Status.SUCCESS)
            .build();
    SpawnResult spawnResult2 =
        new SpawnResult.Builder()
            .setWallTime(Duration.ofMillis(4))
            .setUserTime(Duration.ofMillis(1))
            .setSystemTime(Duration.ofMillis(7))
            .setStatus(SpawnResult.Status.SUCCESS)
            .build();
    SpawnResult spawnResult3 =
        new SpawnResult.Builder()
            .setWallTime(Duration.ofMillis(1))
            .setUserTime(Duration.ofMillis(2))
            .setSystemTime(Duration.ofMillis(2))
            .setStatus(SpawnResult.Status.SUCCESS)
            .build();
    List<SpawnResult> spawnResults = ImmutableList.of(spawnResult1, spawnResult2, spawnResult3);
    ActionResult actionResult = ActionResult.create(spawnResults);
    assertThat(actionResult.cumulativeCommandExecutionWallTime()).isPresent();
    assertThat(actionResult.cumulativeCommandExecutionWallTime()).hasValue(Duration.ofMillis(1984));
    assertThat(actionResult.cumulativeCommandExecutionUserTime()).isPresent();
    assertThat(actionResult.cumulativeCommandExecutionUserTime()).hasValue(Duration.ofMillis(4));
    assertThat(actionResult.cumulativeCommandExecutionSystemTime()).isPresent();
    assertThat(actionResult.cumulativeCommandExecutionSystemTime()).hasValue(Duration.ofMillis(42));
  }

  @Test
  public void testCumulativeCommandExecutionTime_ManyEmptySpawnResults() {
    SpawnResult spawnResult1 =
        new SpawnResult.Builder().setStatus(SpawnResult.Status.SUCCESS).build();
    SpawnResult spawnResult2 =
        new SpawnResult.Builder().setStatus(SpawnResult.Status.SUCCESS).build();
    SpawnResult spawnResult3 =
        new SpawnResult.Builder().setStatus(SpawnResult.Status.SUCCESS).build();
    List<SpawnResult> spawnResults = ImmutableList.of(spawnResult1, spawnResult2, spawnResult3);
    ActionResult actionResult = ActionResult.create(spawnResults);
    assertThat(actionResult.cumulativeCommandExecutionWallTime()).isEmpty();
    assertThat(actionResult.cumulativeCommandExecutionUserTime()).isEmpty();
    assertThat(actionResult.cumulativeCommandExecutionSystemTime()).isEmpty();
  }
}
