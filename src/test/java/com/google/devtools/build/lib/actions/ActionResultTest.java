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

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableSet;
import java.time.Duration;
import java.util.Set;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link ActionResult}. */
@RunWith(JUnit4.class)
public final class ActionResultTest {

  @Test
  public void testCumulativeCommandExecutionWallTime_NoSpawnResults() {
    Set<SpawnResult> spawnResults = ImmutableSet.of();
    ActionResult actionResult = ActionResult.create(spawnResults);
    assertThat(actionResult.cumulativeCommandExecutionWallTime()).isEqualTo(Duration.ZERO);
  }

  @Test
  public void testCumulativeCommandExecutionWallTime_OneSpawnResult() {
    SpawnResult spawnResult =
        new SpawnResult.Builder()
            .setWallTimeMillis(42)
            .setStatus(SpawnResult.Status.SUCCESS)
            .build();
    Set<SpawnResult> spawnResults = ImmutableSet.of(spawnResult);
    ActionResult actionResult = ActionResult.create(spawnResults);
    assertThat(actionResult.cumulativeCommandExecutionWallTime()).isEqualTo(Duration.ofMillis(42));
  }

  @Test
  public void testCumulativeCommandExecutionWallTime_ManySpawnResults() {
    SpawnResult spawnResult1 =
        new SpawnResult.Builder()
            .setWallTimeMillis(1979)
            .setStatus(SpawnResult.Status.SUCCESS)
            .build();
    SpawnResult spawnResult2 =
        new SpawnResult.Builder()
            .setWallTimeMillis(4)
            .setStatus(SpawnResult.Status.SUCCESS)
            .build();
    SpawnResult spawnResult3 =
        new SpawnResult.Builder()
            .setWallTimeMillis(1)
            .setStatus(SpawnResult.Status.SUCCESS)
            .build();
    Set<SpawnResult> spawnResults = ImmutableSet.of(spawnResult1, spawnResult2, spawnResult3);
    ActionResult actionResult = ActionResult.create(spawnResults);
    assertThat(actionResult.cumulativeCommandExecutionWallTime())
        .isEqualTo(Duration.ofMillis(1984));
  }
}
