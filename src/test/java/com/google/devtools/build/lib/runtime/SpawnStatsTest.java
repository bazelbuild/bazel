// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.runtime;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.actions.ActionResult;
import com.google.devtools.build.lib.actions.SpawnResult;
import java.util.ArrayList;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Testing SpawnStats */
@RunWith(JUnit4.class)
public final class SpawnStatsTest {

  SpawnStats stats;

  @Before
  public void setUp() {
    stats = new SpawnStats();
  }

  @Test
  public void emptySet() {
    assertThat(stats.getSummary()).isEqualTo("0 processes.");
  }

  @Test
  public void one() {
    stats.countRunnerName("foo");
    assertThat(stats.getSummary()).isEqualTo("1 process: 1 foo.");
  }

  @Test
  public void oneRemote() {
    stats.countRunnerName("remote cache hit");
    assertThat(stats.getSummary()).isEqualTo("1 process: 1 remote cache hit.");
  }

  @Test
  public void two() {
    stats.countRunnerName("foo");
    stats.countRunnerName("foo");
    assertThat(stats.getSummary()).isEqualTo("2 processes: 2 foo.");
  }

  @Test
  public void order() {
    stats.countRunnerName("a");
    stats.countRunnerName("b");
    stats.countRunnerName("b");
    stats.countRunnerName("c");
    stats.countRunnerName("c");
    stats.countRunnerName("c");
    assertThat(stats.getSummary()).isEqualTo("6 processes: 1 a, 2 b, 3 c.");
  }

  @Test
  public void reverseOrder() {
    stats.countRunnerName("a");
    stats.countRunnerName("a");
    stats.countRunnerName("a");
    stats.countRunnerName("b");
    stats.countRunnerName("b");
    stats.countRunnerName("c");
    assertThat(stats.getSummary()).isEqualTo("6 processes: 3 a, 2 b, 1 c.");
  }

  @Test
  public void cacheFirst() {
    stats.countRunnerName("a");
    stats.countRunnerName("a");
    stats.countRunnerName("a");
    stats.countRunnerName("b");
    stats.countRunnerName("remote cache hit");
    stats.countRunnerName("b");
    stats.countRunnerName("c");
    assertThat(stats.getSummary()).isEqualTo("7 processes: 1 remote cache hit, 3 a, 2 b, 1 c.");
  }

  private final SpawnResult rA =
      new SpawnResult.Builder().setStatus(SpawnResult.Status.SUCCESS).setRunnerName("abc").build();
  private final SpawnResult rB =
      new SpawnResult.Builder().setStatus(SpawnResult.Status.SUCCESS).setRunnerName("cde").build();

  @Test
  public void actionOneSpawn() {

    ArrayList<SpawnResult> spawns = new ArrayList<>();
    spawns.add(rA);

    stats.countActionResult(ActionResult.create(spawns));
    assertThat(stats.getSummary()).isEqualTo("1 process: 1 abc.");
  }

  @Test
  public void actionManySpawn() {
    // Different spawns with the same runner count as one action

    ArrayList<SpawnResult> spawns = new ArrayList<>();
    spawns.add(rA);
    spawns.add(rA);
    spawns.add(rA);

    stats.countActionResult(ActionResult.create(spawns));
    assertThat(stats.getSummary()).isEqualTo("3 processes: 3 abc.");
  }

  @Test
  public void actionManySpawnMixed() {
    // Different spawns mixed runners

    ArrayList<SpawnResult> spawns = new ArrayList<>();
    spawns.add(rA);
    spawns.add(rA);
    spawns.add(rB);

    stats.countActionResult(ActionResult.create(spawns));
    assertThat(stats.getSummary()).isEqualTo("3 processes: 2 abc, 1 cde.");
  }

  @Test
  public void actionManyActionsMixed() {
    // Five actions:
    // abc
    // abc, abc
    // abc, abc, cde
    // abc, abc, cde
    // abc, abc, cde

    ArrayList<SpawnResult> spawns = new ArrayList<>();
    spawns.add(rA);
    stats.countActionResult(ActionResult.create(spawns));

    spawns.add(rA);
    stats.countActionResult(ActionResult.create(spawns));

    spawns.add(rB);
    stats.countActionResult(ActionResult.create(spawns));
    stats.countActionResult(ActionResult.create(spawns));
    stats.countActionResult(ActionResult.create(spawns));

    assertThat(stats.getSummary()).isEqualTo("12 processes: 9 abc, 3 cde.");
  }
}
