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
import com.google.devtools.build.lib.actions.SpawnMetrics;
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
    assertThat(SpawnStats.convertSummaryToString(stats.getSummary())).isEqualTo("0 processes.");
  }

  @Test
  public void one() {
    ArrayList<SpawnResult> spawns = new ArrayList<>();
    spawns.add(
        new SpawnResult.Builder()
            .setStatus(SpawnResult.Status.SUCCESS)
            .setRunnerName("foo")
            .build());
    stats.countActionResult(ActionResult.create(spawns));
    stats.incrementActionCount();
    assertThat(SpawnStats.convertSummaryToString(stats.getSummary()))
        .isEqualTo("1 process: 1 foo.");
  }

  @Test
  public void oneRemote() {
    ArrayList<SpawnResult> spawns = new ArrayList<>();
    spawns.add(
        new SpawnResult.Builder()
            .setStatus(SpawnResult.Status.SUCCESS)
            .setRunnerName("remote cache hit")
            .build());
    stats.countActionResult(ActionResult.create(spawns));
    stats.incrementActionCount();
    assertThat(SpawnStats.convertSummaryToString(stats.getSummary()))
        .isEqualTo("1 process: 1 remote cache hit.");
  }

  @Test
  public void two() {
    for (int i = 0; i < 2; i++) {
      ArrayList<SpawnResult> spawns = new ArrayList<>();
      spawns.add(
          new SpawnResult.Builder()
              .setStatus(SpawnResult.Status.SUCCESS)
              .setRunnerName("foo")
              .build());
      stats.countActionResult(ActionResult.create(spawns));
      stats.incrementActionCount();
    }
    assertThat(SpawnStats.convertSummaryToString(stats.getSummary()))
        .isEqualTo("2 processes: 2 foo.");
  }

  @Test
  public void order() {
    ArrayList<SpawnResult> spawns = new ArrayList<>();
    spawns.add(
        new SpawnResult.Builder().setStatus(SpawnResult.Status.SUCCESS).setRunnerName("a").build());
    stats.countActionResult(ActionResult.create(spawns));
    stats.incrementActionCount();

    for (int i = 0; i < 2; i++) {
      spawns = new ArrayList<>();
      spawns.add(
          new SpawnResult.Builder()
              .setStatus(SpawnResult.Status.SUCCESS)
              .setRunnerName("b")
              .build());
      stats.countActionResult(ActionResult.create(spawns));
      stats.incrementActionCount();
    }

    for (int i = 0; i < 3; i++) {
      spawns = new ArrayList<>();
      spawns.add(
          new SpawnResult.Builder()
              .setStatus(SpawnResult.Status.SUCCESS)
              .setRunnerName("c")
              .build());
      stats.countActionResult(ActionResult.create(spawns));
      stats.incrementActionCount();
    }
    assertThat(SpawnStats.convertSummaryToString(stats.getSummary()))
        .isEqualTo("6 processes: 1 a, 2 b, 3 c.");
  }

  @Test
  public void reverseOrder() {
    for (int i = 0; i < 3; i++) {
      ArrayList<SpawnResult> spawns = new ArrayList<>();
      spawns.add(
          new SpawnResult.Builder()
              .setStatus(SpawnResult.Status.SUCCESS)
              .setRunnerName("a")
              .build());
      stats.countActionResult(ActionResult.create(spawns));
      stats.incrementActionCount();
    }

    for (int i = 0; i < 2; i++) {
      ArrayList<SpawnResult> spawns = new ArrayList<>();
      spawns.add(
          new SpawnResult.Builder()
              .setStatus(SpawnResult.Status.SUCCESS)
              .setRunnerName("b")
              .build());
      stats.countActionResult(ActionResult.create(spawns));
      stats.incrementActionCount();
    }

    ArrayList<SpawnResult> spawns = new ArrayList<>();
    spawns.add(
        new SpawnResult.Builder().setStatus(SpawnResult.Status.SUCCESS).setRunnerName("c").build());
    stats.countActionResult(ActionResult.create(spawns));
    stats.incrementActionCount();

    assertThat(SpawnStats.convertSummaryToString(stats.getSummary()))
        .isEqualTo("6 processes: 3 a, 2 b, 1 c.");
  }

  @Test
  public void cacheFirst() {
    for (int i = 0; i < 3; i++) {
      ArrayList<SpawnResult> spawns = new ArrayList<>();
      spawns.add(
          new SpawnResult.Builder()
              .setStatus(SpawnResult.Status.SUCCESS)
              .setRunnerName("a")
              .build());
      stats.countActionResult(ActionResult.create(spawns));
      stats.incrementActionCount();
    }

    for (int i = 0; i < 2; i++) {
      ArrayList<SpawnResult> spawns = new ArrayList<>();
      spawns.add(
          new SpawnResult.Builder()
              .setStatus(SpawnResult.Status.SUCCESS)
              .setRunnerName("b")
              .build());
      stats.countActionResult(ActionResult.create(spawns));
      stats.incrementActionCount();
    }

    ArrayList<SpawnResult> spawns = new ArrayList<>();
    spawns.add(
        new SpawnResult.Builder()
            .setStatus(SpawnResult.Status.SUCCESS)
            .setRunnerName("remote cache hit")
            .build());
    stats.countActionResult(ActionResult.create(spawns));
    stats.incrementActionCount();

    spawns = new ArrayList<>();
    spawns.add(
        new SpawnResult.Builder().setStatus(SpawnResult.Status.SUCCESS).setRunnerName("c").build());
    stats.countActionResult(ActionResult.create(spawns));
    stats.incrementActionCount();

    assertThat(SpawnStats.convertSummaryToString(stats.getSummary()))
        .isEqualTo("7 processes: 1 remote cache hit, 3 a, 2 b, 1 c.");
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
    stats.incrementActionCount();
    assertThat(SpawnStats.convertSummaryToString(stats.getSummary()))
        .isEqualTo("1 process: 1 abc.");
  }

  @Test
  public void actionManySpawn() {
    // One action with multiple spawns - should count as 1 action with 3 spawns

    ArrayList<SpawnResult> spawns = new ArrayList<>();
    spawns.add(rA);
    spawns.add(rA);
    spawns.add(rA);

    stats.countActionResult(ActionResult.create(spawns));
    stats.incrementActionCount();
    assertThat(SpawnStats.convertSummaryToString(stats.getSummary()))
        .isEqualTo("1 process: 3 abc.");
  }

  @Test
  public void actionManySpawnMixed() {
    // One action with multiple spawns of different runners

    ArrayList<SpawnResult> spawns = new ArrayList<>();
    spawns.add(rA);
    spawns.add(rA);
    spawns.add(rB);

    stats.countActionResult(ActionResult.create(spawns));
    stats.incrementActionCount();
    assertThat(SpawnStats.convertSummaryToString(stats.getSummary()))
        .isEqualTo("1 process: 2 abc, 1 cde.");
  }

  @Test
  public void actionManyActionsMixed() {
    // Five actions:
    // Action 1: 1 spawn (abc)
    // Action 2: 2 spawns (abc, abc)
    // Action 3: 3 spawns (abc, abc, cde)
    // Action 4: 3 spawns (abc, abc, cde)
    // Action 5: 3 spawns (abc, abc, cde)

    ArrayList<SpawnResult> spawns = new ArrayList<>();
    spawns.add(rA);
    stats.countActionResult(ActionResult.create(spawns));
    stats.incrementActionCount();

    spawns = new ArrayList<>();
    spawns.add(rA);
    spawns.add(rA);
    stats.countActionResult(ActionResult.create(spawns));
    stats.incrementActionCount();

    spawns = new ArrayList<>();
    spawns.add(rA);
    spawns.add(rA);
    spawns.add(rB);
    stats.countActionResult(ActionResult.create(spawns));
    stats.incrementActionCount();
    stats.countActionResult(ActionResult.create(spawns));
    stats.incrementActionCount();
    stats.countActionResult(ActionResult.create(spawns));
    stats.incrementActionCount();

    assertThat(SpawnStats.convertSummaryToString(stats.getSummary()))
        .isEqualTo("5 processes: 9 abc, 3 cde.");
  }

  @Test
  public void onlyInternal() {
    stats.incrementActionCount();
    assertThat(SpawnStats.convertSummaryToString(stats.getSummary()))
        .isEqualTo("1 process: 1 internal.");
  }

  @Test
  public void orderCacheInternalRest() {
    for (int i = 0; i < 3; i++) {
      ArrayList<SpawnResult> spawns = new ArrayList<>();
      spawns.add(
          new SpawnResult.Builder()
              .setStatus(SpawnResult.Status.SUCCESS)
              .setRunnerName("a")
              .build());
      stats.countActionResult(ActionResult.create(spawns));
      stats.incrementActionCount();
    }

    for (int i = 0; i < 2; i++) {
      ArrayList<SpawnResult> spawns = new ArrayList<>();
      spawns.add(
          new SpawnResult.Builder()
              .setStatus(SpawnResult.Status.SUCCESS)
              .setRunnerName("b")
              .build());
      stats.countActionResult(ActionResult.create(spawns));
      stats.incrementActionCount();
    }

    ArrayList<SpawnResult> spawns = new ArrayList<>();
    spawns.add(
        new SpawnResult.Builder()
            .setStatus(SpawnResult.Status.SUCCESS)
            .setRunnerName("remote cache hit")
            .build());
    stats.countActionResult(ActionResult.create(spawns));
    stats.incrementActionCount();

    spawns = new ArrayList<>();
    spawns.add(
        new SpawnResult.Builder().setStatus(SpawnResult.Status.SUCCESS).setRunnerName("c").build());
    stats.countActionResult(ActionResult.create(spawns));
    stats.incrementActionCount();

    for (int i = 0; i < 2; i++) {
      spawns = new ArrayList<>();
      spawns.add(
          new SpawnResult.Builder()
              .setStatus(SpawnResult.Status.SUCCESS)
              .setRunnerName("z")
              .build());
      stats.countActionResult(ActionResult.create(spawns));
      stats.incrementActionCount();
    }

    // Add 2 internal actions (no spawns)
    for (int i = 0; i < 2; i++) {
      stats.incrementActionCount();
    }

    assertThat(SpawnStats.convertSummaryToString(stats.getSummary()))
        .isEqualTo("11 processes: 1 remote cache hit, 2 internal, 3 a, 2 b, 1 c, 2 z.");
  }

  private final SpawnResult rC =
      new SpawnResult.Builder()
          .setStatus(SpawnResult.Status.SUCCESS)
          .setSpawnMetrics(SpawnMetrics.Builder.forExec(SpawnMetrics.ExecKind.OTHER).build())
          .setRunnerName("fgh")
          .build();

  @Test
  public void getExecKindDefined() {
    ArrayList<SpawnResult> spawns = new ArrayList<>();
    spawns.add(rC);
    stats.countActionResult(ActionResult.create(spawns));
    assertThat(stats.getExecKindFor("fgh")).isEqualTo(SpawnMetrics.ExecKind.OTHER.toString());
  }

  @Test
  public void getExecKindNotDefined() {
    var unused = stats.getSummary();
    assertThat(stats.getExecKindFor("total")).isNull();
    assertThat(stats.getExecKindFor("internal")).isNull();
  }

  @Test
  public void internalCountWithMultipleSpawnsPerAction() {
    // Action 1: 3 spawns (counts as 1 non-internal action)
    ArrayList<SpawnResult> spawnsA = new ArrayList<>();
    spawnsA.add(rA);
    spawnsA.add(rA);
    spawnsA.add(rA);
    stats.countActionResult(ActionResult.create(spawnsA));
    stats.incrementActionCount();

    // Action 2: 2 spawns (counts as 1 non-internal action)
    ArrayList<SpawnResult> spawnsB = new ArrayList<>();
    spawnsB.add(rB);
    spawnsB.add(rB);
    stats.countActionResult(ActionResult.create(spawnsB));
    stats.incrementActionCount();

    // Action 3: internal action (no spawns)
    stats.incrementActionCount();

    assertThat(SpawnStats.convertSummaryToString(stats.getSummary()))
        .isEqualTo("3 processes: 1 internal, 3 abc, 2 cde.");
  }
}
