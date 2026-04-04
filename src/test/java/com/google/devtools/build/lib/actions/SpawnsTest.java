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

package com.google.devtools.build.lib.actions;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.exec.util.SpawnBuilder;
import java.time.Duration;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link Spawns}. */
@RunWith(JUnit4.class)
public final class SpawnsTest {

  @Test
  public void getTimeout_noTimeout_returnsZero() throws Exception {
    Spawn spawn = new SpawnBuilder("cmd").build();
    assertThat(Spawns.getTimeout(spawn)).isEqualTo(Duration.ZERO);
  }

  @Test
  public void getTimeout_keyValueSeconds_returnsDuration() throws Exception {
    Spawn spawn = new SpawnBuilder("cmd").withExecutionInfo("timeout", "42").build();
    assertThat(Spawns.getTimeout(spawn)).isEqualTo(Duration.ofSeconds(42));
  }

  @Test
  public void getTimeout_tagFormatSeconds_returnsDuration() throws Exception {
    Spawn spawn = new SpawnBuilder("cmd").withExecutionInfo("timeout:42", "").build();
    assertThat(Spawns.getTimeout(spawn)).isEqualTo(Duration.ofSeconds(42));
  }

  @Test
  public void getTimeout_tagFormatMinutes_returnsDuration() throws Exception {
    Spawn spawn = new SpawnBuilder("cmd").withExecutionInfo("timeout:5m", "").build();
    assertThat(Spawns.getTimeout(spawn)).isEqualTo(Duration.ofMinutes(5));
  }

  @Test
  public void getTimeout_tagFormatHours_returnsDuration() throws Exception {
    Spawn spawn = new SpawnBuilder("cmd").withExecutionInfo("timeout:1h", "").build();
    assertThat(Spawns.getTimeout(spawn)).isEqualTo(Duration.ofHours(1));
  }

  @Test
  public void getTimeout_keyValueDurationUnit_returnsDuration() throws Exception {
    Spawn spawn = new SpawnBuilder("cmd").withExecutionInfo("timeout", "30").build();
    assertThat(Spawns.getTimeout(spawn)).isEqualTo(Duration.ofSeconds(30));
  }

  @Test
  public void getTimeout_duplicateTakesLarger() throws Exception {
    // When both key-value and tag formats are present, takes the more permissive (larger) value.
    Spawn spawn =
        new SpawnBuilder("cmd")
            .withExecutionInfo("timeout", "60")
            .withExecutionInfo("timeout:120", "")
            .build();
    assertThat(Spawns.getTimeout(spawn)).isEqualTo(Duration.ofSeconds(120));
  }

  @Test
  public void getTimeout_duplicateTakesLarger_reversed() throws Exception {
    // Same as above but with larger value in key-value format.
    Spawn spawn =
        new SpawnBuilder("cmd")
            .withExecutionInfo("timeout", "300")
            .withExecutionInfo("timeout:60", "")
            .build();
    assertThat(Spawns.getTimeout(spawn)).isEqualTo(Duration.ofSeconds(300));
  }

  @Test
  public void getTimeout_invalidTagValue_throwsExecException() {
    Spawn spawn = new SpawnBuilder("cmd").withExecutionInfo("timeout:abc", "").build();
    assertThrows(ExecException.class, () -> Spawns.getTimeout(spawn));
  }

  @Test
  public void getTimeout_invalidKeyValue_throwsExecException() {
    Spawn spawn = new SpawnBuilder("cmd").withExecutionInfo("timeout", "not_a_number").build();
    assertThrows(ExecException.class, () -> Spawns.getTimeout(spawn));
  }

  @Test
  public void getTimeout_withDefaultTimeout_usesDefault() throws Exception {
    Spawn spawn = new SpawnBuilder("cmd").build();
    assertThat(Spawns.getTimeout(spawn, Duration.ofSeconds(99)))
        .isEqualTo(Duration.ofSeconds(99));
  }

  @Test
  public void getTimeout_withDefaultTimeout_overriddenByExecInfo() throws Exception {
    Spawn spawn = new SpawnBuilder("cmd").withExecutionInfo("timeout", "42").build();
    assertThat(Spawns.getTimeout(spawn, Duration.ofSeconds(99)))
        .isEqualTo(Duration.ofSeconds(42));
  }
}
