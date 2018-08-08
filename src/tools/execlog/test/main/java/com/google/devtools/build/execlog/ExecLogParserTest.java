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
package com.google.devtools.build.execlog;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.exec.Protos.SpawnExec;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Testing ExecLogParser */
@RunWith(JUnit4.class)
public final class ExecLogParserTest {

  private InputStream toInputStream(List<SpawnExec> list) throws Exception {
    ByteArrayOutputStream bos = new ByteArrayOutputStream();
    for (SpawnExec spawnExec : list) {
      spawnExec.writeDelimitedTo(bos);
    }

    return new ByteArrayInputStream(bos.toByteArray());
  }

  @Test
  public void getNextEmpty() throws Exception {
    ExecLogParser p = new ExecLogParser(toInputStream(new ArrayList<SpawnExec>()), null);
    assertThat(p.getNext()).isNull();
  }

  @Test
  public void getNextEmptyWithRunner() throws Exception {
    ExecLogParser p = new ExecLogParser(toInputStream(new ArrayList<SpawnExec>()), "local");
    assertThat(p.getNext()).isNull();
  }

  @Test
  public void getNextSingleSpawn() throws Exception {
    SpawnExec e = SpawnExec.newBuilder().setRunner("runs").addCommandArgs("command").build();
    ExecLogParser p = new ExecLogParser(toInputStream(Arrays.asList(e)), null);
    assertThat(p.getNext()).isEqualTo(e);
    assertThat(p.getNext()).isNull();
  }

  @Test
  public void getNextSingleSpawnRunnerMatch() throws Exception {
    SpawnExec e = SpawnExec.newBuilder().setRunner("runs").addCommandArgs("command").build();
    ExecLogParser p = new ExecLogParser(toInputStream(Arrays.asList(e)), "runs");
    assertThat(p.getNext()).isEqualTo(e);
    assertThat(p.getNext()).isNull();
  }

  @Test
  public void getNextSingleSpawnRunnerNoMatch() throws Exception {
    SpawnExec e = SpawnExec.newBuilder().setRunner("runs").addCommandArgs("command").build();
    ExecLogParser p = new ExecLogParser(toInputStream(Arrays.asList(e)), "run");
    assertThat(p.getNext()).isNull();
  }

  @Test
  public void getNextManyMatches() throws Exception {
    SpawnExec e = SpawnExec.newBuilder().setRunner("run1").addCommandArgs("com1").build();
    SpawnExec e2 = SpawnExec.newBuilder().setRunner("r").addCommandArgs("com2").build();
    SpawnExec e3 = SpawnExec.newBuilder().setRunner("run1").addCommandArgs("com3").build();
    SpawnExec e4 = SpawnExec.newBuilder().setRunner("run1").addCommandArgs("com4").build();
    SpawnExec e5 = SpawnExec.newBuilder().setRunner("ru").addCommandArgs("com5").build();
    ExecLogParser p = new ExecLogParser(toInputStream(Arrays.asList(e, e2, e3, e4, e5)), "run1");
    assertThat(p.getNext()).isEqualTo(e);
    assertThat(p.getNext()).isEqualTo(e3);
    assertThat(p.getNext()).isEqualTo(e4);
    assertThat(p.getNext()).isNull();
  }

  @Test
  public void getNextManyMatches1() throws Exception {
    SpawnExec e = SpawnExec.newBuilder().setRunner("run1").addCommandArgs("com1").build();
    SpawnExec e2 = SpawnExec.newBuilder().setRunner("r").addCommandArgs("com2").build();
    SpawnExec e3 = SpawnExec.newBuilder().setRunner("run1").addCommandArgs("com3").build();
    SpawnExec e4 = SpawnExec.newBuilder().setRunner("run1").addCommandArgs("com4").build();
    SpawnExec e5 = SpawnExec.newBuilder().setRunner("ru").addCommandArgs("com5").build();
    ExecLogParser p = new ExecLogParser(toInputStream(Arrays.asList(e, e2, e3, e4, e5)), "r");
    assertThat(p.getNext()).isEqualTo(e2);
    assertThat(p.getNext()).isNull();
  }

  @Test
  public void getNextManyMatches2() throws Exception {
    SpawnExec e = SpawnExec.newBuilder().setRunner("run1").addCommandArgs("com1").build();
    SpawnExec e2 = SpawnExec.newBuilder().setRunner("r").addCommandArgs("com2").build();
    SpawnExec e3 = SpawnExec.newBuilder().setRunner("run1").addCommandArgs("com3").build();
    SpawnExec e4 = SpawnExec.newBuilder().setRunner("run1").addCommandArgs("com4").build();
    SpawnExec e5 = SpawnExec.newBuilder().setRunner("ru").addCommandArgs("com5").build();
    ExecLogParser p = new ExecLogParser(toInputStream(Arrays.asList(e, e2, e3, e4, e5)), "ru");
    assertThat(p.getNext()).isEqualTo(e5);
    assertThat(p.getNext()).isNull();
  }

  @Test
  public void getNextManyButNoMatch() throws Exception {
    SpawnExec e = SpawnExec.newBuilder().setRunner("run1").addCommandArgs("com1").build();
    SpawnExec e2 = SpawnExec.newBuilder().setRunner("r").addCommandArgs("com2").build();
    SpawnExec e3 = SpawnExec.newBuilder().setRunner("run1").addCommandArgs("com3").build();
    SpawnExec e4 = SpawnExec.newBuilder().setRunner("run1").addCommandArgs("com4").build();
    SpawnExec e5 = SpawnExec.newBuilder().setRunner("ru").addCommandArgs("com5").build();
    ExecLogParser p = new ExecLogParser(toInputStream(Arrays.asList(e, e2, e3, e4, e5)), "none");
    assertThat(p.getNext()).isNull();
  }

  @Test
  public void getNextManyNoMatcher() throws Exception {
    SpawnExec e = SpawnExec.newBuilder().setRunner("run1").addCommandArgs("com1").build();
    SpawnExec e2 = SpawnExec.newBuilder().setRunner("r").addCommandArgs("com2").build();
    SpawnExec e3 = SpawnExec.newBuilder().setRunner("run1").addCommandArgs("com3").build();
    SpawnExec e4 = SpawnExec.newBuilder().setRunner("run1").addCommandArgs("com4").build();
    SpawnExec e5 = SpawnExec.newBuilder().setRunner("ru").addCommandArgs("com5").build();
    ExecLogParser p = new ExecLogParser(toInputStream(Arrays.asList(e, e2, e3, e4, e5)), null);
    assertThat(p.getNext()).isEqualTo(e);
    assertThat(p.getNext()).isEqualTo(e2);
    assertThat(p.getNext()).isEqualTo(e3);
    assertThat(p.getNext()).isEqualTo(e4);
    assertThat(p.getNext()).isEqualTo(e5);
    assertThat(p.getNext()).isNull();
  }
}
