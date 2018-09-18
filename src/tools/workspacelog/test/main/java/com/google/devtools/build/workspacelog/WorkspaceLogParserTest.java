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
package com.google.devtools.build.workspacelog;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.bazel.debug.proto.WorkspaceLogProtos.WorkspaceEvent;
import com.google.devtools.build.workspacelog.WorkspaceLogParser.ExcludingLogParser;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Testing WorkspaceLogParser */
@RunWith(JUnit4.class)
public final class WorkspaceLogParserTest {

  private InputStream toInputStream(List<WorkspaceEvent> list) throws Exception {
    ByteArrayOutputStream bos = new ByteArrayOutputStream();
    for (WorkspaceEvent event : list) {
      event.writeDelimitedTo(bos);
    }

    return new ByteArrayInputStream(bos.toByteArray());
  }

  @Test
  public void getNextEmpty() throws Exception {
    ExcludingLogParser p =
        new ExcludingLogParser(toInputStream(new ArrayList<WorkspaceEvent>()), null);
    assertThat(p.getNext()).isNull();
  }

  @Test
  public void getNextEmptyWithExclusions() throws Exception {
    ExcludingLogParser p =
        new ExcludingLogParser(
            toInputStream(new ArrayList<WorkspaceEvent>()), new HashSet<>(Arrays.asList("a", "b")));
    assertThat(p.getNext()).isNull();
  }

  @Test
  public void getNextSingleExcluded1() throws Exception {
    WorkspaceEvent a = WorkspaceEvent.newBuilder().setRule("a").setLocation("SomeLocation").build();

    // Excluded by first exclusion
    ExcludingLogParser p =
        new ExcludingLogParser(
            toInputStream(Arrays.asList(a)), new HashSet<>(Arrays.asList("a", "b")));
    assertThat(p.getNext()).isNull();
  }

  @Test
  public void getNextSingleExcluded2() throws Exception {
    WorkspaceEvent a = WorkspaceEvent.newBuilder().setRule("a").setLocation("SomeLocation").build();

    // Excluded by second exclusion
    ExcludingLogParser p =
        new ExcludingLogParser(
            toInputStream(Arrays.asList(a)), new HashSet<>(Arrays.asList("b", "a")));
    assertThat(p.getNext()).isNull();
  }

  @Test
  public void getNextSingleIncluded() throws Exception {
    WorkspaceEvent a =
        WorkspaceEvent.newBuilder().setRule("onOnList").setLocation("SomeLocation").build();

    ExcludingLogParser p =
        new ExcludingLogParser(
            toInputStream(Arrays.asList(a)), new HashSet<>(Arrays.asList("b", "a")));
    assertThat(p.getNext()).isEqualTo(a);
    assertThat(p.getNext()).isNull();
  }

  @Test
  public void getNextSingleLongerList1() throws Exception {
    WorkspaceEvent a = WorkspaceEvent.newBuilder().setRule("a").setLocation("a1").build();
    WorkspaceEvent b = WorkspaceEvent.newBuilder().setRule("b").setLocation("b1").build();
    WorkspaceEvent c = WorkspaceEvent.newBuilder().setRule("a").setLocation("a2").build();
    WorkspaceEvent d = WorkspaceEvent.newBuilder().setRule("b").setLocation("b2").build();
    WorkspaceEvent e = WorkspaceEvent.newBuilder().setRule("d").build();

    ExcludingLogParser p =
        new ExcludingLogParser(
            toInputStream(Arrays.asList(a, b, c, d, e)), new HashSet<>(Arrays.asList("b", "a")));
    assertThat(p.getNext()).isEqualTo(e);
    assertThat(p.getNext()).isNull();
  }

  @Test
  public void getNextSingleLongerList2() throws Exception {
    WorkspaceEvent a = WorkspaceEvent.newBuilder().setRule("a").setLocation("a1").build();
    WorkspaceEvent b = WorkspaceEvent.newBuilder().setRule("b").setLocation("b1").build();
    WorkspaceEvent c = WorkspaceEvent.newBuilder().setRule("a").setLocation("a2").build();
    WorkspaceEvent d = WorkspaceEvent.newBuilder().setRule("b").setLocation("b2").build();
    WorkspaceEvent e = WorkspaceEvent.newBuilder().setRule("d").build();

    ExcludingLogParser p =
        new ExcludingLogParser(
            toInputStream(Arrays.asList(a, b, c, d, e)), new HashSet<>(Arrays.asList("d")));
    assertThat(p.getNext()).isEqualTo(a);
    assertThat(p.getNext()).isEqualTo(b);
    assertThat(p.getNext()).isEqualTo(c);
    assertThat(p.getNext()).isEqualTo(d);
    assertThat(p.getNext()).isNull();
  }
}
